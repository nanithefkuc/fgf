//! The quadratic extension GF((2^31 − 1)^2), the QM31 field.
//!
//! Elements are pairs `(re, im)` of Mersenne31 lanes with `i² = −1`.
//! Since `p = 2³¹ − 1 ≡ 3 (mod 4)`, `−1` is a quadratic non-residue, so
//! `Fₚ[i]/(i²+1)` is a field. The stable byte form is the interleaved
//! little-endian pair `[re: u32, im: u32]`, 8 bytes per element.
//!
//! # Totality and canonicalization
//!
//! As with the prime fields, every raw limb bit pattern is a legal input and
//! [`Elem::from_raw`](crate::field::Elem) reduces each limb into canonical
//! storage on the way in. Every arithmetic output is canonical (`< p` per
//! limb). Storage is always canonical, so equality, hashing, and ordering
//! compare raw words directly. Inspect the stored limbs with
//! [`Elem::to_raw`](crate::field::Elem::to_raw).
//!
//! Arithmetic is variable-time and not for secret data.
//!
//! ```
//! use fgf::{Elem, QuadMersenne31};
//!
//! // i² = −1  →  (0 + i)² = (−1 + 0·i)
//! const I: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw(0, 1);
//! const _: () = assert!(I.square().to_raw().0 == 0x7FFF_FFFE);
//! const _: () = assert!(I.square().to_raw().1 == 0);
//!
//! // Known product, pinned against the frozen M31 known answer
//! // a² = 0x71C7_1C71: (a + a·i)² = 0 + 2a²·i with 2a² mod p = 0x638E_38E3.
//! const A: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw(0x5555_5555, 0x5555_5555);
//! const B: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw(0x5555_5555, 0x5555_5555);
//! const _: () = assert!(A.mul(B).to_raw().0 == 0);
//! const _: () = assert!(A.mul(B).to_raw().1 == 0x638E_38E3);
//!
//! // Division total
//! const _: () = assert!(A.div(Elem::<QuadMersenne31>::ZERO).to_raw().0 == 0);
//! const _: () = assert!(A.div(Elem::<QuadMersenne31>::ZERO).to_raw().1 == 0);
//!
//! // Generator has full order p²−1
//! assert_eq!(
//!     Elem::<QuadMersenne31>::GENERATOR.pow(0x3FFF_FFFF_0000_0000),
//!     Elem::<QuadMersenne31>::ONE
//! );
//! ```

use super::mersenne31;
use super::{Elem, Field, FieldBuffer, HasGenerator, PrimeCharacteristic};
use crate::field::Mersenne31;

/// The base modulus `p = 2³¹ − 1`.
pub const MODULUS: u32 = mersenne31::MODULUS;

/// Marker type for GF((2³¹ − 1)²).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct QuadMersenne31;

#[inline]
#[must_use]
const fn limb_canonical(x: u32) -> u32 {
    mersenne31::reduce(x)
}

#[inline]
#[must_use]
const fn limb_add(a: u32, b: u32) -> u32 {
    let a = limb_canonical(a);
    let b = limb_canonical(b);
    let s = a + b;
    if s >= MODULUS { s - MODULUS } else { s }
}

#[inline]
#[must_use]
const fn limb_sub(a: u32, b: u32) -> u32 {
    let a = limb_canonical(a);
    let b = limb_canonical(b);
    let s = a + (MODULUS - b);
    if s >= MODULUS { s - MODULUS } else { s }
}

#[inline]
#[must_use]
const fn limb_neg(a: u32) -> u32 {
    let a = limb_canonical(a);
    if a == 0 { 0 } else { MODULUS - a }
}

#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
const fn limb_mul(a: u32, b: u32) -> u32 {
    let a = limb_canonical(a) as u64;
    let b = limb_canonical(b) as u64;
    let prod = a * b;
    let lo = (prod as u32) & MODULUS;
    let hi = (prod >> 31) as u32;
    mersenne31::reduce(lo + hi)
}

/// Raw product `(a+bi)(c+di)`, over canonical limbs.
const fn raw_mul(left: (u32, u32), right: (u32, u32)) -> (u32, u32) {
    let ac = limb_mul(left.0, right.0);
    let bd = limb_mul(left.1, right.1);
    let ad = limb_mul(left.0, right.1);
    let bc = limb_mul(left.1, right.0);
    (limb_sub(ac, bd), limb_add(ad, bc))
}

/// Raw square `(a+bi)²`, over canonical limbs.
const fn raw_square(value: (u32, u32)) -> (u32, u32) {
    let a2 = limb_mul(value.0, value.0);
    let b2 = limb_mul(value.1, value.1);
    let ab = limb_mul(value.0, value.1);
    (limb_sub(a2, b2), limb_add(ab, ab))
}

/// Raw power by square-and-multiply, over canonical limbs.
const fn raw_pow(mut base: (u32, u32), mut exponent: u64) -> (u32, u32) {
    let mut result = (1, 0);
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = raw_mul(result, base);
        }
        base = raw_square(base);
        exponent >>= 1;
    }
    result
}

/// Whether `(re, im)` is the multiplicative identity.
const fn is_raw_one(value: (u32, u32)) -> bool {
    limb_canonical(value.0) == 1 && limb_canonical(value.1) == 0
}

/// Prime factors of the group order `p² − 1`, for the generator check.
const GROUP_FACTORS: [u64; 7] = [2, 3, 7, 11, 31, 151, 331];

/// Whether `(re, im)` has full multiplicative order `p² − 1`.
const fn is_primitive(candidate: (u32, u32)) -> bool {
    let order = (MODULUS as u64) * (MODULUS as u64) - 1;
    if !is_raw_one(raw_pow(candidate, order)) {
        return false;
    }
    let mut i = 0;
    while i < GROUP_FACTORS.len() {
        if is_raw_one(raw_pow(candidate, order / GROUP_FACTORS[i])) {
            return false;
        }
        i += 1;
    }
    true
}

impl Field for QuadMersenne31 {
    type Raw = (u32, u32);
    type Characteristic = PrimeCharacteristic<{ MODULUS as u64 }>;
    const NAME: &'static str = "GF((2^31 - 1)^2)";
    const DEGREE: u32 = 2;
    const ORDER: u128 = (MODULUS as u128) * (MODULUS as u128);
    const ZERO_RAW: (u32, u32) = (0, 0);
    const ONE_RAW: (u32, u32) = (1, 0);
    const VALID: () = ();

    #[inline]
    fn canonical_raw(raw: (u32, u32)) -> (u32, u32) {
        (limb_canonical(raw.0), limb_canonical(raw.1))
    }

    #[inline]
    fn add_raw(left: (u32, u32), right: (u32, u32)) -> (u32, u32) {
        (limb_add(left.0, right.0), limb_add(left.1, right.1))
    }

    #[inline]
    fn sub_raw(left: (u32, u32), right: (u32, u32)) -> (u32, u32) {
        (limb_sub(left.0, right.0), limb_sub(left.1, right.1))
    }

    #[inline]
    fn neg_raw(value: (u32, u32)) -> (u32, u32) {
        (limb_neg(value.0), limb_neg(value.1))
    }

    #[inline]
    fn mul_raw(left: (u32, u32), right: (u32, u32)) -> (u32, u32) {
        raw_mul(left, right)
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn inv_raw(value: (u32, u32)) -> (u32, u32) {
        let re2 = limb_mul(value.0, value.0);
        let im2 = limb_mul(value.1, value.1);
        let norm = limb_add(re2, im2);
        if norm == 0 {
            return (0, 0);
        }
        let norm_inv = super::powmod(norm as u64, (MODULUS - 2) as u64, MODULUS as u64) as u32;
        (
            limb_mul(value.0, norm_inv),
            limb_mul(limb_neg(value.1), norm_inv),
        )
    }
}

impl HasGenerator for QuadMersenne31 {
    const GENERATOR_RAW: (u32, u32) = {
        assert!(
            is_primitive((1, 12)),
            "QuadMersenne31 generator does not have full order"
        );
        (1, 12)
    };
}

impl FieldBuffer for QuadMersenne31 {
    const BYTES: usize = 8;
    const STORAGE_BITS: u32 = 64;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let bytes: [u8; 8] = bytes
            .try_into()
            .expect("QM31 element has the wrong byte width");
        Elem::<Self>::from_bytes(bytes)
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(bytes.len(), 8, "QM31 element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl Elem<QuadMersenne31> {
    /// The imaginary unit `i` with `i² = −1`.
    pub const I: Self = Elem { raw: (0, 1) };

    /// Decode from stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        let re = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let im = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Self::from_raw(re, im)
    }

    /// Encode to stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 8] {
        let re = self.raw.0.to_le_bytes();
        let im = self.raw.1.to_le_bytes();
        [re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]
    }

    /// Wrap raw limbs, reducing each into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_raw(re: u32, im: u32) -> Self {
        let () = super::Validate::<QuadMersenne31>::OK;
        Elem {
            raw: (limb_canonical(re), limb_canonical(im)),
        }
    }

    /// Real and imaginary components, as base-field elements.
    #[inline]
    #[must_use]
    pub const fn to_components(self) -> (Elem<Mersenne31>, Elem<Mersenne31>) {
        (
            Elem::<Mersenne31> { raw: self.raw.0 },
            Elem::<Mersenne31> { raw: self.raw.1 },
        )
    }

    /// Build from base-field components.
    #[inline]
    #[must_use]
    pub const fn from_components(re: Elem<Mersenne31>, im: Elem<Mersenne31>) -> Self {
        Elem {
            raw: (re.raw, im.raw),
        }
    }

    /// Conjugate `a − bi`.
    #[inline]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Elem {
            raw: (self.raw.0, limb_neg(self.raw.1)),
        }
    }

    /// Norm `a² + b²` as an element of the base field.
    ///
    /// The norm lives in `GF(2^31 − 1)`, not in a raw lane; the returned
    /// value is canonical.
    #[inline]
    #[must_use]
    pub const fn norm(self) -> Elem<Mersenne31> {
        let re2 = limb_mul(self.raw.0, self.raw.0);
        let im2 = limb_mul(self.raw.1, self.raw.1);
        Elem::<Mersenne31> {
            raw: limb_add(re2, im2),
        }
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Elem {
            raw: (
                limb_add(self.raw.0, rhs.raw.0),
                limb_add(self.raw.1, rhs.raw.1),
            ),
        }
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Elem {
            raw: (
                limb_sub(self.raw.0, rhs.raw.0),
                limb_sub(self.raw.1, rhs.raw.1),
            ),
        }
    }

    /// Additive inverse.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Elem {
            raw: (limb_neg(self.raw.0), limb_neg(self.raw.1)),
        }
    }

    /// Field multiplication `(a+bi)(c+di) = (ac−bd)+(ad+bc)i`.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: raw_mul(self.raw, rhs.raw),
        }
    }

    /// Square: `(a+bi)² = (a²−b²) + 2ab·i`.
    ///
    /// Three base multiplies instead of the four a general multiply costs;
    /// `2ab` is one modular add of `ab` with itself.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        Elem {
            raw: raw_square(self.raw),
        }
    }

    /// Multiplicative inverse via conjugate / norm.
    ///
    /// Maps zero to zero.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    pub const fn inv(self) -> Self {
        let re2 = limb_mul(self.raw.0, self.raw.0);
        let im2 = limb_mul(self.raw.1, self.raw.1);
        let norm = limb_add(re2, im2);
        if norm == 0 {
            return Self::ZERO;
        }
        let norm_inv = super::powmod(norm as u64, (MODULUS - 2) as u64, MODULUS as u64) as u32;
        Elem {
            raw: (
                limb_mul(self.raw.0, norm_inv),
                limb_mul(limb_neg(self.raw.1), norm_inv),
            ),
        }
    }

    /// Field division. `x / 0 == 0`.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if limb_canonical(rhs.raw.0) == 0 && limb_canonical(rhs.raw.1) == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power. `pow(_, 0) == ONE`.
    #[inline]
    #[must_use]
    pub const fn pow(self, mut exponent: u128) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }
}

impl core::fmt::Display for Elem<QuadMersenne31> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "({:#x}+{:#x}i)", self.raw.0, self.raw.1)
    }
}

#[cfg(test)]
mod tests {
    use super::{Elem, HasGenerator, MODULUS, QuadMersenne31, is_primitive};

    const fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    /// `g^k` has full order `p² − 1` exactly when `gcd(k, p² − 1) = 1`, so
    /// the check must accept and reject powers of the generator by that rule.
    #[test]
    fn primitivity_follows_generator_power_coprimality() {
        let (re, im) = core::hint::black_box(QuadMersenne31::GENERATOR_RAW);
        let generator = Elem::<QuadMersenne31>::from_raw(re, im);
        let order = u64::from(MODULUS) * u64::from(MODULUS) - 1;
        for k in 1..=48u64 {
            let candidate = generator.pow(u128::from(k)).to_raw();
            assert_eq!(
                is_primitive(core::hint::black_box(candidate)),
                gcd(k, order) == 1,
                "(1 + 12i)^{k}"
            );
        }
    }

    /// Zero has no multiplicative order; base-field elements and `i` have
    /// orders dividing `p − 1` and 4, far below `p² − 1`.
    #[test]
    fn primitivity_rejects_small_orders() {
        for candidate in [(0, 0), (1, 0), (7, 0), (MODULUS - 1, 0), (0, 1)] {
            assert!(
                !is_primitive(core::hint::black_box(candidate)),
                "{candidate:?}"
            );
        }
    }
}
