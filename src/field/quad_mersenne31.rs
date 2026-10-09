//! The quadratic extension GF((2^31 − 1)^2), the QM31 field.
//!
//! [`QuadMersenne31`] is the extension row
//! `FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>`: pairs
//! `[re, im]` of [`Mersenne31`] limbs with `i² = −1`, under the relation
//! `t^2 + 1` that [`ComplexMersenne31`] names. Since `p = 2³¹ − 1 ≡ 3
//! (mod 4)`, `−1` is a quadratic non-residue, so `Fₚ[i]/(i²+1)` is a field.
//! The raw word is `[u32; 2]`, real limb first, and the stable byte form is
//! the interleaved little-endian pair `[re: u32, im: u32]`, 8 bytes per
//! element.
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
//! const I: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw([0, 1]);
//! const _: () = assert!(I.square().to_raw()[0] == 0x7FFF_FFFE);
//! const _: () = assert!(I.square().to_raw()[1] == 0);
//!
//! // Known product, pinned against the frozen M31 known answer
//! // a² = 0x71C7_1C71: (a + a·i)² = 0 + 2a²·i with 2a² mod p = 0x638E_38E3.
//! const A: Elem<QuadMersenne31> = Elem::<QuadMersenne31>::from_raw([0x5555_5555, 0x5555_5555]);
//! const _: () = assert!(A.mul(A).to_raw()[0] == 0);
//! const _: () = assert!(A.mul(A).to_raw()[1] == 0x638E_38E3);
//!
//! // Division total
//! const _: () = assert!(A.div(Elem::<QuadMersenne31>::ZERO).is_zero());
//!
//! // Generator has full order p²−1
//! assert_eq!(
//!     Elem::<QuadMersenne31>::GENERATOR.pow(0x3FFF_FFFF_0000_0000),
//!     Elem::<QuadMersenne31>::ONE
//! );
//! ```

use super::extension::{Components, Extension, ExtensionSpec, Relation};
use super::finite_field::FiniteField;
use super::mersenne31;
use super::{Elem, HasGenerator, Mersenne31, Validate};

/// The base modulus `p = 2³¹ − 1`.
pub const MODULUS: u32 = mersenne31::MODULUS;

/// The defining data of QM31: [`Mersenne31`] extended by `t^2 + 1`, with
/// one base limb per component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ComplexMersenne31;

impl ExtensionSpec<2> for ComplexMersenne31 {
    type Base = Mersenne31;
    type Storage = Components;
    const NAME: &'static str = "GF((2^31 - 1)^2)";
    // `t^2 + 1`: constant coefficient one, linear coefficient zero.
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}

/// GF((2³¹ − 1)²), the QM31 field, as an extension row over
/// [`Mersenne31`].
pub type QuadMersenne31 = FiniteField<0x7FFF_FFFF, 2, Extension<ComplexMersenne31, 2>>;

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

// The shared scalar bodies: the row's `Field` arithmetic and its const
// surface both run these.

/// Canonical limbs.
#[inline]
#[must_use]
pub(crate) const fn raw_canonical(value: [u32; 2]) -> [u32; 2] {
    [limb_canonical(value[0]), limb_canonical(value[1])]
}

/// Raw sum.
#[inline]
#[must_use]
pub(crate) const fn raw_add(left: [u32; 2], right: [u32; 2]) -> [u32; 2] {
    [limb_add(left[0], right[0]), limb_add(left[1], right[1])]
}

/// Raw difference.
#[inline]
#[must_use]
pub(crate) const fn raw_sub(left: [u32; 2], right: [u32; 2]) -> [u32; 2] {
    [limb_sub(left[0], right[0]), limb_sub(left[1], right[1])]
}

/// Raw additive inverse.
#[inline]
#[must_use]
pub(crate) const fn raw_neg(value: [u32; 2]) -> [u32; 2] {
    [limb_neg(value[0]), limb_neg(value[1])]
}

/// Raw product `(a+bi)(c+di) = (ac−bd)+(ad+bc)i`.
#[inline]
#[must_use]
pub(crate) const fn raw_mul(left: [u32; 2], right: [u32; 2]) -> [u32; 2] {
    let ac = limb_mul(left[0], right[0]);
    let bd = limb_mul(left[1], right[1]);
    let ad = limb_mul(left[0], right[1]);
    let bc = limb_mul(left[1], right[0]);
    [limb_sub(ac, bd), limb_add(ad, bc)]
}

/// Raw square `(a+bi)² = (a²−b²) + 2ab·i`: three base multiplies.
#[inline]
#[must_use]
pub(crate) const fn raw_square(value: [u32; 2]) -> [u32; 2] {
    let a2 = limb_mul(value[0], value[0]);
    let b2 = limb_mul(value[1], value[1]);
    let ab = limb_mul(value[0], value[1]);
    [limb_sub(a2, b2), limb_add(ab, ab)]
}

/// The norm `a² + b²` in the base field, canonical.
#[inline]
#[must_use]
const fn raw_norm(value: [u32; 2]) -> u32 {
    let re2 = limb_mul(value[0], value[0]);
    let im2 = limb_mul(value[1], value[1]);
    limb_add(re2, im2)
}

/// Raw inverse via conjugate over norm, zero to zero.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub(crate) const fn raw_inv(value: [u32; 2]) -> [u32; 2] {
    let norm = raw_norm(value);
    if norm == 0 {
        return [0, 0];
    }
    let norm_inv = super::powmod(norm as u64, (MODULUS - 2) as u64, MODULUS as u64) as u32;
    [
        limb_mul(value[0], norm_inv),
        limb_mul(limb_neg(value[1]), norm_inv),
    ]
}

/// Raw power by square-and-multiply, over canonical limbs.
const fn raw_pow(mut base: [u32; 2], mut exponent: u64) -> [u32; 2] {
    let mut result = [1, 0];
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = raw_mul(result, base);
        }
        base = raw_square(base);
        exponent >>= 1;
    }
    result
}

/// Whether `[re, im]` is the multiplicative identity.
const fn is_raw_one(value: [u32; 2]) -> bool {
    limb_canonical(value[0]) == 1 && limb_canonical(value[1]) == 0
}

/// Prime factors of the group order `p² − 1`, for the generator check.
const GROUP_FACTORS: [u64; 7] = [2, 3, 7, 11, 31, 151, 331];

/// Whether `[re, im]` has full multiplicative order `p² − 1`.
const fn is_primitive(candidate: [u32; 2]) -> bool {
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

impl HasGenerator for QuadMersenne31 {
    const GENERATOR_RAW: [u32; 2] = {
        assert!(
            is_primitive([1, 12]),
            "QuadMersenne31 generator does not have full order"
        );
        [1, 12]
    };
}

impl Elem<QuadMersenne31> {
    /// The imaginary unit `i` with `i² = −1`.
    pub const I: Self = Elem { raw: [0, 1] };

    /// Wrap raw limbs `[re, im]`, reducing each into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: [u32; 2]) -> Self {
        let () = Validate::<QuadMersenne31>::OK;
        Elem {
            raw: raw_canonical(value),
        }
    }

    /// Decode from stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        let re = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let im = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Self::from_raw([re, im])
    }

    /// Encode to stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 8] {
        let re = self.raw[0].to_le_bytes();
        let im = self.raw[1].to_le_bytes();
        [re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]
    }

    /// Build from base-field components `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn from_components(components: [Elem<Mersenne31>; 2]) -> Self {
        Elem {
            raw: raw_canonical([components[0].raw, components[1].raw]),
        }
    }

    /// Real and imaginary components `[re, im]`, as base-field elements.
    #[inline]
    #[must_use]
    pub const fn to_components(self) -> [Elem<Mersenne31>; 2] {
        [
            Elem::<Mersenne31> { raw: self.raw[0] },
            Elem::<Mersenne31> { raw: self.raw[1] },
        ]
    }

    /// Conjugate `a − bi`.
    #[inline]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Elem {
            raw: [self.raw[0], limb_neg(self.raw[1])],
        }
    }

    /// Norm `a² + b²` as an element of the base field.
    ///
    /// The norm lives in `GF(2^31 − 1)`, not in a raw lane; the returned
    /// value is canonical.
    #[inline]
    #[must_use]
    pub const fn norm(self) -> Elem<Mersenne31> {
        Elem::<Mersenne31> {
            raw: raw_norm(self.raw),
        }
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Elem {
            raw: raw_add(self.raw, rhs.raw),
        }
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Elem {
            raw: raw_sub(self.raw, rhs.raw),
        }
    }

    /// Additive inverse.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Elem {
            raw: raw_neg(self.raw),
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
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        Elem {
            raw: raw_square(self.raw),
        }
    }

    /// Multiplicative inverse via conjugate / norm. Maps zero to zero.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        Elem {
            raw: raw_inv(self.raw),
        }
    }

    /// Field division. `x / 0 == 0`.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if limb_canonical(rhs.raw[0]) == 0 && limb_canonical(rhs.raw[1]) == 0 {
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

    /// The canonical representative of this element: the element itself,
    /// since storage is canonical by construction.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        self
    }

    /// Whether this element is the additive identity.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.raw[0] == 0 && self.raw[1] == 0
    }

    /// Whether this element is the multiplicative identity.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.raw[0] == 1 && self.raw[1] == 0
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
        let generator =
            Elem::<QuadMersenne31>::from_raw(core::hint::black_box(QuadMersenne31::GENERATOR_RAW));
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
        for candidate in [[0, 0], [1, 0], [7, 0], [MODULUS - 1, 0], [0, 1]] {
            assert!(
                !is_primitive(core::hint::black_box(candidate)),
                "{candidate:?}"
            );
        }
    }
}
