//! The prime field GF(2^31 − 1), the Mersenne31 field.
//!
//! Elements are 32-bit lanes reduced modulo the Mersenne prime
//! `p = 2^31 − 1 = 0x7FFF_FFFF`. The reduction exploits `2^31 ≡ 1 (mod p)`, so
//! folding the high bit back into the low 31 bits (`lo + hi`) plus a single
//! conditional subtract canonicalizes any 32-bit value.
//!
//! # Totality and canonicalization
//!
//! Every raw 32-bit lane is a legal input: [`Elem::from_raw`](crate::field::Elem)
//! and [`FieldBuffer::decode`](crate::field::FieldBuffer) reduce it into the
//! canonical range `0..p` on the way in. Every arithmetic output is canonical
//! with no branch and no panic on out-of-range operands, the prime-field
//! analogue of the crate-wide `inv(0) == 0` convention. Storage is always
//! canonical, so equality, hashing, and ordering compare raw words directly.
//! Inspect the stored bits with [`Elem::to_raw`](crate::field::Elem::to_raw).
//!
//! Arithmetic is variable-time and not intended for secret data.
//!
//! ```
//! use fgf::{Elem, Mersenne31};
//!
//! // Known-answer product, pinned against the modular reduction.
//! const X: Elem<Mersenne31> = Elem::<Mersenne31>::from_raw(0x5555_5555);
//! const _: () = assert!(X.mul(X).to_raw() == 0x71C7_1C71);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem<Mersenne31> = Elem::<Mersenne31>::from_raw(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x4000_0000);
//! const _: () = assert!(Elem::<Mersenne31>::from_raw(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::<Mersenne31>::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1.
//! assert_eq!(
//!     Elem::<Mersenne31>::GENERATOR.pow(0x7FFF_FFFE),
//!     Elem::<Mersenne31>::ONE
//! );
//! ```

use super::{Elem, Field, FieldBuffer, HasGenerator, PrimeCharacteristic};

/// The field modulus, the Mersenne prime `2^31 − 1`.
pub const MODULUS: u32 = 0x7FFF_FFFF;

/// Reduce an arbitrary 32-bit lane to the canonical range `0..p`.
///
/// `2^31 ≡ 1 (mod p)`, so folding the top bit into the low 31 bits and one
/// conditional subtract suffices for any `u32`.
///
/// `reduce` maps a machine integer into the residue range; element storage
/// is canonical by construction.
#[inline]
#[must_use]
pub const fn reduce(x: u32) -> u32 {
    let s = (x & MODULUS) + (x >> 31);
    if s >= MODULUS { s - MODULUS } else { s }
}

/// Marker type for GF(2^31 − 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Mersenne31;

/// Prime factors of the group order `p − 1`, for the generator check.
const GROUP_FACTORS: [u64; 7] = [2, 3, 7, 11, 31, 151, 331];

/// Whether `candidate` has full multiplicative order `p − 1`.
const fn is_primitive(candidate: u64) -> bool {
    let order = (MODULUS - 1) as u64;
    if super::powmod(candidate, order, MODULUS as u64) != 1 {
        return false;
    }
    let mut i = 0;
    while i < GROUP_FACTORS.len() {
        if super::powmod(candidate, order / GROUP_FACTORS[i], MODULUS as u64) == 1 {
            return false;
        }
        i += 1;
    }
    true
}

impl Field for Mersenne31 {
    type Raw = u32;
    type Characteristic = PrimeCharacteristic<{ MODULUS as u64 }>;
    const NAME: &'static str = "GF(2^31 - 1)";
    const DEGREE: u32 = 1;
    const ORDER: u128 = MODULUS as u128;
    const ZERO_RAW: u32 = 0;
    const ONE_RAW: u32 = 1;
    const VALID: () = ();

    #[inline]
    fn canonical_raw(raw: u32) -> u32 {
        reduce(raw)
    }

    #[inline]
    fn add_raw(left: u32, right: u32) -> u32 {
        let sum = reduce(left) + reduce(right);
        if sum >= MODULUS { sum - MODULUS } else { sum }
    }

    #[inline]
    fn sub_raw(left: u32, right: u32) -> u32 {
        let sum = reduce(left) + (MODULUS - reduce(right));
        if sum >= MODULUS { sum - MODULUS } else { sum }
    }

    #[inline]
    fn neg_raw(value: u32) -> u32 {
        let reduced = reduce(value);
        if reduced == 0 { 0 } else { MODULUS - reduced }
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn mul_raw(left: u32, right: u32) -> u32 {
        let product = reduce(left) as u64 * reduce(right) as u64;
        let lo = (product as u32) & MODULUS;
        let hi = (product >> 31) as u32;
        reduce(lo + hi)
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn inv_raw(value: u32) -> u32 {
        if reduce(value) == 0 {
            return 0;
        }
        super::powmod(reduce(value) as u64, (MODULUS - 2) as u64, MODULUS as u64) as u32
    }
}

impl HasGenerator for Mersenne31 {
    const GENERATOR_RAW: u32 = {
        assert!(
            is_primitive(7),
            "Mersenne31 generator does not have full order"
        );
        7
    };
}

impl FieldBuffer for Mersenne31 {
    const BYTES: usize = 4;
    const STORAGE_BITS: u32 = 32;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let bytes: [u8; 4] = bytes
            .try_into()
            .expect("GF(2^31 - 1) element has the wrong byte width");
        Elem::<Self>::from_raw(u32::from_le_bytes(bytes))
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(
            bytes.len(),
            4,
            "GF(2^31 - 1) element has the wrong byte width"
        );
        bytes.copy_from_slice(&value.to_raw().to_le_bytes());
    }
}

impl Elem<Mersenne31> {
    /// Decode a raw little-endian lane, reducing it into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self::from_raw(u32::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        self.raw.to_le_bytes()
    }

    /// Wrap a raw lane, reducing it into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        let () = super::Validate::<Mersenne31>::OK;
        Elem { raw: reduce(value) }
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        let sum = reduce(self.raw) + reduce(rhs.raw);
        Elem {
            raw: if sum >= MODULUS { sum - MODULUS } else { sum },
        }
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        let sum = reduce(self.raw) + (MODULUS - reduce(rhs.raw));
        Elem {
            raw: if sum >= MODULUS { sum - MODULUS } else { sum },
        }
    }

    /// Additive inverse. `neg(0) == 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        let reduced = reduce(self.raw);
        Elem {
            raw: if reduced == 0 { 0 } else { MODULUS - reduced },
        }
    }

    /// Field multiplication, folding the 62-bit product with `2^31 ≡ 1`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    pub const fn mul(self, rhs: Self) -> Self {
        let product = reduce(self.raw) as u64 * reduce(rhs.raw) as u64;
        let lo = (product as u32) & MODULUS;
        let hi = (product >> 31) as u32;
        Elem {
            raw: reduce(lo + hi),
        }
    }

    /// Square.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse by Fermat's little theorem (`a^(p−2)`).
    ///
    /// Maps zero to zero by convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        self.pow((MODULUS - 2) as u128)
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if reduce(rhs.raw) == 0 {
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

impl core::fmt::Display for Elem<Mersenne31> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::{Elem, HasGenerator, MODULUS, Mersenne31, is_primitive};

    const fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    /// `g^k` has full order `p − 1` exactly when `gcd(k, p − 1) = 1`, so the
    /// check must accept and reject powers of the generator by that rule.
    #[test]
    fn primitivity_follows_generator_power_coprimality() {
        let generator =
            Elem::<Mersenne31>::from_raw(core::hint::black_box(Mersenne31::GENERATOR_RAW));
        for k in 1..=96u64 {
            let candidate = u64::from(generator.pow(u128::from(k)).to_raw());
            assert_eq!(
                is_primitive(core::hint::black_box(candidate)),
                gcd(k, u64::from(MODULUS - 1)) == 1,
                "7^{k}"
            );
        }
    }

    /// Zero has no multiplicative order; one and `−1` have orders 1 and 2.
    #[test]
    fn primitivity_rejects_degenerate_orders() {
        for candidate in [0, 1, u64::from(MODULUS - 1)] {
            assert!(
                !is_primitive(core::hint::black_box(candidate)),
                "{candidate}"
            );
        }
    }
}
