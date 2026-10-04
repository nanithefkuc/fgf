//! The prime field GF(2^64 − 2^32 + 1), the Goldilocks field.
//!
//! Elements are 64-bit lanes reduced modulo `p = 2^64 − 2^32 + 1 =
//! 0xFFFF_FFFF_0000_0001`. Reduction of a 128-bit product uses the identity
//! `2^64 ≡ 2^32 − 1 (mod p)` (the split-fold that makes Goldilocks fast on
//! 32-bit-half integer SIMD).
//!
//! # Totality and canonicalization
//!
//! Every raw 64-bit lane is a legal input: [`Elem::from_raw`](crate::field::Elem)
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
//! use fgf::{Elem, Goldilocks};
//!
//! const X: Elem<Goldilocks> = Elem::<Goldilocks>::from_raw(0xFFFF_FFFF_0000_0000);
//! const _: () = assert!(X.mul(X).to_raw() == 0x0000_0000_0000_0001);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem<Goldilocks> = Elem::<Goldilocks>::from_raw(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x7FFF_FFFF_8000_0001);
//! const _: () = assert!(Elem::<Goldilocks>::from_raw(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::<Goldilocks>::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1 = 2^64 − 2^32.
//! assert_eq!(
//!     Elem::<Goldilocks>::GENERATOR.pow(0xFFFF_FFFF_0000_0000),
//!     Elem::<Goldilocks>::ONE
//! );
//! ```

use super::{Elem, Field, FieldBuffer, HasGenerator, PrimeCharacteristic};

/// The field modulus, `2^64 − 2^32 + 1`.
pub const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// `2^32 − 1`, the residue of `2^64` modulo `p` and the fold constant.
const EPSILON: u64 = 0xFFFF_FFFF;

/// Reduce a value already in `0..2^64` to the canonical range `0..p`.
///
/// A single conditional subtract suffices: any `u64` exceeds `p` by less than
/// `2^32`.
///
/// The `reduce*` family maps a machine integer into the residue range;
/// element storage is canonical by construction.
#[inline]
#[must_use]
pub const fn reduce(x: u64) -> u64 {
    if x >= MODULUS { x - MODULUS } else { x }
}

/// Reduce a full 128-bit product modulo `p`, returning a canonical lane.
///
/// Splits the high 64 bits at the 32-bit boundary and folds each part with
/// `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`, the Plonky2 Goldilocks reduction. The
/// constants are derived from the modulus, not copied from a comparable.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub const fn reduce_wide(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // 2^96 ≡ −1, so subtract the top 32 bits of the high word.
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };
    // 2^64 ≡ 2^32 − 1, so scale the low 32 bits of the high word by EPSILON.
    let t1 = x_hi_lo * EPSILON;
    let (res, carry) = t0.overflowing_add(t1);
    let t2 = res.wrapping_add(if carry { EPSILON } else { 0 });
    reduce(t2)
}

/// Marker type for GF(2^64 − 2^32 + 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Goldilocks;

/// Prime factors of the group order `p − 1`, for the generator check.
const GROUP_FACTORS: [u64; 6] = [2, 3, 5, 17, 257, 65537];

/// Whether `candidate` has full multiplicative order `p − 1`.
const fn is_primitive(candidate: u64) -> bool {
    let order = MODULUS - 1;
    if super::powmod(candidate, order, MODULUS) != 1 {
        return false;
    }
    let mut i = 0;
    while i < GROUP_FACTORS.len() {
        if super::powmod(candidate, order / GROUP_FACTORS[i], MODULUS) == 1 {
            return false;
        }
        i += 1;
    }
    true
}

impl Field for Goldilocks {
    type Raw = u64;
    type Characteristic = PrimeCharacteristic<MODULUS>;
    const NAME: &'static str = "GF(2^64 - 2^32 + 1)";
    const DEGREE: u32 = 1;
    const ORDER: u128 = MODULUS as u128;
    const ZERO_RAW: u64 = 0;
    const ONE_RAW: u64 = 1;
    const VALID: () = ();

    #[inline]
    fn canonical_raw(raw: u64) -> u64 {
        reduce(raw)
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn add_raw(left: u64, right: u64) -> u64 {
        let sum = reduce(left) as u128 + reduce(right) as u128;
        let modulus = MODULUS as u128;
        if sum >= modulus {
            (sum - modulus) as u64
        } else {
            sum as u64
        }
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn sub_raw(left: u64, right: u64) -> u64 {
        let sum = reduce(left) as u128 + MODULUS as u128 - reduce(right) as u128;
        let modulus = MODULUS as u128;
        if sum >= modulus {
            (sum - modulus) as u64
        } else {
            sum as u64
        }
    }

    #[inline]
    fn neg_raw(value: u64) -> u64 {
        let reduced = reduce(value);
        if reduced == 0 { 0 } else { MODULUS - reduced }
    }

    #[inline]
    fn mul_raw(left: u64, right: u64) -> u64 {
        reduce_wide(u128::from(reduce(left)) * u128::from(reduce(right)))
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn inv_raw(value: u64) -> u64 {
        if reduce(value) == 0 {
            return 0;
        }
        super::powmod(reduce(value), MODULUS - 2, MODULUS)
    }
}

impl HasGenerator for Goldilocks {
    const GENERATOR_RAW: u64 = {
        assert!(
            is_primitive(7),
            "Goldilocks generator does not have full order"
        );
        7
    };
}

impl FieldBuffer for Goldilocks {
    const BYTES: usize = 8;
    const STORAGE_BITS: u32 = 64;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let bytes: [u8; 8] = bytes
            .try_into()
            .expect("GF(2^64 - 2^32 + 1) element has the wrong byte width");
        Elem::<Self>::from_raw(u64::from_le_bytes(bytes))
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(
            bytes.len(),
            8,
            "GF(2^64 - 2^32 + 1) element has the wrong byte width"
        );
        bytes.copy_from_slice(&value.to_raw().to_le_bytes());
    }
}

impl Elem<Goldilocks> {
    /// Decode from the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self::from_raw(u64::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 8] {
        self.raw.to_le_bytes()
    }

    /// Wrap a raw lane, reducing it into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        let () = super::Validate::<Goldilocks>::OK;
        Elem { raw: reduce(value) }
    }

    /// Field addition.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    pub const fn add(self, rhs: Self) -> Self {
        let sum = reduce(self.raw) as u128 + reduce(rhs.raw) as u128;
        let modulus = MODULUS as u128;
        Elem {
            raw: if sum >= modulus {
                (sum - modulus) as u64
            } else {
                sum as u64
            },
        }
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    pub const fn sub(self, rhs: Self) -> Self {
        let sum = reduce(self.raw) as u128 + MODULUS as u128 - reduce(rhs.raw) as u128;
        let modulus = MODULUS as u128;
        Elem {
            raw: if sum >= modulus {
                (sum - modulus) as u64
            } else {
                sum as u64
            },
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

    /// Field multiplication, folding the 128-bit product modulo `p`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_lossless)]
    pub const fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: reduce_wide(self.raw as u128 * rhs.raw as u128),
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

impl core::fmt::Display for Elem<Goldilocks> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.raw)
    }
}
