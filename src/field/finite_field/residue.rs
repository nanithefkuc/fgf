//! Ordinary-residue rows of the canonical field family.
//!
//! [`Residue`] is the family's prime presentation, implemented at degree one
//! for every prime `u64` characteristic: arithmetic is plain modular
//! arithmetic on one `u64`, with no fold tricks and no generator search. A
//! stored element is exactly its ordinary residue in `0..P`, so raw words,
//! mathematical residues, and [`Display`] output coincide.
//!
//! ```
//! use fgf::{Elem, FiniteField};
//!
//! type P = FiniteField<1_000_000_007, 1>;
//!
//! const X: Elem<P> = Elem::<P>::from_raw(1_000_000_006);
//! const INV: Elem<P> = X.inv();
//! const _: () = assert!(INV.to_raw() == 1_000_000_006);
//! const _: () = assert!(X.mul(INV).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::<P>::ZERO).to_raw() == 0);
//!
//! // Construction outside `0..P` is rejected instead of narrowed.
//! assert!(Elem::<P>::from_residue(1_000_000_007).is_err());
//! ```
//!
//! # Representation and layout
//!
//! One canonical `u64` per element. The stable byte lane is the smallest
//! little-endian lane holding every residue: characteristic two uses an
//! eight-byte lane, and an odd characteristic uses one, two, four, or
//! eight bytes as its residues fit within `2^8`, `2^16`, `2^32`, and
//! `u64`. `FieldBuffer::decode` zero-extends the lane to `u64` and
//! reduces it, so every bit pattern of the lane is accepted, and
//! `FieldBuffer::encode` always emits the canonical residue at the
//! row's lane width. These two methods are the byte API of the rows.
//!
//! # Totality and canonicalization
//!
//! Every raw lane is a legal input to construction and decoding, which
//! reduce it into `0..P` on the way in. Every arithmetic output is
//! canonical with no panic on out-of-range operands: by the crate-wide
//! convention `inv(0) == 0` and `x / 0 == 0`, and `0^0 == 1`. At `P = 2`
//! the Fermat exponent `P - 2` is zero, so the inverse body maps zero to
//! zero explicitly instead of exponentiating. Arithmetic is variable-time
//! and not intended for secret data.
//!
//! Prepared coefficients for the packed surface resolve the canonical
//! element into the row's one-word backend form: the canonical value
//! itself on byte, two-byte, and `P = 2` rows, the Shoup pair on
//! four-byte rows, and the Montgomery word on eight-byte rows at odd `P`.
//! Preparation allocates nothing, packed application allocates nothing,
//! and no workspace carries meaning between calls.

use crate::field::extension::Descriptor;
use crate::field::extension::descriptor::PrimePresentation;
use crate::field::{
    Elem, Field, FieldBuffer, PrimeCharacteristic, PrimeIdentity, RangeError, Validate,
};

use super::{FiniteField, FiniteFieldRepr, checked_order};

/// The ordinary-residue presentation: arithmetic modulo the characteristic.
///
/// The default presentation of [`FiniteField`](super::FiniteField),
/// implemented at degree one only, so `FiniteField<P, 1>` — spelled without
/// the presentation parameter — is the prime field GF(P). Field [`NAME`]:
/// `GF(p)`; this label: `residue`.
///
/// [`NAME`]: Field::NAME
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Residue;

impl super::private::Sealed for Residue {}

impl<const P: u64> FiniteFieldRepr<P, 1> for Residue {
    const NAME: &'static str = "residue";
    // The defining data of the residue presentation is the characteristic
    // itself; the shared metadata check validates it before any remainder
    // arithmetic can run.
    const VALID: () = {
        checked_order::<P, 1>();
    };
}

impl<const P: u64> super::private::Sealed for FiniteField<P, 1, Residue> {}

// ---------------------------------------------------------------------------
// Shared scalar bodies
// ---------------------------------------------------------------------------

/// `a + b` over canonical residues: one conditional subtraction.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
const fn add_body<const P: u64>(left: u64, right: u64) -> u64 {
    let sum = left as u128 + right as u128;
    let modulus = P as u128;
    if sum >= modulus {
        (sum - modulus) as u64
    } else {
        sum as u64
    }
}

/// `a - b` over canonical residues: borrow-safe.
const fn sub_body<const P: u64>(left: u64, right: u64) -> u64 {
    if left >= right {
        left - right
    } else {
        P - (right - left)
    }
}

/// `-a` over a canonical residue. `neg(0) == 0`.
const fn neg_body<const P: u64>(value: u64) -> u64 {
    if value == 0 { 0 } else { P - value }
}

/// `a * b` over canonical residues, through the widened product.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
const fn mul_body<const P: u64>(left: u64, right: u64) -> u64 {
    (left as u128 * right as u128 % P as u128) as u64
}

/// `a^e` over a canonical residue, by square-and-multiply.
const fn pow_body<const P: u64>(mut base: u64, mut exponent: u128) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_body::<P>(result, base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = mul_body::<P>(base, base);
        }
    }
    result
}

/// `a⁻¹` over a canonical residue, by Fermat's little theorem.
///
/// The explicit zero branch is what keeps the convention at `P = 2`, where
/// the Fermat exponent `P - 2` is zero.
#[allow(clippy::cast_lossless)]
const fn inv_body<const P: u64>(value: u64) -> u64 {
    if value == 0 {
        0
    } else {
        pow_body::<P>(value, P as u128 - 2)
    }
}

/// The canonical residue of any raw word.
const fn reduce_body<const P: u64>(value: u64) -> u64 {
    value % P
}

/// The lane width in bytes of one characteristic: the smallest
/// little-endian lane holding every residue, with characteristic two
/// keeping the eight-byte lane. Total over every `u64`, including the
/// values other code rejects as characteristics.
const fn lane_bytes(characteristic: u64) -> usize {
    if characteristic == 2 {
        8
    } else if characteristic <= (1 << 8) {
        1
    } else if characteristic <= (1 << 16) {
        2
    } else if characteristic <= (1 << 32) {
        4
    } else {
        8
    }
}

/// The little-endian word one lane spells, zero-extended to `u64`. The
/// caller bounds the lane to the row's byte width.
#[allow(clippy::cast_lossless)]
const fn read_lane(lane: &[u8]) -> u64 {
    let mut word = 0;
    let mut index = lane.len();
    while index > 0 {
        index -= 1;
        word = (word << 8) | (lane[index] as u64);
    }
    word
}

// ---------------------------------------------------------------------------
// Field row
// ---------------------------------------------------------------------------

impl<const P: u64> Field for FiniteField<P, 1, Residue> {
    type Raw = u64;
    type Characteristic = PrimeCharacteristic<P>;
    const NAME: &'static str = "GF(p)";
    // Degree and order read the metadata proof directly, never the row's
    // identities, so central validation can read them without a cycle.
    const DEGREE: u32 = {
        checked_order::<P, 1>();
        1
    };
    const ORDER: u128 = checked_order::<P, 1>();
    // The identity words evaluate the row's validity first and then read
    // their row-local data, never the reverse.
    const ZERO_RAW: u64 = {
        let () = <Self as Field>::VALID;
        0
    };
    const ONE_RAW: u64 = {
        let () = <Self as Field>::VALID;
        1
    };
    const VALID: () = {
        checked_order::<P, 1>();
        let () = <Residue as FiniteFieldRepr<P, 1>>::VALID;
    };

    #[inline]
    fn canonical_raw(raw: u64) -> u64 {
        let () = Validate::<Self>::OK;
        reduce_body::<P>(raw)
    }

    #[inline]
    fn add_raw(left: u64, right: u64) -> u64 {
        let () = Validate::<Self>::OK;
        add_body::<P>(left, right)
    }

    #[inline]
    fn sub_raw(left: u64, right: u64) -> u64 {
        let () = Validate::<Self>::OK;
        sub_body::<P>(left, right)
    }

    #[inline]
    fn neg_raw(value: u64) -> u64 {
        let () = Validate::<Self>::OK;
        neg_body::<P>(value)
    }

    #[inline]
    fn mul_raw(left: u64, right: u64) -> u64 {
        let () = Validate::<Self>::OK;
        mul_body::<P>(left, right)
    }

    #[inline]
    fn inv_raw(value: u64) -> u64 {
        let () = Validate::<Self>::OK;
        inv_body::<P>(value)
    }
}

impl<const P: u64> super::super::private::RowAccess for FiniteField<P, 1, Residue> {
    const DESCRIPTOR: &'static Descriptor = &Descriptor::prime(P, PrimePresentation::Residue);

    const BINARY_DESCRIPTION: Option<
        &'static super::super::binary::description::BinaryDescription,
    > = None;
    fn prime_coordinates_into(dst: &mut [u64], raw: <Self as Field>::Raw) {
        let () = Validate::<Self>::OK;
        assert_eq!(
            dst.len(),
            1,
            "prime coordinate count differs from the field degree"
        );
        dst[0] = reduce_body::<P>(raw);
    }

    fn from_prime_coordinates(src: &[u64]) -> <Self as Field>::Raw {
        let () = Validate::<Self>::OK;
        assert_eq!(
            src.len(),
            1,
            "prime coordinate count differs from the field degree"
        );
        assert!(
            src[0] < P,
            "prime coordinate is not below the characteristic"
        );
        src[0]
    }

    fn from_prime(value: u64) -> <Self as Field>::Raw {
        // Shares the inherent `from_raw` body, so const and runtime
        // contexts build exactly the same element.
        Elem::<Self>::from_raw(value).to_raw()
    }

    fn to_prime(raw: <Self as Field>::Raw) -> Option<u64> {
        // Shares the `reduce_body` reduction with the inherent
        // construction: the hook additionally reduces alias lanes, which
        // the inherent readout never sees because storage is canonical.
        Some(reduce_body::<P>(raw))
    }

    fn coordinate_word(raw: <Self as Field>::Raw) -> u64 {
        let () = Validate::<Self>::OK;
        assert!(P == 2, "coordinate word on a row of odd characteristic");
        reduce_body::<P>(raw)
    }

    fn from_coordinate_word(word: u64) -> <Self as Field>::Raw {
        let () = Validate::<Self>::OK;
        assert!(P == 2, "coordinate word on a row of odd characteristic");
        assert!(
            word < 2,
            "coordinate word holds bits at or above the field degree"
        );
        word
    }
}

impl<const P: u64> FieldBuffer for FiniteField<P, 1, Residue> {
    // The lane width is the presentation's layout fact of the
    // characteristic.
    const BYTES: usize = lane_bytes(P);
    // The lane never exceeds eight bytes, so the product fits `u32`.
    #[allow(clippy::cast_possible_truncation)]
    const STORAGE_BITS: u32 = (8 * lane_bytes(P)) as u32;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        assert_eq!(
            bytes.len(),
            Self::BYTES,
            "GF(p) element has the wrong byte width"
        );
        Elem::<Self>::from_raw(read_lane(bytes))
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(
            bytes.len(),
            Self::BYTES,
            "GF(p) element has the wrong byte width"
        );
        let raw = value.to_raw().to_le_bytes();
        bytes.copy_from_slice(&raw[..Self::BYTES]);
    }
}

// ---------------------------------------------------------------------------
// Concrete const element surface
// ---------------------------------------------------------------------------

impl<const P: u64> Elem<FiniteField<P, 1, Residue>> {
    /// Wrap a raw word, reducing it into canonical storage.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        let () = Validate::<FiniteField<P, 1, Residue>>::OK;
        Elem {
            raw: reduce_body::<P>(value),
        }
    }

    /// Wrap an ordinary residue in `0..P`.
    ///
    /// Rejects `value >= P` instead of narrowing it.
    ///
    /// # Errors
    /// Returns [`RangeError`] when `value` is at or above the
    /// characteristic.
    #[inline]
    pub const fn from_residue(value: u64) -> Result<Self, RangeError> {
        match super::super::check_residue(
            value,
            <PrimeCharacteristic<P> as PrimeIdentity>::CHARACTERISTIC,
        ) {
            Ok(checked) => Ok(Self::from_raw(checked)),
            Err(error) => Err(error),
        }
    }

    /// The ordinary residue naming this element. Always `Some` at degree
    /// one.
    #[inline]
    #[must_use]
    pub const fn to_residue(self) -> Option<u64> {
        Some(self.raw)
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Elem {
            raw: add_body::<P>(self.raw, rhs.raw),
        }
    }

    /// Field subtraction, borrow-safe.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Elem {
            raw: sub_body::<P>(self.raw, rhs.raw),
        }
    }

    /// Additive inverse. `neg(0) == 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Elem {
            raw: neg_body::<P>(self.raw),
        }
    }

    /// Field multiplication, through the widened product.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: mul_body::<P>(self.raw, rhs.raw),
        }
    }

    /// Square.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse by Fermat's little theorem (`a^(P−2)`).
    ///
    /// Maps zero to zero by convention, including the `P = 2` case.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        Elem {
            raw: inv_body::<P>(self.raw),
        }
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division
    /// total leaves hot loops branch-free and keeps this callable from
    /// `const` context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if rhs.raw == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power. `pow(_, 0) == ONE`, zero
    /// included.
    #[inline]
    #[must_use]
    pub const fn pow(self, exponent: u128) -> Self {
        Elem {
            raw: pow_body::<P>(self.raw, exponent),
        }
    }

    /// The canonical form of this element. Storage is canonical already, so
    /// this preserves every constructed element.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        Self::from_raw(self.raw)
    }

    /// Whether this element is the additive identity.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.raw == 0
    }

    /// Whether this element is the multiplicative identity.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.raw == 1
    }
}

impl<const P: u64> core::fmt::Display for Elem<FiniteField<P, 1, Residue>> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width rule is total over `u64` and switches exactly at each
    /// lane's capacity, with characteristic two on the eight-byte lane.
    #[test]
    fn lane_bytes_switches_at_each_capacity() {
        let cases: [(u64, usize); 12] = [
            (0, 1),
            (1, 1),
            (2, 8),
            (3, 1),
            (1 << 8, 1),
            ((1 << 8) + 1, 2),
            (1 << 16, 2),
            ((1 << 16) + 1, 4),
            (1 << 32, 4),
            ((1 << 32) + 1, 8),
            (u64::MAX - 58, 8),
            (u64::MAX, 8),
        ];
        for (characteristic, bytes) in cases {
            assert_eq!(lane_bytes(characteristic), bytes, "{characteristic}");
        }
    }

    /// `read_lane` spells little-endian words of every width.
    #[test]
    fn read_lane_is_little_endian() {
        assert_eq!(read_lane(&[]), 0);
        assert_eq!(read_lane(&[0xAB]), 0xAB);
        assert_eq!(read_lane(&[0x34, 0x12]), 0x1234);
        assert_eq!(read_lane(&[0xFF; 8]), u64::MAX);
    }

    /// The inherent identity predicates agree with the identities.
    #[test]
    fn identity_predicates() {
        type Seven = FiniteField<7, 1, Residue>;
        assert!(Elem::<Seven>::ONE.is_one());
        assert!(!Elem::<Seven>::ZERO.is_one());
        assert!(Elem::<Seven>::ZERO.is_zero());
        assert!(!Elem::<Seven>::from_raw(8).is_zero());
        assert!(Elem::<Seven>::from_raw(8).is_one());
    }
}
