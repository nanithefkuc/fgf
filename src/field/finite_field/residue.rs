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
//! One canonical `u64` per element. The stable byte encoding is exactly
//! eight little-endian bytes of the canonical residue; decoding accepts any
//! `u64` and reduces it, and encoding always emits the canonical residue.
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
//! Prepared coefficients for the packed surface are the canonical elements
//! themselves: preparation copies the value and allocates nothing, packed
//! application allocates nothing, and no workspace carries meaning between
//! calls.

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

impl<const P: u64> FieldBuffer for FiniteField<P, 1, Residue> {
    const BYTES: usize = 8;
    const STORAGE_BITS: u32 = 64;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let bytes: [u8; 8] = bytes
            .try_into()
            .expect("GF(p) element has the wrong byte width");
        Elem::<Self>::from_raw(u64::from_le_bytes(bytes))
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(bytes.len(), 8, "GF(p) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_raw().to_le_bytes());
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
        if value >= <PrimeCharacteristic<P> as PrimeIdentity>::CHARACTERISTIC {
            return Err(RangeError);
        }
        Ok(Self::from_raw(value))
    }

    /// The ordinary residue naming this element. Always `Some` at degree
    /// one.
    #[inline]
    #[must_use]
    pub const fn to_residue(self) -> Option<u64> {
        Some(self.raw)
    }

    /// Decode from the stable little-endian representation, reducing any
    /// `u64` alias.
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
