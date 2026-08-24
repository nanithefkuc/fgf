//! GF(2), the field with two elements — the base of every binary tower.
//!
//! Addition and subtraction are XOR, multiplication is AND, negation and
//! squaring are the identity (`x² = x`), and the multiplicative group is
//! trivial: `inv(1) = 1`, `inv(0) = 0` by the crate-wide convention, so
//! `x / 1 = x` and `x / 0 = 0`.
//!
//! This is the scalar oracle for the bit-packed kernels in [`crate::bits`].
//! It deliberately has no [`super::Field`] implementation: GF(2) has no
//! byte-per-element vector representation worth shipping — one element is one
//! *bit*, and the packed surface over `&[u8]` buffers is [`crate::bits`], not
//! [`crate::ops`].
//!
//! ```
//! use fgf::field::Elem;
//! use fgf::gf2;
//!
//! // The whole field, exhaustively.
//! for a in [gf2::Elem(0), gf2::Elem(1)] {
//!     for b in [gf2::Elem(0), gf2::Elem(1)] {
//!         assert_eq!(a.add(b).to_raw(), a.to_raw() ^ b.to_raw());
//!         assert_eq!(a.mul(b).to_raw(), a.to_raw() & b.to_raw());
//!         assert_eq!(a.sub(b), a.add(b));
//!         assert_eq!(a.neg(), a);
//!         assert_eq!(a.square(), a);
//!     }
//! }
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(gf2::Elem(1).div(gf2::Elem::ZERO).to_raw() == 0);
//! ```

use core::fmt;

use super::Elem as ElemTrait;

/// Number of elements in the field.
pub const ORDER: u128 = 2;

/// A generator of the multiplicative group.
///
/// The group is trivial — `{1}` — so the generator is one and has order 1.
pub const GENERATOR: Elem = Elem(1);

/// Marker type for GF(2).
///
/// A zero-sized name for the field. Unlike the tower and prime fields it
/// carries no [`super::Field`] implementation: GF(2) is represented one
/// element per bit, and that packed surface lives in [`crate::bits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Gf2;

impl Gf2 {
    /// Human-readable field name.
    pub const NAME: &'static str = "GF(2)";
}

/// An element of GF(2), stored as a byte holding `0` or `1`.
///
/// Only bit 0 of the raw byte is meaningful; [`Elem::from_raw`] masks it away
/// and every arithmetic output is canonical, so a non-canonical [`Elem`] can
/// only come from constructing the tuple directly. The derived [`Ord`] and
/// [`Hash`] order the raw byte.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u8);

impl Elem {
    /// The additive identity, and the absorbing element for multiplication.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Wrap a raw byte, keeping only the low bit.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        Self(value & 1)
    }

    /// Unwrap to the raw byte, as stored.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u8 {
        self.0
    }

    /// Decode from the stable one-byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        Self(bytes[0] & 1)
    }

    /// Encode to the stable one-byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 1] {
        [self.0 & 1]
    }

    /// The canonical representative of this element.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        Self(self.0 & 1)
    }

    /// Field addition. XOR of the low bits.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self((self.0 ^ rhs.0) & 1)
    }

    /// Field subtraction. Identical to [`Elem::add`]: characteristic two.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }

    /// Additive inverse. The identity: `x + x = 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        self.canonical()
    }

    /// Field multiplication. AND of the low bits.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0 & 1)
    }

    /// Square. The identity: `x² = x` in GF(2).
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.canonical()
    }

    /// Multiplicative inverse. `inv(1) = 1` and `inv(0) = 0` by convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        self.canonical()
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if rhs.0 & 1 == 0 {
            Self::ZERO
        } else {
            self.canonical()
        }
    }

    /// Raise to an unsigned integer power. `pow(_, 0) == ONE`.
    ///
    /// Every element satisfies `x² = x`, so any positive power is `x` itself.
    #[inline]
    #[must_use]
    pub const fn pow(self, exponent: u64) -> Self {
        if exponent == 0 {
            Self::ONE
        } else {
            self.canonical()
        }
    }

    /// Whether this element is the additive identity.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 & 1 == 0
    }

    /// Whether this element is the multiplicative identity.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.0 & 1 == 1
    }
}

impl ElemTrait for Elem {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
    #[inline]
    fn neg(self) -> Self {
        Elem::neg(self)
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
    #[inline]
    fn square(self) -> Self {
        Elem::square(self)
    }
    #[inline]
    fn inv(self) -> Self {
        Elem::inv(self)
    }
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
    #[inline]
    fn pow(self, exponent: u64) -> Self {
        Elem::pow(self, exponent)
    }
    #[inline]
    fn is_zero(self) -> bool {
        Elem::is_zero(self)
    }
    #[inline]
    fn is_one(self) -> bool {
        Elem::is_one(self)
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2({})", self.0 & 1)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 & 1)
    }
}

impl core::ops::Add for Elem {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
}

impl core::ops::Sub for Elem {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
}

impl core::ops::Neg for Elem {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Elem::neg(self)
    }
}

impl core::ops::Mul for Elem {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
}

impl core::ops::Div for Elem {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
}

impl core::ops::AddAssign for Elem {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Elem::add(*self, rhs);
    }
}

impl core::ops::SubAssign for Elem {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Elem::sub(*self, rhs);
    }
}

impl core::ops::MulAssign for Elem {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Elem::mul(*self, rhs);
    }
}

impl core::ops::DivAssign for Elem {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Elem::div(*self, rhs);
    }
}

impl core::iter::Sum for Elem {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Elem::add)
    }
}

impl<'a> core::iter::Sum<&'a Elem> for Elem {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| Elem::add(acc, x))
    }
}

impl core::iter::Product for Elem {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Elem::mul)
    }
}

impl<'a> core::iter::Product<&'a Elem> for Elem {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| Elem::mul(acc, x))
    }
}
