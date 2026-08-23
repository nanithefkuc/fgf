//! The prime field GF(2^31 − 1), the Mersenne31 field.
//!
//! Elements are 32-bit lanes reduced modulo the Mersenne prime
//! `p = 2^31 − 1 = 0x7FFF_FFFF`. The reduction exploits `2^31 ≡ 1 (mod p)`, so
//! folding the high bit back into the low 31 bits (`lo + hi`) plus a single
//! conditional subtract canonicalizes any 32-bit value.
//!
//! # Totality and canonicalization
//!
//! Every raw 32-bit lane is a legal input: [`Elem::from_raw`] and [`Field::read`]
//! do not canonicalize or reject. Every arithmetic *output* is canonical — a
//! value in `0..p` — with no branch and no panic on out-of-range operands, the
//! prime-field analogue of the crate-wide `inv(0) == 0` convention. Because a
//! lane may hold a non-canonical bit pattern, the derived [`PartialEq`],
//! [`Ord`], and [`Hash`] compare the raw representation, not the field value;
//! compare arithmetic results (always canonical) or [`Elem::canonical`] when
//! field equality is meant.
//!
//! Arithmetic is variable-time and not intended for secret data.
//!
//! ```
//! use fgf::mersenne31::{self, Elem};
//!
//! // Known-answer product, pinned against the modular reduction.
//! const X: Elem = Elem(0x5555_5555);
//! const _: () = assert!(X.mul(X).to_raw() == 0x71C7_1C71);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem = Elem(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x4000_0000);
//! const _: () = assert!(Elem(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1.
//! assert_eq!(mersenne31::GENERATOR.pow(0x7FFF_FFFE), Elem::ONE);
//! ```

use core::fmt;

use super::{Elem as ElemTrait, Field};

/// The field modulus, the Mersenne prime `2^31 − 1`.
pub const MODULUS: u32 = 0x7FFF_FFFF;

/// A generator of the multiplicative group, of order `p − 1`.
///
/// `7` is the smallest primitive root modulo `2^31 − 1`.
pub const GENERATOR: Elem = Elem(7);

/// Marker type for GF(2^31 − 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Mersenne31;

/// An element of GF(2^31 − 1), stored as a little-endian 32-bit lane.
///
/// The derived [`Ord`]/[`Hash`] are raw-representation order, useful for map
/// keys and deterministic iteration; they carry no field-theoretic meaning and
/// distinguish non-canonical encodings of the same field value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u32);

/// Reduce an arbitrary 32-bit lane to the canonical range `0..p`.
///
/// `2^31 ≡ 1 (mod p)`, so folding the top bit into the low 31 bits and one
/// conditional subtract suffices for any `u32`.
#[inline]
#[must_use]
pub const fn reduce(x: u32) -> u32 {
    let s = (x & MODULUS) + (x >> 31);
    if s >= MODULUS { s - MODULUS } else { s }
}

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Decode from the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Wrap a raw lane. Does not canonicalize.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    /// Unwrap to the raw lane bits, as stored.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        self.0
    }

    /// The canonical representative in `0..p` of this element.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        Self(reduce(self.0))
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        let a = reduce(self.0);
        let b = reduce(rhs.0);
        let s = a + b;
        Self(if s >= MODULUS { s - MODULUS } else { s })
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        let a = reduce(self.0);
        let b = reduce(rhs.0);
        let s = a + (MODULUS - b);
        Self(if s >= MODULUS { s - MODULUS } else { s })
    }

    /// Additive inverse. `neg(0) == 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        let a = reduce(self.0);
        Self(if a == 0 { 0 } else { MODULUS - a })
    }

    /// Field multiplication, folding the 62-bit product with `2^31 ≡ 1`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn mul(self, rhs: Self) -> Self {
        let a = reduce(self.0) as u64;
        let b = reduce(rhs.0) as u64;
        let prod = a * b;
        let lo = (prod as u32) & MODULUS;
        let hi = (prod >> 31) as u32;
        Self(reduce(lo + hi))
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
        self.pow((MODULUS - 2) as u64)
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        let b = reduce(rhs.0);
        if b == 0 {
            return Self::ZERO;
        }
        self.mul(Self(b).inv())
    }

    /// Raise to an unsigned integer power. `pow(_, 0) == ONE`.
    #[inline]
    #[must_use]
    pub const fn pow(self, mut exponent: u64) -> Self {
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
        reduce(self.0) == 0
    }
    #[inline]
    fn is_one(self) -> bool {
        reduce(self.0) == 1
    }
}

impl Field for Mersenne31 {
    type Elem = Elem;

    const NAME: &'static str = "GF(2^31 - 1)";
    const BITS: u32 = 32;
    const BYTES: usize = 4;
    const ORDER: u128 = MODULUS as u128;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let bytes: [u8; 4] = bytes
            .try_into()
            .expect("GF(2^31 - 1) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(
            bytes.len(),
            4,
            "GF(2^31 - 1) element has the wrong byte width"
        );
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mersenne31({:#010x})", self.0)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", reduce(self.0))
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
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Elem::add)
    }
}

impl<'a> core::iter::Sum<&'a Elem> for Elem {
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| Elem::add(acc, x))
    }
}

impl core::iter::Product for Elem {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Elem::mul)
    }
}

impl<'a> core::iter::Product<&'a Elem> for Elem {
    #[inline]
    fn product<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| Elem::mul(acc, x))
    }
}
