//! The prime field GF(2^64 − 2^32 + 1), the Goldilocks field.
//!
//! Elements are 64-bit lanes reduced modulo `p = 2^64 − 2^32 + 1 =
//! 0xFFFF_FFFF_0000_0001`. Reduction of a 128-bit product uses the identity
//! `2^64 ≡ 2^32 − 1 (mod p)` (the split-fold that makes Goldilocks fast on
//! 32-bit-half integer SIMD).
//!
//! # Totality and canonicalization
//!
//! Every raw 64-bit lane is a legal input: [`Elem::from_raw`] and
//! [`Field::read`] do not canonicalize or reject. Every arithmetic *output* is
//! canonical — a value in `0..p` — with no branch and no panic on out-of-range
//! operands, the prime-field analogue of the crate-wide `inv(0) == 0`
//! convention. Because a lane may hold a non-canonical bit pattern, the derived
//! [`PartialEq`], [`Ord`], and [`Hash`] compare the raw representation, not the
//! field value; compare arithmetic results (always canonical) or
//! [`Elem::canonical`] when field equality is meant.
//!
//! Arithmetic is variable-time and not intended for secret data.
//!
//! ```
//! use fgf::goldilocks::{self, Elem};
//!
//! // Known-answer product, pinned against the split-fold reduction.
//! const X: Elem = Elem(0xFFFF_FFFF_0000_0000);
//! const _: () = assert!(X.mul(X).to_raw() == 0x0000_0000_0000_0001);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const HALF: Elem = Elem(2).inv();
//! const _: () = assert!(HALF.to_raw() == 0x7FFF_FFFF_8000_0001);
//! const _: () = assert!(Elem(2).mul(HALF).to_raw() == 1);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::ZERO).to_raw() == 0);
//!
//! // The generator has full multiplicative order p − 1 = 2^64 − 2^32.
//! assert_eq!(goldilocks::GENERATOR.pow(0xFFFF_FFFF_0000_0000), Elem::ONE);
//! ```

use core::fmt;

use super::{Elem as ElemTrait, Field};

/// The field modulus, `2^64 − 2^32 + 1`.
pub const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// `2^32 − 1`, the residue of `2^64` modulo `p` and the fold constant.
const EPSILON: u64 = 0xFFFF_FFFF;

/// A generator of the multiplicative group, of order `p − 1`.
///
/// `7` is the smallest primitive root modulo `2^64 − 2^32 + 1`.
pub const GENERATOR: Elem = Elem(7);

/// Marker type for GF(2^64 − 2^32 + 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Goldilocks;

/// An element of GF(2^64 − 2^32 + 1), stored as a little-endian 64-bit lane.
///
/// The derived [`Ord`]/[`Hash`] are raw-representation order, useful for map
/// keys and deterministic iteration; they carry no field-theoretic meaning and
/// distinguish non-canonical encodings of the same field value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u64);

/// Reduce a value already in `0..2^64` to the canonical range `0..p`.
///
/// A single conditional subtract suffices: any `u64` exceeds `p` by less than
/// `2^32`.
#[inline]
#[must_use]
pub const fn canonical(x: u64) -> u64 {
    if x >= MODULUS { x - MODULUS } else { x }
}

/// Reduce a full 128-bit product modulo `p`, returning a canonical lane.
///
/// Splits the high 64 bits at the 32-bit boundary and folds each part with
/// `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`, the Plonky2 Goldilocks reduction. The
/// constants are derived from the modulus, not copied from a comparable.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn reduce128(x: u128) -> u64 {
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
    canonical(t2)
}

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Decode from the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Wrap a raw lane. Does not canonicalize.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Unwrap to the raw lane bits, as stored.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u64 {
        self.0
    }

    /// The canonical representative in `0..p` of this element.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        Self(canonical(self.0))
    }

    /// Field addition.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn add(self, rhs: Self) -> Self {
        let a = canonical(self.0) as u128;
        let b = canonical(rhs.0) as u128;
        let s = a + b;
        let m = MODULUS as u128;
        Self(if s >= m { (s - m) as u64 } else { s as u64 })
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn sub(self, rhs: Self) -> Self {
        let a = canonical(self.0) as u128;
        let b = canonical(rhs.0) as u128;
        let m = MODULUS as u128;
        let s = a + m - b;
        Self(if s >= m { (s - m) as u64 } else { s as u64 })
    }

    /// Additive inverse. `neg(0) == 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        let a = canonical(self.0);
        Self(if a == 0 { 0 } else { MODULUS - a })
    }

    /// Field multiplication, folding the 128-bit product modulo `p`.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Self(reduce128((self.0 as u128) * (rhs.0 as u128)))
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
        self.pow(MODULUS - 2)
    }

    /// Field division. Returns zero when the divisor is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        let b = canonical(rhs.0);
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
        canonical(self.0) == 0
    }
    #[inline]
    fn is_one(self) -> bool {
        canonical(self.0) == 1
    }
}

impl Field for Goldilocks {
    type Elem = Elem;

    const NAME: &'static str = "GF(2^64 - 2^32 + 1)";
    const BITS: u32 = 64;
    const BYTES: usize = 8;
    const ORDER: u128 = MODULUS as u128;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let bytes: [u8; 8] = bytes
            .try_into()
            .expect("GF(2^64 - 2^32 + 1) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(
            bytes.len(),
            8,
            "GF(2^64 - 2^32 + 1) element has the wrong byte width"
        );
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Goldilocks({:#018x})", self.0)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", canonical(self.0))
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
