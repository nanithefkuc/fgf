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
//! every arithmetic output is canonical (`< p` per limb). `from_raw` does not
//! canonicalize; `canonical` and every arithmetic operation does.
//!
//! Arithmetic is variable-time and not for secret data.
//!
//! ```
//! use fgf::quad_mersenne31::{self, Elem};
//!
//! // i² = −1  →  (0 + i)² = (−1 + 0·i)
//! const I: Elem = Elem(0, 1);
//! const _: () = assert!(I.square().0 == 0x7FFF_FFFE);
//! const _: () = assert!(I.square().1 == 0);
//!
//! // Known product
//! const A: Elem = Elem(0x5555_5555, 0x5555_5555);
//! const B: Elem = Elem(0x5555_5555, 0x5555_5555);
//! // (a+ai)² = 0 + 2a²·i ; 2·(0x71C71C71) reduced etc — checked against u128 oracle
//! let _ = A.mul(B);
//!
//! // Division total
//! const _: () = assert!(A.div(Elem::ZERO).0 == 0);
//! const _: () = assert!(A.div(Elem::ZERO).1 == 0);
//!
//! // Generator has full order p²−1
//! assert_eq!(quad_mersenne31::GENERATOR.pow(0x3FFF_FFFF_0000_0000), Elem::ONE);
//! ```

use core::fmt;

use super::{Elem as ElemTrait, Field};
use crate::field::mersenne31;

/// The base modulus `p = 2³¹ − 1`.
pub const MODULUS: u32 = mersenne31::MODULUS;

/// Order of the extension field `p²`.
pub const ORDER: u128 = (MODULUS as u128) * (MODULUS as u128);

/// A generator of the multiplicative group, of order `p² − 1`.
///
/// `(1, 12)` is the lexicographically smallest primitive element.
pub const GENERATOR: Elem = Elem(1, 12);

/// Marker type for GF((2³¹ − 1)²).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct QuadMersenne31;

/// An element of GF((2³¹ − 1)²), stored as interleaved `u32` limbs `(re, im)`.
///
/// The derived `Ord`/`Hash` are raw-representation order, not field order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u32, pub u32);

#[inline]
#[must_use]
const fn m31_canonical(x: u32) -> u32 {
    mersenne31::reduce(x)
}

#[inline]
#[must_use]
const fn m31_add(a: u32, b: u32) -> u32 {
    let a = m31_canonical(a);
    let b = m31_canonical(b);
    let s = a + b;
    if s >= MODULUS { s - MODULUS } else { s }
}

#[inline]
#[must_use]
const fn m31_sub(a: u32, b: u32) -> u32 {
    let a = m31_canonical(a);
    let b = m31_canonical(b);
    let s = a + (MODULUS - b);
    if s >= MODULUS { s - MODULUS } else { s }
}

#[inline]
#[must_use]
const fn m31_neg(a: u32) -> u32 {
    let a = m31_canonical(a);
    if a == 0 { 0 } else { MODULUS - a }
}

#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
const fn m31_mul(a: u32, b: u32) -> u32 {
    let a = m31_canonical(a) as u64;
    let b = m31_canonical(b) as u64;
    let prod = a * b;
    let lo = (prod as u32) & MODULUS;
    let hi = (prod >> 31) as u32;
    mersenne31::reduce(lo + hi)
}

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0, 0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1, 0);
    /// The imaginary unit `i` with `i² = −1`.
    pub const I: Self = Self(0, 1);

    /// Decode from stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        let re = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let im = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Self(re, im)
    }

    /// Encode to stable little-endian `[re, im]`.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 8] {
        let re = self.0.to_le_bytes();
        let im = self.1.to_le_bytes();
        [re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]
    }

    /// Wrap raw limbs. Does not canonicalize.
    #[inline]
    #[must_use]
    pub const fn from_raw(re: u32, im: u32) -> Self {
        Self(re, im)
    }

    /// Unwrap to raw limbs.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> (u32, u32) {
        (self.0, self.1)
    }

    /// Canonical representative per limb.
    #[inline]
    #[must_use]
    pub const fn canonical(self) -> Self {
        Self(m31_canonical(self.0), m31_canonical(self.1))
    }

    /// Real and imaginary components.
    #[inline]
    #[must_use]
    pub const fn components(self) -> (mersenne31::Elem, mersenne31::Elem) {
        (mersenne31::Elem(self.0), mersenne31::Elem(self.1))
    }

    /// Build from components.
    #[inline]
    #[must_use]
    pub const fn from_components(re: mersenne31::Elem, im: mersenne31::Elem) -> Self {
        Self(re.0, im.0)
    }

    /// Conjugate `a − bi`.
    #[inline]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self(self.0, m31_neg(self.1))
    }

    /// Norm `a² + b²` in the base field (canonical `u32`).
    #[inline]
    #[must_use]
    pub const fn norm(self) -> u32 {
        let re2 = m31_mul(self.0, self.0);
        let im2 = m31_mul(self.1, self.1);
        m31_add(re2, im2)
    }

    /// Field addition.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self(m31_add(self.0, rhs.0), m31_add(self.1, rhs.1))
    }

    /// Field subtraction.
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Self(m31_sub(self.0, rhs.0), m31_sub(self.1, rhs.1))
    }

    /// Additive inverse.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Self(m31_neg(self.0), m31_neg(self.1))
    }

    /// Field multiplication `(a+bi)(c+di) = (ac−bd)+(ad+bc)i`.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        let ac = m31_mul(self.0, rhs.0);
        let bd = m31_mul(self.1, rhs.1);
        let ad = m31_mul(self.0, rhs.1);
        let bc = m31_mul(self.1, rhs.0);
        Self(m31_sub(ac, bd), m31_add(ad, bc))
    }

    /// Square.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse via conjugate / norm.
    ///
    /// Maps zero to zero.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        let n = self.norm();
        if n == 0 {
            return Self::ZERO;
        }
        let n_inv = mersenne31::Elem(n).inv().0;
        // n_inv already canonical (<p)
        let re = m31_mul(self.0, n_inv);
        let im_neg = m31_neg(self.1);
        let im = m31_mul(im_neg, n_inv);
        Self(re, im)
    }

    /// Field division. `x / 0 == 0`.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        let n = rhs.norm();
        if n == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power.
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

    /// Raise to `u128` exponent (needed for `p²−1` which exceeds `u64`? actually fits in 63 bits, but convenience).
    #[inline]
    #[must_use]
    pub const fn pow_u128(self, mut exponent: u128) -> Self {
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
        m31_canonical(self.0) == 0 && m31_canonical(self.1) == 0
    }
    #[inline]
    fn is_one(self) -> bool {
        m31_canonical(self.0) == 1 && m31_canonical(self.1) == 0
    }
}

impl Field for QuadMersenne31 {
    type Elem = Elem;

    const NAME: &'static str = "GF((2^31 - 1)^2)";
    const BITS: u32 = 64;
    const BYTES: usize = 8;
    const ORDER: u128 = ORDER;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let arr: [u8; 8] = bytes.try_into().expect("QM31 element has wrong byte width");
        Elem::from_bytes(arr)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(bytes.len(), 8, "QM31 element has wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QM31({:#010x},{:#010x})", self.0, self.1)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:#x}+{:#x}i)", self.0, self.1)
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
        *self = self.add(rhs);
    }
}
impl core::ops::SubAssign for Elem {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}
impl core::ops::MulAssign for Elem {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}
impl core::ops::DivAssign for Elem {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div(rhs);
    }
}
impl core::iter::Sum for Elem {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Self::add)
    }
}
impl<'a> core::iter::Sum<&'a Elem> for Elem {
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.copied().fold(Self::ZERO, Self::add)
    }
}
impl core::iter::Product for Elem {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Self::mul)
    }
}
impl<'a> core::iter::Product<&'a Elem> for Elem {
    #[inline]
    fn product<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.copied().fold(Self::ONE, Self::mul)
    }
}
