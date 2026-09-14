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
//! canonicalize; `canonical` and every arithmetic operation does. Equality,
//! hashing, and ordering follow the field value — both limbs reduced modulo
//! `p` — so elements whose limbs agree modulo `p` compare equal, hash
//! equally, and sort as one element. Inspect the stored limbs with
//! [`Elem::to_raw`].
//!
//! Arithmetic is variable-time and not for secret data.
//!
//! ```
//! use fgf::quad_mersenne31::{self, Elem};
//!
//! // i² = −1  →  (0 + i)² = (−1 + 0·i)
//! const I: Elem = Elem::from_raw(0, 1);
//! const _: () = assert!(I.square().to_raw().0 == 0x7FFF_FFFE);
//! const _: () = assert!(I.square().to_raw().1 == 0);
//!
//! // Known product, pinned against the frozen M31 known answer
//! // a² = 0x71C7_1C71: (a + a·i)² = 0 + 2a²·i with 2a² mod p = 0x638E_38E3.
//! const A: Elem = Elem::from_raw(0x5555_5555, 0x5555_5555);
//! const B: Elem = Elem::from_raw(0x5555_5555, 0x5555_5555);
//! const _: () = assert!(A.mul(B).to_raw().0 == 0);
//! const _: () = assert!(A.mul(B).to_raw().1 == 0x638E_38E3);
//!
//! // Division total
//! const _: () = assert!(A.div(Elem::ZERO).to_raw().0 == 0);
//! const _: () = assert!(A.div(Elem::ZERO).to_raw().1 == 0);
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
/// The limbs are stored exactly as passed to [`Elem::from_raw`], so either
/// may hold a non-canonical bit pattern (≥ p). Equality, hashing, and
/// ordering follow the field value — both limbs reduced modulo `p` — so
/// elements whose limbs agree modulo `p` compare equal, hash equally, and
/// sort as a single element. Ordering is lexicographic over the canonical
/// `(re, im)` pair: a deterministic total order for maps and sorting, not an
/// order compatible with field arithmetic. [`Elem::to_raw`] exposes the
/// stored limbs.
#[derive(Clone, Copy, Default)]
pub struct Elem(pub(crate) u32, pub(crate) u32);

impl PartialEq for Elem {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        m31_canonical(self.0) == m31_canonical(other.0)
            && m31_canonical(self.1) == m31_canonical(other.1)
    }
}

impl Eq for Elem {}

impl core::hash::Hash for Elem {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(m31_canonical(self.0));
        state.write_u32(m31_canonical(self.1));
    }
}

impl PartialOrd for Elem {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Elem {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        m31_canonical(self.0)
            .cmp(&m31_canonical(other.0))
            .then_with(|| m31_canonical(self.1).cmp(&m31_canonical(other.1)))
    }
}

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

    /// Wrap raw limbs. Does not canonicalize: the stored limbs keep the exact
    /// bits passed in, and equality reduces each modulo `p`.
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
    pub const fn to_components(self) -> (mersenne31::Elem, mersenne31::Elem) {
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

    /// Norm `a² + b²` as an element of the base field.
    ///
    /// The norm lives in `GF(2^31 − 1)`, not in a raw lane; the returned
    /// value is canonical.
    #[inline]
    #[must_use]
    pub const fn norm(self) -> crate::field::mersenne31::Elem {
        let re2 = m31_mul(self.0, self.0);
        let im2 = m31_mul(self.1, self.1);
        mersenne31::Elem::from_raw(m31_add(re2, im2))
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

    /// Square: `(a+bi)² = (a²−b²) + 2ab·i`.
    ///
    /// Three base multiplies instead of the four a general multiply costs;
    /// `2ab` is one modular add of `ab` with itself, cheaper than another
    /// multiply. Measured faster than `square via mul` on the reference host
    /// (BENCHMARKS.md).
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        let a2 = m31_mul(self.0, self.0);
        let b2 = m31_mul(self.1, self.1);
        let ab = m31_mul(self.0, self.1);
        Self(m31_sub(a2, b2), m31_add(ab, ab))
    }

    /// Multiplicative inverse via conjugate / norm.
    ///
    /// Maps zero to zero.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        let n = self.norm();
        if n.to_raw() == 0 {
            return Self::ZERO;
        }
        let n_inv = n.inv().0;
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
        if n.to_raw() == 0 {
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

    /// Raise to a `u128` exponent.
    ///
    /// The group order `p² − 1 = 2^62 − 2^32` fits comfortably in a `u64`,
    /// so [`Elem::pow`] covers every exponent this field's arithmetic can
    /// produce; this variant exists for callers that already hold a `u128`
    /// exponent.
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
    const CHARACTERISTIC: u64 = MODULUS as u64;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem {
        let arr: [u8; 8] = bytes.try_into().expect("QM31 element has wrong byte width");
        Elem::from_bytes(arr)
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem) {
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
