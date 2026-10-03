//! Binary fields parameterized by degree and representation.
//!
//! [`Gf<const N, R>`](Gf) names GF(2^N) under the representation `R` — a
//! zero-sized marker carrying no runtime data — and [`Elem<const N, R>`] is
//! its element, stored as one raw storage word of `R`'s choosing. The
//! [`Repr`] trait is the sealed vocabulary of representations this crate
//! ships: a representation fixes the storage type, the field's name, and the
//! raw encodings of one and of the canonical generator. [`ByteRepr`] narrows
//! it to the degree-eight representations whose arithmetic is shared: every
//! additive byte basis multiplies through discrete-logarithm tables, so one
//! `const` implementation in this module serves all of them.
//!
//! # Layout
//!
//! Raw storage is private to the crate. Every bit pattern of `R::Raw` is a
//! distinct canonical field value, and the stable byte encoding of an element
//! is the little-endian encoding of its raw word.
//!
//! # Totality and canonicalization
//!
//! By crate-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
//! profile and under `const` evaluation alike. The shared degree-eight
//! arithmetic keeps that convention, and so does the degree-one field
//! [`Gf1`](crate::field::gf1::Gf1).

use core::fmt;

use super::poly::Poly;
use super::{Field, FieldElem};

mod private {
    pub trait Sealed {}
}

/// A sealed representation of GF(2^N).
///
/// `Raw` is the storage type of one element and is fixed per degree: `u8`
/// for N in {1, 8}, `u16` for N = 16. Implementations are fixed inside the
/// crate.
#[allow(private_bounds)]
pub trait Repr<const N: u8>:
    Copy + core::fmt::Debug + core::hash::Hash + PartialEq + Eq + 'static + private::Sealed
{
    /// Storage of one element in this representation.
    ///
    /// Every bit pattern is a distinct canonical field value. Raw storage is
    /// private to the crate; each element exposes it through an inherent
    /// `to_raw` method.
    type Raw: Copy + PartialEq + Eq + core::hash::Hash + core::fmt::Debug + Default + Ord;

    /// Human-readable field name, e.g. `"GF(2^8)/0x11B"`.
    const NAME: &'static str;
    /// Raw storage of the multiplicative identity. Representation-specific:
    /// polynomial bases use one low bit; normal bases use all-ones.
    const ONE_RAW: Self::Raw;
    /// Raw storage of the canonical generator of the multiplicative group.
    const GENERATOR_RAW: Self::Raw;
}

/// A one-byte representation of GF(2^8) whose arithmetic is shared.
///
/// Every additive byte basis — polynomial, normal, Cantor — multiplies by
/// discrete logarithms, so one table pair per representation drives the one
/// `const` arithmetic implementation on [`Elem<8, R>`](Elem). The tables are
/// built at compile time from each representation's reference construction.
pub trait ByteRepr: Repr<8, Raw = u8> {
    /// `EXP[i] = GENERATOR^i` in this representation's encoding.
    const EXP: &'static [u8; 255];
    /// `LOG[EXP[i]] = i`; `LOG[0]` is undefined and callers short-circuit on
    /// zero before indexing.
    const LOG: &'static [u8; 256];
    /// Use-time validity check for the representation's defining data, so
    /// every element constructor rejects an invalid representation at
    /// monomorphization.
    const VALID: ();
}

/// Marker for GF(2^N) under the representation `R`. Zero-sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf<const N: u8, R: Repr<N>>(core::marker::PhantomData<R>);

/// An element of [`Gf<N, R>`](Gf), stored in `R`'s raw form.
///
/// Every bit pattern of the raw word is a distinct field value, so the
/// [`PartialEq`]/[`Hash`]/[`Ord`] implementations — raw-bit order — compare
/// field values. That order is a deterministic total order for map keys and
/// sorting; no order compatible with addition exists in characteristic two.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Elem<const N: u8, R: Repr<N>>(pub(crate) R::Raw);

// Raw-bit order over the storage word: the derived shape would work, but a
// derive places its `PartialOrd`/`Ord` bounds on `R` itself rather than on
// `R::Raw`, which the representation contract does not require.
impl<const N: u8, R: Repr<N>> PartialOrd for Elem<N, R> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: u8, R: Repr<N>> Ord for Elem<N, R> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// The GF(2^8) field under any degree-eight representation `R`.
pub type Gf8<R> = Gf<8, R>;

impl<R: ByteRepr> Elem<8, R> {
    /// The additive identity, and the absorbing element for multiplication.
    pub const ZERO: Self = Self::from_raw(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self::from_raw(R::ONE_RAW);
    /// The canonical generator of the multiplicative group.
    ///
    /// The generator the discrete-log tables of every degree-eight
    /// representation are built from.
    pub const GENERATOR: Self = Self::from_raw(R::GENERATOR_RAW);

    /// Wrap a raw storage word. Every raw word is a distinct field value.
    ///
    /// Referencing this constructor evaluates the representation's validity
    /// check, so a representation with invalid defining data fails to
    /// compile at the use site.
    // Reading `VALID` forces its evaluation; the unit value is the point.
    #[allow(clippy::let_unit_value)]
    #[allow(clippy::ignored_unit_patterns)]
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        let _ = R::VALID;
        Self(value)
    }

    /// Unwrap to the raw storage word.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u8 {
        self.0
    }

    /// Decode from the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        Self::from_raw(bytes[0])
    }

    /// Encode to the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 1] {
        [self.0]
    }

    /// Field addition. Identical to subtraction and to bitwise XOR.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    /// Field subtraction. Identical to [`Elem::add`].
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }

    /// Additive inverse. The identity: `x + x = 0`.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        self
    }

    /// Field multiplication through the representation's discrete-log tables.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        // LOG has no entry for zero, so the absorbing case must
        // short-circuit.
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        let la = R::LOG[self.0 as usize] as usize;
        let lb = R::LOG[rhs.0 as usize] as usize;
        Self(R::EXP[(la + lb) % 255])
    }

    /// Square.
    ///
    /// A flat byte field has no cheaper form than a general multiply; the
    /// tower fields do. Present for parity with them.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse. Maps zero to zero by crate convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        // The `% 255` matters only for `self == 1`, where `LOG` is 0 and
        // `255 - 0` would run off the end of the 255-entry table.
        let l = R::LOG[self.0 as usize] as usize;
        Self(R::EXP[(255 - l) % 255])
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division
    /// total leaves hot loops branch-free and keeps this callable from
    /// `const` context.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power.
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

impl<R: ByteRepr> FieldElem for Elem<8, R> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::sub(self, rhs)
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::mul(self, rhs)
    }
    #[inline]
    fn square(self) -> Self {
        Self::square(self)
    }
    #[inline]
    fn inv(self) -> Self {
        Self::inv(self)
    }
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::div(self, rhs)
    }
    #[inline]
    fn pow(self, exponent: u64) -> Self {
        Self::pow(self, exponent)
    }
}

impl<R: ByteRepr> Field for Gf<8, R> {
    type Elem = Elem<8, R>;

    const NAME: &'static str = R::NAME;
    const BITS: u32 = 8;
    const BYTES: usize = 1;
    const ORDER: u128 = 256;
    const CHARACTERISTIC: u64 = 2;
    const GENERATOR: Elem<8, R> = Elem::<8, R>::GENERATOR;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<8, R> {
        let bytes: [u8; 1] = bytes
            .try_into()
            .expect("GF(2^8) element has the wrong byte width");
        Self::Elem::from_bytes(bytes)
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<8, R>) {
        assert_eq!(bytes.len(), 1, "GF(2^8) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl<R: ByteRepr> fmt::Display for Elem<8, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}", self.0)
    }
}

// The remaining element surface is generic over `(N, R)` and routes through
// `FieldElem`, so it covers exactly the degrees with arithmetic: the shared
// byte-field impl above and the concrete degree-one impl in `poly`.
impl<const N: u8, R: Repr<N>> fmt::Debug for Elem<N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:?})", R::NAME, self.0)
    }
}

impl<const N: u8, R: Repr<N>> Default for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    fn default() -> Self {
        <Self as FieldElem>::ZERO
    }
}

impl<const N: u8, R: Repr<N>> core::ops::Add for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        <Self as FieldElem>::add(self, rhs)
    }
}

impl<const N: u8, R: Repr<N>> core::ops::Sub for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        <Self as FieldElem>::sub(self, rhs)
    }
}

impl<const N: u8, R: Repr<N>> core::ops::Neg for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        <Self as FieldElem>::neg(self)
    }
}

impl<const N: u8, R: Repr<N>> core::ops::Mul for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        <Self as FieldElem>::mul(self, rhs)
    }
}

impl<const N: u8, R: Repr<N>> core::ops::Div for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        <Self as FieldElem>::div(self, rhs)
    }
}

impl<const N: u8, R: Repr<N>> core::ops::AddAssign for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = <Self as FieldElem>::add(*self, rhs);
    }
}

impl<const N: u8, R: Repr<N>> core::ops::SubAssign for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = <Self as FieldElem>::sub(*self, rhs);
    }
}

impl<const N: u8, R: Repr<N>> core::ops::MulAssign for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = <Self as FieldElem>::mul(*self, rhs);
    }
}

impl<const N: u8, R: Repr<N>> core::ops::DivAssign for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = <Self as FieldElem>::div(*self, rhs);
    }
}

impl<const N: u8, R: Repr<N>> core::iter::Sum for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as FieldElem>::ZERO, <Self as FieldElem>::add)
    }
}

impl<'a, const N: u8, R: Repr<N>> core::iter::Sum<&'a Elem<N, R>> for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem<N, R>>>(iter: I) -> Self {
        iter.fold(<Self as FieldElem>::ZERO, |acc, &x| {
            <Self as FieldElem>::add(acc, x)
        })
    }
}

impl<const N: u8, R: Repr<N>> core::iter::Product for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as FieldElem>::ONE, <Self as FieldElem>::mul)
    }
}

impl<'a, const N: u8, R: Repr<N>> core::iter::Product<&'a Elem<N, R>> for Elem<N, R>
where
    Elem<N, R>: FieldElem,
{
    #[inline]
    fn product<I: Iterator<Item = &'a Elem<N, R>>>(iter: I) -> Self {
        iter.fold(<Self as FieldElem>::ONE, |acc, &x| {
            <Self as FieldElem>::mul(acc, x)
        })
    }
}

impl<const P: u32> private::Sealed for Poly<P> {}
