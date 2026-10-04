//! Recursive quadratic towers through degree 64.
//!
//! A [`Tower<S>`] is GF(2^N) as `a + b*t` over its base field, under the
//! relation `t^2 + A*t + B = 0` pinned by [`TowerSpec`]. The low `N/2`
//! coordinate bits hold `a` and the high `N/2` bits hold `b`; storage bits at
//! or above `N` are zero. One is the base one in the low component. The
//! scalar identities are:
//!
//! ```text
//! product:   (a*c + B*b*d, a*d + b*c + A*b*d)
//! square:    (a^2 + B*b^2, A*b^2)
//! conjugate: (a + A*b, b)
//! norm:      a^2 + A*a*b + B*b^2
//! ```
//!
//! [`TowerSpec`] is open: downstream crates define custom quadratic towers
//! over supported binary bases with raw `A`/`B` coordinate words. Validation
//! runs through description arithmetic: the base first, excess coefficient
//! bits, `A != 0`, and absolute `trace(B/A^2) == 1`. [`TowerGeneratorSpec`]
//! optionally pins a selected multiplicative generator, checked to full
//! order against the complete prime factors of `2^N - 1`.
//!
//! Shipped specs nest from GF(2): [`Rijndael16`], [`Rijndael32`],
//! [`Rijndael64`] root at the AES field, and the Fan-Paar specs nest through
//! the Wiedemann relation `X^2 + alpha*X + 1`, where `alpha` is the previous
//! level's indeterminate (`1` over GF(2), then the raw `base_one << (h/2)`
//! in its own packing). [`Gf16`], [`Gf32`], [`Gf64`], [`FanPaar8`],
//! [`FanPaar16`], [`FanPaar32`], and [`FanPaar64`] alias the corresponding
//! `Gf<N, Tower<…>>` types with their frozen generators.
//!
//! Scalar arithmetic is one `const` body per degree row. Degrees 2, 4, and 8
//! share the small-row bodies with the flat presentations: the
//! representation's log/exp tables serve pinned polynomial leaves and the
//! descriptor interpreter serves every other description. Degrees 16, 32,
//! and 64 branch on structural equality with the pinned Rijndael and
//! Fan-Paar descriptions, taking that presentation's specialized recurrence;
//! every other description uses the general descriptor interpreter. Custom
//! specs cannot claim a strategy.
//!
//! ```
//! use fgf::{Elem, Gf16};
//!
//! let x = Elem::<Gf16>::from_raw(0x1234);
//! assert_eq!(x.mul(Elem::<Gf16>::ONE), x);
//! assert_eq!(Elem::<Gf16>::GENERATOR.pow(65_535), Elem::<Gf16>::ONE);
//! ```
//!
//! A degree-128 tower has no representation row:
//!
//! ```compile_fail,E0277
//! use fgf::{Gf, Tower, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct Over64;
//! impl TowerSpec for Over64 {
//!     type Base = fgf::Gf64;
//!     const A: u64 = 1;
//!     const B: u64 = 0x2000_0000_0000_0000;
//!     const NAME: &'static str = "too wide";
//! }
//! let _ = fgf::Elem::<Gf<128, Tower<Over64>>>::ZERO;
//! ```
//!
//! A tower over the wrong base degree is rejected by the trait bounds:
//!
//! ```compile_fail,E0277
//! use fgf::{Gf, Tower, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct WrongBase;
//! impl TowerSpec for WrongBase {
//!     type Base = fgf::Gf16;
//!     const A: u64 = 1;
//!     const B: u64 = 0x2000;
//!     const NAME: &'static str = "wrong base";
//! }
//! let _ = fgf::Elem::<Gf<64, Tower<WrongBase>>>::ZERO;
//! ```
//!
//! The correct-base positive case for the same shape:
//!
//! ```
//! use fgf::{Gf, Tower, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct RightBase;
//! impl TowerSpec for RightBase {
//!     type Base = fgf::Gf16;
//!     const A: u64 = 1;
//!     const B: u64 = 0x2000;
//!     const NAME: &'static str = "right base";
//! }
//! let _ = fgf::Elem::<Gf<32, Tower<RightBase>>>::ONE;
//! ```
//!
//! A zero linear coefficient is rejected at value use:
//!
//! ```compile_fail,E0080
//! use fgf::{Gf, Tower, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct ZeroA;
//! impl TowerSpec for ZeroA {
//!     type Base = fgf::Gf<8, fgf::Poly<0x11B>>;
//!     const A: u64 = 0;
//!     const B: u64 = 0x20;
//!     const NAME: &'static str = "zero a";
//! }
//! let _ = fgf::Elem::<Gf<16, Tower<ZeroA>>>::ONE;
//! ```
//!
//! A reducible relation is rejected at value use:
//!
//! ```compile_fail,E0080
//! use fgf::{Gf, Tower, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct TraceZero;
//! impl TowerSpec for TraceZero {
//!     type Base = fgf::Gf<8, fgf::Poly<0x11D>>;
//!     const A: u64 = 1;
//!     const B: u64 = 1;
//!     const NAME: &'static str = "reducible";
//! }
//! let _ = fgf::Elem::<Gf<16, Tower<TraceZero>>>::ONE;
//! ```
//!
//! A non-generator is rejected at generator use:
//!
//! ```compile_fail,E0080
//! use fgf::{Gf, Tower, TowerGeneratorSpec, TowerSpec};
//!
//! #[derive(Clone, Copy)]
//! struct BadGen;
//! impl TowerSpec for BadGen {
//!     type Base = fgf::Gf<8, fgf::Poly<0x11B>>;
//!     const A: u64 = 1;
//!     const B: u64 = 0x20;
//!     const NAME: &'static str = "bad gen";
//! }
//! impl TowerGeneratorSpec for BadGen {
//!     const GENERATOR: u64 = 2;
//! }
//! let _ = fgf::Elem::<Gf<16, Tower<BadGen>>>::GENERATOR;
//! ```

use core::fmt;

use super::description::{BinaryDescription, ByteLogExp};
use super::poly::{AES, Poly};
use super::{BinaryDegree, BinaryField, BinaryRepr, Gf, private};
use crate::field::{Elem, Field, FieldBuffer, HasGenerator};

/// The defining data of one quadratic tower.
#[allow(private_bounds)]
pub trait TowerSpec: Copy + 'static {
    /// The half-degree base field.
    type Base: BinaryField;
    /// Linear coefficient `A` of `t^2 + A*t + B`, as a base-coordinate word.
    const A: u64;
    /// Constant coefficient `B` of `t^2 + A*t + B`, as a base-coordinate word.
    const B: u64;
    /// Human-readable presentation name.
    const NAME: &'static str;
}

/// An optional selected multiplicative generator for a tower.
#[allow(private_bounds)]
pub trait TowerGeneratorSpec: TowerSpec {
    /// Raw word of the selected generator, of full multiplicative order.
    const GENERATOR: u64;
}

/// The quadratic tower representation over the spec's base.
pub struct Tower<S: TowerSpec>(core::marker::PhantomData<fn() -> S>);

impl<S: TowerSpec> private::Sealed for Tower<S> {}

// A derive would bound the impl on `S: Clone`; the manual impl bounds
// only through `S: Copy`, which the spec contract already supplies.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<S: TowerSpec> Clone for Tower<S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: TowerSpec> Copy for Tower<S> {}

impl<S: TowerSpec> fmt::Debug for Tower<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(S::NAME)
    }
}

impl<S: TowerSpec> Default for Tower<S> {
    #[inline]
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

// ---------------------------------------------------------------------------
// Fan-Paar recurrence: the specialized arithmetic of the Wiedemann tower.
//
// Starting from GF(2), each level doubles the degree with
// `X^2 + alpha*X + 1 = 0`, where `alpha` is the previous level's
// indeterminate. Elements pack low component first; these helpers select the
// pinned Fan-Paar strategy inside each wide degree-row body.

#[inline]
const fn fp_low_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Multiply by the top-level indeterminate of a `bits`-wide level.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than 64.
#[must_use]
pub const fn fp_mul_alpha(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = fp_low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    a1 | ((a0 ^ fp_mul_alpha(a1, half)) << half)
}

/// Table-free recursive multiplication: the Karatsuba path for every
/// width, including the byte level. The log-table builder below uses this
/// rather than [`fp_multiply`] so table construction never reads the tables.
const fn fp_multiply_rec(lhs: u64, rhs: u64, bits: u32) -> u64 {
    if bits == 1 {
        return lhs & rhs & 1;
    }
    let half = bits / 2;
    let mask = fp_low_mask(half);
    let a0 = lhs & mask;
    let a1 = lhs >> half;
    let b0 = rhs & mask;
    let b1 = rhs >> half;
    let z0 = fp_multiply_rec(a0, b0, half);
    let z2 = fp_multiply_rec(a1, b1, half);
    let z1 = fp_multiply_rec(a0 ^ a1, b0 ^ b1, half) ^ z0 ^ z2;
    (z0 ^ z2) | ((z1 ^ fp_mul_alpha(z2, half)) << half)
}

const fn fp_build_exp8() -> [u8; 255] {
    let mut table = [0u8; 255];
    let mut value = 1u8;
    let mut i = 0;
    while i < table.len() {
        table[i] = value;
        // Widening a byte into the tower word is exact.
        #[allow(clippy::cast_lossless)]
        let next = fp_multiply_rec(value as u64, 0x2d, 8);
        // The byte tower closes over one byte.
        #[allow(clippy::cast_possible_truncation)]
        let narrowed = next as u8;
        value = narrowed;
        i += 1;
    }
    table
}

const FP_EXP8: [u8; 255] = fp_build_exp8();

#[allow(clippy::cast_possible_truncation)]
const fn fp_build_log8() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < FP_EXP8.len() {
        table[FP_EXP8[i] as usize] = i as u8;
        i += 1;
    }
    table
}

const FP_LOG8: [u8; 256] = fp_build_log8();

/// Recursive Fan-Paar multiplication with a log-table byte base.
///
/// The independent schoolbook recurrence for the whole Wiedemann tower:
/// the kernel table banks and the descriptor interpreter are both checked
/// against it, and the pinned Fan-Paar descriptions select it as their
/// `const` arithmetic strategy.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than 64.
#[must_use]
pub const fn fp_multiply(lhs: u64, rhs: u64, bits: u32) -> u64 {
    if bits == 8 {
        if lhs == 0 || rhs == 0 {
            return 0;
        }
        // Byte-field table indices; the operands are validated bytes here.
        #[allow(clippy::cast_possible_truncation)]
        let log = FP_LOG8[lhs as usize] as usize + FP_LOG8[rhs as usize] as usize;
        return FP_EXP8[log % 255] as u64;
    }
    fp_multiply_rec(lhs, rhs, bits)
}

/// Recursive squaring through the linear recurrence.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than 64.
#[must_use]
pub const fn fp_square(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = fp_low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    let z0 = fp_square(a0, half);
    let z2 = fp_square(a1, half);
    (z0 ^ z2) | (fp_mul_alpha(z2, half) << half)
}

/// Recursive conjugate-and-norm inversion, with `inv(0) = 0`.
///
/// # Panics
///
/// Panics unless `bits` is a power of two no greater than 64.
#[must_use]
pub const fn fp_invert(value: u64, bits: u32) -> u64 {
    if bits == 1 {
        return value & 1;
    }
    let half = bits / 2;
    let mask = fp_low_mask(half);
    let a0 = value & mask;
    let a1 = value >> half;
    let a0z1 = a0 ^ fp_mul_alpha(a1, half);
    let norm = fp_multiply(a0, a0z1, half) ^ fp_square(a1, half);
    let norm_inv = fp_invert(norm, half);
    fp_multiply(norm_inv, a0z1, half) | (fp_multiply(norm_inv, a1, half) << half)
}

// ---------------------------------------------------------------------------
// Shipped specs.

/// The Rijndael-rooted tower over the AES field: `t^2 + t + 0x20`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael16;

impl TowerSpec for Rijndael16 {
    type Base = Gf<8, Poly<AES>>;
    const A: u64 = 1;
    const B: u64 = 0x20;
    const NAME: &'static str = "GF(2^16)";
}

impl TowerGeneratorSpec for Rijndael16 {
    const GENERATOR: u64 = 0x0108;
}

/// GF(2^16): [`Gf<16, Tower<Rijndael16>>`](Gf).
pub type Gf16 = Gf<16, Tower<Rijndael16>>;

/// The Rijndael-rooted tower over [`Gf16`]: `v^2 + v + 0x2000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael32;

impl TowerSpec for Rijndael32 {
    type Base = Gf16;
    const A: u64 = 1;
    const B: u64 = 0x2000;
    const NAME: &'static str = "GF(2^32)";
}

impl TowerGeneratorSpec for Rijndael32 {
    const GENERATOR: u64 = 0x0001_0002;
}

/// GF(2^32): [`Gf<32, Tower<Rijndael32>>`](Gf).
pub type Gf32 = Gf<32, Tower<Rijndael32>>;

/// The Rijndael-rooted tower over [`Gf32`]: `w^2 + w + 0x2000_0000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael64;

impl TowerSpec for Rijndael64 {
    type Base = Gf32;
    const A: u64 = 1;
    const B: u64 = 0x2000_0000;
    const NAME: &'static str = "GF(2^64)";
}

impl TowerGeneratorSpec for Rijndael64 {
    const GENERATOR: u64 = 0x0000_0001_0000_0004;
}

/// GF(2^64): [`Gf<64, Tower<Rijndael64>>`](Gf).
pub type Gf64 = Gf<64, Tower<Rijndael64>>;

/// Fan-Paar degree-2 tower over GF(2): `t^2 + t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar2Spec;

impl TowerSpec for FanPaar2Spec {
    type Base = crate::field::Gf1;
    const A: u64 = 1;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^2)";
}

impl TowerGeneratorSpec for FanPaar2Spec {
    const GENERATOR: u64 = 0x2;
}

/// Fan-Paar degree-4 tower over Fan-Paar GF(2^2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar4Spec;

impl TowerSpec for FanPaar4Spec {
    type Base = Gf<2, Tower<FanPaar2Spec>>;
    const A: u64 = 0x2;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^4)";
}

impl TowerGeneratorSpec for FanPaar4Spec {
    const GENERATOR: u64 = 0x7;
}

/// Fan-Paar degree-8 tower over Fan-Paar GF(2^4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar8Spec;

impl TowerSpec for FanPaar8Spec {
    type Base = Gf<4, Tower<FanPaar4Spec>>;
    const A: u64 = 0x4;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^8)";
}

impl TowerGeneratorSpec for FanPaar8Spec {
    const GENERATOR: u64 = 0x2d;
}

/// Fan-Paar GF(2^8): [`Gf<8, Tower<FanPaar8Spec>>`](Gf).
pub type FanPaar8 = Gf<8, Tower<FanPaar8Spec>>;

/// Fan-Paar degree-16 tower over Fan-Paar GF(2^8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar16Spec;

impl TowerSpec for FanPaar16Spec {
    type Base = FanPaar8;
    const A: u64 = 0x10;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^16)";
}

impl TowerGeneratorSpec for FanPaar16Spec {
    const GENERATOR: u64 = 0xe2de;
}

/// Fan-Paar GF(2^16): [`Gf<16, Tower<FanPaar16Spec>>`](Gf).
pub type FanPaar16 = Gf<16, Tower<FanPaar16Spec>>;

/// Fan-Paar degree-32 tower over Fan-Paar GF(2^16).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar32Spec;

impl TowerSpec for FanPaar32Spec {
    type Base = FanPaar16;
    const A: u64 = 0x0100;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^32)";
}

impl TowerGeneratorSpec for FanPaar32Spec {
    const GENERATOR: u64 = 0x03e2_1cea;
}

/// Fan-Paar GF(2^32): [`Gf<32, Tower<FanPaar32Spec>>`](Gf).
pub type FanPaar32 = Gf<32, Tower<FanPaar32Spec>>;

/// Fan-Paar degree-64 tower over Fan-Paar GF(2^32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar64Spec;

impl TowerSpec for FanPaar64Spec {
    type Base = FanPaar32;
    const A: u64 = 0x0001_0000;
    const B: u64 = 1;
    const NAME: &'static str = "Fan-Paar GF(2^64)";
}

impl TowerGeneratorSpec for FanPaar64Spec {
    const GENERATOR: u64 = 0x070f_870d_cd9c_1d88;
}

/// Fan-Paar GF(2^64): [`Gf<64, Tower<FanPaar64Spec>>`](Gf).
pub type FanPaar64 = Gf<64, Tower<FanPaar64Spec>>;

/// Fan-Paar GF(2^2), the public intermediate level.
pub type FanPaar2 = Gf<2, Tower<FanPaar2Spec>>;
/// Fan-Paar GF(2^4), the public intermediate level.
pub type FanPaar4 = Gf<4, Tower<FanPaar4Spec>>;

// ---------------------------------------------------------------------------
// Representation rows: degrees 2, 4, 8, 16, 32, 64.

macro_rules! tower_repr_row {
    ($n:literal, $h:literal) => {
        impl<S: TowerSpec> BinaryRepr<$n> for Tower<S>
        where
            S::Base: BinaryDegree<$h>,
        {
            const NAME: &'static str = S::NAME;
            const DESCRIPTION: &'static BinaryDescription = &BinaryDescription::append_quadratic(
                <S::Base as BinaryField>::DESCRIPTION,
                S::A,
                S::B,
            );
            const LOG_EXP: Option<&'static ByteLogExp> = None;
            const VALID: () = {
                let () = <S::Base as Field>::VALID;
                <Self as BinaryRepr<$n>>::DESCRIPTION.validate();
            };
        }
    };
}

tower_repr_row!(2, 1);
tower_repr_row!(4, 2);
tower_repr_row!(8, 4);
tower_repr_row!(16, 8);
tower_repr_row!(32, 16);
tower_repr_row!(64, 32);

// Pinned descriptions for strategy selection in the wide rows and the
// kernel dispatch, which routes pinned presentations to their SIMD bodies.
pub(crate) const RIJNDAEL16_DESC: &BinaryDescription =
    <Tower<Rijndael16> as BinaryRepr<16>>::DESCRIPTION;
pub(crate) const RIJNDAEL32_DESC: &BinaryDescription =
    <Tower<Rijndael32> as BinaryRepr<32>>::DESCRIPTION;
pub(crate) const RIJNDAEL64_DESC: &BinaryDescription =
    <Tower<Rijndael64> as BinaryRepr<64>>::DESCRIPTION;
pub(crate) const FANPAAR16_DESC: &BinaryDescription =
    <Tower<FanPaar16Spec> as BinaryRepr<16>>::DESCRIPTION;
pub(crate) const FANPAAR32_DESC: &BinaryDescription =
    <Tower<FanPaar32Spec> as BinaryRepr<32>>::DESCRIPTION;
pub(crate) const FANPAAR64_DESC: &BinaryDescription =
    <Tower<FanPaar64Spec> as BinaryRepr<64>>::DESCRIPTION;

// ---------------------------------------------------------------------------
// Rijndael specialized helpers over concrete bases.
//
// The generic wide-row bodies below delegate to these when the description
// matches the pinned Rijndael presentation, keeping the existing tower
// arithmetic (Karatsuba with the relation fold, single base inversion).

const fn aes_mul(a: u8, b: u8) -> u8 {
    Elem::<Gf<8, Poly<AES>>>::from_raw(a)
        .mul(Elem::<Gf<8, Poly<AES>>>::from_raw(b))
        .to_raw()
}

const fn aes_square(a: u8) -> u8 {
    let e = Elem::<Gf<8, Poly<AES>>>::from_raw(a);
    e.mul(e).to_raw()
}

const fn aes_inv(a: u8) -> u8 {
    Elem::<Gf<8, Poly<AES>>>::from_raw(a).inv().to_raw()
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael16_mul(a: u16, b: u16) -> u16 {
    let (x0, x1) = (a as u8, (a >> 8) as u8);
    let (y0, y1) = (b as u8, (b >> 8) as u8);
    let ac = aes_mul(x0, y0);
    let bd = aes_mul(x1, y1);
    let lo = ac ^ aes_mul(0x20, bd);
    let hi = aes_mul(x0 ^ x1, y0 ^ y1) ^ ac;
    (lo as u16) | ((hi as u16) << 8)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael16_square(a: u16) -> u16 {
    let (x0, x1) = (a as u8, (a >> 8) as u8);
    let x02 = aes_square(x0);
    let x12 = aes_square(x1);
    let lo = x02 ^ aes_mul(0x20, x12);
    (lo as u16) | ((x12 as u16) << 8)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael16_inv(a: u16) -> u16 {
    if a == 0 {
        return 0;
    }
    let (x0, x1) = (a as u8, (a >> 8) as u8);
    let x02 = aes_square(x0);
    let x12 = aes_square(x1);
    let middle = aes_mul(x0, x1);
    let norm = x02 ^ middle ^ aes_mul(0x20, x12);
    let ni = aes_inv(norm);
    let lo = aes_mul(x0 ^ x1, ni);
    let hi = aes_mul(x1, ni);
    (lo as u16) | ((hi as u16) << 8)
}

// Rijndael helpers one level up: schoolbook over the concrete subfield
// with the level's relation, so the wide bodies below stay raw-word code.

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_mul(a: u32, b: u32) -> u32 {
    let x0 = Elem::<Gf16>::from_raw(a as u16);
    let x1 = Elem::<Gf16>::from_raw((a >> 16) as u16);
    let y0 = Elem::<Gf16>::from_raw(b as u16);
    let y1 = Elem::<Gf16>::from_raw((b >> 16) as u16);
    let ac = x0.mul(y0);
    let bd = x1.mul(y1);
    let lo = ac.add(Elem::<Gf16>::from_raw(0x2000).mul(bd));
    let hi = x0.add(x1).mul(y0.add(y1)).add(ac);
    (lo.to_raw() as u32) | ((hi.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_square(a: u32) -> u32 {
    let x0 = Elem::<Gf16>::from_raw(a as u16);
    let x1 = Elem::<Gf16>::from_raw((a >> 16) as u16);
    let x02 = x0.square();
    let x12 = x1.square();
    let lo = x02.add(Elem::<Gf16>::from_raw(0x2000).mul(x12));
    (lo.to_raw() as u32) | ((x12.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_inv(a: u32) -> u32 {
    if a == 0 {
        return 0;
    }
    let x0 = Elem::<Gf16>::from_raw(a as u16);
    let x1 = Elem::<Gf16>::from_raw((a >> 16) as u16);
    let delta = Elem::<Gf16>::from_raw(0x2000);
    let norm = x0.square().add(x0.mul(x1)).add(delta.mul(x1.square()));
    let ni = norm.inv();
    let lo = x0.add(x1).mul(ni);
    let hi = x1.mul(ni);
    (lo.to_raw() as u32) | ((hi.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_mul(a: u64, b: u64) -> u64 {
    let x0 = Elem::<Gf32>::from_raw(a as u32);
    let x1 = Elem::<Gf32>::from_raw((a >> 32) as u32);
    let y0 = Elem::<Gf32>::from_raw(b as u32);
    let y1 = Elem::<Gf32>::from_raw((b >> 32) as u32);
    let ac = x0.mul(y0);
    let bd = x1.mul(y1);
    let lo = ac.add(Elem::<Gf32>::from_raw(0x2000_0000).mul(bd));
    let hi = x0.add(x1).mul(y0.add(y1)).add(ac);
    (lo.to_raw() as u64) | ((hi.to_raw() as u64) << 32)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_square(a: u64) -> u64 {
    let x0 = Elem::<Gf32>::from_raw(a as u32);
    let x1 = Elem::<Gf32>::from_raw((a >> 32) as u32);
    let x02 = x0.square();
    let x12 = x1.square();
    let lo = x02.add(Elem::<Gf32>::from_raw(0x2000_0000).mul(x12));
    (lo.to_raw() as u64) | ((x12.to_raw() as u64) << 32)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_inv(a: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    let x0 = Elem::<Gf32>::from_raw(a as u32);
    let x1 = Elem::<Gf32>::from_raw((a >> 32) as u32);
    let delta = Elem::<Gf32>::from_raw(0x2000_0000);
    let norm = x0.square().add(x0.mul(x1)).add(delta.mul(x1.square()));
    let ni = norm.inv();
    let lo = x0.add(x1).mul(ni);
    let hi = x1.mul(ni);
    (lo.to_raw() as u64) | ((hi.to_raw() as u64) << 32)
}

// ---------------------------------------------------------------------------
// Wide degree rows: 16, 32, 64.
//
// Field, BinaryField, BinaryDegree, HasGenerator, FieldBuffer, inherent const
// arithmetic with the one-body strategy branch, and Display. Degrees 2, 4,
// and 8 reuse the small-row surface in `super` (Field, coordinates, and the
// log/exp-or-interpreter arithmetic) over every `BinaryRepr`, towers
// included.

/// Emit the tower-only surface of one wide degree row: the checked
/// generator provider, the byte-buffer encoding, and the component
/// accessors. The field surface and inherent arithmetic come from the
/// degree-row macros in `super`, shared with the flat presentations.
macro_rules! tower_wide_row {
    ($n:literal, $h:literal, $raw:ty, $baseraw:ty, $half:literal,
     $bytes:literal, $order:expr, $factors:expr) => {
        impl<S: TowerGeneratorSpec> HasGenerator for Gf<$n, Tower<S>>
        where
            S::Base: BinaryDegree<$h>,
        {
            #[allow(clippy::cast_possible_truncation)]
            const GENERATOR_RAW: $raw = {
                assert!(
                    $n == 64 || S::GENERATOR >> $n == 0,
                    "tower generator has excess bits"
                );
                // Direct storage construction: `from_raw` reads field
                // validity but not this check, while this check reads the
                // arithmetic below; routing through the constructor would
                // close the const-eval cycle.
                let g = Elem::<Gf<$n, Tower<S>>> {
                    raw: S::GENERATOR as $raw,
                };
                let one = <Gf<$n, Tower<S>> as Field>::ONE_RAW;
                let order: u128 = $order - 1;
                assert!(
                    g.pow(order).to_raw() == one,
                    "tower generator does not have full order"
                );
                let mut i = 0;
                while i < $factors.len() {
                    assert!(
                        g.pow(order / $factors[i] as u128).to_raw() != one,
                        "tower generator does not have full order"
                    );
                    i += 1;
                }
                S::GENERATOR as $raw
            };
        }

        impl<S: TowerSpec> FieldBuffer for Gf<$n, Tower<S>>
        where
            S::Base: BinaryDegree<$h>,
        {
            const BYTES: usize = $bytes;
            const STORAGE_BITS: u32 = 8 * $bytes;

            #[inline]
            fn decode(bytes: &[u8]) -> Elem<Self> {
                let bytes: [u8; $bytes] = bytes
                    .try_into()
                    .expect("tower element has the wrong byte width");
                Elem::<Self>::from_bytes(bytes)
            }

            #[inline]
            fn encode(bytes: &mut [u8], value: Elem<Self>) {
                assert_eq!(
                    bytes.len(),
                    $bytes,
                    "tower element has the wrong byte width"
                );
                bytes.copy_from_slice(&value.to_bytes());
            }
        }

        impl<S: TowerSpec> Elem<Gf<$n, Tower<S>>>
        where
            S::Base: BinaryDegree<$h>,
        {
            /// Construct `a + b*t` from its two base-field components.
            #[inline]
            #[must_use]
            pub const fn from_components(a: Elem<S::Base>, b: Elem<S::Base>) -> Self
            where
                S::Base: Field<Raw = $baseraw>,
            {
                // Widening a component word into its tower half is exact.
                #[allow(clippy::cast_lossless)]
                let raw = (a.to_raw() as $raw) | ((b.to_raw() as $raw) << $half);
                Elem { raw }
            }

            /// Return the `(a, b)` components of `a + b*t`.
            #[inline]
            #[must_use]
            // Splitting the raw word into halves: the truncation IS the operation.
            #[allow(clippy::cast_possible_truncation)]
            pub const fn to_components(self) -> (Elem<S::Base>, Elem<S::Base>)
            where
                S::Base: Field<Raw = $baseraw>,
            {
                (
                    Elem::<S::Base> {
                        raw: self.raw as $baseraw,
                    },
                    Elem::<S::Base> {
                        raw: (self.raw >> $half) as $baseraw,
                    },
                )
            }
        }
    };
}

tower_wide_row!(16, 8, u16, u8, 8, 2, 1u128 << 16, &[3u64, 5, 17, 257]);
tower_wide_row!(
    32,
    16,
    u32,
    u16,
    16,
    4,
    1u128 << 32,
    &[3u64, 5, 17, 257, 65_537]
);
tower_wide_row!(
    64,
    32,
    u64,
    u32,
    32,
    8,
    1u128 << 64,
    &[3u64, 5, 17, 257, 641, 65_537, 6_700_417]
);

// ---------------------------------------------------------------------------
// Generators for the small tower rows, checked through the description.

macro_rules! tower_small_generator {
    ($n:literal, $h:literal) => {
        impl<S: TowerGeneratorSpec> HasGenerator for Gf<$n, Tower<S>>
        where
            S::Base: BinaryDegree<$h>,
        {
            #[allow(clippy::cast_possible_truncation)]
            const GENERATOR_RAW: u8 = {
                assert!(S::GENERATOR >> $n == 0, "tower generator has excess bits");
                assert!(
                    <Tower<S> as BinaryRepr<$n>>::DESCRIPTION.is_generator(S::GENERATOR),
                    "tower generator does not have full order"
                );
                S::GENERATOR as u8
            };
        }
    };
}

tower_small_generator!(2, 1);
tower_small_generator!(4, 2);
tower_small_generator!(8, 4);

// Multiply by the tower indeterminate, for the nibble-table builders that
// resolve Fan-Paar factors in coefficient preparation. The small-row
// arithmetic itself stays shared; only this factor helper is Fan-Paar
// specific.

impl Elem<FanPaar8> {
    /// Multiply by this level's tower indeterminate `X`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub const fn mul_alpha(self) -> Self {
        Elem {
            raw: fp_mul_alpha(self.raw as u64, 8) as u8,
        }
    }
}

impl Elem<FanPaar16> {
    /// Multiply by this level's tower indeterminate `X`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub const fn mul_alpha(self) -> Self {
        Elem {
            raw: fp_mul_alpha(self.raw as u64, 16) as u16,
        }
    }
}

impl Elem<FanPaar32> {
    /// Multiply by this level's tower indeterminate `X`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub const fn mul_alpha(self) -> Self {
        Elem {
            raw: fp_mul_alpha(self.raw as u64, 32) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use super::*;

    /// Independent schoolbook product over one tower's base, reducing
    /// `(a + b t)(c + d t)` by `t^2 = A t + B` with the base's own
    /// arithmetic, never the tower formulas under test.
    #[allow(clippy::cast_possible_truncation)]
    fn schoolbook16<S: TowerSpec>(x: u16, y: u16, a: u64, b: u64) -> u16
    where
        S::Base: BinaryDegree<8>,
    {
        let mul = |p: u8, q: u8| {
            let desc = <S::Base as BinaryField>::DESCRIPTION;
            desc.mul(u64::from(p), u64::from(q)) as u8
        };
        let (a0, a1) = (x as u8, (x >> 8) as u8);
        let (b0, b1) = (y as u8, (y >> 8) as u8);
        let ac = mul(a0, b0);
        let bd = mul(a1, b1);
        let ad = mul(a0, b1);
        let bc = mul(a1, b0);
        let lo = ac ^ mul(b as u8, bd);
        let hi = ad ^ bc ^ mul(a as u8, bd);
        u16::from_le_bytes([lo, hi])
    }

    #[allow(clippy::cast_possible_truncation)]
    fn tower_sample16(seeds: &[u16]) -> Vec<u16> {
        let mut values = vec![0u16, 1, 2, 0x00ff, 0x0100, 0xff00, 0xffff];
        values.extend_from_slice(seeds);
        let mut state = 0x0123_4567_89ab_cdefu64;
        for _ in 0..64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            values.push((state >> 32) as u16);
        }
        values
    }

    fn check_matches_schoolbook16<S: TowerSpec>(a: u64, b: u64, generator: u16)
    where
        S::Base: BinaryDegree<8>,
    {
        for &x in &tower_sample16(&[generator]) {
            for &y in &tower_sample16(&[generator]) {
                let got = Elem::<Gf<16, Tower<S>>>::from_raw(x)
                    .mul(Elem::<Gf<16, Tower<S>>>::from_raw(y))
                    .to_raw();
                assert_eq!(got, schoolbook16::<S>(x, y, a, b), "{x:04x} * {y:04x}");
            }
        }
    }

    fn check_field_laws16<S: TowerSpec>(generator: u16)
    where
        S::Base: BinaryDegree<8>,
    {
        let one = Elem::<Gf<16, Tower<S>>>::ONE;
        for &x in &tower_sample16(&[generator]) {
            let a = Elem::<Gf<16, Tower<S>>>::from_raw(x);
            assert_eq!(a.add(a), Elem::<Gf<16, Tower<S>>>::ZERO);
            assert_eq!(a.mul(one), a);
            assert_eq!(a.square(), a.mul(a));
            if !a.is_zero() {
                assert_eq!(a.mul(a.inv()), one);
            }
            assert_eq!(
                a.div(Elem::<Gf<16, Tower<S>>>::ZERO),
                Elem::<Gf<16, Tower<S>>>::ZERO
            );
            assert_eq!(a.pow(0), one);
        }
    }

    fn check_generator_order16<S: TowerGeneratorSpec>(factors: &[u128])
    where
        S::Base: BinaryDegree<8>,
    {
        let g = Elem::<Gf<16, Tower<S>>>::GENERATOR;
        assert_eq!(g.pow(65_535), Elem::<Gf<16, Tower<S>>>::ONE);
        for &q in factors {
            assert_ne!(g.pow(65_535 / q), Elem::<Gf<16, Tower<S>>>::ONE);
        }
    }

    #[test]
    fn rijndael16_matches_schoolbook() {
        check_matches_schoolbook16::<Rijndael16>(1, 0x20, 0x0108);
    }

    #[test]
    fn rijndael16_field_laws() {
        check_field_laws16::<Rijndael16>(0x0108);
    }

    #[test]
    fn rijndael16_generator_order() {
        check_generator_order16::<Rijndael16>(&[3, 5, 17, 257]);
    }

    #[test]
    fn fanpaar16_matches_schoolbook() {
        check_matches_schoolbook16::<FanPaar16Spec>(0x10, 1, 0xe2de);
    }

    #[test]
    fn fanpaar16_field_laws() {
        check_field_laws16::<FanPaar16Spec>(0xe2de);
    }

    #[test]
    fn fanpaar16_generator_order() {
        check_generator_order16::<FanPaar16Spec>(&[3, 5, 17, 257]);
    }

    #[test]
    fn fanpaar_small_levels_match_recurrence() {
        // Known Wiedemann vectors, independent of the descriptor interpreter.
        assert_eq!(fp_multiply(0b10, 0b11, 2), 0b01);
        assert_eq!(fp_square(0x10, 8), 0x41);
        assert_eq!(fp_invert(0b0100, 4), 0b0110);
        assert_eq!(fp_invert(0x10, 8), 0x14);
        // Every level satisfies its defining quadratic.
        for bits in [2u32, 4, 8, 16, 32, 64] {
            let half = bits / 2;
            let generator = 1u64 << half;
            let sub = if half == 1 { 1 } else { 1u64 << (half / 2) };
            assert_eq!(fp_square(generator, bits), 1 | (sub << half));
        }
        // Small tower elements agree with the recurrence.
        for (x, y) in [(0x2u8, 0x3u8), (0x1u8, 0x1u8), (0x0u8, 0x2u8)] {
            let got = Elem::<FanPaar2>::from_raw(x)
                .mul(Elem::<FanPaar2>::from_raw(y))
                .to_raw();
            assert_eq!(u64::from(got), fp_multiply(u64::from(x), u64::from(y), 2));
        }
        let g2 = Elem::<FanPaar2>::GENERATOR;
        assert_eq!(g2.pow(3), Elem::<FanPaar2>::ONE);
        let g4 = Elem::<FanPaar4>::GENERATOR;
        assert_eq!(g4.pow(15), Elem::<FanPaar4>::ONE);
        let g8 = Elem::<FanPaar8>::GENERATOR;
        assert_eq!(g8.pow(255), Elem::<FanPaar8>::ONE);
    }

    #[test]
    fn rijndael32_matches_schoolbook() {
        // Splitting words into halves and widening products back is exact
        // over these fixed widths.
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        let mul = |x: u32, y: u32| {
            let (a0, a1) = (x as u16, (x >> 16) as u16);
            let (b0, b1) = (y as u16, (y >> 16) as u16);
            let ac = rijndael16_mul(a0, b0);
            let bd = rijndael16_mul(a1, b1);
            let ad = rijndael16_mul(a0, b1);
            let bc = rijndael16_mul(a1, b0);
            let lo = ac ^ rijndael16_mul(0x2000, bd);
            let hi = ad ^ bc ^ bd;
            (lo as u32) | ((hi as u32) << 16)
        };
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut values = vec![
            0u32,
            1,
            2,
            0x00ff,
            0x0001_0000,
            0xffff_0000,
            0xffff_ffff,
            0x0001_0002,
        ];
        for _ in 0..64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            values.push((state >> 32) as u32);
        }
        for &x in &values {
            for &y in &values {
                let got = Elem::<Gf32>::from_raw(x)
                    .mul(Elem::<Gf32>::from_raw(y))
                    .to_raw();
                assert_eq!(got, mul(x, y), "{x:08x} * {y:08x}");
            }
        }
        for &x in &values {
            let a = Elem::<Gf32>::from_raw(x);
            assert_eq!(a.square(), a.mul(a));
            if !a.is_zero() {
                assert_eq!(a.mul(a.inv()), Elem::<Gf32>::ONE);
            }
        }
        let g = Elem::<Gf32>::GENERATOR;
        assert_eq!(g.pow(u128::from(u32::MAX)), Elem::<Gf32>::ONE);
        for q in [3u128, 5, 17, 257, 65_537] {
            assert_ne!(g.pow(u128::from(u32::MAX) / q), Elem::<Gf32>::ONE);
        }
    }

    #[test]
    fn fanpaar32_matches_schoolbook() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        let base_mul = |x: u32, y: u32| (fp_multiply(x as u64, y as u64, 16) & 0xffff) as u16;
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        let mul = |x: u32, y: u32| {
            let (a0, a1) = (x & 0xffff, x >> 16);
            let (b0, b1) = (y & 0xffff, y >> 16);
            let ac = u32::from(base_mul(a0, b0));
            let bd = u32::from(base_mul(a1, b1));
            let ad = u32::from(base_mul(a0, b1));
            let bc = u32::from(base_mul(a1, b0));
            let alpha_bd = u32::from(base_mul(0x0100, bd));
            let lo = ac ^ bd;
            let hi = ad ^ bc ^ alpha_bd;
            (lo & 0xffff) | ((hi & 0xffff) << 16)
        };
        let mut state = 0x0245_68ac_e135_79bdu64;
        let mut values = vec![
            0u32,
            1,
            2,
            0x00ff,
            0x0001_0000,
            0xffff_0000,
            0xffff_ffff,
            0x03e2_1cea,
        ];
        for _ in 0..64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            values.push((state >> 32) as u32);
        }
        for &x in &values {
            for &y in &values {
                let got = Elem::<FanPaar32>::from_raw(x)
                    .mul(Elem::<FanPaar32>::from_raw(y))
                    .to_raw();
                assert_eq!(got, mul(x, y), "{x:08x} * {y:08x}");
            }
        }
        let g = Elem::<FanPaar32>::GENERATOR;
        assert_eq!(g.pow(u128::from(u32::MAX)), Elem::<FanPaar32>::ONE);
        for q in [3u128, 5, 17, 257, 65_537] {
            assert_ne!(g.pow(u128::from(u32::MAX) / q), Elem::<FanPaar32>::ONE);
        }
    }

    #[test]
    fn wide_field_facts() {
        assert_eq!(<Gf64 as Field>::DEGREE, 64);
        assert_eq!(<Gf64 as Field>::ORDER, 1u128 << 64);
        assert_eq!(<Gf64 as FieldBuffer>::BYTES, 8);
        assert_eq!(<Gf32 as FieldBuffer>::BYTES, 4);
        let g = Elem::<Gf64>::GENERATOR;
        assert_eq!(g.pow(u128::from(u64::MAX)), Elem::<Gf64>::ONE);
        for q in [3u128, 5, 17, 257, 641, 65_537, 6_700_417] {
            assert_ne!(g.pow(u128::from(u64::MAX) / q), Elem::<Gf64>::ONE);
        }
        // Coordinate round trip of `u64::MAX`.
        let max = <Gf64 as BinaryField>::from_coordinates(u64::MAX).expect("full-width word");
        assert_eq!(<Gf64 as BinaryField>::to_coordinates(max), u64::MAX);
        // Excess-bit rejection below the cap.
        assert!(<Gf32 as BinaryField>::from_coordinates(1u64 << 32).is_err());
        assert!(<Gf16 as BinaryField>::from_coordinates(1u64 << 16).is_err());
        // Masks never shift by 64: the full word is accepted.
        assert!(<Gf64 as BinaryField>::from_coordinates(u64::MAX).is_ok());
    }

    #[test]
    fn fanpaar64_matches_reference_pair() {
        // Forward fixture from the pre-cutover tower, via the recurrence.
        assert_eq!(
            fp_multiply(0xc84d_6191_1083_1cef, 0x0000_0000_0000_a14f, 64),
            0x3565_086d_6b9e_f595
        );
        let g = Elem::<FanPaar64>::GENERATOR;
        assert_eq!(g.pow(u128::from(u64::MAX)), Elem::<FanPaar64>::ONE);
    }

    /// A custom tower over a normal-basis degree-8 base.
    #[derive(Clone, Copy)]
    struct CustomNormal8;

    impl TowerSpec for CustomNormal8 {
        type Base = Gf<8, crate::field::Normal<0x11B, 0x20>>;
        const A: u64 = 0xFF;
        const B: u64 = transport_poly(0x20);
        const NAME: &'static str = "custom normal tower";
    }

    const fn transport_poly(word: u64) -> u64 {
        let from = crate::field::binary::normal::inverse_columns(
            crate::field::binary::normal::to_base::<0x11B, 0x20, 8>(),
            8,
        );
        let mut image = 0u64;
        let mut i = 0;
        while i < 8 {
            if (word >> i) & 1 == 1 {
                image ^= from[i] as u64;
            }
            i += 1;
        }
        image
    }

    #[test]
    fn custom_normal_tower_zero_one_inverse() {
        type F = Gf<16, Tower<CustomNormal8>>;
        assert_eq!(Elem::<F>::ZERO.to_raw(), 0);
        // The actual one holds the base one in the low component.
        let base_one = <CustomNormal8 as TowerSpec>::Base::DESCRIPTION.one();
        assert_eq!(base_one, 0xFF);
        assert_eq!(Elem::<F>::ONE.to_raw(), 0xFF);
        let mut state = 0x1357_9bdf_0246_8aceu64;
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_possible_truncation)]
            let x = (state >> 32) as u16;
            let a = Elem::<F>::from_raw(x);
            if !a.is_zero() {
                assert_eq!(a.mul(a.inv()), Elem::<F>::ONE, "{x:04x}");
            }
        }
    }
}
