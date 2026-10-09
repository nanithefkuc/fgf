//! The shipped quadratic binary extension instances.
//!
//! The Rijndael and Fan-Paar towers are [`ExtensionSpec<2>`] instances: each
//! names the field `FiniteField<2, N, Extension<S, 2>>` as `a + b*t` over
//! the previous level, under the relation `t^2 + c1*t + c0 = 0`. The low
//! `N/2` coordinate bits hold `a` and the high `N/2` bits hold `b`; storage
//! bits at or above `N` are zero. One is the base one in the low component.
//! The scalar identities are:
//!
//! ```text
//! product:   (a*c + c0*b*d, a*d + b*c + c1*b*d)
//! square:    (a^2 + c0*b^2, c1*b^2)
//! conjugate: (a + c1*b, b)
//! norm:      a^2 + c1*a*b + c0*b^2
//! ```
//!
//! [`Rijndael16`], [`Rijndael32`], and [`Rijndael64`] root at the AES field
//! with `c1 = 1` and `c0` equal to `0x20`, `0x2000`, and `0x2000_0000`. The
//! Fan-Paar specs nest from GF(2) through the Wiedemann relation
//! `t^2 + alpha*t + 1`, where `alpha` is the previous level's
//! indeterminate (`1` over GF(2), then the raw `base_one << (h/2)` in its
//! own packing). [`FanPaar2`] and [`FanPaar4`] keep their coordinate word
//! in a padded byte and carry no byte buffer; every other instance fills
//! its `u8`, `u16`, `u32`, or `u64` word exactly and encodes it
//! little-endian.
//!
//! Each instance carries its frozen generator and a full `const` surface on
//! its concrete element type: constructors, byte and component conversions,
//! and arithmetic, plus `mul_alpha` on Fan-Paar GF(2^8), GF(2^16), and
//! GF(2^32). The `const` methods and the row's [`Field`](crate::Field)
//! arithmetic share one specialized body per instance: the Rijndael rows run
//! the Karatsuba recurrence over the AES field, the wide Fan-Paar rows the
//! Wiedemann recurrence, and Fan-Paar GF(2^2) through GF(2^8) the
//! descriptor interpreter. An extension row whose presentation equals a
//! shipped instance's — base presentation, relation, and storage — runs
//! that instance's body; every other row runs the generic extension body.
//!
//! ```
//! use fgf::{Elem, Extension, FiniteField, Rijndael16};
//!
//! type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
//!
//! let x = Elem::<Gf16>::from_raw(0x1234);
//! assert_eq!(x.mul(Elem::<Gf16>::ONE), x);
//! assert_eq!(Elem::<Gf16>::GENERATOR.pow(65_535), Elem::<Gf16>::ONE);
//!
//! // Components are the base-field coefficients of `1` and `t`.
//! let [low, high] = x.to_components();
//! assert_eq!((low.to_raw(), high.to_raw()), (0x34, 0x12));
//! assert_eq!(Elem::<Gf16>::from_components([low, high]), x);
//! ```

use super::BinaryField;
use super::description::BinaryDescription;
use super::poly::{AES, Polynomial};
use crate::field::extension::{Extension, ExtensionSpec, Packed, PackedPadded, Relation};
use crate::field::{Elem, FiniteField, HasGenerator, Validate};

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

/// The Rijndael-rooted extension of the AES field: `t^2 + t + 0x20`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael16;

impl ExtensionSpec<2> for Rijndael16 {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<AES>>>(&[0x20, 0x01]);
}

/// The Rijndael-rooted extension of Rijndael GF(2^16): `v^2 + v + 0x2000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael32;

impl ExtensionSpec<2> for Rijndael32 {
    type Base = Rijndael16Field;
    type Storage = Packed<u32>;
    const NAME: &'static str = "GF(2^32)";
    const RELATION: Relation = Relation::from_binary_words::<Rijndael16Field>(&[0x2000, 0x0001]);
}

/// The Rijndael-rooted extension of Rijndael GF(2^32):
/// `w^2 + w + 0x2000_0000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rijndael64;

impl ExtensionSpec<2> for Rijndael64 {
    type Base = Rijndael32Field;
    type Storage = Packed<u64>;
    const NAME: &'static str = "GF(2^64)";
    const RELATION: Relation =
        Relation::from_binary_words::<Rijndael32Field>(&[0x2000_0000, 0x0000_0001]);
}

/// Fan-Paar GF(2^2) over GF(2): `t^2 + t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar2;

impl ExtensionSpec<2> for FanPaar2 {
    type Base = FiniteField<2, 1, Polynomial<3>>;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "Fan-Paar GF(2^2)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[1, 1]);
}

/// Fan-Paar GF(2^4) over Fan-Paar GF(2^2): `t^2 + 0x2*t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar4;

impl ExtensionSpec<2> for FanPaar4 {
    type Base = FanPaar2Field;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "Fan-Paar GF(2^4)";
    const RELATION: Relation = Relation::from_binary_words::<FanPaar2Field>(&[1, 0x2]);
}

/// Fan-Paar GF(2^8) over Fan-Paar GF(2^4): `t^2 + 0x4*t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar8;

impl ExtensionSpec<2> for FanPaar8 {
    type Base = FanPaar4Field;
    type Storage = Packed<u8>;
    const NAME: &'static str = "Fan-Paar GF(2^8)";
    const RELATION: Relation = Relation::from_binary_words::<FanPaar4Field>(&[1, 0x4]);
}

/// Fan-Paar GF(2^16) over Fan-Paar GF(2^8): `t^2 + 0x10*t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar16;

impl ExtensionSpec<2> for FanPaar16 {
    type Base = FanPaar8Field;
    type Storage = Packed<u16>;
    const NAME: &'static str = "Fan-Paar GF(2^16)";
    const RELATION: Relation = Relation::from_binary_words::<FanPaar8Field>(&[1, 0x10]);
}

/// Fan-Paar GF(2^32) over Fan-Paar GF(2^16): `t^2 + 0x0100*t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar32;

impl ExtensionSpec<2> for FanPaar32 {
    type Base = FanPaar16Field;
    type Storage = Packed<u32>;
    const NAME: &'static str = "Fan-Paar GF(2^32)";
    const RELATION: Relation = Relation::from_binary_words::<FanPaar16Field>(&[1, 0x0100]);
}

/// Fan-Paar GF(2^64) over Fan-Paar GF(2^32): `t^2 + 0x0001_0000*t + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FanPaar64;

impl ExtensionSpec<2> for FanPaar64 {
    type Base = FanPaar32Field;
    type Storage = Packed<u64>;
    const NAME: &'static str = "Fan-Paar GF(2^64)";
    const RELATION: Relation = Relation::from_binary_words::<FanPaar32Field>(&[1, 0x0001_0000]);
}

// The shipped rows, named for the crate's own kernels, tables, and tests.
pub(crate) type Rijndael16Field = FiniteField<2, 16, Extension<Rijndael16, 2>>;
pub(crate) type Rijndael32Field = FiniteField<2, 32, Extension<Rijndael32, 2>>;
pub(crate) type Rijndael64Field = FiniteField<2, 64, Extension<Rijndael64, 2>>;
pub(crate) type FanPaar2Field = FiniteField<2, 2, Extension<FanPaar2, 2>>;
pub(crate) type FanPaar4Field = FiniteField<2, 4, Extension<FanPaar4, 2>>;
pub(crate) type FanPaar8Field = FiniteField<2, 8, Extension<FanPaar8, 2>>;
pub(crate) type FanPaar16Field = FiniteField<2, 16, Extension<FanPaar16, 2>>;
pub(crate) type FanPaar32Field = FiniteField<2, 32, Extension<FanPaar32, 2>>;
pub(crate) type FanPaar64Field = FiniteField<2, 64, Extension<FanPaar64, 2>>;

// The shipped rows' binary descriptions: the descriptor-interpreter bodies
// of the small Fan-Paar rows, the embedding reference ladder, and the
// kernel tests' structural checks.
pub(crate) const RIJNDAEL16_DESC: &BinaryDescription =
    <Rijndael16Field as BinaryField>::DESCRIPTION;
pub(crate) const RIJNDAEL32_DESC: &BinaryDescription =
    <Rijndael32Field as BinaryField>::DESCRIPTION;
pub(crate) const RIJNDAEL64_DESC: &BinaryDescription =
    <Rijndael64Field as BinaryField>::DESCRIPTION;
pub(crate) const FANPAAR2_DESC: &BinaryDescription = <FanPaar2Field as BinaryField>::DESCRIPTION;
pub(crate) const FANPAAR4_DESC: &BinaryDescription = <FanPaar4Field as BinaryField>::DESCRIPTION;
pub(crate) const FANPAAR8_DESC: &BinaryDescription = <FanPaar8Field as BinaryField>::DESCRIPTION;

// ---------------------------------------------------------------------------
// Rijndael specialized helpers over concrete bases.
//
// The generic wide-row bodies below delegate to these when the description
// matches the pinned Rijndael presentation, keeping the existing tower
// arithmetic (Karatsuba with the relation fold, single base inversion).

const fn aes_mul(a: u8, b: u8) -> u8 {
    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(a)
        .mul(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(b))
        .to_raw()
}

const fn aes_square(a: u8) -> u8 {
    let e = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(a);
    e.mul(e).to_raw()
}

const fn aes_inv(a: u8) -> u8 {
    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(a)
        .inv()
        .to_raw()
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
// with the level's relation, so the instance bodies stay raw-word code.

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_mul(a: u32, b: u32) -> u32 {
    let x0 = Elem::<Rijndael16Field>::from_raw(a as u16);
    let x1 = Elem::<Rijndael16Field>::from_raw((a >> 16) as u16);
    let y0 = Elem::<Rijndael16Field>::from_raw(b as u16);
    let y1 = Elem::<Rijndael16Field>::from_raw((b >> 16) as u16);
    let ac = x0.mul(y0);
    let bd = x1.mul(y1);
    let lo = ac.add(Elem::<Rijndael16Field>::from_raw(0x2000).mul(bd));
    let hi = x0.add(x1).mul(y0.add(y1)).add(ac);
    (lo.to_raw() as u32) | ((hi.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_square(a: u32) -> u32 {
    let x0 = Elem::<Rijndael16Field>::from_raw(a as u16);
    let x1 = Elem::<Rijndael16Field>::from_raw((a >> 16) as u16);
    let x02 = x0.square();
    let x12 = x1.square();
    let lo = x02.add(Elem::<Rijndael16Field>::from_raw(0x2000).mul(x12));
    (lo.to_raw() as u32) | ((x12.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael32_inv(a: u32) -> u32 {
    if a == 0 {
        return 0;
    }
    let x0 = Elem::<Rijndael16Field>::from_raw(a as u16);
    let x1 = Elem::<Rijndael16Field>::from_raw((a >> 16) as u16);
    let delta = Elem::<Rijndael16Field>::from_raw(0x2000);
    let norm = x0.square().add(x0.mul(x1)).add(delta.mul(x1.square()));
    let ni = norm.inv();
    let lo = x0.add(x1).mul(ni);
    let hi = x1.mul(ni);
    (lo.to_raw() as u32) | ((hi.to_raw() as u32) << 16)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_mul(a: u64, b: u64) -> u64 {
    let x0 = Elem::<Rijndael32Field>::from_raw(a as u32);
    let x1 = Elem::<Rijndael32Field>::from_raw((a >> 32) as u32);
    let y0 = Elem::<Rijndael32Field>::from_raw(b as u32);
    let y1 = Elem::<Rijndael32Field>::from_raw((b >> 32) as u32);
    let ac = x0.mul(y0);
    let bd = x1.mul(y1);
    let lo = ac.add(Elem::<Rijndael32Field>::from_raw(0x2000_0000).mul(bd));
    let hi = x0.add(x1).mul(y0.add(y1)).add(ac);
    (lo.to_raw() as u64) | ((hi.to_raw() as u64) << 32)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_square(a: u64) -> u64 {
    let x0 = Elem::<Rijndael32Field>::from_raw(a as u32);
    let x1 = Elem::<Rijndael32Field>::from_raw((a >> 32) as u32);
    let x02 = x0.square();
    let x12 = x1.square();
    let lo = x02.add(Elem::<Rijndael32Field>::from_raw(0x2000_0000).mul(x12));
    (lo.to_raw() as u64) | ((x12.to_raw() as u64) << 32)
}

// Splitting the raw word into halves: the truncation IS the operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn rijndael64_inv(a: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    let x0 = Elem::<Rijndael32Field>::from_raw(a as u32);
    let x1 = Elem::<Rijndael32Field>::from_raw((a >> 32) as u32);
    let delta = Elem::<Rijndael32Field>::from_raw(0x2000_0000);
    let norm = x0.square().add(x0.mul(x1)).add(delta.mul(x1.square()));
    let ni = norm.inv();
    let lo = x0.add(x1).mul(ni);
    let hi = x1.mul(ni);
    (lo.to_raw() as u64) | ((hi.to_raw() as u64) << 32)
}

// ---------------------------------------------------------------------------
// Fan-Paar bodies at each instance's raw width.
//
// The wide rows run the Wiedemann recurrence; GF(2^2) through GF(2^8) run
// the descriptor interpreter over their own description. Every word holds
// at most the row's coordinate bits, so the narrowing casts drop only zero
// bits.

macro_rules! fan_paar_bodies {
    ($raw:ty, $mul:ident, $square:ident, $inv:ident, recurrence: $bits:literal) => {
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        pub(crate) const fn $mul(a: $raw, b: $raw) -> $raw {
            fp_multiply(a as u64, b as u64, $bits) as $raw
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        pub(crate) const fn $square(a: $raw) -> $raw {
            fp_square(a as u64, $bits) as $raw
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        pub(crate) const fn $inv(a: $raw) -> $raw {
            fp_invert(a as u64, $bits) as $raw
        }
    };
    ($mul:ident, $square:ident, $inv:ident, description: $description:ident) => {
        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        pub(crate) const fn $mul(a: u8, b: u8) -> u8 {
            $description.mul(a as u64, b as u64) as u8
        }

        pub(crate) const fn $square(a: u8) -> u8 {
            $mul(a, a)
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
        pub(crate) const fn $inv(a: u8) -> u8 {
            $description.inv(a as u64) as u8
        }
    };
}

fan_paar_bodies!(fanpaar2_mul, fanpaar2_square, fanpaar2_inv, description: FANPAAR2_DESC);
fan_paar_bodies!(fanpaar4_mul, fanpaar4_square, fanpaar4_inv, description: FANPAAR4_DESC);
fan_paar_bodies!(fanpaar8_mul, fanpaar8_square, fanpaar8_inv, description: FANPAAR8_DESC);
fan_paar_bodies!(u16, fanpaar16_mul, fanpaar16_square, fanpaar16_inv, recurrence: 16);
fan_paar_bodies!(u32, fanpaar32_mul, fanpaar32_square, fanpaar32_inv, recurrence: 32);
fan_paar_bodies!(u64, fanpaar64_mul, fanpaar64_square, fanpaar64_inv, recurrence: 64);

// ---------------------------------------------------------------------------
// Instance const surfaces and generators.

/// Emit the inherent `const` surface of one shipped instance: construction,
/// bytes, and arithmetic over the instance's specialized bodies.
macro_rules! instance_surface {
    (
        $field:ty, $raw:ty, $bytes:literal, mask: $mask:expr,
        mul: $mul:path, square: $square:path, inv: $inv:path
    ) => {
        impl Elem<$field> {
            /// Wrap a raw storage word, keeping the coordinate bits.
            #[inline]
            #[must_use]
            pub const fn from_raw(value: $raw) -> Self {
                let () = Validate::<$field>::OK;
                Elem { raw: value & $mask }
            }

            /// Decode from the stable little-endian representation.
            #[inline]
            #[must_use]
            pub const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                Self::from_raw(<$raw>::from_le_bytes(bytes))
            }

            /// Encode to the stable little-endian representation.
            #[inline]
            #[must_use]
            pub const fn to_bytes(self) -> [u8; $bytes] {
                self.raw.to_le_bytes()
            }

            /// Field addition. Identical to subtraction and to bitwise XOR.
            #[inline]
            #[must_use]
            pub const fn add(self, rhs: Self) -> Self {
                Elem {
                    raw: self.raw ^ rhs.raw,
                }
            }

            /// Field subtraction. Identical to [`Elem::add`](Self::add).
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

            /// Field multiplication.
            #[inline]
            #[must_use]
            pub const fn mul(self, rhs: Self) -> Self {
                Elem {
                    raw: $mul(self.raw, rhs.raw),
                }
            }

            /// Square.
            #[inline]
            #[must_use]
            pub const fn square(self) -> Self {
                Elem {
                    raw: $square(self.raw),
                }
            }

            /// Multiplicative inverse. Maps zero to zero by crate convention.
            #[inline]
            #[must_use]
            pub const fn inv(self) -> Self {
                Elem {
                    raw: $inv(self.raw),
                }
            }

            /// Field division. Returns zero when either operand is zero.
            #[inline]
            #[must_use]
            pub const fn div(self, rhs: Self) -> Self {
                if self.raw == 0 || rhs.raw == 0 {
                    return Self::ZERO;
                }
                self.mul(rhs.inv())
            }

            /// Raise to an unsigned integer power. `pow(_, 0) == ONE`.
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

            /// The canonical representative of this element: the element
            /// itself, since storage is canonical by construction.
            #[inline]
            #[must_use]
            pub const fn canonical(self) -> Self {
                self
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
                self.raw == Self::ONE.raw
            }
        }
    };
}

/// Emit the component conversions of one wide instance, matching
/// [`ExtensionField<2>`](crate::ExtensionField) as `const` methods.
macro_rules! instance_components {
    ($field:ty, $base:ty, $raw:ty, $baseraw:ty, $half:literal) => {
        impl Elem<$field> {
            /// Construct `a + b*t` from its base-field components `[a, b]`.
            #[inline]
            #[must_use]
            pub const fn from_components(components: [Elem<$base>; 2]) -> Self {
                // Widening a component word into its half is exact.
                #[allow(clippy::cast_lossless)]
                let raw = (components[0].raw as $raw) | ((components[1].raw as $raw) << $half);
                Elem { raw }
            }

            /// Return the components `[a, b]` of `a + b*t`.
            #[inline]
            #[must_use]
            // Splitting the raw word into halves: the truncation IS the
            // operation.
            #[allow(clippy::cast_possible_truncation)]
            pub const fn to_components(self) -> [Elem<$base>; 2] {
                [
                    Elem {
                        raw: self.raw as $baseraw,
                    },
                    Elem {
                        raw: (self.raw >> $half) as $baseraw,
                    },
                ]
            }
        }
    };
}

/// Emit one instance's generator, proven to full multiplicative order in
/// const against the prime factors of the group order.
macro_rules! instance_generator {
    ($field:ty, $raw:ty, $generator:literal, $order:expr, $factors:expr) => {
        impl HasGenerator for $field {
            const GENERATOR_RAW: $raw = {
                // Direct storage construction keeps the proof independent
                // of `Elem::GENERATOR`, which reads this constant.
                let g = Elem::<$field> { raw: $generator };
                let one = Elem::<$field>::ONE;
                let order: u128 = $order;
                let factors: &[u128] = $factors;
                assert!(
                    g.pow(order).raw == one.raw,
                    "shipped generator does not have full order"
                );
                let mut i = 0;
                while i < factors.len() {
                    assert!(
                        g.pow(order / factors[i]).raw != one.raw,
                        "shipped generator does not have full order"
                    );
                    i += 1;
                }
                $generator
            };
        }
    };
}

instance_surface!(Rijndael16Field, u16, 2, mask: u16::MAX,
    mul: rijndael16_mul, square: rijndael16_square, inv: rijndael16_inv);
instance_surface!(Rijndael32Field, u32, 4, mask: u32::MAX,
    mul: rijndael32_mul, square: rijndael32_square, inv: rijndael32_inv);
instance_surface!(Rijndael64Field, u64, 8, mask: u64::MAX,
    mul: rijndael64_mul, square: rijndael64_square, inv: rijndael64_inv);
instance_surface!(FanPaar2Field, u8, 1, mask: 0x3,
    mul: fanpaar2_mul, square: fanpaar2_square, inv: fanpaar2_inv);
instance_surface!(FanPaar4Field, u8, 1, mask: 0xF,
    mul: fanpaar4_mul, square: fanpaar4_square, inv: fanpaar4_inv);
instance_surface!(FanPaar8Field, u8, 1, mask: u8::MAX,
    mul: fanpaar8_mul, square: fanpaar8_square, inv: fanpaar8_inv);
instance_surface!(FanPaar16Field, u16, 2, mask: u16::MAX,
    mul: fanpaar16_mul, square: fanpaar16_square, inv: fanpaar16_inv);
instance_surface!(FanPaar32Field, u32, 4, mask: u32::MAX,
    mul: fanpaar32_mul, square: fanpaar32_square, inv: fanpaar32_inv);
instance_surface!(FanPaar64Field, u64, 8, mask: u64::MAX,
    mul: fanpaar64_mul, square: fanpaar64_square, inv: fanpaar64_inv);

instance_components!(Rijndael16Field, FiniteField<2, 8, Polynomial<AES>>, u16, u8, 8);
instance_components!(Rijndael32Field, Rijndael16Field, u32, u16, 16);
instance_components!(Rijndael64Field, Rijndael32Field, u64, u32, 32);
instance_components!(FanPaar16Field, FanPaar8Field, u16, u8, 8);
instance_components!(FanPaar32Field, FanPaar16Field, u32, u16, 16);
instance_components!(FanPaar64Field, FanPaar32Field, u64, u32, 32);

instance_generator!(
    Rijndael16Field,
    u16,
    0x0108,
    (1u128 << 16) - 1,
    &[3, 5, 17, 257]
);
instance_generator!(
    Rijndael32Field,
    u32,
    0x0001_0002,
    (1u128 << 32) - 1,
    &[3, 5, 17, 257, 65_537]
);
instance_generator!(
    Rijndael64Field,
    u64,
    0x0000_0001_0000_0004,
    (1u128 << 64) - 1,
    &[3, 5, 17, 257, 641, 65_537, 6_700_417]
);
instance_generator!(FanPaar2Field, u8, 0x2, 3, &[3]);
instance_generator!(FanPaar4Field, u8, 0x7, 15, &[3, 5]);
instance_generator!(FanPaar8Field, u8, 0x2d, 255, &[3, 5, 17]);
instance_generator!(
    FanPaar16Field,
    u16,
    0xe2de,
    (1u128 << 16) - 1,
    &[3, 5, 17, 257]
);
instance_generator!(
    FanPaar32Field,
    u32,
    0x03e2_1cea,
    (1u128 << 32) - 1,
    &[3, 5, 17, 257, 65_537]
);
instance_generator!(
    FanPaar64Field,
    u64,
    0x070f_870d_cd9c_1d88,
    (1u128 << 64) - 1,
    &[3, 5, 17, 257, 641, 65_537, 6_700_417]
);

// Multiply by the tower indeterminate, for the nibble-table builders that
// resolve Fan-Paar factors in coefficient preparation.

impl Elem<FanPaar8Field> {
    /// Multiply by this level's indeterminate `X`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub const fn mul_alpha(self) -> Self {
        Elem {
            raw: fp_mul_alpha(self.raw as u64, 8) as u8,
        }
    }
}

impl Elem<FanPaar16Field> {
    /// Multiply by this level's indeterminate `X`.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub const fn mul_alpha(self) -> Self {
        Elem {
            raw: fp_mul_alpha(self.raw as u64, 16) as u16,
        }
    }
}

impl Elem<FanPaar32Field> {
    /// Multiply by this level's indeterminate `X`.
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
    use crate::field::{Field, FieldElem};

    /// Independent schoolbook product over one degree-16 instance's base,
    /// reducing `(a + b t)(c + d t)` by `t^2 = c1 t + c0` with the base's
    /// own description, never the instance formulas under test.
    #[allow(clippy::cast_possible_truncation)]
    fn schoolbook16(base: &BinaryDescription, x: u16, y: u16, c1: u64, c0: u64) -> u16 {
        let mul = |p: u8, q: u8| base.mul(u64::from(p), u64::from(q)) as u8;
        let (a0, a1) = (x as u8, (x >> 8) as u8);
        let (b0, b1) = (y as u8, (y >> 8) as u8);
        let ac = mul(a0, b0);
        let bd = mul(a1, b1);
        let ad = mul(a0, b1);
        let bc = mul(a1, b0);
        let lo = ac ^ mul(c0 as u8, bd);
        let hi = ad ^ bc ^ mul(c1 as u8, bd);
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

    /// The const product and the `Field` product of one degree-16 instance
    /// both match the schoolbook reference.
    fn check_matches_schoolbook16<F: Field<Raw = u16>>(
        mul: fn(u16, u16) -> u16,
        base: &BinaryDescription,
        c1: u64,
        c0: u64,
        generator: u16,
    ) {
        for &x in &tower_sample16(&[generator]) {
            for &y in &tower_sample16(&[generator]) {
                let want = schoolbook16(base, x, y, c1, c0);
                assert_eq!(mul(x, y), want, "const {x:04x} * {y:04x}");
                assert_eq!(F::mul_raw(x, y), want, "Field {x:04x} * {y:04x}");
            }
        }
    }

    macro_rules! check_field_laws16 {
        ($field:ty, $generator:expr) => {{
            let one = Elem::<$field>::ONE;
            for &x in &tower_sample16(&[$generator]) {
                let a = Elem::<$field>::from_raw(x);
                assert_eq!(a.add(a), Elem::<$field>::ZERO);
                assert_eq!(a.mul(one), a);
                assert_eq!(a.square(), a.mul(a));
                if !a.is_zero() {
                    assert_eq!(a.mul(a.inv()), one);
                }
                assert_eq!(a.div(Elem::<$field>::ZERO), Elem::<$field>::ZERO);
                assert_eq!(a.pow(0), one);
            }
        }};
    }

    macro_rules! check_generator_order16 {
        ($field:ty, $factors:expr) => {{
            let g = Elem::<$field>::GENERATOR;
            assert_eq!(g.pow(65_535), Elem::<$field>::ONE);
            for &q in $factors {
                assert_ne!(g.pow(65_535 / q), Elem::<$field>::ONE);
            }
        }};
    }

    #[test]
    fn rijndael16_matches_schoolbook() {
        check_matches_schoolbook16::<Rijndael16Field>(
            rijndael16_mul,
            <FiniteField<2, 8, Polynomial<AES>> as BinaryField>::DESCRIPTION,
            1,
            0x20,
            0x0108,
        );
    }

    #[test]
    fn rijndael16_field_laws() {
        check_field_laws16!(Rijndael16Field, 0x0108);
    }

    #[test]
    fn rijndael16_generator_order() {
        check_generator_order16!(Rijndael16Field, &[3u128, 5, 17, 257]);
    }

    #[test]
    fn fanpaar16_matches_schoolbook() {
        check_matches_schoolbook16::<FanPaar16Field>(fanpaar16_mul, FANPAAR8_DESC, 0x10, 1, 0xe2de);
    }

    #[test]
    fn fanpaar16_field_laws() {
        check_field_laws16!(FanPaar16Field, 0xe2de);
    }

    #[test]
    fn fanpaar16_generator_order() {
        check_generator_order16!(FanPaar16Field, &[3u128, 5, 17, 257]);
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
        // Small instance elements agree with the recurrence, through the
        // const surface and the `Field` body alike.
        for x in 0..4u8 {
            for y in 0..4u8 {
                let want = fp_multiply(u64::from(x), u64::from(y), 2);
                let got = Elem::<FanPaar2Field>::from_raw(x)
                    .mul(Elem::<FanPaar2Field>::from_raw(y))
                    .to_raw();
                assert_eq!(u64::from(got), want);
                assert_eq!(u64::from(FanPaar2Field::mul_raw(x, y)), want);
            }
        }
        for x in 0..16u8 {
            for y in 0..16u8 {
                let want = fp_multiply(u64::from(x), u64::from(y), 4);
                assert_eq!(u64::from(fanpaar4_mul(x, y)), want);
                assert_eq!(u64::from(FanPaar4Field::mul_raw(x, y)), want);
            }
        }
        for x in 0..=u8::MAX {
            for y in [0u8, 1, 2, 0x2d, 0x80, 0xff] {
                let want = fp_multiply(u64::from(x), u64::from(y), 8);
                assert_eq!(u64::from(fanpaar8_mul(x, y)), want);
                assert_eq!(u64::from(FanPaar8Field::mul_raw(x, y)), want);
            }
        }
        let g2 = Elem::<FanPaar2Field>::GENERATOR;
        assert_eq!(g2.pow(3), Elem::<FanPaar2Field>::ONE);
        let g4 = Elem::<FanPaar4Field>::GENERATOR;
        assert_eq!(g4.pow(15), Elem::<FanPaar4Field>::ONE);
        let g8 = Elem::<FanPaar8Field>::GENERATOR;
        assert_eq!(g8.pow(255), Elem::<FanPaar8Field>::ONE);
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
                let want = mul(x, y);
                let got = Elem::<Rijndael32Field>::from_raw(x)
                    .mul(Elem::<Rijndael32Field>::from_raw(y))
                    .to_raw();
                assert_eq!(got, want, "const {x:08x} * {y:08x}");
                assert_eq!(
                    Rijndael32Field::mul_raw(x, y),
                    want,
                    "Field {x:08x} * {y:08x}"
                );
            }
        }
        for &x in &values {
            let a = Elem::<Rijndael32Field>::from_raw(x);
            assert_eq!(a.square(), a.mul(a));
            if !a.is_zero() {
                assert_eq!(a.mul(a.inv()), Elem::<Rijndael32Field>::ONE);
            }
        }
        let g = Elem::<Rijndael32Field>::GENERATOR;
        assert_eq!(g.pow(u128::from(u32::MAX)), Elem::<Rijndael32Field>::ONE);
        for q in [3u128, 5, 17, 257, 65_537] {
            assert_ne!(
                g.pow(u128::from(u32::MAX) / q),
                Elem::<Rijndael32Field>::ONE
            );
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
                let want = mul(x, y);
                let got = Elem::<FanPaar32Field>::from_raw(x)
                    .mul(Elem::<FanPaar32Field>::from_raw(y))
                    .to_raw();
                assert_eq!(got, want, "const {x:08x} * {y:08x}");
                assert_eq!(
                    FanPaar32Field::mul_raw(x, y),
                    want,
                    "Field {x:08x} * {y:08x}"
                );
            }
        }
        let g = Elem::<FanPaar32Field>::GENERATOR;
        assert_eq!(g.pow(u128::from(u32::MAX)), Elem::<FanPaar32Field>::ONE);
        for q in [3u128, 5, 17, 257, 65_537] {
            assert_ne!(g.pow(u128::from(u32::MAX) / q), Elem::<FanPaar32Field>::ONE);
        }
    }

    #[test]
    fn wide_field_facts() {
        assert_eq!(<Rijndael64Field as Field>::DEGREE, 64);
        assert_eq!(<Rijndael64Field as Field>::ORDER, 1u128 << 64);
        assert_eq!(<Rijndael64Field as crate::field::FieldBuffer>::BYTES, 8);
        assert_eq!(<Rijndael32Field as crate::field::FieldBuffer>::BYTES, 4);
        let g = Elem::<Rijndael64Field>::GENERATOR;
        assert_eq!(g.pow(u128::from(u64::MAX)), Elem::<Rijndael64Field>::ONE);
        for q in [3u128, 5, 17, 257, 641, 65_537, 6_700_417] {
            assert_ne!(
                g.pow(u128::from(u64::MAX) / q),
                Elem::<Rijndael64Field>::ONE
            );
        }
        // Coordinate round trip of `u64::MAX`.
        let max =
            <Rijndael64Field as BinaryField>::from_coordinates(u64::MAX).expect("full-width word");
        assert_eq!(
            <Rijndael64Field as BinaryField>::to_coordinates(max),
            u64::MAX
        );
        // Excess-bit rejection below the cap.
        assert!(<Rijndael32Field as BinaryField>::from_coordinates(1u64 << 32).is_err());
        assert!(<Rijndael16Field as BinaryField>::from_coordinates(1u64 << 16).is_err());
    }

    #[test]
    fn fanpaar64_matches_reference_pair() {
        // Forward fixture from the pre-cutover tower, via the recurrence.
        assert_eq!(
            fp_multiply(0xc84d_6191_1083_1cef, 0x0000_0000_0000_a14f, 64),
            0x3565_086d_6b9e_f595
        );
        assert_eq!(
            FanPaar64Field::mul_raw(0xc84d_6191_1083_1cef, 0x0000_0000_0000_a14f),
            0x3565_086d_6b9e_f595
        );
        let g = Elem::<FanPaar64Field>::GENERATOR;
        assert_eq!(g.pow(u128::from(u64::MAX)), Elem::<FanPaar64Field>::ONE);
    }

    /// A custom quadratic extension over a normal-basis degree-8 base.
    struct CustomNormal8;

    impl ExtensionSpec<2> for CustomNormal8 {
        type Base = FiniteField<2, 8, crate::field::Normal<0x11B, 0x20>>;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom normal tower";
        const RELATION: Relation = Relation::from_binary_words::<
            FiniteField<2, 8, crate::field::Normal<0x11B, 0x20>>,
        >(&[transport_poly(0x20), 0xFF]);
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
        type F = FiniteField<2, 16, Extension<CustomNormal8, 2>>;
        assert_eq!(Elem::<F>::ZERO.to_raw(), 0);
        // The actual one holds the base one in the low component.
        let base_one = <CustomNormal8 as ExtensionSpec<2>>::Base::DESCRIPTION.one();
        assert_eq!(base_one, 0xFF);
        assert_eq!(Elem::<F>::ONE.to_raw(), 0xFF);
        let mut state = 0x1357_9bdf_0246_8aceu64;
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_possible_truncation)]
            let a = <Elem<F> as FieldElem>::from_raw(((state >> 32) as u16).max(1));
            assert_eq!(a * a.inv(), Elem::<F>::ONE, "{:04x}", a.to_raw());
        }
    }

    /// The transport helper the custom spec's constant is written with is a
    /// field isomorphism from the `0x11B` polynomial basis onto the normal
    /// basis: it carries one to the normal-basis one and every polynomial
    /// product to the normal-basis product of the images.
    #[test]
    fn transport_poly_is_a_field_isomorphism() {
        type Poly8 = FiniteField<2, 8, crate::field::Polynomial<0x11B>>;
        type Normal8 = FiniteField<2, 8, crate::field::Normal<0x11B, 0x20>>;
        #[allow(clippy::cast_possible_truncation)]
        let transport = |word: u8| transport_poly(core::hint::black_box(u64::from(word))) as u8;
        assert_eq!(transport(1), Elem::<Normal8>::ONE.to_raw());
        let mut images = [false; 256];
        for a in 0..=u8::MAX {
            let image = transport(a);
            assert!(!images[usize::from(image)], "{a:#04x} collides");
            images[usize::from(image)] = true;
            for b in [0u8, 1, 2, 0x20, 0x53, 0xca, 0xff] {
                let product = Elem::<Poly8>::from_raw(a)
                    .mul(Elem::<Poly8>::from_raw(b))
                    .to_raw();
                let image_product = Elem::<Normal8>::from_raw(image)
                    .mul(Elem::<Normal8>::from_raw(transport(b)))
                    .to_raw();
                assert_eq!(transport(product), image_product, "{a:#04x} * {b:#04x}");
            }
        }
    }

    /// Schoolbook Wiedemann product, written directly from the relation
    /// `X^2 = alpha*X + 1` with no Karatsuba split and no tables.
    fn wiedemann_schoolbook(lhs: u64, rhs: u64, bits: u32) -> u64 {
        if bits == 1 {
            return lhs & rhs & 1;
        }
        let half = bits / 2;
        let mask = (1u64 << half) - 1;
        let (a0, a1) = (lhs & mask, lhs >> half);
        let (b0, b1) = (rhs & mask, rhs >> half);
        let high = wiedemann_schoolbook(a1, b1, half);
        // `alpha` is the half level's own indeterminate, or one over GF(2).
        let alpha_high = if half == 1 {
            high
        } else {
            wiedemann_schoolbook(high, 1u64 << (half / 2), half)
        };
        let low = wiedemann_schoolbook(a0, b0, half) ^ high;
        let cross =
            wiedemann_schoolbook(a0, b1, half) ^ wiedemann_schoolbook(a1, b0, half) ^ alpha_high;
        low | (cross << half)
    }

    #[test]
    fn byte_log_tables_enumerate_the_generator_powers() {
        let exp = fp_build_exp8();
        let log = fp_build_log8();
        let mut seen = [false; 256];
        let mut power = 1u64;
        for (i, &entry) in exp.iter().enumerate() {
            assert_eq!(u64::from(entry), power, "0x2d^{i}");
            assert!(!seen[usize::from(entry)], "0x2d^{i} repeats");
            seen[usize::from(entry)] = true;
            assert_eq!(usize::from(log[usize::from(entry)]), i, "log 0x2d^{i}");
            power = wiedemann_schoolbook(power, 0x2d, 8);
        }
        // Full multiplicative order: the 255 powers are every nonzero byte
        // and the cycle closes.
        assert!(!seen[0]);
        assert_eq!(power, 1);
    }
}
