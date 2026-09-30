//! GFNI GF(2^8) kernels over 32-byte lanes.
//!
//! `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply, so a `Gf8B`
//! coefficient is a broadcast byte; `VGF2P8AFFINEQB` applies an arbitrary 8x8
//! GF(2) map per lane and so multiplies `Gf8D` under `0x11D`. Both are
//! pipelined but not single-cycle, which shapes every loop: the single-buffer
//! AXPY keeps four multiply chains in flight, and the blocked kernels hold a
//! destination tile in registers so the multiplier, not memory, is the limit.
//!
//! Submodules split on fan shape: `single`, `scatter`, `gather`, `matrix` over
//! the register-blocked row-group bodies in `rows`, and `elementwise`. Every
//! entry takes an `X64V3GfniCryptoToken`.
//!
//! This file holds the seam the blocked shapes share. The two byte fields
//! differ in exactly three places: how a coefficient becomes a factor vector,
//! which one-instruction multiply folds it in, and which nibble bank the
//! sub-lane tail reads. Everything else — the tiling, the register budget, the
//! remainder ladder — is identical, so the kernels are generic over
//! [`Blocked`] and monomorphize back to the code each field had alone.

mod elementwise;
mod gather;
mod matrix;
mod rows;
mod scatter;
mod single;

#[cfg(feature = "simd512")]
pub(super) use elementwise::ISO_11D_TO_11B;
pub use elementwise::{
    mul_elementwise_assign_gfni, mul_elementwise_assign_gfni_8d, mul_elementwise_gfni,
    mul_elementwise_gfni_8d,
};
pub use gather::{mul_add_gather_gfni, mul_add_gather_gfni_8d};
pub use matrix::{
    mul_add_matrix_at_gfni, mul_add_matrix_at_gfni_8d, mul_add_matrix_gfni, mul_add_matrix_gfni_8d,
    mul_add_matrix_gfni_8d_with, mul_add_matrix_gfni_with, mul_into_matrix_gfni,
    mul_into_matrix_gfni_8d, mul_into_matrix_gfni_8d_with, mul_into_matrix_gfni_with,
};
pub use scatter::{mul_add_scatter_gfni, mul_add_scatter_gfni_8d};
pub use single::{
    mul_add_gfni, mul_add_gfni_8d, mul_assign_gfni, mul_assign_gfni_8d, mul_into_gfni,
    mul_into_gfni_8d,
};
#[cfg(feature = "simd512")]
pub(super) use single::{mul_add_gfni_8d_impl, mul_assign_gfni_8d_impl, mul_into_gfni_8d_impl};

use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::tables::{ScaleTable, affine_8d, scale_table, scale_table_8d};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// The per-field multiply seam for the blocked GF(2^8) kernels.
pub(super) trait Blocked {
    /// Coefficient form selected by the strategy.
    type Coeff: Copy;
    /// The additive identity, for staging arrays.
    fn zero() -> Self::Coeff;
    /// `true` when the multiply is `VGF2P8AFFINEQB`, `false` for `GF2P8MULB`.
    const AFFINE: bool;
    /// Raw coefficient byte.
    fn byte(coeff: Self::Coeff) -> u8;
    /// The `VGF2P8AFFINEQB` matrix qword (unused when `AFFINE` is false).
    fn map(coeff: Self::Coeff) -> u64;
    /// The coefficient's sub-lane nibble tables, from this field's bank.
    fn table(coeff: Self::Coeff) -> &'static ScaleTable;
}

/// Native GFNI multiply in the AES field `0x11B` (`Gf8B`).
pub(super) enum Gfni {}

impl Blocked for Gfni {
    type Coeff = Elem;
    const AFFINE: bool = false;
    #[inline]
    fn zero() -> Elem {
        Elem(0)
    }
    #[inline]
    fn byte(coeff: Elem) -> u8 {
        coeff.0
    }
    #[inline]
    fn map(_coeff: Elem) -> u64 {
        0
    }
    #[inline]
    fn table(coeff: Elem) -> &'static ScaleTable {
        scale_table(coeff)
    }
}

/// Affine-map multiply in the Reed–Solomon field `0x11D` (`Gf8D`).
pub(super) enum Affine8D {}

impl Blocked for Affine8D {
    type Coeff = gf8d::Elem;
    const AFFINE: bool = true;
    #[inline]
    fn zero() -> gf8d::Elem {
        gf8d::Elem(0)
    }
    #[inline]
    fn byte(coeff: gf8d::Elem) -> u8 {
        coeff.0
    }
    #[inline]
    fn map(coeff: gf8d::Elem) -> u64 {
        affine_8d(coeff)
    }
    #[inline]
    fn table(coeff: gf8d::Elem) -> &'static ScaleTable {
        scale_table_8d(coeff)
    }
}

/// Broadcast a coefficient into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_gfni<S: Blocked>(coeff: S::Coeff) -> __m256i {
    if S::AFFINE {
        _mm256_set1_epi64x(S::map(coeff).cast_signed())
    } else {
        _mm256_set1_epi8(S::byte(coeff).cast_signed())
    }
}

/// Broadcast a coefficient into a 128-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_half_gfni<S: Blocked>(coeff: S::Coeff) -> __m128i {
    if S::AFFINE {
        _mm_set1_epi64x(S::map(coeff).cast_signed())
    } else {
        _mm_set1_epi8(S::byte(coeff).cast_signed())
    }
}

/// `coeff * x` over a 256-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul_gfni<S: Blocked>(x: __m256i, factor: __m256i) -> __m256i {
    if S::AFFINE {
        _mm256_gf2p8affine_epi64_epi8::<0>(x, factor)
    } else {
        _mm256_gf2p8mul_epi8(x, factor)
    }
}

/// `coeff * x` over a 128-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul_half_gfni<S: Blocked>(x: __m128i, factor: __m128i) -> __m128i {
    if S::AFFINE {
        _mm_gf2p8affine_epi64_epi8::<0>(x, factor)
    } else {
        _mm_gf2p8mul_epi8(x, factor)
    }
}

/// Single-row remainder AXPY (`dst ^= coeff * src`) for this field: whole
/// 32/16-byte lanes on the one-instruction multiply, sub-XMM tail on the
/// nibble bank.
///
/// Runs in the row-group bodies' matching feature context, so it calls the
/// shared `#[rite]` multiply bodies directly rather than the token-bearing
/// entries.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn brem_gfni<S: Blocked>(dst: &mut [u8], coeff: S::Coeff, src: &[u8]) {
    if S::AFFINE {
        single::mul_add_gfni_8d_impl(dst, S::map(coeff), S::table(coeff), src);
    } else {
        single::mul_add_gfni_impl(dst, Elem(S::byte(coeff)), src);
    }
}

/// The multiply factor of one coefficient, as a word the tile loop can
/// broadcast without touching a table: the `VGF2P8AFFINEQB` map on the affine
/// fields, the raw coefficient byte on the native `GF2P8MULB` one.
#[inline]
fn factor_word_gfni<S: Blocked>(coeff: S::Coeff) -> u64 {
    if S::AFFINE {
        S::map(coeff)
    } else {
        u64::from(S::byte(coeff))
    }
}

/// Broadcast a resolved factor word into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn wfactor_gfni<S: Blocked>(word: u64) -> __m256i {
    if S::AFFINE {
        _mm256_set1_epi64x(word.cast_signed())
    } else {
        // The byte fields park the coefficient in the word's low byte.
        _mm256_set1_epi8(word.to_le_bytes()[0].cast_signed())
    }
}
