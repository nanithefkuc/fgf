//! AVX-512 GF(2^8) kernels over 64-byte lanes, under `simd512`.
//!
//! Submodules split on fan shape: `single`, `scatter`, `gather`, `matrix`,
//! and `elementwise`. `Gf8B` multiplies with `GF2P8MULB`; `Gf8D` folds each
//! coefficient in with `VGF2P8AFFINEQB`, whose immediate is the XOR constant,
//! so `<0>` selects the pure linear map and each factor replicates one 64-bit
//! map qword to all eight 64-bit lanes. Complete 64-byte lanes run here; every
//! remainder descends through the matching GFNI entry.
//!
//! This file holds what the shapes share: the [`MapCoeff`] seam and the
//! 64-byte factor, multiply, and remainder helpers.

mod elementwise;
mod gather;
mod matrix;
mod scatter;
mod single;

pub use elementwise::{
    mul_elementwise_assign_avx512, mul_elementwise_assign_avx512_8d, mul_elementwise_avx512,
    mul_elementwise_avx512_8d,
};
#[cfg(test)]
pub(crate) use gather::GATHER_PEEL_MIN;
pub use gather::mul_add_gather_avx512;
#[cfg(test)]
pub(crate) use matrix::MATRIX_PEEL_MIN;
pub use matrix::{
    mul_add_matrix_at_avx512, mul_add_matrix_avx512, mul_add_matrix_avx512_8d_with,
    mul_add_matrix_avx512_with, mul_into_matrix_avx512, mul_into_matrix_avx512_8d_with,
    mul_into_matrix_avx512_with,
};
#[cfg(test)]
pub(crate) use scatter::SCATTER_PEEL_MIN;
pub use scatter::mul_add_scatter_avx512;
pub use single::{mul_add_avx512, mul_assign_avx512, mul_into_avx512};

use crate::field::gf8d;
use crate::kernel::tables::{ScaleTable, affine_8d, scale_table_8d};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Coefficient facts a 512-bit matrix tile needs: the affine map word, the
/// sub-lane nibble tables, and the zero test. Implemented for raw
/// [`gf8d::Elem`] (map re-derived per tile) and for
/// [`crate::kernel::gf8::Prepared8D`] (map read from the stored field).
pub trait MapCoeff: Copy {
    /// The `VGF2P8AFFINEQB` map word that multiplies by `coeff`.
    fn map(coeff: Self) -> u64;
    /// The nibble tables of `coeff`, for sub-lane remainders.
    fn table(coeff: Self) -> &'static ScaleTable;
    /// Whether `coeff` is the zero element.
    fn is_zero(coeff: Self) -> bool;
    /// The zero element.
    fn zero() -> Self;
}

impl MapCoeff for gf8d::Elem {
    #[inline]
    fn map(coeff: Self) -> u64 {
        affine_8d(coeff)
    }
    #[inline]
    fn table(coeff: Self) -> &'static ScaleTable {
        scale_table_8d(coeff)
    }
    #[inline]
    fn is_zero(coeff: Self) -> bool {
        coeff.0 == 0
    }
    #[inline]
    fn zero() -> Self {
        gf8d::Elem(0)
    }
}

impl MapCoeff for crate::field::gf8b::Elem {
    #[inline]
    fn map(coeff: Self) -> u64 {
        crate::kernel::tables::affine_8b(coeff)
    }
    #[inline]
    fn table(coeff: Self) -> &'static ScaleTable {
        crate::kernel::tables::scale_table(coeff)
    }
    #[inline]
    fn is_zero(coeff: Self) -> bool {
        coeff.0 == 0
    }
    #[inline]
    fn zero() -> Self {
        crate::field::gf8b::Elem(0)
    }
}

impl MapCoeff for &'static ScaleTable {
    #[inline]
    fn map(coeff: Self) -> u64 {
        crate::kernel::tables::affine_8b(coeff.coeff)
    }
    #[inline]
    fn table(coeff: Self) -> &'static ScaleTable {
        coeff
    }
    #[inline]
    fn is_zero(coeff: Self) -> bool {
        coeff.coeff.0 == 0
    }
    #[inline]
    fn zero() -> Self {
        crate::kernel::tables::scale_table(crate::field::gf8b::Elem(0))
    }
}

impl MapCoeff for crate::kernel::gf8::Prepared8D {
    #[inline]
    fn map(coeff: Self) -> u64 {
        coeff.affine
    }
    #[inline]
    fn table(coeff: Self) -> &'static ScaleTable {
        coeff.table
    }
    #[inline]
    fn is_zero(coeff: Self) -> bool {
        coeff.table.coeff.0 == 0
    }
    #[inline]
    fn zero() -> Self {
        Self {
            table: scale_table_8d(gf8d::Elem(0)),
            affine: affine_8d(gf8d::Elem(0)),
        }
    }
}

/// Broadcast a `0x11D` coefficient map into a 512-bit multiply factor.
///
/// One 64-bit map qword replicated to all eight lanes: each 8-byte group of
/// the source meets the same map.
#[inline]
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn bfactor_avx512(map: u64) -> __m512i {
    _mm512_set1_epi64(map.cast_signed())
}

/// `map * x` over a 512-bit lane: the affine map applied per 8-byte group.
///
/// The `<0>` immediate is the XOR constant (zero: pure multiply), not a lane
/// selector.
#[inline]
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn bmul_avx512(x: __m512i, factor: __m512i) -> __m512i {
    _mm512_gf2p8affine_epi64_epi8::<0>(x, factor)
}

/// Single-row remainder AXPY over 64-byte lanes, then the AVX2 ladder.
///
/// Runs in the row-group bodies' matching feature context; the shared
/// `#[rite]` multiply bodies are called directly rather than the
/// token-bearing entry.
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn brem_avx512(dst: &mut [u8], map: u64, table: &ScaleTable, src: &[u8]) {
    single::mul_add_avx512_impl(dst, map, table, src);
}

/// Bytes of lead-in that put 64-byte accesses on a cache-line boundary.
#[inline]
pub(super) fn peel_avx512(ptr: *const u8, len: usize) -> usize {
    let head = ptr.align_offset(64);
    if head == usize::MAX || head >= len {
        0
    } else {
        head
    }
}
