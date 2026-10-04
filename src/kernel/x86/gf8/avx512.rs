//! AVX-512 GF(2^8) kernels over 64-byte lanes, under `simd512`.
//!
//! Submodules split on fan shape: `single`, `scatter`, `gather`, `matrix`,
//! and `elementwise`. Every fixed-coefficient shape folds its coefficient in
//! with `VGF2P8AFFINEQB`, whose immediate is the XOR constant, so `<0>`
//! selects the pure linear map and each factor replicates one 64-bit map
//! qword to all eight 64-bit lanes; that one form serves every polynomial,
//! the AES field included. Complete 64-byte lanes run here; every remainder
//! descends through the matching GFNI body under the same affine multiply.
//!
//! This file holds what the shapes share: the 64-byte factor, multiply, and
//! remainder helpers over the representation-erased
//! [`Prepared`](crate::kernel::gf8::Prepared) coefficient.

mod elementwise;
mod gather;
mod matrix;
mod scatter;
mod single;

pub use elementwise::{mul_elementwise_assign_avx512, mul_elementwise_avx512};
#[cfg(test)]
pub(crate) use gather::GATHER_PEEL_MIN;
pub use gather::mul_add_gather_avx512;
#[cfg(test)]
pub(crate) use matrix::MATRIX_PEEL_MIN;
pub use matrix::{
    mul_add_matrix_at_avx512_with, mul_add_matrix_avx512_with, mul_into_matrix_avx512_with,
};
#[cfg(test)]
pub(crate) use scatter::SCATTER_PEEL_MIN;
pub use scatter::mul_add_scatter_avx512;
pub use single::{mul_add_avx512, mul_assign_avx512, mul_into_avx512};

use super::PeelBody;
use crate::kernel::gf8::Prepared;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Broadcast a coefficient map into a 512-bit multiply factor.
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
pub(super) fn brem_avx512(dst: &mut [u8], coeff: Prepared, src: &[u8]) {
    single::mul_add_avx512_impl::<PeelBody>(dst, coeff, src);
}

/// The sub-lane remainder below one 64-byte lane: the 32/16-byte affine
/// bodies and the nibble tail.
#[inline]
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn brem_half_avx512(dst: &mut [u8], coeff: Prepared, src: &[u8]) {
    super::gfni::mul_add_gfni_impl::<false, PeelBody>(dst, coeff, src);
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
