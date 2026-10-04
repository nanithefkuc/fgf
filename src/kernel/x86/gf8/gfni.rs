//! GFNI GF(2^8) kernels over 32-byte lanes.
//!
//! `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply, so an AES-native
//! coefficient is a broadcast byte; `VGF2P8AFFINEQB` applies an arbitrary
//! 8x8 GF(2) map per lane and so multiplies under every other encoding. Both
//! are pipelined but not single-cycle, which shapes every loop: the
//! single-buffer AXPY keeps four multiply chains in flight, and the blocked
//! kernels hold a destination tile in registers so the multiplier, not
//! memory, is the limit.
//!
//! Submodules split on fan shape: `single`, `scatter`, `gather`, `matrix` over
//! the register-blocked row-group bodies in `rows`, and `elementwise`. Every
//! entry takes an `X64V3GfniCryptoToken`.
//!
//! This file holds the seam the blocked shapes share. The coefficient is the
//! representation-erased [`Prepared`] — its affine map qword under the affine
//! multiply, its raw byte under the native one — and the bodies are generic
//! over nothing but `NATIVE`, the native-versus-affine selection the dispatch
//! layer resolves from the representation's `AES_NATIVE` constant. Each value
//! monomorphizes to one multiply with the same unroll and the same lane
//! descent.

mod elementwise;
mod gather;
mod matrix;
mod rows;
mod scatter;
mod single;

pub use elementwise::{mul_elementwise_assign_gfni, mul_elementwise_gfni};
pub use gather::mul_add_gather_gfni;
pub use matrix::{
    mul_add_matrix_at_gfni_with, mul_add_matrix_gfni_with, mul_into_matrix_gfni_with,
};
#[cfg(test)]
pub(crate) use rows::MATRIX_PREFETCH_MIN;
pub use scatter::mul_add_scatter_gfni;
pub use single::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};
#[cfg(feature = "simd512")]
pub(super) use single::{mul_add_gfni_impl, mul_assign_gfni_impl, mul_into_gfni_impl};

use super::PeelBody;
use crate::kernel::gf8::Prepared;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Broadcast a coefficient into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_gfni<const NATIVE: bool>(coeff: Prepared) -> __m256i {
    if NATIVE {
        _mm256_set1_epi8(coeff.byte().cast_signed())
    } else {
        _mm256_set1_epi64x(coeff.affine.cast_signed())
    }
}

/// Broadcast a coefficient into a 128-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_half_gfni<const NATIVE: bool>(coeff: Prepared) -> __m128i {
    if NATIVE {
        _mm_set1_epi8(coeff.byte().cast_signed())
    } else {
        _mm_set1_epi64x(coeff.affine.cast_signed())
    }
}

/// `coeff * x` over a 256-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul_gfni<const NATIVE: bool>(x: __m256i, factor: __m256i) -> __m256i {
    if NATIVE {
        _mm256_gf2p8mul_epi8(x, factor)
    } else {
        _mm256_gf2p8affine_epi64_epi8::<0>(x, factor)
    }
}

/// `coeff * x` over a 128-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul_half_gfni<const NATIVE: bool>(x: __m128i, factor: __m128i) -> __m128i {
    if NATIVE {
        _mm_gf2p8mul_epi8(x, factor)
    } else {
        _mm_gf2p8affine_epi64_epi8::<0>(x, factor)
    }
}

/// Single-row remainder AXPY (`dst ^= coeff * src`) for this field: whole
/// 32/16-byte lanes on the one-instruction multiply, sub-XMM tail on the
/// nibble bank.
///
/// Runs in the row-group bodies' matching feature context, so it calls the
/// shared `#[rite]` multiply body directly rather than the token-bearing
/// entry.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn brem_gfni<const NATIVE: bool>(dst: &mut [u8], coeff: Prepared, src: &[u8]) {
    single::mul_add_gfni_impl::<NATIVE, PeelBody>(dst, coeff, src);
}

/// The multiply factor of one coefficient, as a word the tile loop can
/// broadcast without touching a table: the `VGF2P8AFFINEQB` map on the affine
/// multiply, the raw coefficient byte on the native `GF2P8MULB` one.
#[inline]
fn factor_word_gfni<const NATIVE: bool>(coeff: Prepared) -> u64 {
    if NATIVE {
        u64::from(coeff.byte())
    } else {
        coeff.affine
    }
}

/// Broadcast a resolved factor word into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn wfactor_gfni<const NATIVE: bool>(word: u64) -> __m256i {
    if NATIVE {
        // The byte fields park the coefficient in the word's low byte.
        _mm256_set1_epi8(word.to_le_bytes()[0].cast_signed())
    } else {
        _mm256_set1_epi64x(word.cast_signed())
    }
}
