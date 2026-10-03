//! GFNI GF(2^8) kernels over 32-byte lanes.
//!
//! `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply, so a `Gf8<AES>`
//! coefficient is a broadcast byte; `VGF2P8AFFINEQB` applies an arbitrary 8x8
//! GF(2) map per lane and so multiplies under every other polynomial. Both
//! are pipelined but not single-cycle, which shapes every loop: the
//! single-buffer AXPY keeps four multiply chains in flight, and the blocked
//! kernels hold a destination tile in registers so the multiplier, not
//! memory, is the limit.
//!
//! Submodules split on fan shape: `single`, `scatter`, `gather`, `matrix` over
//! the register-blocked row-group bodies in `rows`, and `elementwise`. Every
//! entry takes an `X64V3GfniCryptoToken`.
//!
//! This file holds the seam the blocked shapes share. The polynomials differ
//! in exactly two places: how a coefficient becomes a factor vector and which
//! one-instruction multiply folds it in. Everything else — the tiling, the
//! register budget, the remainder ladder — is identical, so the kernels are
//! generic over [`Blocked`] and monomorphize to the native multiply under
//! [`AES`] and the affine multiply under every other polynomial.

mod elementwise;
mod gather;
mod matrix;
mod rows;
mod scatter;
mod single;

pub use elementwise::{mul_elementwise_assign_gfni, mul_elementwise_gfni};
pub use gather::mul_add_gather_gfni;
pub use matrix::{
    mul_add_matrix_at_gfni, mul_add_matrix_gfni, mul_add_matrix_gfni_with, mul_into_matrix_gfni,
    mul_into_matrix_gfni_with,
};
#[cfg(test)]
pub(crate) use rows::MATRIX_PREFETCH_MIN;
pub use scatter::mul_add_scatter_gfni;
pub use single::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};
#[cfg(feature = "simd512")]
pub(super) use single::{mul_add_gfni_impl, mul_assign_gfni_impl, mul_into_gfni_impl};

use crate::field::gf8::{AES, Elem};
use crate::kernel::tables::{affine_map, scale_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// The per-coefficient multiply seam for the GFNI and AVX-512 GF(2^8)
/// kernels.
///
/// Every [`Elem`] implements it, and so does every prepared coefficient;
/// the polynomial selects the multiply at compile time, and each fact comes
/// from the coefficient's own polynomial bank.
pub(crate) trait Blocked: Copy {
    /// `true` when the multiply is `VGF2P8AFFINEQB`, `false` for the native
    /// AES `GF2P8MULB`.
    const AFFINE: bool;
    /// The additive identity, for staging arrays.
    fn zero() -> Self;
    /// Raw coefficient byte.
    fn byte(self) -> u8;
    /// The `VGF2P8AFFINEQB` matrix qword that multiplies by this
    /// coefficient.
    fn map(self) -> u64;
    /// The coefficient's sub-lane nibble tables.
    fn table(self) -> &'static crate::kernel::tables::ScaleTable;
    /// Whether this is the zero coefficient.
    #[inline]
    fn is_zero(self) -> bool {
        self.byte() == 0
    }
}

impl<const POLY: u16> Blocked for Elem<POLY> {
    const AFFINE: bool = POLY != AES;
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline]
    fn byte(self) -> u8 {
        self.0
    }
    #[inline]
    fn map(self) -> u64 {
        affine_map(self)
    }
    #[inline]
    fn table(self) -> &'static crate::kernel::tables::ScaleTable {
        scale_table(self)
    }
}

impl<const POLY: u16> Blocked for crate::kernel::gf8::Prepared<POLY> {
    const AFFINE: bool = POLY != AES;
    #[inline]
    fn zero() -> Self {
        Self::new(Elem::ZERO)
    }
    #[inline]
    fn byte(self) -> u8 {
        self.table.coeff
    }
    #[inline]
    fn map(self) -> u64 {
        self.affine
    }
    #[inline]
    fn table(self) -> &'static crate::kernel::tables::ScaleTable {
        self.table
    }
}

/// A coefficient forced onto the `VGF2P8AFFINEQB` multiply, whatever its
/// polynomial.
///
/// The 512-bit kernels fold every coefficient in through its affine map,
/// including the AES field's, and their sub-lane remainders descend through
/// the 256-bit bodies with the same multiply.
#[cfg(feature = "simd512")]
#[derive(Clone, Copy)]
pub(super) struct Affine<C>(pub(super) C);

#[cfg(feature = "simd512")]
impl<C: Blocked> Blocked for Affine<C> {
    const AFFINE: bool = true;
    #[inline]
    fn zero() -> Self {
        Self(C::zero())
    }
    #[inline]
    fn byte(self) -> u8 {
        self.0.byte()
    }
    #[inline]
    fn map(self) -> u64 {
        self.0.map()
    }
    #[inline]
    fn table(self) -> &'static crate::kernel::tables::ScaleTable {
        self.0.table()
    }
}

/// Broadcast a coefficient into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_gfni<S: Blocked>(coeff: S) -> __m256i {
    if S::AFFINE {
        _mm256_set1_epi64x(coeff.map().cast_signed())
    } else {
        _mm256_set1_epi8(coeff.byte().cast_signed())
    }
}

/// Broadcast a coefficient into a 128-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor_half_gfni<S: Blocked>(coeff: S) -> __m128i {
    if S::AFFINE {
        _mm_set1_epi64x(coeff.map().cast_signed())
    } else {
        _mm_set1_epi8(coeff.byte().cast_signed())
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
/// shared `#[rite]` multiply body directly rather than the token-bearing
/// entry.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn brem_gfni<S: Blocked>(dst: &mut [u8], coeff: S, src: &[u8]) {
    single::mul_add_gfni_impl::<S>(dst, coeff, src);
}

/// The multiply factor of one coefficient, as a word the tile loop can
/// broadcast without touching a table: the `VGF2P8AFFINEQB` map on the affine
/// fields, the raw coefficient byte on the native `GF2P8MULB` one.
#[inline]
fn factor_word_gfni<S: Blocked>(coeff: S) -> u64 {
    if S::AFFINE {
        coeff.map()
    } else {
        u64::from(coeff.byte())
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
