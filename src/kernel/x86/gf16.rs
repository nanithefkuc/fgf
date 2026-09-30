//! GF(2^16) tower kernels for x86 / `x86_64`.
//!
//! Every kernel here is the same identity, spelled three ways. Interleaved
//! bytes `[a, b]` are `a + b*u`; multiplying by `c0 + c1*u` gives
//!
//! ```text
//! even lane = c0*a       ^ (DELTA*c1)*b
//! odd  lane = (c0+c1)*b  ^ c1*a
//! ```
//!
//! Both lanes are therefore one alternating-coefficient byte multiply of the
//! source, `XORed` with one of the source with adjacent bytes exchanged. No
//! planar de-interleave, no 16-bit multiply, no 128 KiB table.
//!
//! - **GFNI** does each byte multiply with `GF2P8MULB` and gets the two
//!   alternating coefficients straight out of
//!   [`TowerCoeff`] — one `vpbroadcastw`
//!   each, no table at all.
//! - **AVX2 / SSSE3** have no field multiply, so each of the four base-field
//!   factors becomes a split-nibble `PSHUFB` pair against
//!   [`TowerTables`]; the even and odd
//!   byte lanes are then selected with a `0x00ff` halfword mask.
//!
//! Multi-row wiring differs by backend. SSSE3 blocks gather and matrix;
//! GFNI also blocks both, while AVX2 uses repeated AXPY for those shapes.
//! On `V4x`, matrix tiles widen; scatter and gather retain their GFNI lanes.
//! A coefficient occupies two broadcast words or four nibble tables, so the
//! amount of live coefficient state determines which row shapes benefit from
//! blocking. See BENCHMARKS.md for the dispatch measurements.
//!
//! ## Layout
//!
//! The single-buffer kernels split by multiply strategy — `gfni` for narrow
//! `GF2P8MULB`, `wide` for AVX-512 GFNI, and `nibble` for AVX2 and SSSE3
//! shuffles. Multi-row kernels split by fan shape: `gather`, `scatter`, and
//! `matrix`, with `matrix512` for the wide register-blocked tile; `elementwise`
//! handles varying operands. Shared coefficient words, shuffle controls, and
//! narrow blocked state live in this parent module.
//!
//! Every public kernel is a safe [`archmage`] entrypoint taking the exact
//! capability token its instructions require (`X64V4xToken` for AVX-512,
//! `X64V3GfniCryptoToken` for GFNI, `X64V3Token` for AVX2, and `X64V2Token`
//! for SSSE3) and validating its geometry. Inner helpers are `#[rite]`.
//!
//! Owned unsafe is confined to the narrow non-temporal stores and the
//! offset-addressed scatter and matrix groups. Wide matrix groups borrow
//! disjoint row slices without raw pointers.
//!
//! [`archmage`]: https://docs.rs/archmage

use crate::field::gf16::Elem;
use crate::kernel::gf16::Prepared;
use crate::kernel::tables::{TowerCoeff, TowerTables};

// The `x86` root helpers the submodules below reach through `super::`:
// non-temporal store selection, the destination alignment peel, and the
// GF(2^8) vector multiply the elementwise kernels borrow.
use super::{PREFETCH_AHEAD, gf8, nt_split, peel_to_align, prefetch_dst, prefetch_tile, store256};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

mod elementwise;
mod gather;
mod gfni;
mod matrix;
#[cfg(feature = "simd512")]
mod matrix512;
mod nibble;
mod scatter;
#[cfg(feature = "simd512")]
mod wide;

/// Rejects byte lengths that end inside a GF(2^16) element.
#[inline]
fn check_elements(name: &str, bytes: usize) {
    assert!(
        bytes.is_multiple_of(2),
        "{name}: buffer of {bytes} bytes ends with a partial GF(2^16) element",
    );
}

pub use elementwise::{
    mul_elementwise_assign_avx2, mul_elementwise_assign_gfni, mul_elementwise_assign_ssse3,
    mul_elementwise_avx2, mul_elementwise_gfni, mul_elementwise_ssse3,
};
pub use gather::{mul_add_gather_gfni, mul_add_gather_ssse3};
pub use gfni::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};
pub use matrix::{
    mul_add_matrix_gfni, mul_add_matrix_gfni_with, mul_add_matrix_ssse3, mul_add_matrix_ssse3_with,
};
#[cfg(feature = "simd512")]
pub use matrix512::{mul_add_matrix512, mul_add_matrix512_with};
pub use nibble::{
    mul_add_avx2, mul_add_ssse3, mul_assign_avx2, mul_assign_ssse3, mul_into_avx2, mul_into_ssse3,
};
pub use scatter::{mul_add_scatter_avx2, mul_add_scatter_gfni, mul_add_scatter_ssse3};
#[cfg(all(test, feature = "simd512"))]
pub(crate) use wide::MUL_ADD_PEEL_MIN;
#[cfg(feature = "simd512")]
pub use wide::{
    mul_add512, mul_assign512, mul_elementwise_assign512, mul_elementwise512, mul_into512,
};

// Cores shared with the Fan–Paar tower kernels and the blocked fan shapes.
pub(crate) use nibble::{
    NibbleAvx2, NibbleSsse3, nibble_avx2, nibble_ssse3, scale_avx2, scale_ssse3,
};

/// A GF(2^16) coefficient in a form the shuffle kernels can consume.
pub trait TableCoefficient {
    /// The raw coefficient element.
    fn coefficient(&self) -> Elem;
    /// Borrow the four nibble tables, building them on the fly if needed.
    fn with_tables<R>(&self, consume: impl FnOnce(&TowerTables) -> R) -> R;
}

impl TableCoefficient for Elem {
    #[inline]
    fn coefficient(&self) -> Elem {
        *self
    }

    #[inline]
    fn with_tables<R>(&self, consume: impl FnOnce(&TowerTables) -> R) -> R {
        consume(&TowerTables::new(*self))
    }
}

impl TableCoefficient for Prepared {
    #[inline]
    fn coefficient(&self) -> Elem {
        self.coeff()
    }

    #[inline]
    fn with_tables<R>(&self, consume: impl FnOnce(&TowerTables) -> R) -> R {
        match self {
            Prepared::Tables(tables) => consume(tables),
            other => consume(&TowerTables::new(other.coeff())),
        }
    }
}

/// `PSHUFB` control that exchanges the two bytes of every element.
///
/// The 16-byte pattern is repeated because `PSHUFB` indexes within each
/// 128-bit lane independently — which costs nothing here, since an element
/// never straddles a lane boundary.
const SWAP_ADJACENT: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];

/// The same control repeated for AVX2's two 128-bit lanes.
const SWAP_ADJACENT_WIDE: [u8; 32] = {
    let mut wide = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        wide[i] = SWAP_ADJACENT[i & 15];
        i += 1;
    }
    wide
};

/// The two broadcast words of a coefficient, as the halfwords `set1_epi16`
/// takes.
///
/// `to_ne_bytes`/`from_ne_bytes` rather than a cast: this is a
/// reinterpretation of the same 16 bits, not a numeric conversion.
#[inline]
fn broadcast_words(coeff: TowerCoeff) -> (i16, i16) {
    (
        i16::from_ne_bytes(coeff.same.to_ne_bytes()),
        i16::from_ne_bytes(coeff.cross.to_ne_bytes()),
    )
}

/// Load the 32-byte adjacent-exchange shuffle control.
#[archmage::rite(v3, import_intrinsics)]
fn swap_mask256() -> __m256i {
    _mm256_loadu_si256(&SWAP_ADJACENT_WIDE)
}

/// Load the 16-byte adjacent-exchange shuffle control.
#[archmage::rite(v2, import_intrinsics)]
fn swap_mask128() -> __m128i {
    _mm_loadu_si128(&SWAP_ADJACENT)
}

/// `coeff * src` for one 32-byte lane, given the source and its
/// adjacent-exchanged self.
///
/// The exchange is the caller's job so that the multi-row kernels can shuffle
/// once and reuse the result for every destination row.
#[archmage::rite(v3_gfni_crypto)]
fn scale_gfni(src: __m256i, swapped: __m256i, same: __m256i, cross: __m256i) -> __m256i {
    _mm256_xor_si256(
        _mm256_gf2p8mul_epi8(src, same),
        _mm256_gf2p8mul_epi8(swapped, cross),
    )
}

// ---------------------------------------------------------------------------
// Blocked state, shared by the gather and matrix fan shapes.
// ---------------------------------------------------------------------------

const TERM_TILE: usize = 8;
