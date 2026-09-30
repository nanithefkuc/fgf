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
//! Submodules split on instruction set: `ssse3`, `avx2`, `gfni` (itself split
//! on fan shape), and `avx512` under `simd512`. The coefficient words, shuffle
//! controls, and blocked-state sizes every instruction set shares live in this
//! file.
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

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;
mod gfni;
mod ssse3;

pub use avx2::{
    mul_add_avx2, mul_add_scatter_avx2, mul_assign_avx2, mul_elementwise_assign_avx2,
    mul_elementwise_avx2, mul_into_avx2,
};
#[cfg(all(test, feature = "simd512"))]
pub(crate) use avx512::MUL_ADD_PEEL_MIN;
#[cfg(feature = "simd512")]
pub use avx512::{
    mul_add_avx512, mul_add_matrix_avx512, mul_add_matrix_avx512_with, mul_assign_avx512,
    mul_elementwise_assign_avx512, mul_elementwise_avx512, mul_into_avx512,
};
pub use gfni::{
    mul_add_gather_gfni, mul_add_gfni, mul_add_matrix_gfni, mul_add_matrix_gfni_with,
    mul_add_scatter_gfni, mul_assign_gfni, mul_elementwise_assign_gfni, mul_elementwise_gfni,
    mul_into_gfni,
};
pub use ssse3::{
    mul_add_gather_ssse3, mul_add_matrix_ssse3, mul_add_matrix_ssse3_with, mul_add_scatter_ssse3,
    mul_add_ssse3, mul_assign_ssse3, mul_elementwise_assign_ssse3, mul_elementwise_ssse3,
    mul_into_ssse3,
};

// Cores shared with the Fan–Paar tower kernels.
pub(crate) use avx2::{NibbleAvx2, nibble_avx2, scale_avx2};
pub(crate) use ssse3::{nibble_ssse3, scale_ssse3};

use crate::field::gf16::Elem;
use crate::kernel::gf16::Prepared;
use crate::kernel::tables::{TowerCoeff, TowerTables};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Rejects byte lengths that end inside a GF(2^16) element.
#[inline]
fn check_elements(name: &str, bytes: usize) {
    assert!(
        bytes.is_multiple_of(2),
        "{name}: buffer of {bytes} bytes ends with a partial GF(2^16) element",
    );
}

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
const SWAP_ADJACENT_AVX2: [u8; 32] = {
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
fn swap_mask_avx2() -> __m256i {
    _mm256_loadu_si256(&SWAP_ADJACENT_AVX2)
}

/// Load the 16-byte adjacent-exchange shuffle control.
#[archmage::rite(v2, import_intrinsics)]
fn swap_mask_ssse3() -> __m128i {
    _mm_loadu_si128(&SWAP_ADJACENT)
}

/// Prepared coefficient sets the blocked gather and matrix shapes hold at
/// once.
const TERM_TILE: usize = 8;
