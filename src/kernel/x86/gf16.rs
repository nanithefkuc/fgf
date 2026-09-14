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
//! Multi-row wiring differs by backend, and deliberately so. SSSE3 blocks
//! both gather and matrix; GFNI blocks the matrix but gathers with repeated
//! AXPY; AVX2 uses repeated AXPY for both. A GF(2^16) coefficient costs four
//! nibble tables or two broadcast words, so how many of them can stay
//! resident is what decides whether blocking beats re-reading the
//! destination — see the crossover notes on the `kernel::gf16` dispatch arms,
//! and BENCHMARKS.md, before rewiring any of these.
//!
//! ## Layout
//!
//! The single-buffer kernels split by multiply strategy — `gfni` for
//! `GF2P8MULB`, `nibble` for the AVX2 and SSSE3 `PSHUFB` pairs — and the
//! multi-row kernels split by fan shape: `gather`, `scatter`, `matrix`, with
//! `elementwise` for the one shape whose operands both
//! vary. What stays here is what more than one of them needs: the
//! coefficient trait, the adjacent-exchange control and its two loads, the
//! 32-byte GFNI core, and the blocked AVX2 lane and split state the gather
//! and matrix tiles share.
//!
//! Every public kernel is a safe [`archmage`] entrypoint taking the exact
//! capability token its instructions require (`X64V3GfniCryptoToken` for
//! GFNI, `X64V3Token` for AVX2, `X64V2Token` for SSSE3) and asserting its
//! own geometry; inner helpers are `#[rite]`. The unsafe that remains is the
//! sanctioned residue: the non-temporal stores in the `mul_into` lanes and
//! the offset-addressed destination rows of the scatter and matrix group
//! bodies, each with a per-item `#[allow(unsafe_code)]` and a SINCE–THUS
//! proof.
//!
//! [`archmage`]: https://docs.rs/archmage

use crate::field::gf16::Elem;
use crate::kernel::gf16::Prepared;
use crate::kernel::tables::{ScaleTable, TowerCoeff, TowerTables};

// The `x86` root helpers the submodules below reach through `super::`:
// non-temporal store selection, the destination alignment peel, and the
// GF(2^8) vector multiply the elementwise kernels borrow.
use super::{gf8, nt_split, peel_to_align, store256};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

mod elementwise;
mod gather;
mod gfni;
mod matrix;
mod nibble;
mod scatter;
/// Rejects byte lengths that end inside a GF(2^16) element.
#[inline]
fn check_elements(name: &str, bytes: usize) {
    assert!(
        bytes.is_multiple_of(2),
        "{name}: buffer of {bytes} bytes ends with a partial GF(2^16) element",
    );
}

pub use elementwise::{mul_elementwise_avx2, mul_elementwise_gfni, mul_elementwise_ssse3};
pub use gather::{mul_add_gather_avx2, mul_add_gather_gfni, mul_add_gather_ssse3};
pub use gfni::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};
pub use matrix::{
    mul_add_matrix_avx2, mul_add_matrix_gfni, mul_add_matrix_gfni_with, mul_add_matrix_ssse3,
    mul_add_matrix_ssse3_with,
};
pub use nibble::{
    mul_add_avx2, mul_add_ssse3, mul_assign_avx2, mul_assign_ssse3, mul_into_avx2, mul_into_ssse3,
};
pub use scatter::{mul_add_scatter_avx2, mul_add_scatter_gfni, mul_add_scatter_ssse3};

// Cores shared with the Fan–Paar tower kernels and the blocked fan shapes.
pub(crate) use nibble::{
    NibbleAvx2, NibbleSsse3, lookup_avx2, nibble_avx2, nibble_ssse3, scale_avx2, scale_ssse3,
    split_avx2,
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

/// Lane constants every AVX2 nibble multiply needs, materialized once per
/// kernel call.
///
/// [`NibbleAvx2`] carries its own copy, which suits the single-coefficient
/// kernels: there is exactly one table set, so the constants ride along for
/// free. The blocked kernels hold state per *live coefficient*, and three of
/// every eleven vectors would then be the same constant over again. Split
/// out, only the eight vectors that genuinely differ scale with the block.
struct LaneAvx2 {
    /// `0x0f` in every byte: the nibble-extraction mask.
    nibble: __m256i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    even: __m256i,
    /// Adjacent-byte exchange control.
    swap: __m256i,
}

/// Materialize the shared lane constants.
#[archmage::rite(v3)]
fn lane_avx2() -> LaneAvx2 {
    LaneAvx2 {
        nibble: _mm256_set1_epi8(0x0f),
        even: _mm256_set1_epi16(0x00ff),
        swap: swap_mask256(),
    }
}

/// One source vector reduced to everything that does not depend on the
/// coefficient applied to it.
///
/// This is what pays for a blocked tile. [`scale_avx2`] redoes the adjacent
/// exchange and both nibble splits for every coefficient; a tile that folds
/// one source into several rows derives them once here and hands the same
/// four index vectors to every row.
struct SplitAvx2 {
    /// Low and high nibbles of the source.
    direct: (__m256i, __m256i),
    /// Low and high nibbles of the adjacent-exchanged source.
    crossed: (__m256i, __m256i),
}

/// Exchange and split `src` once for a whole tile.
#[archmage::rite(v3)]
fn split_source_avx2(src: __m256i, lanes: &LaneAvx2) -> SplitAvx2 {
    SplitAvx2 {
        direct: split_avx2(src, lanes.nibble),
        crossed: split_avx2(_mm256_shuffle_epi8(src, lanes.swap), lanes.nibble),
    }
}

/// One base-field byte multiply, broadcasting a shared bank entry at the
/// point of use.
///
/// The blocked kernels keep four such borrows per live coefficient rather
/// than [`NibbleAvx2`]'s eleven vectors: the bank is rodata that is already
/// resident and shared by every coefficient with a factor in common, so the
/// state that scales with the block stays at four pointers.
#[archmage::rite(v3, import_intrinsics)]
fn lookup_bank_avx2(factor: &ScaleTable, split: (__m256i, __m256i)) -> __m256i {
    let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(&factor.lo));
    let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(&factor.hi));
    lookup_avx2(lo, hi, split)
}

/// `coeff * src` for one 32-byte lane, over a source already exchanged and
/// split by [`split_source_avx2`].
///
/// The same four nibble multiplies and the same parity-before-mask grouping
/// as [`scale_avx2`]; only the coefficient-independent prologue is gone.
#[archmage::rite(v3)]
fn scale_split_avx2(
    split: &SplitAvx2,
    factors: &[&'static ScaleTable; 4],
    even_mask: __m256i,
) -> __m256i {
    let even = _mm256_xor_si256(
        lookup_bank_avx2(factors[0], split.direct),
        lookup_bank_avx2(factors[2], split.crossed),
    );
    let odd = _mm256_xor_si256(
        lookup_bank_avx2(factors[1], split.direct),
        lookup_bank_avx2(factors[3], split.crossed),
    );
    _mm256_xor_si256(
        _mm256_and_si256(even, even_mask),
        _mm256_andnot_si256(even_mask, odd),
    )
}
