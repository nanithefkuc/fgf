//! GF(2^8) kernels for x86 and `x86_64`.
//!
//! Submodules split on instruction set; the larger two split again on fan
//! shape. `ssse3` and `avx2` hold the nibble-shuffle kernels, `gfni` the
//! `GF2P8MULB`/`VGF2P8AFFINEQB` kernels at 32-byte lanes, and `avx512` the
//! same multiplies at 64-byte lanes under `simd512`. The caller's resolved
//! [`Backend`](crate::kernel::Backend) selects among them.
//!
//! - **GFNI.** `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply across
//!   byte lanes, so a coefficient is nothing but a broadcast byte.
//! - **Nibble shuffle (AVX2, SSSE3).** With no byte-wide multiply, `c * x`
//!   splits into `c * (x & 0xf) ^ c * (x & 0xf0)`, two `PSHUFB` lookups
//!   against a [`ScaleTable`](crate::kernel::tables::ScaleTable).
//!
//! Buffer lengths are arbitrary, so x86 kernels descend through narrower SIMD
//! lanes before handing only the sub-XMM remainder to the scalar nibble
//! kernels in `crate::kernel::gf8`.
//!
//! Every public entry is a safe [`archmage`] capability-token function taking
//! the exact token its instructions require and asserting its own buffer
//! geometry before the loops run. This file holds the geometry checks every
//! instruction set shares.

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;
mod gfni;
mod ssse3;

pub use avx2::{
    mul_add_avx2, mul_add_gather_avx2, mul_add_matrix_avx2, mul_add_matrix_avx2_with,
    mul_add_scatter_avx2, mul_assign_avx2, mul_elementwise_assign_avx2, mul_elementwise_avx2,
    mul_into_avx2, multiply_vectors_avx2,
};
#[cfg(feature = "simd512")]
#[cfg(test)]
pub(crate) use avx512::{GATHER_PEEL_MIN, MATRIX_PEEL_MIN, SCATTER_PEEL_MIN};
#[cfg(feature = "simd512")]
pub use avx512::{
    mul_add_avx512, mul_add_gather_avx512, mul_add_matrix_at_avx512, mul_add_matrix_avx512,
    mul_add_matrix_avx512_8d_with, mul_add_matrix_avx512_with, mul_add_scatter_avx512,
    mul_assign_avx512, mul_elementwise_assign_avx512, mul_elementwise_assign_avx512_8d,
    mul_elementwise_avx512, mul_elementwise_avx512_8d, mul_into_avx512, mul_into_matrix_avx512,
    mul_into_matrix_avx512_8d_with, mul_into_matrix_avx512_with,
};
pub use gfni::{
    mul_add_gather_gfni, mul_add_gather_gfni_8d, mul_add_gfni, mul_add_gfni_8d,
    mul_add_matrix_at_gfni, mul_add_matrix_at_gfni_8d, mul_add_matrix_gfni, mul_add_matrix_gfni_8d,
    mul_add_matrix_gfni_8d_with, mul_add_matrix_gfni_with, mul_add_scatter_gfni,
    mul_add_scatter_gfni_8d, mul_assign_gfni, mul_assign_gfni_8d, mul_elementwise_assign_gfni,
    mul_elementwise_assign_gfni_8d, mul_elementwise_gfni, mul_elementwise_gfni_8d, mul_into_gfni,
    mul_into_gfni_8d, mul_into_matrix_gfni, mul_into_matrix_gfni_8d, mul_into_matrix_gfni_8d_with,
    mul_into_matrix_gfni_with,
};
pub use ssse3::{
    mul_add_gather_ssse3, mul_add_matrix_ssse3, mul_add_matrix_ssse3_with, mul_add_scatter_ssse3,
    mul_add_ssse3, mul_assign_ssse3, mul_elementwise_assign_ssse3, mul_elementwise_ssse3,
    mul_into_ssse3, multiply_vectors_ssse3,
};

/// How many terms a matrix row group resolves into one stack array before
/// its tile loops run.
///
/// Resolving a coefficient costs one map (or byte) read per term per row, so
/// doing it per group instead of per tile removes the per-(tile, term)
/// coefficient load, its slice bounds check, and the term pointer walk from
/// the loop the multiplier runs in. The chunk bounds the scratch array
/// (`32 * 4 * 8` bytes at the widest group); term counts above it split into
/// further passes, each of which re-reads the destination it accumulates
/// into, so the chunk is sized to keep the erasure widths that matter
/// (`k <= 32`) in a single pass.
pub(crate) const RESOLVE_CHUNK: usize = 32;

/// Geometry contract shared by the gather entries: one coefficient per
/// source, and every source exactly as long as the destination.
fn check_gather(name: &str, dst: &[u8], coeffs: usize, srcs: &[&[u8]]) {
    assert_eq!(coeffs, srcs.len(), "{name}: one coefficient per source");
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            dst.len(),
            "{name}: source {index} does not match the destination length"
        );
    }
}

/// Scattered-row geometry shared by the `_at` entries: every row in bounds and
/// pairwise disjoint.
fn check_scattered(name: &str, dst_len: usize, row_len: usize, row_starts: &[usize]) {
    for (j, &start) in row_starts.iter().enumerate() {
        let end = start
            .checked_add(row_len)
            .unwrap_or_else(|| panic!("{name}: row offset plus length overflows"));
        assert!(
            end <= dst_len,
            "{name}: row {j} spans {start}..{end} but dst is {dst_len} bytes"
        );
    }
    for (a, &sa) in row_starts.iter().enumerate() {
        for &sb in &row_starts[a + 1..] {
            let (lo, hi) = if sa <= sb { (sa, sb) } else { (sb, sa) };
            assert!(
                hi - lo >= row_len,
                "{name}: rows at {sa} and {sb} overlap for {row_len}-byte rows"
            );
        }
    }
}
