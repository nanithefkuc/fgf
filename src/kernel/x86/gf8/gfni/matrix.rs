//! Many sources into many rows.
//!
//! Register-blocked over the destination: a group of rows is loaded into
//! accumulators once, every term is folded in, and the tile is stored once, so
//! destination traffic is independent of the term count. This module holds the
//! checked entry points and the row-group walks, which split long rows into
//! column blocks when more than one row group runs; the group bodies
//! themselves are in `rows`.
//!
//! Every entry is a safe [`archmage`] capability-token function carrying the
//! geometry contract. The row-group walks keep the offset-addressed-row
//! residue in [`archmage::rite`] helpers, and the `_with` forms trust the
//! [`Matrix`](crate::kernel::Matrix) provider for per-term coefficient counts
//! — the one provider-callback residue in this family.

use super::super::{ColumnWindow, RESOLVE_CHUNK, check_scattered, column_blocked, column_blocks};
use super::rows::{MAX_GROUP_ROWS, group_rows, matrix_group};
use crate::kernel::Matrix;
use crate::kernel::gf8::Prepared;

/// For each `(coeffs, src)` term and each row `j < nrows`,
/// `rows[j] ^= coeffs[j] * src`.
///
/// Register-blocked over the destination: a group of rows is loaded into
/// accumulators once, every term is folded in, and the tile is stored once.
/// Destination traffic is therefore independent of `terms.len()`, which is
/// the whole point of the fused shape. Rows are split into the fewest
/// width-balanced groups the register budget holds (see `rows::group_rows`).
///
/// Unlike [`mul_add_scatter_gfni`](super::mul_add_scatter_gfni) the term loop does not skip zero coefficients. The
/// predicate depends only on the term and the row group, never on the tile,
/// so testing it where it would have to live — innermost, once per tile — is
/// pure overhead on a dense matrix, and it costs more than the branch: the
/// coefficients have to reach a GPR to be tested, which stops each factor
/// broadcast folding into a memory-operand `vpbroadcastb`.
/// Sparsity belongs in the scatter shape, which drops zero rows before
/// grouping and outside any loop.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies a source of `row_len` bytes. Coefficient counts per term are the
/// provider's contract.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni_with<const NATIVE: bool, M: Matrix<Prepared> + ?Sized>(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_matrix_gfni: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if terms.len() == 0 {
        return;
    }
    mul_add_matrix_impl::<NATIVE, M, false>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_gfni_with`] with overwrite semantics:
/// `rows[j] = sum_t coeffs[t][j] * src[t]`, over a generic matrix source.
///
/// Accumulators are seeded in registers — from the first product in a
/// multi-row group, from zero in a one-row group — instead of from the
/// destination, so prior destination contents are ignored and no separate
/// `fill(0)` runs: the erasure-encode shape. Term counts above one resolve
/// chunk accumulate later chunks into what the first wrote.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_gfni_with<const NATIVE: bool, M: Matrix<Prepared> + ?Sized>(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_into_matrix_gfni: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    // Overwrite seeds accumulators from zero, so no destination bytes are
    // read; an empty term list still writes the zeroed sum.
    mul_add_matrix_impl::<NATIVE, M, true>(rows, row_len, nrows, terms);
}

/// Contiguous matrix walk shared by every matrix entry above.
///
/// With more terms than one resolve chunk, on the geometry the shared
/// `column_blocked` trigger selects, the walk splits the rows into column
/// blocks and runs every chunk and row group over one block before the next,
/// so each later pass re-reads sources and destination blocks that are still
/// cached. Up to one chunk the row groups run over whole rows, where the
/// source prefetch in `rows` keeps the streams fed.
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_impl<const NATIVE: bool, M: Matrix<Prepared> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    if !gfni_column_blocked(nrows, row_len, terms.len()) {
        matrix_block::<NATIVE, M, OVERWRITE>(rows, row_len, 0, row_len, nrows, terms);
        return;
    }
    for (start, len) in column_blocks(row_len, 0) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_block::<NATIVE, _, OVERWRITE>(rows, row_len, start, len, nrows, &window);
    }
}

/// Contiguous row-group walk over one column block: the
/// `block_start..block_start + block_len` span of each of `nrows` rows that
/// sit `pitch` bytes apart, against sources that span the block.
///
/// Residue: the rows of each group are addressed as runtime offsets into one
/// region. Every entry asserts `nrows * pitch <= rows.len()` before this walk
/// runs, the block lies inside one row, and the `_with` provider contract
/// covers per-term coefficient counts.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn matrix_block<const NATIVE: bool, M: Matrix<Prepared> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    pitch: usize,
    block_start: usize,
    block_len: usize,
    nrows: usize,
    terms: &M,
) {
    debug_assert!(block_start + block_len <= pitch);
    let base = rows.as_mut_ptr();
    let mut g = 0;
    while g < nrows {
        let width = group_rows(nrows, g, terms.len());
        let mut ptrs = [core::ptr::null_mut::<u8>(); MAX_GROUP_ROWS];
        for (j, slot) in ptrs[..width].iter_mut().enumerate() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `g + j < nrows`, row `g + j`'s block starts at
            //        `base + (g + j) * pitch + block_start` and spans
            //        `block_len` bytes, `block_start + block_len <= pitch`,
            //        and the entry asserted `nrows * pitch <= rows.len()`.
            // THUS: each staged pointer addresses an in-bounds, whole block
            //       of its row.
            //
            // ALIASING
            // SINCE: distinct rows sit `pitch` apart in one region and each
            //        block lies inside its own row, so the group's windows
            //        are pairwise disjoint.
            // THUS: no row's store conflicts with another row's window.
            *slot = unsafe { base.add((g + j) * pitch + block_start) };
        }
        // The provider contract supplies a coefficient per term for rows `g`
        // through `g + width - 1`.
        matrix_group::<NATIVE, M, OVERWRITE>(&ptrs[..width], block_len, g, terms);
        g += width;
    }
}

/// [`mul_add_matrix_gfni_with`], but the destination rows are disjoint and
/// scattered through `dst` at byte offsets `row_starts` rather than laid out
/// contiguously. Reuses the same register-blocked row groups; only the base
/// pointer of each row differs.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, the rows are
/// pairwise disjoint, and every term supplies a `row_len`-byte source.
/// Coefficient counts per term are the provider's contract.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_gfni_with<const NATIVE: bool, M: Matrix<Prepared> + ?Sized>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    check_scattered("mul_add_matrix_at_gfni", dst.len(), row_len, row_starts);
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if terms.len() == 0 || row_starts.is_empty() {
        return;
    }
    mul_add_matrix_at_impl::<NATIVE, M>(dst, row_len, row_starts, terms);
}

/// Scattered matrix walk: the same column blocking as
/// [`mul_add_matrix_impl`], with each row starting at an arbitrary disjoint
/// offset.
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_at_impl<const NATIVE: bool, M: Matrix<Prepared> + ?Sized>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    if !gfni_column_blocked(row_starts.len(), row_len, terms.len()) {
        matrix_at_block::<NATIVE, M>(dst, row_len, 0, row_len, row_starts, terms);
        return;
    }
    for (start, len) in column_blocks(row_len, 0) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_at_block::<NATIVE, _>(dst, row_len, start, len, row_starts, &window);
    }
}

/// Whether the GFNI matrix walk runs in column blocks: the shared geometry
/// trigger, restricted to term counts above one resolve chunk. Whole-row
/// bodies prefetch their sources only while every term fits in one chunk;
/// the selector and its threshold are measured parameters.
fn gfni_column_blocked(nrows: usize, row_len: usize, nterms: usize) -> bool {
    nterms > RESOLVE_CHUNK && column_blocked(nrows, row_len)
}

/// Scattered row-group walk over one column block: the
/// `block_start..block_start + block_len` span of each `row_len`-byte row
/// starting at `row_starts[j]`, against sources that span the block.
///
/// Residue: the rows are addressed at runtime offsets the caller proved
/// in-bounds and pairwise disjoint, which no safe primitive expresses.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn matrix_at_block<const NATIVE: bool, M: Matrix<Prepared> + ?Sized>(
    dst: &mut [u8],
    row_len: usize,
    block_start: usize,
    block_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    debug_assert!(block_start + block_len <= row_len);
    let base = dst.as_mut_ptr();
    let nrows = row_starts.len();
    let mut g = 0;
    while g < nrows {
        let width = group_rows(nrows, g, terms.len());
        let mut ptrs = [core::ptr::null_mut::<u8>(); MAX_GROUP_ROWS];
        for (j, slot) in ptrs[..width].iter_mut().enumerate() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: each `row_starts[j] + row_len <= dst.len()` was asserted
            //        by the entry, and `block_start + block_len <= row_len`.
            // THUS: each staged pointer addresses an in-bounds, whole block
            //       of its row.
            //
            // ALIASING
            // SINCE: the entry asserted every pair of rows disjoint, and each
            //        block lies inside its own row.
            // THUS: no row's store conflicts with another row's window.
            *slot = unsafe { base.add(row_starts[g + j] + block_start) };
        }
        matrix_group::<NATIVE, M, false>(&ptrs[..width], block_len, g, terms);
        g += width;
    }
}
