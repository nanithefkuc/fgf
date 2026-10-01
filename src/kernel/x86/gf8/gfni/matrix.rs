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
use super::rows::{matrix_rows1, matrix_rows2, matrix_rows4};
use super::{Affine8D, Blocked, Gfni};
use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::Matrix;

/// For each `(coeffs, src)` term and each row `j < nrows`,
/// `rows[j] ^= coeffs[j] * src`.
///
/// Register-blocked over the destination: a group of rows is loaded into
/// accumulators once, every term is folded in, and the tile is stored once.
/// Destination traffic is therefore independent of `terms.len()`, which is
/// the whole point of the fused shape. Rows are grouped in fours (64-byte
/// tiles, eight accumulators), then a pair and a single (128-byte tiles).
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
/// supplies `nrows` coefficients for a source of `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_gfni: term {t} needs {nrows} coefficients"
        );
    }
    mul_add_matrix_gfni_with(_token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, GFNI backend, over a generic matrix source.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies a source of `row_len` bytes. Coefficient counts per term are the
/// provider's contract.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni_with<M: Matrix<Elem> + ?Sized>(
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
    mul_add_matrix_impl::<Gfni, M, false>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_gfni`] under `0x11D`, folding every term in with its affine map.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_gfni_8d: term {t} needs {nrows} coefficients"
        );
    }
    mul_add_matrix_gfni_8d_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_gfni_8d`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni_8d_with<M: Matrix<gf8d::Elem> + ?Sized>(
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
        "mul_add_matrix_gfni_8d: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if terms.len() == 0 {
        return;
    }
    mul_add_matrix_impl::<Affine8D, M, false>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_gfni`] with overwrite semantics: `rows[j] = sum_t coeffs[t][j] * src[t]`.
///
/// Accumulators are seeded from zero in registers instead of the destination,
/// so prior destination contents are ignored and no separate `fill(0)` runs —
/// the erasure-encode shape. Term counts above one resolve chunk accumulate
/// later chunks into what the first wrote.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix_gfni: term {t} needs {nrows} coefficients"
        );
    }
    mul_into_matrix_gfni_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_into_matrix_gfni`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_gfni_with<M: Matrix<Elem> + ?Sized>(
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
    mul_add_matrix_impl::<Gfni, M, true>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_gfni_8d`] with overwrite semantics under `0x11D`.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_gfni_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix_gfni_8d: term {t} needs {nrows} coefficients"
        );
    }
    mul_into_matrix_gfni_8d_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_into_matrix_gfni_8d`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_gfni_8d_with<M: Matrix<gf8d::Elem> + ?Sized>(
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
        "mul_into_matrix_gfni_8d: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    // As `mul_into_matrix_gfni_with`, with the affine map multiply.
    mul_add_matrix_impl::<Affine8D, M, true>(rows, row_len, nrows, terms);
}

/// Contiguous matrix walk shared by every matrix entry above.
///
/// With more than one row group over long rows and more terms than one resolve
/// chunk, the walk splits the rows into column blocks and runs every row group
/// over one block before the next, so later groups re-read sources the first
/// group just loaded. Up to one chunk the row groups run over whole rows,
/// where the source prefetch in `rows` keeps the streams fed.
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_impl<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    if !gfni_column_blocked(nrows, row_len, terms.len()) {
        matrix_block::<S, M, OVERWRITE>(rows, row_len, 0, row_len, nrows, terms);
        return;
    }
    for (start, len) in column_blocks(row_len, 0) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_block::<S, _, OVERWRITE>(rows, row_len, start, len, nrows, &window);
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
fn matrix_block<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    pitch: usize,
    block_start: usize,
    block_len: usize,
    nrows: usize,
    terms: &M,
) {
    debug_assert!(block_start + block_len <= pitch);
    let base = rows.as_mut_ptr();
    // Rows are grouped four at a time, then a pair, then whatever single row
    // is left: `nrows - g` is at most three after the first loop and at most
    // one after the pair.
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: row `j`'s block starts at `base + j * pitch + block_start`
        //        and spans `block_len` bytes, `block_start + block_len <=
        //        pitch`, and the entry asserted `nrows * pitch <=
        //        rows.len()`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: distinct rows sit `pitch` apart in one region and each block
        //        lies inside its own row, so the four windows are pairwise
        //        disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(g * pitch + block_start),
                base.add((g + 1) * pitch + block_start),
                base.add((g + 2) * pitch + block_start),
                base.add((g + 3) * pitch + block_start),
            ]
        };
        // The provider contract supplies four coefficients per term for rows
        // `g` through `g + 3`.
        matrix_rows4::<S, M, OVERWRITE>(ptrs, block_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as for the four-row group, rows `g` and `g + 1` start at
        //        `base + j * pitch`, `block_start + block_len <= pitch`, and
        //        `nrows * pitch <= rows.len()`.
        // THUS: each staged pointer addresses an in-bounds block of its row.
        //
        // ALIASING
        // SINCE: the rows sit `pitch` apart and each block lies inside its
        //        own row.
        // THUS: the two windows are disjoint.
        let ptrs = unsafe {
            [
                base.add(g * pitch + block_start),
                base.add((g + 1) * pitch + block_start),
            ]
        };
        matrix_rows2::<S, M, OVERWRITE>(ptrs, block_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, row `g` starts at `base + g * pitch` with its block
        //        inside the row and `nrows * pitch <= rows.len()`.
        // THUS: the staged pointer addresses an in-bounds block of the row.
        let ptr = unsafe { base.add(g * pitch + block_start) };
        matrix_rows1::<S, M, OVERWRITE>(ptr, block_len, g, terms);
    }
}

/// [`mul_add_matrix_gfni`], but the destination rows are disjoint and scattered
/// through `dst` at byte offsets `row_starts` rather than laid out
/// contiguously. Reuses the same register-blocked row groups; only the base
/// pointer of each row differs.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, the rows are
/// pairwise disjoint, and every term supplies `row_starts.len()` coefficients
/// for a `row_len`-byte source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem], &[u8])],
) {
    check_scattered("mul_add_matrix_at_gfni", dst.len(), row_len, row_starts);
    for (t, (coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            row_starts.len(),
            "mul_add_matrix_at_gfni: term {t} needs {} coefficients",
            row_starts.len()
        );
        assert_eq!(src.len(), row_len);
    }
    if terms.is_empty() || row_starts.is_empty() {
        return;
    }
    mul_add_matrix_at_impl::<Gfni, _>(dst, row_len, row_starts, terms);
}

/// [`mul_add_matrix_at_gfni`] under `0x11D`, folding terms in with the affine
/// map.
///
/// # Panics
/// As [`mul_add_matrix_at_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_gfni_8d(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[gf8d::Elem], &[u8])],
) {
    check_scattered("mul_add_matrix_at_gfni_8d", dst.len(), row_len, row_starts);
    for (t, (coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            row_starts.len(),
            "mul_add_matrix_at_gfni_8d: term {t} needs {} coefficients",
            row_starts.len()
        );
        assert_eq!(src.len(), row_len);
    }
    if terms.is_empty() || row_starts.is_empty() {
        return;
    }
    mul_add_matrix_at_impl::<Affine8D, _>(dst, row_len, row_starts, terms);
}

/// Scattered matrix walk: the same column blocking as
/// [`mul_add_matrix_impl`], with each row starting at an arbitrary disjoint
/// offset.
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_at_impl<S: Blocked, M: Matrix<S::Coeff> + ?Sized>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    if !gfni_column_blocked(row_starts.len(), row_len, terms.len()) {
        matrix_at_block::<S, M>(dst, row_len, 0, row_len, row_starts, terms);
        return;
    }
    for (start, len) in column_blocks(row_len, 0) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_at_block::<S, _>(dst, row_len, start, len, row_starts, &window);
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
fn matrix_at_block<S: Blocked, M: Matrix<S::Coeff> + ?Sized>(
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
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: each `row_starts[j] + row_len <= dst.len()` was asserted by
        //        the entry, and `block_start + block_len <= row_len`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint, and each
        //        block lies inside its own row.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(row_starts[g] + block_start),
                base.add(row_starts[g + 1] + block_start),
                base.add(row_starts[g + 2] + block_start),
                base.add(row_starts[g + 3] + block_start),
            ]
        };
        matrix_rows4::<S, M, false>(ptrs, block_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the entry asserted `row_starts[j] + row_len <= dst.len()`
        //        and `block_start + block_len <= row_len`.
        // THUS: each staged pointer addresses an in-bounds block of its row.
        //
        // ALIASING
        // SINCE: the entry asserted the rows pairwise disjoint, and each
        //        block lies inside its own row.
        // THUS: the two windows are disjoint.
        let ptrs = unsafe {
            [
                base.add(row_starts[g] + block_start),
                base.add(row_starts[g + 1] + block_start),
            ]
        };
        matrix_rows2::<S, M, false>(ptrs, block_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, `row_starts[g] + row_len <= dst.len()` and the
        //        block lies inside the row.
        // THUS: the staged pointer addresses an in-bounds block of the row.
        let ptr = unsafe { base.add(row_starts[g] + block_start) };
        matrix_rows1::<S, M, false>(ptr, block_len, g, terms);
    }
}
