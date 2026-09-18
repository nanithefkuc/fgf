//! Many sources into many rows: the production matrix kernels.
//!
//! Register-blocked over the destination: a group of rows is loaded into
//! accumulators once, every term is folded in, and the tile is stored once, so
//! destination traffic is independent of the term count. This module holds the
//! checked entry points and the row-group walk; the group bodies themselves
//! are in `rows`.
//!
//! Every entry is a safe [`archmage`] capability-token function carrying the
//! geometry contract. The row-group walks keep the offset-addressed-row
//! residue in [`archmage::rite`] helpers, and the `_with` forms trust the
//! [`Matrix`](crate::kernel::Matrix) provider for per-term coefficient counts
//! — the one provider-callback residue in this family.

use super::rows::{matrix_rows1, matrix_rows2, matrix_rows4};
use super::{Affine8D, Blocked, Gfni};
use super::{Affine8DPrepared, PreparedMatrix};
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
/// broadcast folding into a memory-operand `vpbroadcastb` (BENCHMARKS.md).
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
pub fn mul_add_matrix_affine(
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
            "mul_add_matrix_affine: term {t} needs {nrows} coefficients"
        );
    }
    mul_add_matrix_affine_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_affine`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_affine_with<M: Matrix<gf8d::Elem> + ?Sized>(
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
        "mul_add_matrix_affine: rows buffer does not hold {nrows} rows of {row_len} bytes"
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
/// so the destination is written once with no read and no separate `fill(0)` —
/// the erasure-encode shape.
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

/// [`mul_add_matrix_affine`] with overwrite semantics under `0x11D`.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_affine(
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
            "mul_into_matrix_affine: term {t} needs {nrows} coefficients"
        );
    }
    mul_into_matrix_affine_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_into_matrix_affine`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_affine_with<M: Matrix<gf8d::Elem> + ?Sized>(
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
        "mul_into_matrix_affine: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    // As `mul_into_matrix_gfni_with`, with the affine map multiply.
    mul_add_matrix_impl::<Affine8D, M, true>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_affine_with`] over already-prepared coefficients: accumulate
/// `rows[j] ^= sum_t coeffs[t][j] * src[t]` with the affine map read from each
/// prepared coefficient instead of recomputed.
///
/// `prepared` is term-major, `term * nrows + row`, the order a plan stores.
///
/// # Panics
/// As [`mul_add_matrix_gfni_with`], plus unless `prepared` holds `nrows` coefficients per
/// source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_affine_prepared_with(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    prepared: &[crate::kernel::gf8::Prepared8D],
    srcs: &[&[u8]],
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "matrix_affine_prepared: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    assert_eq!(
        prepared.len(),
        nrows * srcs.len(),
        "matrix_affine_prepared: one prepared coefficient per (term, row)"
    );
    for src in srcs {
        assert_eq!(src.len(), row_len);
    }
    if srcs.is_empty() {
        return;
    }
    let terms = PreparedMatrix {
        prepared,
        nrows,
        sources: srcs,
    };
    mul_add_matrix_impl::<Affine8DPrepared, PreparedMatrix, false>(rows, row_len, nrows, &terms);
}

/// [`mul_add_matrix_affine_prepared_with`] with overwrite semantics: the erasure-
/// encode shape over prepared coefficients.
///
/// # Panics
/// As [`mul_add_matrix_affine_prepared_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_affine_prepared_with(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    prepared: &[crate::kernel::gf8::Prepared8D],
    srcs: &[&[u8]],
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "matrix_overwrite_affine_prepared: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    assert_eq!(
        prepared.len(),
        nrows * srcs.len(),
        "matrix_overwrite_affine_prepared: one prepared coefficient per (term, row)"
    );
    for src in srcs {
        assert_eq!(src.len(), row_len);
    }
    if srcs.is_empty() {
        // The empty overwrite sum is zero: match the resolved path, whose
        // tile loop stores zeroed accumulators even with no terms.
        rows[..nrows * row_len].fill(0);
        return;
    }
    let terms = PreparedMatrix {
        prepared,
        nrows,
        sources: srcs,
    };
    mul_add_matrix_impl::<Affine8DPrepared, PreparedMatrix, true>(rows, row_len, nrows, &terms);
}

/// Contiguous row-group walk shared by every matrix entry above.
///
/// Residue: the rows of each group are addressed as runtime offsets into one
/// region. Every entry asserts `nrows * row_len <= rows.len()` before this
/// walk runs, distinct rows sit `row_len` apart, and the `_with` provider
/// contract covers per-term coefficient counts.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_impl<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    let base = rows.as_mut_ptr();
    // Rows are grouped four at a time, then a pair, then whatever single row
    // is left: `nrows - g` is at most three after the first loop and at most
    // one after the pair.
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: row `j` starts at `base + j * row_len` and spans `row_len`
        //        bytes, and the entry asserted `nrows * row_len <=
        //        rows.len()`.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: distinct rows sit `row_len` apart in one region, so the four
        //        windows are pairwise disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(g * row_len),
                base.add((g + 1) * row_len),
                base.add((g + 2) * row_len),
                base.add((g + 3) * row_len),
            ]
        };
        // The provider contract supplies four coefficients per term for rows
        // `g` through `g + 3`.
        matrix_rows4::<S, M, OVERWRITE>(ptrs, row_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // As above, for the two rows at `g` and `g + 1`.
        let ptrs = unsafe { [base.add(g * row_len), base.add((g + 1) * row_len)] };
        matrix_rows2::<S, M, OVERWRITE>(ptrs, row_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // As above, for the last row.
        let ptr = unsafe { base.add(g * row_len) };
        matrix_rows1::<S, M, OVERWRITE>(ptr, row_len, g, terms);
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
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            row_starts.len(),
            "mul_add_matrix_at_gfni: term {t} needs {} coefficients",
            row_starts.len()
        );
    }
    mul_add_matrix_at_gfni_with(_token, dst, row_len, row_starts, terms);
}

/// [`mul_add_matrix_at_gfni`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_at_gfni`], with the coefficient count per term left to
/// the provider contract.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_gfni_with<M: Matrix<Elem> + ?Sized>(
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
    mul_add_matrix_at_impl::<Gfni, M>(dst, row_len, row_starts, terms);
}

/// [`mul_add_matrix_at_gfni`] under `0x11D`, folding terms in with the affine
/// map.
///
/// # Panics
/// As [`mul_add_matrix_at_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_affine(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[gf8d::Elem], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            row_starts.len(),
            "mul_add_matrix_at_affine: term {t} needs {} coefficients",
            row_starts.len()
        );
    }
    mul_add_matrix_at_affine_with(_token, dst, row_len, row_starts, terms);
}

/// [`mul_add_matrix_at_affine`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_at_gfni_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_affine_with<M: Matrix<gf8d::Elem> + ?Sized>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    check_scattered("mul_add_matrix_at_affine", dst.len(), row_len, row_starts);
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if terms.len() == 0 || row_starts.is_empty() {
        return;
    }
    mul_add_matrix_at_impl::<Affine8D, M>(dst, row_len, row_starts, terms);
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

/// Scattered row-group walk: the same blocking as [`mul_add_matrix_impl`],
/// with each row starting at an arbitrary disjoint offset.
///
/// Residue: the rows are addressed at runtime offsets the caller proved
/// in-bounds and pairwise disjoint, which no safe primitive expresses.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_matrix_at_impl<S: Blocked, M: Matrix<S::Coeff> + ?Sized>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    let base = dst.as_mut_ptr();
    let nrows = row_starts.len();
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: each `row_starts[j] + row_len <= dst.len()` was asserted by
        //        the entry.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(row_starts[g]),
                base.add(row_starts[g + 1]),
                base.add(row_starts[g + 2]),
                base.add(row_starts[g + 3]),
            ]
        };
        matrix_rows4::<S, M, false>(ptrs, row_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // As above, for the two rows at `g` and `g + 1`.
        let ptrs = unsafe { [base.add(row_starts[g]), base.add(row_starts[g + 1])] };
        matrix_rows2::<S, M, false>(ptrs, row_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // As above, for the last row.
        let ptr = unsafe { base.add(row_starts[g]) };
        matrix_rows1::<S, M, false>(ptr, row_len, g, terms);
    }
}
