//! Matrix operations: apply many sources to many rows at once.
//!
//! The erasure-decode shape: `nrows` destination rows, one source per term,
//! one coefficient per `(source, row)` pair — contiguous rows for staging,
//! offset-addressed rows for writes straight into a codeword buffer.

use crate::field::{Elem, FieldElem};
use crate::kernel::{FieldKernels, RawDispatch};

use super::check_width;

#[cfg(feature = "alloc")]
use super::CoeffMatrix;

/// Apply many sources to many rows: for each `(coeffs, src)` term,
/// `rows[j] += coeffs[j] * src` for every `j` in `0..nrows`.
///
/// Semantically a [`mul_add_scatter`](crate::ops::mul_add_scatter) per term.
/// The difference is memory traffic: blocked backends keep a destination tile
/// in registers while every source is folded in, so the destination is read
/// and written once per tile rather than once per term. That is the whole
/// game once `nrows * row_len` exceeds L1 — which is the normal case for
/// erasure reconstruction.
///
/// A zero-length row with zero-length sources is a valid zero-work shape and
/// a no-op; the structural checks above still apply.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes, every
/// term supplies `nrows` coefficients, and every source is `row_len` bytes.
pub fn mul_add_matrix<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    check_width::<F>("mul_add_matrix", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 || terms.is_empty() || all_zero(terms) {
        return;
    }
    F::mul_add_matrix(RawDispatch, &mut rows[..used], row_len, nrows, terms);
}

/// Apply many sources to many rows using a prepared source-major coefficient
/// matrix.
///
/// The matrix must have dimensions `(srcs.len(), n)` in `(sources, outputs)`:
/// matrix row `s` holds the coefficients pairing source `s` with every output
/// row. The output-row count is
/// [`coeffs.output_count()`](CoeffMatrix::output_count) — it is not a
/// parameter.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.output_count()` rows of
/// `row_len` bytes, `coeffs.source_count() == srcs.len()`, and every source
/// is `row_len` bytes.
#[cfg(feature = "alloc")]
pub fn mul_add_matrix_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_add_matrix_with", row_len);
    let nrows = coeffs.output_count();
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_add_matrix_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if nrows == 0 || row_len == 0 || srcs.is_empty() {
        return;
    }
    F::mul_add_matrix_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        nrows,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Overwrite many rows with field dot products: for each output row `j`,
/// `rows[j] = sum_t coeffs[t][j] * src[t]`.
///
/// The overwrite counterpart of [`mul_add_matrix`] and the erasure-encode
/// shape: the previous destination is ignored, so callers need not zero it.
/// A register-blocked backend seeds accumulators from zero and writes each row
/// once, with no destination read and no separate `fill(0)`.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes, every
/// term supplies `nrows` coefficients, and every source is `row_len` bytes.
pub fn mul_into_matrix<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    check_width::<F>("mul_into_matrix", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_into_matrix: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_into_matrix: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 {
        return;
    }
    if terms.is_empty() {
        rows[..used].fill(0);
        return;
    }
    F::mul_into_matrix(RawDispatch, &mut rows[..used], row_len, nrows, terms);
}

/// Overwrite many rows with field dot products using a prepared source-major
/// coefficient matrix: `rows[j] = sum_t coeffs[t][j] * src[t]`.
///
/// The overwrite counterpart of [`mul_add_matrix_with`]. The matrix must have
/// `srcs.len()` sources; the output-row count is
/// [`coeffs.output_count()`](CoeffMatrix::output_count).
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.output_count()` rows of
/// `row_len` bytes, `coeffs.source_count() == srcs.len()`, and every source
/// is `row_len` bytes.
#[cfg(feature = "alloc")]
pub fn mul_into_matrix_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_into_matrix_with", row_len);
    let nrows = coeffs.output_count();
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_into_matrix_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_into_matrix_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_into_matrix_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if nrows == 0 || row_len == 0 {
        return;
    }
    if srcs.is_empty() {
        rows[..used].fill(0);
        return;
    }
    F::mul_into_matrix_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        nrows,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Apply many sources to many disjoint rows scattered through `dst`: for each
/// `(coeffs, src)` term, `dst[row_starts[j]..][..row_len] += coeffs[j] * src`
/// for every `j` in `0..row_starts.len()`.
///
/// The reconstruction shape when recovered rows land at scattered positions in
/// a larger buffer — their natural slots in a codeword. Unlike
/// [`mul_add_matrix`], no contiguous staging buffer and copy-out is required:
/// a blocked backend writes each recovered row straight to its final location.
///
/// `row_starts` must address pairwise-disjoint, in-bounds rows of `row_len`
/// bytes, each starting on an element boundary. Disjointness is the safety
/// contract the kernel relies on and is checked here, before dispatch.
///
/// A zero-length row is a valid zero-work shape and a no-op; the structural
/// checks above (including disjointness of the named rows) still apply.
/// The `_at` in the name: the destination rows are addressed by explicit
/// offsets into `dst`, not held contiguously from its start.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, every row start
/// is a whole number of elements, the rows are pairwise disjoint, every term
/// supplies `row_starts.len()` coefficients, and every source is `row_len`
/// bytes; or on a partial trailing element.
pub fn mul_add_matrix_at<F: FieldKernels>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    check_at::<F>("mul_add_matrix_at", dst.len(), row_len, row_starts);
    let nrows = row_starts.len();
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_at: term supplies {} coefficients for {nrows} rows",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_at: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 || terms.is_empty() || all_zero(terms) {
        return;
    }
    F::mul_add_matrix_at(RawDispatch, dst, row_len, row_starts, terms);
}

/// Apply many sources to many disjoint rows scattered through `dst` using a
/// prepared source-major coefficient matrix: for each source `s`,
/// `dst[row_starts[j]..][..row_len] += coeffs[s * outputs + j] * srcs[s]`.
///
/// The prepared counterpart of [`mul_add_matrix_at`]; the matrix must have
/// `srcs.len()` sources and `row_starts.len()` outputs. Geometry checks match
/// [`mul_add_matrix_at`], including the pairwise-disjointness contract the
/// kernel relies on.
///
/// A zero-length row is a valid zero-work shape and a no-op; the structural
/// checks above (including disjointness of the named rows) still apply.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, every row start
/// is a whole number of elements, the rows are pairwise disjoint,
/// `coeffs.source_count() == srcs.len()`,
/// `coeffs.output_count() == row_starts.len()`, and every source is `row_len`
/// bytes; or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_add_matrix_at_with<F: FieldKernels>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_at::<F>("mul_add_matrix_at_with", dst.len(), row_len, row_starts);
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_add_matrix_at_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    assert_eq!(
        coeffs.output_count(),
        row_starts.len(),
        "mul_add_matrix_at_with: matrix holds coefficients for {} rows but there are {}",
        coeffs.output_count(),
        row_starts.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_at_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if row_starts.is_empty() || row_len == 0 || srcs.is_empty() {
        return;
    }
    F::mul_add_matrix_at_plan(
        RawDispatch,
        dst,
        row_len,
        row_starts,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Overwrite many disjoint rows scattered through `dst` with field dot
/// products: for each output row `j`,
/// `dst[row_starts[j]..][..row_len] = sum_t coeffs[t][j] * src[t]`.
///
/// The overwrite counterpart of [`mul_add_matrix_at`]: the previous contents
/// of the named rows are ignored, so callers need not zero them. Bytes
/// outside the named rows are not outputs of the call. With no terms the
/// named rows are zeroed, matching [`mul_into_matrix`]. Geometry checks match
/// [`mul_add_matrix_at`], including the pairwise-disjointness contract the
/// kernel relies on.
///
/// A zero-length row is a valid zero-work shape and a no-op; the structural
/// checks above (including disjointness of the named rows) still apply.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, every row start
/// is a whole number of elements, the rows are pairwise disjoint, every term
/// supplies `row_starts.len()` coefficients, and every source is `row_len`
/// bytes; or on a partial trailing element.
pub fn mul_into_matrix_at<F: FieldKernels>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    check_at::<F>("mul_into_matrix_at", dst.len(), row_len, row_starts);
    let nrows = row_starts.len();
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix_at: term supplies {} coefficients for {nrows} rows",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix_at: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 {
        return;
    }
    if terms.is_empty() {
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        return;
    }
    F::mul_into_matrix_at(RawDispatch, dst, row_len, row_starts, terms);
}

/// Overwrite many disjoint rows scattered through `dst` with field dot
/// products using a prepared source-major coefficient matrix:
/// `dst[row_starts[j]..][..row_len] = sum_t coeffs[t][j] * srcs[t]`.
///
/// The prepared counterpart of [`mul_into_matrix_at`] and the overwrite
/// counterpart of [`mul_add_matrix_at_with`]. The matrix must have
/// `srcs.len()` sources and `row_starts.len()` outputs. With no sources the
/// named rows are zeroed. Geometry checks match [`mul_add_matrix_at`],
/// including the pairwise-disjointness contract the kernel relies on.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, every row start
/// is a whole number of elements, the rows are pairwise disjoint,
/// `coeffs.source_count() == srcs.len()`,
/// `coeffs.output_count() == row_starts.len()`, and every source is `row_len`
/// bytes; or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_into_matrix_at_with<F: FieldKernels>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_at::<F>("mul_into_matrix_at_with", dst.len(), row_len, row_starts);
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_into_matrix_at_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    assert_eq!(
        coeffs.output_count(),
        row_starts.len(),
        "mul_into_matrix_at_with: matrix holds coefficients for {} rows but there are {}",
        coeffs.output_count(),
        row_starts.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix_at_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if row_starts.is_empty() || row_len == 0 {
        return;
    }
    if srcs.is_empty() {
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        return;
    }
    F::mul_into_matrix_at_plan(
        RawDispatch,
        dst,
        row_len,
        row_starts,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Scattered-row geometry shared by the `_at` entries: every row in bounds,
/// element-aligned, and pairwise disjoint.
fn check_at<F: FieldKernels>(name: &str, dst_len: usize, row_len: usize, row_starts: &[usize]) {
    check_width::<F>(name, row_len);
    for (j, &start) in row_starts.iter().enumerate() {
        let end = start
            .checked_add(row_len)
            .unwrap_or_else(|| panic!("{name}: row offset + length overflows"));
        assert!(
            end <= dst_len,
            "{name}: row {j} spans {start}..{end} but dst is {dst_len} bytes",
        );
        assert!(
            start.is_multiple_of(F::BYTES),
            "{name}: row {j} offset {start} is not a whole number of {} elements",
            F::NAME,
        );
    }
    // Pairwise disjointness. `O(r^2)` in the erasure count `r`, but this is a
    // per-solve setup cost paid once per pattern, not per byte, and it
    // allocates nothing. Overlap is a caller error: the kernel writes each row
    // through a raw pointer and cannot tolerate aliasing.
    for (a, &sa) in row_starts.iter().enumerate() {
        for &sb in &row_starts[a + 1..] {
            let (lo, hi) = if sa <= sb { (sa, sb) } else { (sb, sa) };
            assert!(
                hi - lo >= row_len,
                "{name}: rows at {sa} and {sb} overlap for {row_len}-byte rows",
            );
        }
    }
}

/// Whether every term's coefficients are zero: accumulation is then a no-op
/// that leaves the destination bytes untouched.
fn all_zero<F: FieldKernels>(terms: &[(&[Elem<F>], &[u8])]) -> bool {
    terms
        .iter()
        .all(|(coeffs, _)| coeffs.iter().all(|coeff| coeff.is_zero()))
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::binary::tower::Rijndael16Field;
    use crate::field::poly::AES;
    use crate::field::{Elem as E8, FiniteField, Polynomial};
    use alloc::vec::Vec;

    /// Term coefficient counts are still validated at zero row length.
    #[test]
    #[should_panic(expected = "term supplies")]
    fn zero_row_length_still_checks_term_counts() {
        let coeffs = [E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x07); 2];
        let mut rows = [0u8; 8];
        mul_add_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, 3, &[(&coeffs, &[])]);
    }

    #[test]
    fn prepared_matrix_ops_match_one_shot() {
        // source-major matrix: source s contributes coeffs[s * outputs + o].
        let values: Vec<Elem<Rijndael16Field>> = (0u16..3 * 4)
            .map(|i| Elem::<Rijndael16Field>::from_raw(i * 511 + 3))
            .collect();
        let matrix = CoeffMatrix::<Rijndael16Field>::from_source_major(3, 4, &values);
        let srcs: Vec<Vec<u8>> = (0u8..3).map(|s| alloc::vec![s + 1; 8]).collect();
        let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();

        let terms: Vec<(&[Elem<_>], &[u8])> = (0..3)
            .map(|s| (&values[s * 4..s * 4 + 4], src_refs[s]))
            .collect();

        let mut one_shot = [0x11u8; 32];
        mul_add_matrix::<Rijndael16Field>(&mut one_shot, 8, 4, &terms);
        let mut prepared = [0x11u8; 32];
        mul_add_matrix_with::<Rijndael16Field>(&mut prepared, 8, &matrix, &src_refs);
        assert_eq!(one_shot, prepared);

        let mut one_shot = [0x22u8; 32];
        mul_into_matrix::<Rijndael16Field>(&mut one_shot, 8, 4, &terms);
        let mut prepared = [0x22u8; 32];
        mul_into_matrix_with::<Rijndael16Field>(&mut prepared, 8, &matrix, &src_refs);
        assert_eq!(one_shot, prepared);
    }
}
