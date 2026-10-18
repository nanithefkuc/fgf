//! Scatter operations: fan one source out to many rows.
//!
//! The systematic-encode shape: one arriving source is multiplied by one
//! coefficient per destination row and accumulated into it.

use crate::field::{Elem, FieldElem};
use crate::kernel::{FieldKernels, RawDispatch};

use super::check_pair;

#[cfg(feature = "alloc")]
use super::CoeffVecRef;

/// Fan one source out to many rows: `rows[j] += coeffs[j] * src`.
///
/// `rows` is a flat buffer of `coeffs.len()` contiguous rows, each `row_len`
/// bytes and the same length as `src`. This is the systematic-encode shape:
/// one arriving data symbol updates every parity row.
///
/// A zero-length row with a zero-length source is a valid zero-work shape
/// and a no-op; the structural checks above still apply.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
pub fn mul_add_scatter<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_scatter(RawDispatch, &mut rows[..used], row_len, coeffs, src);
}

/// Overwrite many rows from one source: `rows[j] = coeffs[j] * src`.
///
/// The overwrite counterpart of [`mul_add_scatter`]: the previous contents
/// of the rows are ignored, so callers need not zero them. `rows` is a flat
/// buffer of `coeffs.len()` contiguous rows, each `row_len` bytes and the
/// same length as `src`. A zero coefficient fills its row with zero lanes
/// and a unit coefficient copies the source bytes, matching
/// [`mul_into`](crate::ops::mul_into).
///
/// A zero-length row with a zero-length source is a valid zero-work shape
/// and a no-op; the structural checks above still apply.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
pub fn mul_into_scatter<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    check_pair::<F>("mul_into_scatter", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_into_scatter: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_into_scatter: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    F::mul_into_scatter(RawDispatch, &mut rows[..used], row_len, coeffs, src);
}

/// Fan one source out to many rows using prepared coefficients.
///
/// Unlike [`mul_add_scatter`], repeated calls do not rebuild backend-specific
/// coefficient tables. The view comes from
/// [`CoeffVec::as_ref`](crate::ops::CoeffVec::as_ref) or, for one column of a
/// coding matrix, [`CoeffMatrix::source`](crate::ops::CoeffMatrix::source).
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_add_scatter_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: CoeffVecRef<'_, F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter_with", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter_with: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.values().all(FieldElem::is_zero) {
        return;
    }
    F::mul_add_scatter_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        coeffs.values,
        coeffs.prepared,
        src,
    );
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::poly::AES;
    use crate::field::{Elem as E8, FiniteField, Polynomial};

    /// Zero-work shapes still validate structure: a non-empty source with a
    /// zero row length is a geometry error, not a silent skip.
    #[test]
    #[should_panic(expected = "mul_add_scatter")]
    fn zero_row_length_still_checks_pairing() {
        let coeffs = [E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x07); 3];
        let mut rows = [0u8; 8];
        mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &coeffs, &[1, 2, 3]);
    }
}
