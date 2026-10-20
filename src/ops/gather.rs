//! Gather operations: fold many sources into one row.
//!
//! The reconstruction shape: a dot product over sources, unit coefficients
//! for offset-addressed rows of one backing region, prepared coefficients for
//! the repeated forms.

use crate::field::{Elem, FieldElem};
use crate::kernel::{FieldKernels, RawDispatch};

use super::check_width;
use super::single::mul_into;

#[cfg(feature = "alloc")]
use super::CoeffVecRef;
#[cfg(feature = "alloc")]
use super::single::mul_into_with;

/// Fold byte-offset sources out of one backing region into one row, every
/// coefficient implicitly one: `dst += sum(region[off..off + dst.len()])`
/// (field addition; byte XOR only in characteristic two).
///
/// The unit-coefficient gather: sources are rows of one buffer — a solved
/// solution block, a table of packed elements — addressed by their start
/// offsets, so a caller folds an index table without staging a slice per
/// source. Fields whose addition is XOR hold the destination in registers
/// across the whole source list; the rest fold one
/// [`add_assign`](crate::ops::add_assign) per source.
/// The `_offsets` in the name: the *sources* are the offset-addressed rows
/// of one backing region; the destination is the single row that comes first.
///
/// ```
/// use fgf::{FiniteField, Polynomial, poly::AES, ops};
///
/// // Two 2-byte rows in one backing region, folded into `dst`.
/// let region = [0x01u8, 0x02, 0x10, 0x20];
/// let mut dst = [0x40u8, 0x80];
/// ops::add_gather_offsets::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, &region, &[0, 2]);
/// assert_eq!(dst, [0x51, 0xa2]);
/// ```
///
/// # Panics
/// Panics if `dst` is not a whole number of field elements, or if any
/// offset plus `dst.len()` falls outside `region`. Every offset is checked,
/// with overflow-safe arithmetic, before any destination byte is written.
/// `region` and `dst` borrow disjoint memory by construction, as with every
/// two-slice operation in this crate.
#[inline]
pub fn add_gather_offsets<F: FieldKernels>(dst: &mut [u8], region: &[u8], offsets: &[u32]) {
    check_width::<F>("add_gather_offsets", dst.len());
    let live = dst.len();
    for (index, &start) in offsets.iter().enumerate() {
        let start = start as usize;
        let end = start
            .checked_add(live)
            .expect("add_gather_offsets: offset + length overflows");
        assert!(
            end <= region.len(),
            "add_gather_offsets: offset {index} ({start}) + {live} exceeds region of {} bytes",
            region.len(),
        );
    }
    F::add_gather_offsets(RawDispatch, region, dst, offsets);
}

/// Fold many sources into one row: `dst += sum(coeffs[i] * srcs[i])`.
///
/// The transpose of [`mul_add_scatter`](crate::ops::mul_add_scatter), and the
/// shape that rebuilds a single lost symbol from its survivors. Blocked
/// backends hold the destination tile in registers while every source is
/// folded in, so `dst` is read and written once per tile rather than once per
/// source — prefer this over a loop of [`mul_add`](crate::ops::mul_add)
/// whenever the sources share a destination.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length, or on a partial trailing element.
pub fn mul_add_gather<F: FieldKernels>(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
    check_width::<F>("mul_add_gather", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_add_gather: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len()
    );
    for &src in srcs {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather: source is {} bytes, expected {}",
            src.len(),
            dst.len()
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_gather(RawDispatch, dst, coeffs, srcs);
}

/// Fold many sources into one row using prepared coefficients.
///
/// One coefficient per source. The view comes from
/// [`CoeffVec::as_ref`](crate::ops::CoeffVec::as_ref).
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_add_gather_with<F: FieldKernels>(
    dst: &mut [u8],
    coeffs: CoeffVecRef<'_, F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_add_gather_with", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_add_gather_with: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather_with: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.values().all(FieldElem::is_zero) {
        return;
    }
    F::mul_add_gather_plan(RawDispatch, dst, coeffs.values, coeffs.prepared, srcs);
}

/// Overwrite one row with a field dot product:
/// `dst = sum(coeffs[i] * srcs[i])`.
///
/// The overwrite counterpart of [`mul_add_gather`]: the previous destination
/// is ignored. Zero sources or all-zero coefficients fill `dst` with zero;
/// one source is the same operation as [`mul_into`].
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length, or on a partial trailing element.
pub fn mul_into_gather<F: FieldKernels>(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
    check_width::<F>("mul_into_gather", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_into_gather: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len()
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_into_gather: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len()
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.iter().all(|coeff| coeff.is_zero()) {
        dst.fill(0);
        return;
    }
    if let [coeff] = coeffs {
        mul_into::<F>(dst, *coeff, srcs[0]);
        return;
    }
    F::mul_into_gather(RawDispatch, dst, coeffs, srcs);
}

/// Overwrite one row with a field dot product using prepared coefficients.
///
/// One coefficient per source. The view comes from
/// [`CoeffVec::as_ref`](crate::ops::CoeffVec::as_ref).
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_into_gather_with<F: FieldKernels>(
    dst: &mut [u8],
    coeffs: CoeffVecRef<'_, F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_into_gather_with", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_into_gather_with: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_into_gather_with: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.values().all(FieldElem::is_zero) {
        dst.fill(0);
        return;
    }
    if coeffs.len() == 1 {
        let coeff = coeffs
            .get(0)
            .expect("single-entry view has one coefficient");
        mul_into_with::<F>(dst, &coeff, srcs[0]);
        return;
    }
    F::mul_into_gather_plan(RawDispatch, dst, coeffs.values, coeffs.prepared, srcs);
}
