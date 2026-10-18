//! Single-row operations: field addition and subtraction, coefficient
//! scaling, elementwise products, and broadcast scalar arithmetic.
//!
//! Every entry takes one destination row. Fixed-coefficient and prepared
//! (`_with`) spellings sit beside each other; the coefficient-shortcut
//! behavior (zero fills or skips, one copies or adds) is defined here and
//! reused by the multi-row shapes.

use crate::field::{Elem, FieldElem};
use crate::kernel::{FieldKernels, RawDispatch};

use super::{PreparedCoefficient, check_pair, check_width};

/// `dst += src`, field addition.
///
/// XOR in characteristic two; a canonical modular fold in the prime fields.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn add_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("add_assign", "dst", dst.len(), "src", src.len());
    F::add_assign(RawDispatch, dst, src);
}

/// Add pairwise rows: `dst_row[j] += src_row[j]` for every row.
///
/// Both buffers are interpreted as the same number of contiguous
/// `row_len`-byte rows. Semantically identical to [`add_assign`] — row
/// boundaries cannot change elementwise addition. Stating the row geometry
/// lets a backend schedule independent row streams; callers that know it
/// should use this form.
///
/// ```
/// use fgf::{FiniteField, Polynomial, poly::AES, ops};
///
/// let src = [0x01u8, 0x02, 0x03, 0x04];
/// let mut dst = [0x10u8, 0x20, 0x30, 0x40];
///
/// // Two rows of two bytes each, added pairwise.
/// ops::add_assign_rows::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, 2, &src);
/// assert_eq!(dst, [0x11, 0x22, 0x33, 0x44]);
/// ```
///
/// # Panics
/// Panics if `row_len == 0`, if `row_len` is not a whole number of field
/// elements, if the buffers differ in length, or if their length is not a
/// whole number of rows.
#[inline]
pub fn add_assign_rows<F: FieldKernels>(dst: &mut [u8], row_len: usize, src: &[u8]) {
    assert_ne!(row_len, 0, "add_assign_rows: row length must be nonzero");
    check_width::<F>("add_assign_rows", row_len);
    assert_eq!(
        dst.len(),
        src.len(),
        "add_assign_rows: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert_eq!(
        dst.len() % row_len,
        0,
        "add_assign_rows: partial trailing row",
    );
    F::add_assign_rows(RawDispatch, dst, src, row_len);
}

/// `dst -= src`. Identical to [`add_assign`] in characteristic two.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn sub_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("sub_assign", "dst", dst.len(), "src", src.len());
    F::sub_assign(RawDispatch, dst, src);
}

/// Subtract pairwise rows: `dst_row[j] -= src_row[j]` for every row.
///
/// The row-geometry counterpart of [`sub_assign`], mirroring
/// [`add_assign_rows`]: both buffers hold the same number of contiguous
/// `row_len`-byte rows. In characteristic two subtraction is addition, so
/// this coincides with [`add_assign_rows`] there; in the prime fields it
/// subtracts with a canonicalizing fold per lane.
///
/// # Panics
/// Panics if `row_len == 0`, if `row_len` is not a whole number of field
/// elements, if the buffers differ in length, or if their length is not a
/// whole number of rows.
#[inline]
pub fn sub_assign_rows<F: FieldKernels>(dst: &mut [u8], row_len: usize, src: &[u8]) {
    assert_ne!(row_len, 0, "sub_assign_rows: row length must be nonzero");
    check_width::<F>("sub_assign_rows", row_len);
    assert_eq!(
        dst.len(),
        src.len(),
        "sub_assign_rows: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert_eq!(
        dst.len() % row_len,
        0,
        "sub_assign_rows: partial trailing row",
    );
    F::sub_assign_rows(RawDispatch, dst, src, row_len);
}

/// `dst += coeff * src`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add<F: FieldKernels>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    check_pair::<F>("mul_add", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        F::add_assign(RawDispatch, dst, src);
        return;
    }
    F::mul_add(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
}

/// `dst += coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        return;
    }
    if value.is_one() {
        F::add_assign(RawDispatch, dst, src);
        return;
    }
    F::mul_add(RawDispatch, dst, coeff.repr(RawDispatch), src);
}

/// `dst = coeff * src`, overwriting `dst`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into<F: FieldKernels>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    check_pair::<F>("mul_into", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    if coeff.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
}

/// `dst = coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_into_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    if value.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(RawDispatch, dst, coeff.repr(RawDispatch), src);
}

/// `dst *= coeff`, in place.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign<F: FieldKernels>(dst: &mut [u8], coeff: Elem<F>) {
    check_width::<F>("mul_assign", dst.len());
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(RawDispatch, dst, &F::prepare(RawDispatch, coeff));
}

/// `dst *= coeff`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign_with<F: FieldKernels>(dst: &mut [u8], coeff: &impl PreparedCoefficient<F>) {
    check_width::<F>("mul_assign_with", dst.len());
    let value = coeff.value();
    if value.is_one() {
        return;
    }
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(RawDispatch, dst, coeff.repr(RawDispatch));
}

/// Elementwise product: `dst[i] = a[i] * b[i]`.
///
/// Both operands vary per lane, so there is no coefficient to broadcast and
/// no nibble table to index. GFNI multiplies the vectors directly; the AVX-512
/// and AVX2 backends use a lane-parallel shift/reduce sequence. Shuffle-only
/// x86 backends and non-x86 targets fall back to the scalar reference path.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
pub fn mul_elementwise<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "a", a.len());
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "b", b.len());
    F::mul_elementwise(RawDispatch, dst, a, b);
}

/// In-place elementwise product: `dst[i] *= src[i]`.
///
/// The two-slice spelling of [`mul_elementwise`]: the destination is also the
/// first operand, so consumers composing lane-parallel Horner steps update
/// their accumulator without a second output buffer.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_elementwise_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("mul_elementwise_assign", "dst", dst.len(), "src", src.len());
    F::mul_elementwise_assign(RawDispatch, dst, src);
}

/// `dst[i] += value`, one field element broadcast across every lane.
///
/// The value is canonicalized once before dispatch, matching
/// [`mul_add`]'s treatment of its coefficient. Destination lanes holding
/// non-canonical prime-field words are canonicalized on load, exactly as in
/// [`add_assign`].
///
/// # Panics
/// Panics if `dst` holds a partial trailing element.
#[inline]
pub fn add_assign_scalar<F: FieldKernels>(dst: &mut [u8], value: Elem<F>) {
    check_width::<F>("add_assign_scalar", dst.len());
    F::add_assign_scalar(RawDispatch, dst, &F::prepare(RawDispatch, value));
}

/// `dst[i] -= value`, one field element broadcast across every lane.
///
/// The value is canonicalized once before dispatch. Destination lanes
/// holding non-canonical prime-field words are canonicalized on load,
/// exactly as in [`sub_assign`].
///
/// # Panics
/// Panics if `dst` holds a partial trailing element.
#[inline]
pub fn sub_assign_scalar<F: FieldKernels>(dst: &mut [u8], value: Elem<F>) {
    check_width::<F>("sub_assign_scalar", dst.len());
    F::sub_assign_scalar(RawDispatch, dst, &F::prepare(RawDispatch, value));
}
