//! Portable prime-field kernels.
//!
//! The reference and fallback path for the prime fields, and the differential
//! oracle for their SIMD backends. These are field-generic and elementwise:
//! correct on any [`Field`] whose [`Elem`] implements modular arithmetic.
//!
//! Unlike [`crate::kernel::scalar`], addition here is the field's modular
//! `add`, never XOR — these are wrong for the binary fields and right for the
//! prime fields. As with the scalar module, keeping this obviously correct is
//! worth more than making it fast: the vector backends carry the speed.

use crate::field::{Elem, Field};

/// `dst[i] += src[i]`, modular field addition.
pub fn add_assign<F: Field>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::read(d).add(F::read(s));
        F::write(d, value);
    }
}

/// `dst[i] -= src[i]`, modular field subtraction.
pub fn sub_assign<F: Field>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::read(d).sub(F::read(s));
        F::write(d, value);
    }
}

/// `dst[i] += coeff * src[i]`, elementwise.
pub fn mul_add<F: Field>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        add_assign::<F>(dst, src);
        return;
    }
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::read(d).add(F::read(s).mul(coeff));
        F::write(d, value);
    }
}

/// `dst[i] = coeff * dst[i]`, in place.
pub fn mul_assign<F: Field>(dst: &mut [u8], coeff: F::Elem) {
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    for d in dst.chunks_exact_mut(F::BYTES) {
        let value = F::read(d).mul(coeff);
        F::write(d, value);
    }
}

/// `rows[j] += coeffs[j] * src` for every row `j`.
pub fn mul_add_scatter<F: Field>(rows: &mut [u8], row_len: usize, coeffs: &[F::Elem], src: &[u8]) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        mul_add::<F>(row, coeff, src);
    }
}

/// `dst += sum(coeffs[i] * srcs[i])`.
pub fn mul_add_gather<F: Field>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add::<F>(dst, coeff, src);
    }
}

/// Apply every `(coeffs, src)` term to all `nrows` rows.
pub fn mul_add_matrix<F: Field>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Elem], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
            mul_add::<F>(row, coeff, src);
        }
    }
}

/// `dst[i] = a[i] * b[i]`, elementwise.
pub fn mul_elementwise<F: Field>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::write(d, F::read(x).mul(F::read(y)));
    }
}
