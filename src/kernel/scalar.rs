//! Portable scalar kernels.
//!
//! These are field-generic and elementwise: correct everywhere, dependent on
//! nothing but [`Field`]. They provide the fallback on targets without a
//! supported vector unit and process sub-lane tails from vector loops. Backend
//! tests use them as the scalar reference except for the Fan–Paar family, whose
//! optimized element arithmetic is checked against the independent
//! [`crate::field::wiedemann`] recurrence.
//!
//! Hot scalar paths that beat the generic form live in the per-field dispatch
//! modules rather than here.

use crate::field::{Elem, FieldBuffer, FieldElem};

/// `dst ^= src`, eight bytes at a time.
///
/// Inlining lets short-buffer callers avoid an out-of-line kernel boundary.
///
/// # Panics
/// Panics if the slices differ in length.
#[inline]
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "xor: length mismatch");

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mixed =
            u64::from_ne_bytes(d.try_into().unwrap()) ^ u64::from_ne_bytes(s.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= s;
    }
}

/// `dst ^= coeff * src`, elementwise.
pub fn mul_add<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());

    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        xor(dst, src);
        return;
    }
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::decode(d).add(F::decode(s).mul(coeff));
        F::encode(d, value);
    }
}

/// `dst *= coeff`, elementwise, in place.
pub fn mul_assign<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>) {
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    for d in dst.chunks_exact_mut(F::BYTES) {
        let value = F::decode(d).mul(coeff);
        F::encode(d, value);
    }
}

/// `rows[j] ^= coeffs[j] * src` for every row `j`.
pub fn mul_add_scatter<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        mul_add::<F>(row, coeff, src);
    }
}

/// Apply every `(coeffs, src)` term to all `nrows` rows.
pub fn mul_add_matrix<F: FieldBuffer>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
            mul_add::<F>(row, coeff, src);
        }
    }
}

/// `dst ^= sum(coeffs[i] * srcs[i])`.
pub fn mul_add_gather<F: FieldBuffer>(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add::<F>(dst, coeff, src);
    }
}

/// `dst[i] = a[i] * b[i]`, elementwise.
pub fn mul_elementwise<F: FieldBuffer>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(x).mul(F::decode(y)));
    }
}

/// `dst[i] *= src[i]`, elementwise, in place.
pub fn mul_elementwise_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).mul(F::decode(s)));
    }
}

/// [`byte_ops::xor_broadcast`]'s portable body: XOR the cyclic `value_bytes`
/// pattern over the whole buffer, regardless of element boundaries.
///
/// The vector broadcast kernels peel whole lanes and hand every remaining
/// byte here, and the scalar arm of the dispatch runs it over the whole
/// destination; lane boundaries sit on element boundaries, so the phase of
/// `value_bytes` matches the destination's element phase throughout.
///
/// # Panics
/// Panics if `value_bytes` is empty.
#[inline]
pub(crate) fn xor_broadcast_bytes(dst: &mut [u8], value_bytes: &[u8]) {
    assert!(!value_bytes.is_empty(), "xor_broadcast_bytes: empty value");
    for (d, &p) in dst.iter_mut().zip(value_bytes.iter().cycle()) {
        *d ^= p;
    }
}

/// Implement [`crate::kernel::FieldKernels`] (default queries) and
/// [`crate::kernel::KernelDispatch`] (portable raw kernels) for a supported
/// field that has no hand-written SIMD backend.
macro_rules! impl_field_kernels {
    ($field:ty) => {
        impl crate::kernel::FieldKernels for $field {}

        impl crate::kernel::KernelDispatch for $field {
            type Prepared = crate::field::Elem<Self>;

            #[inline]
            fn prepare(
                _proof: crate::kernel::RawDispatch,
                coeff: crate::field::Elem<Self>,
            ) -> Self::Prepared {
                coeff
            }

            fn add_assign(_proof: crate::kernel::RawDispatch, dst: &mut [u8], src: &[u8]) {
                crate::kernel::xor(dst, src);
            }

            fn add_gather_offsets(
                _proof: crate::kernel::RawDispatch,
                region: &[u8],
                dst: &mut [u8],
                offsets: &[u32],
            ) {
                crate::kernel::xor_gather(region, dst, offsets);
            }

            fn sub_assign(_proof: crate::kernel::RawDispatch, dst: &mut [u8], src: &[u8]) {
                crate::kernel::xor(dst, src);
            }

            #[inline]
            fn prepared_coeff(
                _proof: crate::kernel::RawDispatch,
                prepared: &Self::Prepared,
            ) -> crate::field::Elem<Self> {
                *prepared
            }

            fn mul_add(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                coeff: &Self::Prepared,
                src: &[u8],
            ) {
                crate::kernel::scalar::mul_add::<Self>(dst, *coeff, src);
            }

            fn mul_assign(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                coeff: &Self::Prepared,
            ) {
                crate::kernel::scalar::mul_assign::<Self>(dst, *coeff);
            }

            fn mul_add_scatter(
                _proof: crate::kernel::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                coeffs: &[crate::field::Elem<Self>],
                src: &[u8],
            ) {
                crate::kernel::scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
            }

            fn mul_add_gather(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                coeffs: &[crate::field::Elem<Self>],
                srcs: &[&[u8]],
            ) {
                crate::kernel::scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
            }

            fn mul_add_matrix(
                _proof: crate::kernel::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[crate::field::Elem<Self>], &[u8])],
            ) {
                crate::kernel::scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
            }

            // `Prepared` is `Elem` for these fields, so the prepared forms
            // are the same call: there is nothing a backend could have
            // resolved ahead of time.
            #[cfg(feature = "alloc")]
            fn mul_add_scatter_with(
                _proof: crate::kernel::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                coeffs: &[Self::Prepared],
                src: &[u8],
            ) {
                crate::kernel::scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
            }

            #[cfg(feature = "alloc")]
            fn mul_add_gather_with(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                coeffs: &[Self::Prepared],
                srcs: &[&[u8]],
            ) {
                crate::kernel::scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
            }

            #[cfg(test)]
            fn mul_add_matrix_with(
                _proof: crate::kernel::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[Self::Prepared], &[u8])],
            ) {
                crate::kernel::scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
            }

            fn mul_elementwise(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                a: &[u8],
                b: &[u8],
            ) {
                crate::kernel::scalar::mul_elementwise::<Self>(dst, a, b);
            }
        }
    };
}

pub(crate) use impl_field_kernels;
