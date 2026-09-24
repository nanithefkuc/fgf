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

use crate::field::{Elem, Field};

/// `dst ^= src`, eight bytes at a time.
///
/// `#[inline]`: the short-buffer callers reach it as their whole kernel,
/// and the call boundary alone cost more than the body below one vector
/// (see `kernel::xor_impl`'s short path and BENCHMARKS.md).
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
pub fn mul_add<F: Field>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
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
pub fn mul_assign<F: Field>(dst: &mut [u8], coeff: F::Elem) {
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
pub fn mul_add_scatter<F: Field>(rows: &mut [u8], row_len: usize, coeffs: &[F::Elem], src: &[u8]) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        mul_add::<F>(row, coeff, src);
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

/// Apply every `(coeffs, src)` term to disjoint `row_len`-byte rows scattered
/// through `dst` at byte offsets `row_starts`: row `j` is
/// `dst[row_starts[j] .. row_starts[j] + row_len]`.
pub fn mul_add_matrix_at<F: Field>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[F::Elem], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (&start, &coeff) in row_starts.iter().zip(coeffs) {
            mul_add::<F>(&mut dst[start..start + row_len], coeff, src);
        }
    }
}

/// `dst ^= sum(coeffs[i] * srcs[i])`.
pub fn mul_add_gather<F: Field>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add::<F>(dst, coeff, src);
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
        F::encode(d, F::decode(x).mul(F::decode(y)));
    }
}

/// `dst[i] *= src[i]`, elementwise, in place.
pub fn mul_elementwise_assign<F: Field>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).mul(F::decode(s)));
    }
}

/// `dst ^= value`, one element broadcast across every lane.
///
/// Characteristic-two fields only: the broadcast bytes are exclusive-ORed into
/// every lane, which is the field's addition there. The prime fields use the
/// modular fold of the portable prime-field reference instead.
pub fn xor_broadcast<F: Field>(dst: &mut [u8], value: F::Elem) {
    let mut encoded = [0u8; 8];
    F::encode(&mut encoded[..F::BYTES], value);
    let pattern = &encoded[..F::BYTES];
    for lane in dst.chunks_exact_mut(F::BYTES) {
        for (d, &p) in lane.iter_mut().zip(pattern) {
            *d ^= p;
        }
    }
}

/// [`xor_broadcast`]'s tail: XOR the cyclic `value_bytes` pattern over the
/// remaining bytes, regardless of element boundaries.
///
/// Vector broadcast kernels peel whole lanes and hand every remaining byte
/// here; lane boundaries sit on element boundaries, so the phase of
/// `value_bytes` matches the destination's element phase throughout.
///
/// # Panics
/// Panics if `value_bytes` is empty.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
#[allow(dead_code)]
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
            type Prepared = <Self as crate::field::Field>::Elem;

            #[inline]
            fn prepare(_proof: crate::kernel::RawDispatch, coeff: Self::Elem) -> Self::Prepared {
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
            ) -> Self::Elem {
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
                coeffs: &[Self::Elem],
                src: &[u8],
            ) {
                crate::kernel::scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
            }

            fn mul_add_gather(
                _proof: crate::kernel::RawDispatch,
                dst: &mut [u8],
                coeffs: &[Self::Elem],
                srcs: &[&[u8]],
            ) {
                crate::kernel::scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
            }

            fn mul_add_matrix(
                _proof: crate::kernel::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[Self::Elem], &[u8])],
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
