//! The default bodies of the dispatch contract's entries.
//!
//! The field layer's [`KernelDispatch`] and
//! [`StorageDispatch`](crate::field::extension::storage::StorageDispatch)
//! contracts are declarations only; this module owns every body an
//! implementation may forward to, so a row or storage that omits an entry
//! runs exactly the composition the contract's defaults used to carry.
//! Each free function reaches other entries through the row `F` — never
//! through another fallback — so specialized overrides keep composing.
//! The macros at the bottom emit the forwarding methods an implementation
//! lists by operation name.

use crate::field::dispatch::{KernelDispatch, RawDispatch};
use crate::field::{Elem, Field, FieldBuffer, PrimeIdentity};

use super::{bytes, prime};

/// Folds byte-offset sources one at a time through `add_assign`.
pub(crate) fn add_gather_offsets<F: KernelDispatch>(
    region: &[u8],
    dst: &mut [u8],
    offsets: &[u32],
) {
    let live = dst.len();
    for &start in offsets {
        F::add_assign(
            RawDispatch,
            dst,
            &region[start as usize..start as usize + live],
        );
    }
}

/// Pairwise row addition over the whole buffers.
pub(crate) fn add_assign_rows<F: KernelDispatch>(dst: &mut [u8], src: &[u8], _row_len: usize) {
    F::add_assign(RawDispatch, dst, src);
}

/// Pairwise row subtraction over the whole buffers.
pub(crate) fn sub_assign_rows<F: KernelDispatch>(dst: &mut [u8], src: &[u8], _row_len: usize) {
    F::sub_assign(RawDispatch, dst, src);
}

/// One source overwrites many rows through the fused single-row kernel.
pub(crate) fn mul_into_scatter<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<F>],
    src: &[u8],
) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        F::mul_into(RawDispatch, row, &F::prepare(RawDispatch, coeff), src);
    }
}

/// Many sources overwrite one row: fused first term, then accumulation.
pub(crate) fn mul_into_gather<F: KernelDispatch>(
    dst: &mut [u8],
    coeffs: &[Elem<F>],
    srcs: &[&[u8]],
) {
    let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
    let Some((first, src)) = pairs.next() else {
        dst.fill(0);
        return;
    };
    F::mul_into(RawDispatch, dst, &F::prepare(RawDispatch, first), src);
    for (coeff, src) in pairs {
        F::mul_add(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
    }
}

/// Prepared single-coefficient AXPY per row.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_scatter_with<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[F::Prepared],
    src: &[u8],
) {
    for (row, coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        F::mul_add(RawDispatch, row, coeff, src);
    }
}

/// Prepared-plan scatter: the row's prepared scatter entry.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_scatter_plan<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    _values: &[Elem<F>],
    coeffs: &[F::Prepared],
    src: &[u8],
) {
    F::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src);
}

/// Prepared single-coefficient AXPY per source.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_gather_with<F: KernelDispatch>(
    dst: &mut [u8],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    for (coeff, &src) in coeffs.iter().zip(srcs) {
        F::mul_add(RawDispatch, dst, coeff, src);
    }
}

/// Prepared-plan gather: the row's prepared gather entry.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_gather_plan<F: KernelDispatch>(
    dst: &mut [u8],
    _values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    F::mul_add_gather_with(RawDispatch, dst, coeffs, srcs);
}

/// Prepared overwrite of one row: fused first term, then accumulation.
#[cfg(any(feature = "alloc", test))]
pub(crate) fn mul_into_gather_with<F: KernelDispatch>(
    dst: &mut [u8],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    let mut pairs = coeffs.iter().zip(srcs.iter().copied());
    let Some((first, src)) = pairs.next() else {
        dst.fill(0);
        return;
    };
    F::mul_into(RawDispatch, dst, first, src);
    for (coeff, src) in pairs {
        F::mul_add(RawDispatch, dst, coeff, src);
    }
}

/// Prepared-plan dot product: the row's prepared overwrite entry.
#[cfg(feature = "alloc")]
pub(crate) fn mul_into_gather_plan<F: KernelDispatch>(
    dst: &mut [u8],
    _values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    F::mul_into_gather_with(RawDispatch, dst, coeffs, srcs);
}

/// Prepared matrix over raw-element terms, row by row.
#[cfg(test)]
pub(crate) fn mul_add_matrix_with<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Prepared], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
            F::mul_add(RawDispatch, row, coeff, src);
        }
    }
}

/// Allocation-free repeated prepared AXPY over flat row-major
/// coefficients.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_matrix_plan<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    let _ = values;
    for (term, &src) in srcs.iter().enumerate() {
        let start = term * nrows;
        for (row, coeff) in rows
            .chunks_exact_mut(row_len)
            .take(nrows)
            .zip(&coeffs[start..start + nrows])
        {
            F::mul_add(RawDispatch, row, coeff, src);
        }
    }
}

/// Zero the output rows, then accumulate the raw-element terms.
pub(crate) fn mul_into_matrix<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<F>], &[u8])],
) {
    for row in rows.chunks_exact_mut(row_len).take(nrows) {
        row.fill(0);
    }
    F::mul_add_matrix(RawDispatch, rows, row_len, nrows, terms);
}

/// Zero the output rows, then run the prepared plan.
#[cfg(feature = "alloc")]
pub(crate) fn mul_into_matrix_plan<F: KernelDispatch>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    for row in rows.chunks_exact_mut(row_len).take(nrows) {
        row.fill(0);
    }
    F::mul_add_matrix_plan(RawDispatch, rows, row_len, nrows, values, coeffs, srcs);
}

/// Scattered accumulation through the per-row fallback.
pub(crate) fn mul_add_matrix_at<F: KernelDispatch>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    F::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms);
}

/// One dispatched `mul_add` per scattered row and term.
pub(crate) fn mul_add_matrix_at_rows<F: KernelDispatch>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (&start, &coeff) in row_starts.iter().zip(coeffs) {
            F::mul_add(
                RawDispatch,
                &mut dst[start..start + row_len],
                &F::prepare(RawDispatch, coeff),
                src,
            );
        }
    }
}

/// Allocation-free repeated prepared AXPY over the named scattered rows.
#[cfg(feature = "alloc")]
pub(crate) fn mul_add_matrix_at_plan<F: KernelDispatch>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    let _ = values;
    let nrows = row_starts.len();
    for (term, &src) in srcs.iter().enumerate() {
        let start = term * nrows;
        for (&at, coeff) in row_starts.iter().zip(&coeffs[start..start + nrows]) {
            F::mul_add(RawDispatch, &mut dst[at..at + row_len], coeff, src);
        }
    }
}

/// Zero the named rows, then accumulate.
pub(crate) fn mul_into_matrix_at<F: KernelDispatch>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[Elem<F>], &[u8])],
) {
    for &start in row_starts {
        dst[start..start + row_len].fill(0);
    }
    F::mul_add_matrix_at(RawDispatch, dst, row_len, row_starts, terms);
}

/// Zero the named rows, then run the prepared plan.
#[cfg(feature = "alloc")]
pub(crate) fn mul_into_matrix_at_plan<F: KernelDispatch>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    values: &[Elem<F>],
    coeffs: &[F::Prepared],
    srcs: &[&[u8]],
) {
    for &start in row_starts {
        dst[start..start + row_len].fill(0);
    }
    F::mul_add_matrix_at_plan(RawDispatch, dst, row_len, row_starts, values, coeffs, srcs);
}

/// Broadcast addition: XOR of the encoded value in characteristic two,
/// the modular path otherwise.
pub(crate) fn add_assign_scalar<F: FieldBuffer>(dst: &mut [u8], value: &F::Prepared) {
    let elem = F::prepared_coeff(RawDispatch, value);
    if <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2 {
        let mut encoded = [0u8; 8];
        F::encode(&mut encoded[..F::BYTES], elem);
        bytes::xor_broadcast(dst, &encoded[..F::BYTES]);
    } else {
        prime::add_assign_scalar::<F>(dst, elem);
    }
}

/// Broadcast subtraction, as [`add_assign_scalar`].
pub(crate) fn sub_assign_scalar<F: FieldBuffer>(dst: &mut [u8], value: &F::Prepared) {
    let elem = F::prepared_coeff(RawDispatch, value);
    if <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2 {
        let mut encoded = [0u8; 8];
        F::encode(&mut encoded[..F::BYTES], elem);
        bytes::xor_broadcast(dst, &encoded[..F::BYTES]);
    } else {
        prime::sub_assign_scalar::<F>(dst, elem);
    }
}

/// Emit one forwarding entry of the dispatch contracts, selected by
/// operation name.
///
/// The `row` form implements a [`KernelDispatch`] entry for `Self`; the
/// `storage` form implements a
/// [`StorageDispatch`](crate::field::extension::storage::StorageDispatch)
/// entry over the row parameter `F`. Each emitted body forwards to the
/// free function in this module holding the contract's default
/// composition, so an implementation listing a name here keeps exactly the
/// behavior the declaration-level default used to carry.
macro_rules! fallback_entry {
    (row backend) => {
        #[inline]
        fn backend(_proof: $crate::field::dispatch::RawDispatch) -> Backend {
            Backend::Scalar
        }
    };
    (row has_vector_elementwise) => {
        #[inline]
        fn has_vector_elementwise(_proof: $crate::field::dispatch::RawDispatch) -> bool {
            false
        }
    };
    (row vector_elementwise_min_bytes) => {
        #[inline]
        fn vector_elementwise_min_bytes(_proof: $crate::field::dispatch::RawDispatch) -> usize {
            0
        }
    };
    (row add_gather_offsets) => {
        #[inline]
        fn add_gather_offsets(
            _proof: $crate::field::dispatch::RawDispatch,
            region: &[u8],
            dst: &mut [u8],
            offsets: &[u32],
        ) {
            $crate::kernel::fallback::add_gather_offsets::<Self>(region, dst, offsets)
        }
    };
    (row add_assign_rows) => {
        #[inline]
        fn add_assign_rows(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            src: &[u8],
            row_len: usize,
        ) {
            $crate::kernel::fallback::add_assign_rows::<Self>(dst, src, row_len)
        }
    };
    (row sub_assign_rows) => {
        #[inline]
        fn sub_assign_rows(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            src: &[u8],
            row_len: usize,
        ) {
            $crate::kernel::fallback::sub_assign_rows::<Self>(dst, src, row_len)
        }
    };
    (row mul_into_scatter) => {
        #[inline]
        fn mul_into_scatter(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            coeffs: &[Elem<Self>],
            src: &[u8],
        ) {
            $crate::kernel::fallback::mul_into_scatter::<Self>(rows, row_len, coeffs, src)
        }
    };
    (row mul_into_gather) => {
        #[inline]
        fn mul_into_gather(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            coeffs: &[Elem<Self>],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_into_gather::<Self>(dst, coeffs, srcs)
        }
    };
    (row mul_add_scatter_with) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_scatter_with(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            coeffs: &[Self::Prepared],
            src: &[u8],
        ) {
            $crate::kernel::fallback::mul_add_scatter_with::<Self>(rows, row_len, coeffs, src)
        }
    };
    (row mul_add_scatter_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_scatter_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            src: &[u8],
        ) {
            $crate::kernel::fallback::mul_add_scatter_plan::<Self>(
                rows, row_len, values, coeffs, src,
            )
        }
    };
    (row mul_add_gather_with) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_gather_with(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_gather_with::<Self>(dst, coeffs, srcs)
        }
    };
    (row mul_add_gather_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_gather_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_gather_plan::<Self>(dst, values, coeffs, srcs)
        }
    };
    (row mul_into_gather_with) => {
        #[cfg(any(feature = "alloc", test))]
        #[inline]
        fn mul_into_gather_with(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_into_gather_with::<Self>(dst, coeffs, srcs)
        }
    };
    (row mul_into_gather_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_into_gather_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_into_gather_plan::<Self>(dst, values, coeffs, srcs)
        }
    };
    (row mul_add_matrix_with) => {
        #[cfg(test)]
        #[inline]
        fn mul_add_matrix_with(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Self::Prepared], &[u8])],
        ) {
            $crate::kernel::fallback::mul_add_matrix_with::<Self>(rows, row_len, nrows, terms)
        }
    };
    (row mul_add_matrix_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_matrix_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_matrix_plan::<Self>(
                rows, row_len, nrows, values, coeffs, srcs,
            )
        }
    };
    (row mul_into_matrix) => {
        #[inline]
        fn mul_into_matrix(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem<Self>], &[u8])],
        ) {
            $crate::kernel::fallback::mul_into_matrix::<Self>(rows, row_len, nrows, terms)
        }
    };
    (row mul_into_matrix_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_into_matrix_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_into_matrix_plan::<Self>(
                rows, row_len, nrows, values, coeffs, srcs,
            )
        }
    };
    (row mul_add_matrix_at) => {
        #[inline]
        fn mul_add_matrix_at(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            row_len: usize,
            row_starts: &[usize],
            terms: &[(&[Elem<Self>], &[u8])],
        ) {
            $crate::kernel::fallback::mul_add_matrix_at::<Self>(dst, row_len, row_starts, terms)
        }
    };
    (row mul_add_matrix_at_rows) => {
        #[inline]
        fn mul_add_matrix_at_rows(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            row_len: usize,
            row_starts: &[usize],
            terms: &[(&[Elem<Self>], &[u8])],
        ) {
            $crate::kernel::fallback::mul_add_matrix_at_rows::<Self>(
                dst, row_len, row_starts, terms,
            )
        }
    };
    (row mul_add_matrix_at_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_matrix_at_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            row_len: usize,
            row_starts: &[usize],
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_matrix_at_plan::<Self>(
                dst, row_len, row_starts, values, coeffs, srcs,
            )
        }
    };
    (row mul_into_matrix_at) => {
        #[inline]
        fn mul_into_matrix_at(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            row_len: usize,
            row_starts: &[usize],
            terms: &[(&[Elem<Self>], &[u8])],
        ) {
            $crate::kernel::fallback::mul_into_matrix_at::<Self>(dst, row_len, row_starts, terms)
        }
    };
    (row mul_into_matrix_at_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_into_matrix_at_plan(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            row_len: usize,
            row_starts: &[usize],
            values: &[Elem<Self>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_into_matrix_at_plan::<Self>(
                dst, row_len, row_starts, values, coeffs, srcs,
            )
        }
    };
    (row add_assign_scalar) => {
        #[inline]
        fn add_assign_scalar(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            value: &Self::Prepared,
        ) {
            $crate::kernel::fallback::add_assign_scalar::<Self>(dst, value)
        }
    };
    (row sub_assign_scalar) => {
        #[inline]
        fn sub_assign_scalar(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            value: &Self::Prepared,
        ) {
            $crate::kernel::fallback::sub_assign_scalar::<Self>(dst, value)
        }
    };

    (storage backend) => {
        #[inline]
        fn backend<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
        ) -> Backend {
            Backend::Scalar
        }
    };
    (storage has_vector_elementwise) => {
        #[inline]
        fn has_vector_elementwise<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
        ) -> bool {
            false
        }
    };
    (storage vector_elementwise_min_bytes) => {
        #[inline]
        fn vector_elementwise_min_bytes<
            F: $crate::field::extension::storage::StorageRow<B, D, Self>,
        >(
            _proof: $crate::field::dispatch::RawDispatch,
        ) -> usize {
            0
        }
    };
    (storage add_gather_offsets) => {
        #[inline]
        fn add_gather_offsets<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            region: &[u8],
            dst: &mut [u8],
            offsets: &[u32],
        ) {
            $crate::kernel::fallback::add_gather_offsets::<F>(region, dst, offsets)
        }
    };
    (storage mul_add_scatter_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_scatter_plan<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            values: &[Elem<F>],
            coeffs: &[Self::Prepared],
            src: &[u8],
        ) {
            $crate::kernel::fallback::mul_add_scatter_plan::<F>(rows, row_len, values, coeffs, src)
        }
    };
    (storage mul_add_gather_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_gather_plan<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            values: &[Elem<F>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_gather_plan::<F>(dst, values, coeffs, srcs)
        }
    };
    (storage mul_add_matrix_plan) => {
        #[cfg(feature = "alloc")]
        #[inline]
        fn mul_add_matrix_plan<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            values: &[Elem<F>],
            coeffs: &[Self::Prepared],
            srcs: &[&[u8]],
        ) {
            $crate::kernel::fallback::mul_add_matrix_plan::<F>(
                rows, row_len, nrows, values, coeffs, srcs,
            )
        }
    };
    (storage mul_elementwise_assign) => {
        #[inline]
        fn mul_elementwise_assign<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            src: &[u8],
        ) {
            $crate::kernel::scalar::mul_elementwise_assign::<F>(dst, src)
        }
    };
    (storage add_assign_scalar) => {
        #[inline]
        fn add_assign_scalar<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            value: &Self::Prepared,
        ) {
            $crate::kernel::fallback::add_assign_scalar::<F>(dst, value)
        }
    };
    (storage sub_assign_scalar) => {
        #[inline]
        fn sub_assign_scalar<F: $crate::field::extension::storage::StorageRow<B, D, Self>>(
            _proof: $crate::field::dispatch::RawDispatch,
            dst: &mut [u8],
            value: &Self::Prepared,
        ) {
            $crate::kernel::fallback::sub_assign_scalar::<F>(dst, value)
        }
    };
}

/// Emit a set of forwarding entries of the dispatch contracts, selected by
/// operation name.
///
/// `row:` lists [`KernelDispatch`] entries an implementation leaves to the
/// contract's fallback; `storage:` lists
/// [`StorageDispatch`](crate::field::extension::storage::StorageDispatch)
/// entries likewise. Each listed name forwards to the corresponding engine
/// default body.
macro_rules! fallback_entries {
    (row: $($name:ident),* $(,)?) => {
        $( fallback_entry!(row $name); )*
    };
    (storage: $($name:ident),* $(,)?) => {
        $( fallback_entry!(storage $name); )*
    };
}
