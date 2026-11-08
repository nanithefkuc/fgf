//! Kernel dispatch for extension rows, routed by storage.
//!
//! One `KernelDispatch` implementation covers every extension row with a
//! byte form; each entry forwards to the row's storage through the field
//! layer's `StorageDispatch` contract. The storage decides the prepared
//! coefficient and the arms a row can reach:
//!
//! - `Packed<u16>`, `Packed<u32>`, and `Packed<u64>` rows run the tower
//!   dispatch in [`super::tower`], whose vector arms serve the rows
//!   structurally equal to a shipped Rijndael or Fan-Paar instance and whose
//!   portable arm serves every other row.
//! - `Packed<u8>` rows run the scalar byte loops with XOR addition.
//! - `Components` rows structurally equal to the shipped prime extension run
//!   the QM31 kernels in [`super::quad_mersenne31`]; every other component
//!   row runs the portable [`prime`] path.
//!
//! A row outside the shipped instances reports [`Backend::Scalar`], no
//! vector elementwise capability, and a zero vector-elementwise threshold.

use crate::field::dispatch::{KernelDispatch, RawDispatch};
use crate::field::extension::pinned::{self, Instance};
use crate::field::extension::storage::{StorageBuffer, StorageDispatch, StorageRepr, StorageRow};
use crate::field::extension::{Components, Extension, ExtensionSpec, ExtensionStorage, Packed};
use crate::field::finite_field::FiniteField;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::field::{Elem, Field, FieldBuffer};
use crate::kernel::{Backend, prime, quad_mersenne31, scalar};

/// The storage of an extension row over `S`.
type StorageOf<S, const D: usize> = <S as ExtensionSpec<D>>::Storage;

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> KernelDispatch
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageBuffer<S::Base, D>,
{
    type Prepared = <StorageOf<S, D> as StorageDispatch<S::Base, D>>::Prepared;

    #[inline]
    fn backend(_proof: RawDispatch) -> Backend {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::backend::<Self>(RawDispatch)
    }

    #[inline]
    fn prepare(proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::prepare::<Self>(proof, coeff)
    }

    #[inline]
    fn prepared_coeff(proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self> {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::prepared_coeff::<Self>(proof, prepared)
    }

    #[inline]
    fn add_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::add_assign::<Self>(proof, dst, src);
    }

    #[inline]
    fn add_gather_offsets(proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::add_gather_offsets::<Self>(
            proof, region, dst, offsets,
        );
    }

    #[inline]
    fn sub_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::sub_assign::<Self>(proof, dst, src);
    }

    #[inline]
    fn add_assign_rows(proof: RawDispatch, dst: &mut [u8], src: &[u8], row_len: usize) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::add_assign_rows::<Self>(
            proof, dst, src, row_len,
        );
    }

    #[inline]
    fn sub_assign_rows(proof: RawDispatch, dst: &mut [u8], src: &[u8], row_len: usize) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::sub_assign_rows::<Self>(
            proof, dst, src, row_len,
        );
    }

    #[inline]
    fn mul_add(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add::<Self>(proof, dst, coeff, src);
    }

    #[inline]
    fn mul_into(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into::<Self>(proof, dst, coeff, src);
    }

    #[inline]
    fn mul_assign(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_assign::<Self>(proof, dst, coeff);
    }

    #[inline]
    fn mul_add_scatter(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_scatter::<Self>(
            proof, rows, row_len, coeffs, src,
        );
    }

    #[inline]
    fn mul_into_scatter(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_scatter::<Self>(
            proof, rows, row_len, coeffs, src,
        );
    }

    #[inline]
    fn mul_add_gather(proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_gather::<Self>(
            proof, dst, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_into_gather(proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_gather::<Self>(
            proof, dst, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_add_matrix(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_matrix::<Self>(
            proof, rows, row_len, nrows, terms,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_scatter_with(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_scatter_with::<Self>(
            proof, rows, row_len, coeffs, src,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_scatter_plan(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_scatter_plan::<Self>(
            proof, rows, row_len, values, coeffs, src,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_gather_with(
        proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_gather_with::<Self>(
            proof, dst, coeffs, srcs,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_gather_plan(
        proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_gather_plan::<Self>(
            proof, dst, values, coeffs, srcs,
        );
    }

    #[cfg(any(feature = "alloc", test))]
    #[inline]
    fn mul_into_gather_with(
        proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_gather_with::<Self>(
            proof, dst, coeffs, srcs,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_gather_plan(
        proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_gather_plan::<Self>(
            proof, dst, values, coeffs, srcs,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_matrix_plan(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_matrix_plan::<Self>(
            proof, rows, row_len, nrows, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_into_matrix(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_matrix::<Self>(
            proof, rows, row_len, nrows, terms,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_matrix_plan(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_matrix_plan::<Self>(
            proof, rows, row_len, nrows, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_add_matrix_at(
        proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_matrix_at::<Self>(
            proof, dst, row_len, row_starts, terms,
        );
    }

    #[inline]
    fn mul_add_matrix_at_rows(
        proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_matrix_at_rows::<Self>(
            proof, dst, row_len, row_starts, terms,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_matrix_at_plan(
        proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_add_matrix_at_plan::<Self>(
            proof, dst, row_len, row_starts, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_into_matrix_at(
        proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_matrix_at::<Self>(
            proof, dst, row_len, row_starts, terms,
        );
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_matrix_at_plan(
        proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_into_matrix_at_plan::<Self>(
            proof, dst, row_len, row_starts, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_elementwise(proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_elementwise::<Self>(proof, dst, a, b);
    }

    #[inline]
    fn mul_elementwise_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::mul_elementwise_assign::<Self>(
            proof, dst, src,
        );
    }

    #[inline]
    fn add_assign_scalar(proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::add_assign_scalar::<Self>(
            proof, dst, value,
        );
    }

    #[inline]
    fn sub_assign_scalar(proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        <StorageOf<S, D> as StorageDispatch<S::Base, D>>::sub_assign_scalar::<Self>(
            proof, dst, value,
        );
    }
}

// ---------------------------------------------------------------------------
// One-byte packed rows
// ---------------------------------------------------------------------------

/// The scalar byte loops: XOR addition, and multiplication through the
/// row's own arithmetic. The prepared coefficient is the raw byte.
impl<B: crate::field::BinaryField, const D: usize> StorageDispatch<B, D> for Packed<u8> {
    type Prepared = u8;

    #[inline]
    fn prepare<F: StorageRow<B, D, Self>>(_proof: RawDispatch, coeff: Elem<F>) -> u8 {
        coeff.raw
    }

    #[inline]
    fn prepared_coeff<F: StorageRow<B, D, Self>>(_proof: RawDispatch, prepared: &u8) -> Elem<F> {
        Elem { raw: *prepared }
    }

    #[inline]
    fn add_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn add_gather_offsets<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        region: &[u8],
        dst: &mut [u8],
        offsets: &[u32],
    ) {
        crate::kernel::xor_gather(region, dst, offsets);
    }

    #[inline]
    fn sub_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    fn mul_add<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &u8,
        src: &[u8],
    ) {
        scalar::mul_add::<F>(dst, Elem { raw: *coeff }, src);
    }

    fn mul_into<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &u8,
        src: &[u8],
    ) {
        dst.copy_from_slice(src);
        scalar::mul_assign::<F>(dst, Elem { raw: *coeff });
    }

    fn mul_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], coeff: &u8) {
        scalar::mul_assign::<F>(dst, Elem { raw: *coeff });
    }

    fn mul_add_scatter<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<F>],
        src: &[u8],
    ) {
        scalar::mul_add_scatter::<F>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        scalar::mul_add_gather::<F>(dst, coeffs, srcs);
    }

    fn mul_add_matrix<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        scalar::mul_add_matrix::<F>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        scalar::mul_elementwise::<F>(dst, a, b);
    }

    fallback_defaults!(
        storage: backend, mul_add_scatter_plan, mul_add_gather_plan,
        mul_add_matrix_plan, mul_elementwise_assign, add_assign_scalar,
        sub_assign_scalar,
    );
}

// ---------------------------------------------------------------------------
// Component rows
// ---------------------------------------------------------------------------

/// Whether the row is the shipped prime extension, whose QM31 kernels it
/// runs.
#[inline]
const fn is_quad_mersenne<F: Field>() -> bool {
    matches!(
        const { pinned::instance(F::DESCRIPTOR) },
        Some(Instance::QuadMersenne31)
    )
}

/// The QM31 element of a component row structurally equal to the shipped
/// prime extension, through its instance word.
#[inline]
fn quad<B: Field, const D: usize>(
    raw: <Components as ExtensionStorage<B, D>>::Raw,
) -> Elem<QuadMersenne31> {
    let word = <Components as StorageRepr<B, D>>::to_pinned_word(raw);
    // Splitting the instance word into its limbs: the truncation is the
    // operation.
    #[allow(clippy::cast_possible_truncation)]
    let limbs = [word as u32, (word >> 32) as u32];
    Elem { raw: limbs }
}

/// The portable [`prime`] path, or the QM31 kernels for the shipped prime
/// extension. The prepared coefficient is the raw component array.
impl<B: FieldBuffer, const D: usize> StorageDispatch<B, D> for Components {
    type Prepared = [B::Raw; D];

    #[inline]
    fn backend<F: StorageRow<B, D, Self>>(_proof: RawDispatch) -> Backend {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::backend_report();
        }
        Backend::Scalar
    }

    #[inline]
    fn prepare<F: StorageRow<B, D, Self>>(_proof: RawDispatch, coeff: Elem<F>) -> [B::Raw; D] {
        coeff.raw
    }

    #[inline]
    fn prepared_coeff<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        prepared: &[B::Raw; D],
    ) -> Elem<F> {
        Elem { raw: *prepared }
    }

    fn add_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::add_assign(dst, src);
        }
        prime::add_assign::<F>(dst, src);
    }

    fn sub_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::sub_assign(dst, src);
        }
        prime::sub_assign::<F>(dst, src);
    }

    fn mul_add<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &[B::Raw; D],
        src: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_add(dst, quad::<B, D>(*coeff), src);
        }
        prime::mul_add::<F>(dst, Elem { raw: *coeff }, src);
    }

    /// One pass: decode, multiply, encode, never reading the old
    /// destination.
    fn mul_into<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &[B::Raw; D],
        src: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_into(dst, quad::<B, D>(*coeff), src);
        }
        prime::mul_into::<F>(dst, Elem { raw: *coeff }, src);
    }

    fn mul_assign<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &[B::Raw; D],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_assign(dst, quad::<B, D>(*coeff));
        }
        prime::mul_assign::<F>(dst, Elem { raw: *coeff });
    }

    fn mul_add_scatter<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<F>],
        src: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            for (row, coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                quad_mersenne31::mul_add(row, quad::<B, D>(coeff.raw), src);
            }
            return;
        }
        prime::mul_add_scatter::<F>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        if is_quad_mersenne::<F>() {
            for (coeff, &src) in coeffs.iter().zip(srcs) {
                quad_mersenne31::mul_add(dst, quad::<B, D>(coeff.raw), src);
            }
            return;
        }
        prime::mul_add_gather::<F>(dst, coeffs, srcs);
    }

    fn mul_add_matrix<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        if is_quad_mersenne::<F>() {
            for &(coeffs, src) in terms {
                for (row, coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                    quad_mersenne31::mul_add(row, quad::<B, D>(coeff.raw), src);
                }
            }
            return;
        }
        prime::mul_add_matrix::<F>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_elementwise(dst, a, b);
        }
        prime::mul_elementwise::<F>(dst, a, b);
    }

    fn mul_elementwise_assign<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        src: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_elementwise_assign(dst, src);
        }
        scalar::mul_elementwise_assign::<F>(dst, src);
    }

    /// Modular broadcast addition. The explicit override keeps every
    /// characteristic, two included, off the binary XOR default: a
    /// component row's lanes are base encodings.
    fn add_assign_scalar<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        value: &[B::Raw; D],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::add_assign_scalar(dst, quad::<B, D>(*value));
        }
        prime::add_assign_scalar::<F>(dst, Elem { raw: *value });
    }

    /// Modular broadcast subtraction, for the same reason.
    fn sub_assign_scalar<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        value: &[B::Raw; D],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::sub_assign_scalar(dst, quad::<B, D>(*value));
        }
        prime::sub_assign_scalar::<F>(dst, Elem { raw: *value });
    }
    fallback_defaults!(
        storage: add_gather_offsets, mul_add_scatter_plan, mul_add_gather_plan,
        mul_add_matrix_plan,
    );
}
