//! Kernel dispatch for extension rows, routed by storage.
//!
//! One [`FieldKernels`] and one `KernelDispatch` implementation cover every
//! extension row with a byte form; each entry forwards to the row's
//! storage through the crate-private [`StorageKernels`]. The storage decides
//! the prepared coefficient and the arms a row can reach:
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

use crate::field::extension::pinned::{self, Instance};
use crate::field::extension::storage::{StorageBuffer, StorageRepr};
use crate::field::extension::{Components, Extension, ExtensionSpec, ExtensionStorage, Packed};
use crate::field::finite_field::FiniteField;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::field::{Elem, Field, FieldBuffer, FieldElem};
use crate::kernel::{
    Backend, FieldKernels, KernelDispatch, RawDispatch, fallback, prime, quad_mersenne31, scalar,
};

/// A field row whose kernels a storage `St` runs: its raw word is the
/// storage's and its prepared coefficient the storage's.
pub(crate) trait StorageRow<B: Field, const D: usize, St: StorageKernels<B, D> + ?Sized>:
    KernelDispatch<Prepared = St::Prepared> + Field<Raw = <St as ExtensionStorage<B, D>>::Raw>
{
}

impl<B: Field, const D: usize, St: StorageKernels<B, D> + ?Sized, F> StorageRow<B, D, St> for F where
    F: KernelDispatch<Prepared = St::Prepared> + Field<Raw = <St as ExtensionStorage<B, D>>::Raw>
{
}

/// The kernel entries of one extension storage, generic over the row.
///
/// Each method is the row's `KernelDispatch` or [`FieldKernels`] entry of
/// the same name; the defaults are that trait's defaults. A method
/// reaching another entry calls it through the row `F`, so overrides and
/// defaults compose exactly as they do on a field's own implementation.
pub(crate) trait StorageKernels<B: Field, const D: usize>: StorageBuffer<B, D> {
    /// The backend-ready form of one coefficient.
    type Prepared: Clone + Send + Sync + core::fmt::Debug;

    /// [`FieldKernels::backend`].
    fn backend<F: StorageRow<B, D, Self>>() -> Backend {
        Backend::Scalar
    }

    /// [`FieldKernels::has_vector_elementwise`].
    fn has_vector_elementwise<F: StorageRow<B, D, Self>>() -> bool {
        false
    }

    /// [`FieldKernels::vector_elementwise_min_bytes`].
    fn vector_elementwise_min_bytes<F: StorageRow<B, D, Self>>() -> usize {
        0
    }

    /// `KernelDispatch::prepare`.
    fn prepare<F: StorageRow<B, D, Self>>(proof: RawDispatch, coeff: Elem<F>) -> Self::Prepared;

    /// `KernelDispatch::prepared_coeff`.
    fn prepared_coeff<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        prepared: &Self::Prepared,
    ) -> Elem<F>;

    /// `KernelDispatch::add_assign`.
    fn add_assign<F: StorageRow<B, D, Self>>(proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// `KernelDispatch::add_gather_offsets`.
    fn add_gather_offsets<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        region: &[u8],
        dst: &mut [u8],
        offsets: &[u32],
    ) {
        fallback::add_gather_offsets::<F>(region, dst, offsets);
    }

    /// `KernelDispatch::sub_assign`.
    fn sub_assign<F: StorageRow<B, D, Self>>(proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// `KernelDispatch::mul_add`.
    fn mul_add<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        coeff: &Self::Prepared,
        src: &[u8],
    );

    /// Applies a nonzero, non-unit coefficient value without unnecessary preparation.
    fn mul_add_unprepared<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        coeff: Elem<F>,
        src: &[u8],
    ) {
        let prepared = Self::prepare::<F>(proof, coeff);
        Self::mul_add::<F>(RawDispatch, dst, &prepared, src);
    }

    /// `KernelDispatch::mul_add_portable`.
    fn mul_add_portable<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: Elem<F>,
        src: &[u8],
    ) {
        prime::mul_add::<F>(dst, coeff, src);
    }

    /// `KernelDispatch::mul_into`.
    fn mul_into<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &Self::Prepared,
        src: &[u8],
    ) {
        fallback::mul_into::<F>(dst, coeff, src);
    }

    /// `KernelDispatch::mul_assign`.
    fn mul_assign<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        coeff: &Self::Prepared,
    );

    /// `KernelDispatch::mul_add_scatter`.
    fn mul_add_scatter<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<F>],
        src: &[u8],
    );

    /// `KernelDispatch::mul_add_gather`.
    fn mul_add_gather<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    );

    /// `KernelDispatch::mul_into_gather`.
    fn mul_into_gather<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        fallback::mul_into_gather::<F>(dst, coeffs, srcs);
    }

    /// `KernelDispatch::mul_into_gather_with`.
    #[cfg(any(feature = "alloc", test))]
    fn mul_into_gather_with<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        fallback::mul_into_gather_with::<F>(dst, coeffs, srcs);
    }

    /// `KernelDispatch::mul_add_matrix`.
    fn mul_add_matrix<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    );

    /// `KernelDispatch::mul_add_scatter_plan`.
    #[cfg(feature = "alloc")]
    fn mul_add_scatter_plan<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        _values: &[Elem<F>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        F::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src);
    }

    /// `KernelDispatch::mul_add_gather_plan`.
    #[cfg(feature = "alloc")]
    fn mul_add_gather_plan<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[Elem<F>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        F::mul_add_gather_with(RawDispatch, dst, coeffs, srcs);
    }

    /// `KernelDispatch::mul_add_matrix_plan`.
    #[cfg(feature = "alloc")]
    fn mul_add_matrix_plan<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        _values: &[Elem<F>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        fallback::mul_add_matrix_plan::<F>(rows, row_len, nrows, coeffs, srcs);
    }

    /// `KernelDispatch::mul_elementwise`.
    fn mul_elementwise<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    );

    /// `KernelDispatch::mul_elementwise_assign`.
    fn mul_elementwise_assign<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        src: &[u8],
    ) {
        scalar::mul_elementwise_assign::<F>(dst, src);
    }

    /// `KernelDispatch::add_assign_scalar`.
    fn add_assign_scalar<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        value: &Self::Prepared,
    ) {
        fallback::add_assign_scalar::<F>(dst, value);
    }

    /// `KernelDispatch::sub_assign_scalar`.
    fn sub_assign_scalar<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        value: &Self::Prepared,
    ) {
        fallback::sub_assign_scalar::<F>(dst, value);
    }
}

/// The storage of an extension row over `S`.
type StorageOf<S, const D: usize> = <S as ExtensionSpec<D>>::Storage;

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> FieldKernels
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageKernels<S::Base, D>,
{
    #[inline]
    fn backend() -> Backend {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::backend::<Self>()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::has_vector_elementwise::<Self>()
    }

    #[inline]
    fn vector_elementwise_min_bytes() -> usize {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::vector_elementwise_min_bytes::<Self>()
    }
}

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> KernelDispatch
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageKernels<S::Base, D>,
{
    type Prepared = <StorageOf<S, D> as StorageKernels<S::Base, D>>::Prepared;

    #[inline]
    fn prepare(proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::prepare::<Self>(proof, coeff)
    }

    #[inline]
    fn prepared_coeff(proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self> {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::prepared_coeff::<Self>(proof, prepared)
    }

    #[inline]
    fn add_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::add_assign::<Self>(proof, dst, src);
    }

    #[inline]
    fn add_gather_offsets(proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::add_gather_offsets::<Self>(
            proof, region, dst, offsets,
        );
    }

    #[inline]
    fn sub_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::sub_assign::<Self>(proof, dst, src);
    }

    #[inline]
    fn mul_add(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add::<Self>(proof, dst, coeff, src);
    }

    #[inline]
    fn mul_add_unprepared(proof: RawDispatch, dst: &mut [u8], coeff: Elem<Self>, src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_unprepared::<Self>(
            proof, dst, coeff, src,
        );
    }

    #[inline]
    fn mul_add_portable(proof: RawDispatch, dst: &mut [u8], coeff: Elem<Self>, src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_portable::<Self>(
            proof, dst, coeff, src,
        );
    }

    #[inline]
    fn mul_into(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_into::<Self>(proof, dst, coeff, src);
    }

    #[inline]
    fn mul_assign(proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_assign::<Self>(proof, dst, coeff);
    }

    #[inline]
    fn mul_add_scatter(
        proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    ) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_scatter::<Self>(
            proof, rows, row_len, coeffs, src,
        );
    }

    #[inline]
    fn mul_add_gather(proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_gather::<Self>(
            proof, dst, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_into_gather(proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_into_gather::<Self>(
            proof, dst, coeffs, srcs,
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
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_into_gather_with::<Self>(
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
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_matrix::<Self>(
            proof, rows, row_len, nrows, terms,
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
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_scatter_plan::<Self>(
            proof, rows, row_len, values, coeffs, src,
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
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_gather_plan::<Self>(
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
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_add_matrix_plan::<Self>(
            proof, rows, row_len, nrows, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_elementwise(proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_elementwise::<Self>(proof, dst, a, b);
    }

    #[inline]
    fn mul_elementwise_assign(proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::mul_elementwise_assign::<Self>(
            proof, dst, src,
        );
    }

    #[inline]
    fn add_assign_scalar(proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::add_assign_scalar::<Self>(
            proof, dst, value,
        );
    }

    #[inline]
    fn sub_assign_scalar(proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        <StorageOf<S, D> as StorageKernels<S::Base, D>>::sub_assign_scalar::<Self>(
            proof, dst, value,
        );
    }
}

// ---------------------------------------------------------------------------
// One-byte packed rows
// ---------------------------------------------------------------------------

/// The scalar byte loops: XOR addition, and multiplication through the
/// row's own arithmetic. The prepared coefficient is the raw byte.
impl<B: crate::field::BinaryField, const D: usize> StorageKernels<B, D> for Packed<u8> {
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

/// Smallest component row, in bytes, whose addition and subtraction reuse
/// the base field's whole-buffer dispatch. Shorter rows keep the component
/// loop. Set by the paired composition record in `benchmarks/v3/compositions.md`.
const COMPONENT_BASE_ADD_MIN_BYTES: usize = 2048;

/// Whether a component row of `len` bytes reuses base-field addition.
///
/// Order-two bases keep the component loop at every length.
const fn reuses_base_addition<B: Field>(len: usize) -> bool {
    B::ORDER != 2 && len >= COMPONENT_BASE_ADD_MIN_BYTES
}

/// The portable [`prime`] path, or the QM31 kernels for the shipped prime
/// extension. The prepared coefficient is the raw component array.
impl<B: FieldBuffer, const D: usize> StorageKernels<B, D> for Components
where
    B::Raw: Send + Sync,
{
    type Prepared = [B::Raw; D];

    #[inline]
    fn backend<F: StorageRow<B, D, Self>>() -> Backend {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::backend_report();
        }
        Backend::Scalar
    }

    #[inline]
    fn has_vector_elementwise<F: StorageRow<B, D, Self>>() -> bool {
        is_quad_mersenne::<F>() && quad_mersenne31::has_vector_elementwise()
    }

    #[inline]
    fn vector_elementwise_min_bytes<F: StorageRow<B, D, Self>>() -> usize {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::vector_elementwise_min_bytes();
        }
        0
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
        if reuses_base_addition::<B>(dst.len()) {
            return F::add_assign_bytes(RawDispatch, dst, src);
        }
        prime::add_assign::<F>(dst, src);
    }

    fn sub_assign<F: StorageRow<B, D, Self>>(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::sub_assign(dst, src);
        }
        if reuses_base_addition::<B>(dst.len()) {
            return F::sub_assign_bytes(RawDispatch, dst, src);
        }
        prime::sub_assign::<F>(dst, src);
    }

    fn mul_add<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: &[B::Raw; D],
        src: &[u8],
    ) {
        let value = Elem::<F> { raw: *coeff };
        if value.is_zero() {
            return;
        }
        if value.is_one() {
            return F::add_assign(RawDispatch, dst, src);
        }
        // Canonical zero and unit comparisons exclude both trivial coefficients.
        #[cfg(test)]
        let _general_trace = crate::kernel::trace::term(crate::kernel::trace::Term::General);
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_add(dst, quad::<B, D>(*coeff), src);
        }
        prime::mul_add::<F>(dst, value, src);
    }

    fn mul_add_unprepared<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: Elem<F>,
        src: &[u8],
    ) {
        if is_quad_mersenne::<F>() {
            return quad_mersenne31::mul_add(dst, quad::<B, D>(coeff.raw), src);
        }
        prime::mul_add::<F>(dst, coeff, src);
    }

    /// The unprepared row body is already portable and accepts every
    /// coefficient, including zero and one.
    fn mul_add_portable<F: StorageRow<B, D, Self>>(
        proof: RawDispatch,
        dst: &mut [u8],
        coeff: Elem<F>,
        src: &[u8],
    ) {
        Self::mul_add_unprepared::<F>(proof, dst, coeff, src);
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
        fallback::mul_add_scatter::<F>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        fallback::mul_add_gather::<F>(dst, coeffs, srcs);
    }

    fn mul_add_matrix<F: StorageRow<B, D, Self>>(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        fallback::mul_add_matrix::<F>(rows, row_len, nrows, terms);
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
}
