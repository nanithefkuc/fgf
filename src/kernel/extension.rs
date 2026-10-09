//! Scalar kernel dispatch for generic extension rows.
//!
//! Every extension row with a byte form runs the portable
//! [`crate::kernel::prime`] path over its own field arithmetic: the
//! generic family ships no vector kernels, so
//! [`FieldKernels::backend`] reports [`Backend::Scalar`] on every host,
//! elementwise multiplication is never vectorized, and the
//! vector-elementwise threshold is zero.
//!
//! The prepared coefficient is the canonical element itself: preparation
//! copies the value and allocates nothing, every packed application runs
//! allocation-free scalar loops, and no workspace carries meaning between
//! calls, so a prepared value may be reused any number of times.

use crate::field::Elem;
use crate::field::extension::storage::StorageBuffer;
use crate::field::extension::{Extension, ExtensionSpec, ExtensionStorage};
use crate::field::finite_field::FiniteField;
use crate::kernel::{FieldKernels, KernelDispatch, RawDispatch, prime};

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> FieldKernels
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageBuffer<S::Base, D>,
    ExtensionRaw<S, D>: Send + Sync,
{
}
/// The raw word of an extension row over `S`, whose element travels as the
/// prepared coefficient.
type ExtensionRaw<S, const D: usize> =
    <<S as ExtensionSpec<D>>::Storage as ExtensionStorage<<S as ExtensionSpec<D>>::Base, D>>::Raw;

impl<const P: u64, const N: u32, const D: usize, S: ExtensionSpec<D>> KernelDispatch
    for FiniteField<P, N, Extension<S, D>>
where
    S::Storage: StorageBuffer<S::Base, D>,
    ExtensionRaw<S, D>: Send + Sync,
{
    /// The canonical element; nothing to resolve.
    type Prepared = Elem<Self>;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared {
        coeff
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self> {
        *prepared
    }

    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        prime::add_assign::<Self>(dst, src);
    }

    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        prime::sub_assign::<Self>(dst, src);
    }

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        prime::mul_add::<Self>(dst, *coeff, src);
    }

    /// One pass: decode, multiply, encode, never reading the old
    /// destination.
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        prime::mul_into::<Self>(dst, *coeff, src);
    }

    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        prime::mul_assign::<Self>(dst, *coeff);
    }

    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    ) {
        prime::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        prime::mul_add_gather::<Self>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        prime::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        prime::mul_elementwise::<Self>(dst, a, b);
    }

    /// Modular broadcast addition. The explicit override keeps every
    /// characteristic, two included, off the binary XOR default: a
    /// component-storage row's lanes are base encodings, and a packed
    /// row's scalar add is the same modular path.
    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::add_assign_scalar::<Self>(dst, *value);
    }

    /// Modular broadcast subtraction, for the same reason.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::sub_assign_scalar::<Self>(dst, *value);
    }
}
