//! Residue prime-field kernel dispatch.
//!
//! Every `FiniteField<P, 1, Residue>` row runs the portable
//! [`crate::kernel::prime`] path: the generic family ships no vector
//! kernels, so [`FieldKernels::backend`] reports [`Backend::Scalar`] on
//! every host, elementwise multiplication is never vectorized, and the
//! vector-elementwise threshold is zero.
//!
//! The prepared coefficient is the canonical element itself: preparation
//! copies the value and allocates nothing, every packed application runs
//! allocation-free scalar loops, and no workspace carries meaning between
//! calls, so a prepared value may be reused any number of times.
//!
//! Broadcast add/sub route through the modular path for every
//! characteristic. The characteristic-two XOR default is a binary-field
//! convention; a residue row's lanes are eight-byte residues, and XOR
//! broadcast over them would leave noncanonical lane words behind at
//! `P = 2`.

use crate::field::Elem;
use crate::field::finite_field::{FiniteField, Residue};
use crate::kernel::{FieldKernels, KernelDispatch, RawDispatch, prime};

impl<const P: u64> FieldKernels for FiniteField<P, 1, Residue> {}

impl<const P: u64> KernelDispatch for FiniteField<P, 1, Residue> {
    /// The canonical element; nothing to resolve.
    type Prepared = Elem<FiniteField<P, 1, Residue>>;

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
    /// characteristic, `P = 2` included, off the binary XOR default.
    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::add_assign_scalar::<Self>(dst, *value);
    }

    /// Modular broadcast subtraction, for the same reason.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::sub_assign_scalar::<Self>(dst, *value);
    }
}
