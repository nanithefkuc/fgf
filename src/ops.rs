//! The public vector operation surface.
//!
//! Every arithmetic entry is generic over [`FieldKernels`] and monomorphizes
//! to that row's dispatched SIMD kernel. Encoding and map rows are
//! [`FieldBuffer`]-only instead: packing (`pack`,
//! `unpack`, `pack_to_vec`) and the prepared maps in `map` work over the
//! stable little-endian byte form, never a kernel operand. Buffers are plain
//! `&[u8]` holding packed elements in that representation. Canonically
//! encoded payloads need no repacking; Section "Prime fields" below states
//! the packed-lane contract. GF(2)-endpoint map rows over the bit-packed
//! [`crate::bits`] representation live in `map_bits`.
//!
//! Implementations split by operation shape: single-row, scatter, gather,
//! and matrix. Coefficient storage and packing conversions have separate
//! modules; this module retains the shared contracts and public paths.
//!
//! # Naming
//!
//! `mul_add` is the fused `dst += coeff * src` (an AXPY; XOR in
//! characteristic two). `_scatter` fans one source out to many rows,
//! `_gather` folds many sources into one row, and `_matrix` does
//! many-to-many with the destination held in registers across sources.
//! These three shapes cover systematic encoding, symbol reconstruction,
//! and erasure decoding respectively.
//!
//! # Reusing a coefficient
//!
//! Each `mul_*` call resolves its coefficient into the form the host backend
//! wants. For GF(2^8) that is an array index; for GF(2^16) on a shuffle
//! backend it is four nibble tables. Amortized over a large buffer that cost
//! vanishes, but across many short buffers with the same coefficient it does
//! not. [`Coeff`] hoists it:
//!
//! ```
//! use fgf::{Elem, Extension, FiniteField, Rijndael16, ops};
//!
//! type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
//! let coeff = ops::Coeff::<F>::new(Elem::<F>::from_raw(0x0108));
//! let src = [0u8; 64];
//! for _ in 0..3 {
//!     let mut symbol = [0u8; 64];
//!     ops::mul_add_with(&mut symbol, &coeff, &src);   // prepared once
//! }
//! ```
//!
//! # Preconditions
//!
//! Buffer lengths must be whole multiples of `F::BYTES`, paired buffers must
//! be equal in length, and `dst`/`src` must not alias. Length constraints are
//! checked and violations panic; Rust's borrowing rules enforce disjointness.
//!
//! # Prime fields: any lanes in, canonical lanes out, with byte-preserving shortcuts
//!
//! The packed operations over the prime fields ([`Mersenne31`], [`Goldilocks`],
//! [`QuadMersenne31`], and the generic residue rows `FiniteField<P, 1, Residue>`)
//! accept every lane bit pattern a decode accepts and
//! produce canonical lanes on output — accumulation, in-place, and tails
//! included — with three byte-preserving exceptions: a zero coefficient
//! fills zero lanes in overwrite and leaves the destination untouched in
//! accumulation; a unit coefficient copies the source bytes unchanged in
//! overwrite, adds them in accumulation, and leaves the destination
//! untouched in place. Those paths preserve noncanonical lane bytes
//! instead of canonicalizing them. That is the invariant a codec maintains
//! over its buffers, and it is what these kernels preserve. Scalar element
//! arithmetic shares the same totality: the [`FieldElem`](crate::field::FieldElem) operations accept
//! any raw lane value and always reduce canonically.
//!
//! [`Mersenne31`]: crate::Mersenne31
//! [`Goldilocks`]: crate::Goldilocks
//! [`QuadMersenne31`]: crate::QuadMersenne31

use crate::field::{Elem, FieldBuffer};
use crate::kernel::{FieldKernels, KernelDispatch, RawDispatch};

mod coefficient;
mod gather;
#[cfg(feature = "alloc")]
pub mod map;
#[cfg(feature = "alloc")]
pub mod map_bits;
mod matrix;
mod packing;
mod recip;
mod scatter;
mod single;

pub use coefficient::Coeff;
pub use gather::{add_gather_offsets, mul_add_gather, mul_into_gather};
pub use matrix::{mul_add_matrix, mul_add_matrix_at, mul_into_matrix, mul_into_matrix_at};
pub use packing::{pack, unpack};
pub use recip::{div_elementwise, div_elementwise_assign, inv_assign, inv_into};
pub use scatter::mul_add_scatter;
pub use single::{
    add_assign, add_assign_rows, add_assign_scalar, mul_add, mul_add_with, mul_assign,
    mul_assign_with, mul_elementwise, mul_elementwise_assign, mul_into, mul_into_with, sub_assign,
    sub_assign_scalar,
};

#[cfg(feature = "alloc")]
pub use coefficient::{CoeffMatrix, CoeffRef, CoeffVec, CoeffVecRef};
#[cfg(feature = "alloc")]
pub use gather::{mul_add_gather_with, mul_into_gather_with};
#[cfg(feature = "alloc")]
pub use map::{
    apply_into_with, contains_into_with, embed_into_with, frobenius_assign_with,
    frobenius_into_with, norm_into_with, restrict_into_with, trace_into_with, unapply_into_with,
};
#[cfg(feature = "alloc")]
pub use matrix::{
    mul_add_matrix_at_with, mul_add_matrix_with, mul_into_matrix_at_with, mul_into_matrix_with,
};
#[cfg(feature = "alloc")]
pub use packing::pack_to_vec;
#[cfg(feature = "alloc")]
pub use scatter::mul_add_scatter_with;

mod private {
    pub trait Sealed {}
}

/// A borrowed or owned backend-prepared coefficient.
///
/// This trait is sealed: values come from [`Coeff`] or, with `alloc`, a
/// `CoeffVec` or `CoeffMatrix`.
/// It exists so the `_with` operations can consume either without cloning the
/// prepared tables. It exposes the coefficient's field value only — the
/// backend representation stays crate-private, so no consumer can extract or
/// depend on a preparation format.
// The private supertrait seals access to backend preparation.
#[allow(private_bounds)]
pub trait PreparedCoefficient<F: FieldKernels>: private::Sealed + PreparedRepr<F> {
    /// The field element this prepared representation multiplies by.
    fn value(&self) -> Elem<F>;
}

/// Crate-private access to the backend representation behind
/// [`PreparedCoefficient`].
///
/// A supertrait with an unnameable proof argument: external generic code
/// bound on `PreparedCoefficient` can see the method through bound
/// elaboration but cannot construct the [`RawDispatch`] it takes, so the
/// representation is unreachable outside the crate.
pub(crate) trait PreparedRepr<F: FieldKernels>: private::Sealed {
    /// Borrow the backend-private representation.
    fn repr(&self, _proof: RawDispatch) -> &<F as KernelDispatch>::Prepared;
}

#[inline]
fn check_width<F: FieldBuffer>(name: &str, len: usize) {
    assert!(
        len.is_multiple_of(F::BYTES),
        "{name}: buffer of {len} bytes is not a whole number of {} elements",
        F::NAME
    );
}

#[inline]
fn check_pair<F: FieldBuffer>(
    name: &str,
    left_name: &str,
    left: usize,
    right_name: &str,
    right: usize,
) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
    check_width::<F>(name, left);
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::binary::tower::Rijndael16Field;
    use crate::field::poly::AES;
    use crate::field::{Elem as E8, FiniteField, Polynomial};

    /// `row_len == 0` with otherwise valid geometry is a no-op everywhere —
    /// including the scalar backend, whose zero-length row chunking is an
    /// empty loop rather than a panic.
    #[test]
    // Term geometry nests the unified element spelling; the slices stay slices.
    #[allow(clippy::type_complexity)]
    fn zero_row_length_is_a_no_op() {
        let coeffs = [E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x07); 3];
        let mut rows = [0xA5u8; 8];
        mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &coeffs, &[]);
        assert_eq!(rows, [0xA5; 8]);

        let coeffs_vec = CoeffVec::<FiniteField<2, 8, Polynomial<AES>>>::new(&coeffs);
        mul_add_scatter_with::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut rows,
            0,
            coeffs_vec.as_ref(),
            &[],
        );
        assert_eq!(rows, [0xA5; 8]);

        let terms: &[(&[E8<FiniteField<2, 8, Polynomial<AES>>>], &[u8])] = &[(&coeffs, &[])];
        mul_add_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, 3, terms);
        assert_eq!(rows, [0xA5; 8]);
        mul_into_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, 3, terms);
        assert_eq!(rows, [0xA5; 8]);
        mul_add_matrix_at::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut rows,
            0,
            &[1, 4],
            &[(&coeffs[..2], &[])],
        );
        assert_eq!(rows, [0xA5; 8]);

        let matrix =
            CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(1, 3, &coeffs);
        mul_add_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &matrix, &[&[]]);
        assert_eq!(rows, [0xA5; 8]);
        mul_into_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &matrix, &[&[]]);
        assert_eq!(rows, [0xA5; 8]);

        // GF(2^16): the field whose odd-length buffers make the element
        // check matter.
        let wide = [Elem::<Rijndael16Field>::from_raw(0x0103); 2];
        let mut rows16 = [0x5Au8; 4];
        mul_add_scatter::<Rijndael16Field>(&mut rows16, 0, &wide, &[]);
        assert_eq!(rows16, [0x5A; 4]);
    }
}
