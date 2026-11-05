//! The public vector operation surface.
//!
//! Every arithmetic entry is generic over [`FieldKernels`](crate::FieldKernels) and monomorphizes
//! to that row's dispatched SIMD kernel. Encoding rows are
//! [`FieldBuffer`](crate::FieldBuffer)-only instead: packing (`pack`, `unpack`, `pack_to_vec`)
//! works over the stable little-endian byte form, never a kernel operand.
//! Buffers are plain `&[u8]` holding packed elements in that representation.
//! Canonically encoded payloads need no repacking.
//!
//! Implementations split along named axes: the operation-shape modules
//! `single`, `scatter`, `gather`, `matrix`, `dot`, and `recip`; the
//! `coefficient`, `packing`, and `geometry` modules; and the public
//! [`bits`] module for bit-packed GF(2) operands.
//!
//! `mul_add` is the fused `dst += coeff * src` (an AXPY; XOR in
//! characteristic two). `_scatter` fans one source out to many rows,
//! `_gather` folds many sources into one row, and `_matrix` does
//! many-to-many with the destination held in registers across sources.
//! These three shapes cover systematic encoding, symbol reconstruction,
//! and erasure decoding respectively.
//!
//! # Examples
//!
//! Each `mul_*` call resolves its coefficient into the form the host backend
//! wants. For GF(2^8) that is an array index; for GF(2^16) on a shuffle
//! backend it is four nibble tables. Amortized over a large buffer that cost
//! vanishes, but across many short buffers with the same coefficient it does
//! not. [`Coeff`] hoists it:
//!
//! ```
//! use fgf::{Elem, FiniteField, binary::Rijndael16, extension::Extension, ops};
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
//! # Totality and canonicalization
//!
//! The packed operations over the prime fields
//! ([`Mersenne31`](crate::prime::Mersenne31), [`Goldilocks`](crate::prime::Goldilocks),
//! [`QuadMersenne31`](crate::extension::QuadMersenne31), and the generic
//! residue rows `FiniteField<P, 1, Residue>`)
//! accept every lane bit pattern a decode accepts and
//! produce canonical lanes on output — accumulation, in-place, and tails
//! included — with three byte-preserving exceptions: a zero coefficient
//! fills zero lanes in overwrite and leaves the destination untouched in
//! accumulation; a unit coefficient copies the source bytes unchanged in
//! overwrite, adds them in accumulation, and leaves the destination
//! untouched in place. Those paths preserve noncanonical lane bytes
//! instead of canonicalizing them. That is the invariant a codec maintains
//! over its buffers, and it is what these kernels preserve. Scalar element
//! arithmetic shares the same totality: the [`FieldElem`](crate::FieldElem) operations accept
//! any raw lane value and always reduce canonically.

mod coefficient;
mod dot;
mod gather;
mod geometry;
mod matrix;
mod packing;
mod recip;
mod scatter;
mod single;

pub mod bits;

pub use coefficient::{Coeff, PreparedCoefficient};
pub use dot::dot_product;
pub use gather::{add_gather_offsets, mul_add_gather, mul_into_gather};
pub use matrix::{mul_add_matrix, mul_add_matrix_at, mul_into_matrix, mul_into_matrix_at};
pub use packing::{pack, unpack};
pub use recip::{div_elementwise, div_elementwise_assign, inv_assign, inv_into};
pub use scatter::{mul_add_scatter, mul_into_scatter};
pub use single::{
    add_assign, add_assign_rows, add_assign_scalar, mul_add, mul_add_with, mul_assign,
    mul_assign_with, mul_elementwise, mul_elementwise_assign, mul_into, mul_into_with, sub_assign,
    sub_assign_rows, sub_assign_scalar,
};

#[cfg(feature = "alloc")]
pub use coefficient::{CoeffMatrix, CoeffRef, CoeffVec, CoeffVecRef};
#[cfg(feature = "alloc")]
pub use gather::{mul_add_gather_with, mul_into_gather_with};
#[cfg(feature = "alloc")]
pub use matrix::{
    mul_add_matrix_at_with, mul_add_matrix_with, mul_into_matrix_at_with, mul_into_matrix_with,
};
#[cfg(feature = "alloc")]
pub use packing::{pack_to_vec, unpack_to_vec};
#[cfg(feature = "alloc")]
pub use scatter::mul_add_scatter_with;

#[cfg(all(test, feature = "alloc"))]
mod tests;
