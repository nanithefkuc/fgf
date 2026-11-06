//! Bit-packed GF(2) buffers: one element per bit.
//!
//! GF(2) is the one field whose natural element width is sub-byte, so it does
//! not fit the byte-per-element lane surface of [`crate::ops`]. Instead,
//! elements are packed eight per byte and the operations here are
//! whole-buffer or masked bit ranges over plain `&[u8]`:
//!
//! - element `i` lives in **bit `i % 8` (LSB-first) of byte `i / 8`**;
//! - a word is the little-endian `u64` assembly of eight bytes, so element
//!   `i` is bit `i % 64` of word `i / 64`;
//! - bits past the logical length are **padding**. The invariant is that
//!   they are zero; it is *caller-maintained*. Range operations leave bits
//!   outside their range untouched. Whole-buffer operations also process
//!   padding and preserve zero padding when their inputs satisfy the
//!   invariant. No operation promises to sanitize nonzero padding.
//!
//! Because a byte holds eight elements and the last byte may be partial, the
//! element count is not recoverable from the byte length: operations whose
//! meaning depends on the exact bit count take an explicit `bits`, and every
//! range operation checks `from <= to <= bits`. Buffers may be longer than
//! [`bytes_for`]`(bits)` — a consumer with word-aligned storage passes its
//! whole row and the surplus bits are padding like any other.
//!
//! Operations split by kind: whole-buffer Boolean algebra, masked bit
//! ranges with their prepared window, subset combination, and counting,
//! over the shared byte geometry.
//!
//! Multiplying by a coefficient is masked out of the API on purpose: the
//! coefficient of GF(2) is a bit — multiply by zero is "skip", by one is
//! "XOR" — so there is no prepared-coefficient form. What *is* prepared
//! here is geometry: a [`XorRange`] derives a
//! bit range's window and masks once for [`xor_range_with`] to apply to
//! many buffer pairs. The scalar oracle for this surface is
//! [`Elem`](crate::Elem) over GF(2) as
//! [`FiniteField<2, 1, Polynomial<3>>`](crate::FiniteField).
//!
//! ```
//! use fgf::ops::bits;
//! use fgf::FieldElem;
//! // Seven elements: 1,0,0,1,1,1,1 -> LSB-first byte 0b0111_1001.
//! let mut a = [0u8; 1];
//! bits::set_range(&mut a, 7, 0, 7);
//! bits::clear_range(&mut a, 7, 1, 3);
//! assert_eq!(a[0], 0b0111_1001);
//! assert_eq!(bits::weight(&a, 7), 5);
//!
//! // The GF(2) inner product of a vector with itself is its weight mod 2.
//! let parity = bits::dot_product(&a, &a, 7);
//! assert!(parity.is_one());
//!
//! // Whole-buffer addition is XOR of equal-length buffers.
//! let b = [0xff];
//! bits::xor_assign(&mut a, &b);
//! assert_eq!(bits::weight(&a, 7), 2);
//! ```

mod count;
mod gather;
mod geometry;
mod logic;
mod range;

pub use count::{dot_product, weight};
pub use gather::xor_gather;
pub use geometry::bytes_for;
pub use logic::{and_assign, and_into, andnot_assign, xor_assign};
pub use range::{XorRange, clear_range, set_range, xor_range, xor_range_with};
