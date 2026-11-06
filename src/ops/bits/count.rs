//! Population count and inner product over the live bits.

use super::geometry::{check_holds, check_pair};
use crate::field::binary::Polynomial;
use crate::field::{Elem, FiniteField};
use crate::kernel::gf1;

/// Hamming weight of the live bits `[0, bits)`.
///
/// Padding bits never count, even when the buffer holds more bytes than
/// [`bytes_for`](super::bytes_for)`(bits)`. The count is a `usize` and is exact for
/// every buffer length.
///
/// # Panics
/// Panics if `buf` is shorter than [`bytes_for`](super::bytes_for)`(bits)`.
#[must_use]
pub fn weight(buf: &[u8], bits: usize) -> usize {
    const SEGMENT_BITS: usize = (u32::MAX as usize / 8) * 8;
    check_holds("bits::weight", "buf", buf.len(), bits);
    // The shared kernel folds into `u32` accumulators, exact for any
    // segment whose set bits fit a `u32`. Split on byte-aligned segments
    // (the kernel is byte-indexed from the slice start) and sum in `usize`.
    let mut total = 0usize;
    let mut done = 0;
    while done < bits {
        let segment = (bits - done).min(SEGMENT_BITS);
        // Each segment carries at most SEGMENT_BITS ≤ u32::MAX set bits, so
        // the `u32` count widens to `usize` without loss on every target
        // this crate builds for (`u32 as usize` is total for the value).
        total += gf1::weight(&buf[done / 8..], segment) as usize;
        done += segment;
    }
    total
}

/// The GF(2) inner product of two packed vectors: the parity of
/// `popcount(a & b)` over the live bits.
///
/// Returned as an [`Elem`] over GF(2) as
/// [`FiniteField<2, 1, Polynomial<3>>`](crate::FiniteField); use
/// [`is_one`](crate::FieldElem::is_one) to test it. Padding bits
/// contribute nothing.
///
/// # Panics
/// Panics if the slices differ in length or are shorter than
/// [`bytes_for`](super::bytes_for)`(bits)`.
#[must_use]
pub fn dot_product(a: &[u8], b: &[u8], bits: usize) -> Elem<FiniteField<2, 1, Polynomial<3>>> {
    check_pair("bits::dot_product", "a", a.len(), "b", b.len());
    check_holds("bits::dot_product", "a", a.len(), bits);
    #[allow(clippy::cast_possible_truncation)] // parity is exactly 0 or 1
    Elem::<FiniteField<2, 1, Polynomial<3>>>::from_raw(gf1::parity(a, b, bits) as u8)
}
