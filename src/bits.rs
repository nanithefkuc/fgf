//! Bit-packed GF(2) buffers: one element per bit.
//!
//! GF(2) is the one field whose natural element width is sub-byte, so it does
//! not fit the byte-per-element [`crate::ops`] surface. Instead, elements are
//! packed eight per byte and the operations here are whole-buffer or masked
//! bit ranges over plain `&[u8]`:
//!
//! - element `i` lives in **bit `i % 8` (LSB-first) of byte `i / 8`**;
//! - a word is the little-endian `u64` assembly of eight bytes, so element
//!   `i` is bit `i % 64` of word `i / 64`;
//! - bits past the logical length are **padding and are always zero** — a
//!   maintained invariant of every output, not a don't-care.
//!
//! Because a byte holds eight elements and the last byte may be partial, the
//! element count is not recoverable from the byte length: operations whose
//! meaning depends on the exact bit count take an explicit `bits`, and every
//! range operation checks `from <= to <= bits`. Buffers may be longer than
//! [`bytes_for`]`(bits)` — a consumer with word-aligned storage passes its
//! whole row and the surplus bits are padding like any other.
//!
//! Multiplying by a coefficient is masked out of the API on purpose: the
//! coefficient of GF(2) is a bit — multiply by zero is "skip", by one is
//! "XOR" — so there is no prepared-coefficient form and no [`crate::ops`]
//! counterpart. What *is* prepared is geometry: a [`RangeXor`] derives a
//! bit range's window and masks once for [`xor_range_with`] to apply to
//! many buffer pairs. The scalar oracle for this surface is
//! [`crate::gf2::Elem`](crate::field::gf2::Elem).
//!
//! ```
//! use fgf::bits;
//! use fgf::field::Elem;
//! // Seven elements: 1,0,0,1,1,1,1 -> LSB-first byte 0b0111_1001.
//! let mut a = [0u8; 1];
//! bits::set_range(&mut a, 7, 0, 7);
//! bits::clear_range(&mut a, 7, 1, 3);
//! assert_eq!(a[0], 0b0111_1001);
//! assert_eq!(bits::weight(&a, 7), 5);
//!
//! // The GF(2) inner product of a vector with itself is its weight mod 2.
//! let parity = bits::parity_dot(&a, &a, 7);
//! assert!(parity.is_one());
//!
//! // Whole-buffer addition is XOR of equal-length buffers.
//! let b = [0xff];
//! bits::xor(&mut a, &b);
//! assert_eq!(bits::weight(&a, 7), 2);
//! ```

use crate::field::gf2;
use crate::kernel;

/// Number of bytes needed to hold `bits` packed elements.
///
/// ```text
/// bits  0  1  8  9 64 65
/// bytes 0  1  1  2  8  9
/// ```
#[must_use]
pub const fn bytes_for(bits: usize) -> usize {
    bits.div_ceil(8)
}

#[inline]
fn check_pair(name: &str, left_name: &str, left: usize, right_name: &str, right: usize) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
}

#[inline]
fn check_range(name: &str, bits: usize, from: usize, to: usize) {
    assert!(from <= to, "{name}: range start {from} exceeds end {to}");
    assert!(
        to <= bits,
        "{name}: range end {to} exceeds the {bits} live bits"
    );
}

#[inline]
fn check_holds(name: &str, operand: &str, len: usize, bits: usize) {
    assert!(
        bytes_for(bits) <= len,
        "{name}: {bits} bits need at least {} bytes but {operand} holds {len}",
        bytes_for(bits)
    );
}

/// `dst ^= src`: GF(2) addition over whole buffers.
///
/// XOR of zero padding stays zero, so no bit count is needed. Buffers must
/// not alias.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor(dst: &mut [u8], src: &[u8]) {
    check_pair("bits::xor", "dst", dst.len(), "src", src.len());
    kernel::gf2::xor(dst, src);
}

/// `dst ^= src` over the bit range `[from, to)`.
///
/// Bits outside the range — including padding — are untouched. `dst` and
/// `src` must not alias.
///
/// The checked one-shot form of [`xor_range_with`]: an operation applied
/// once pays one construction; an operation applied to many buffer pairs
/// over the same range should prepare a [`RangeXor`] instead.
///
/// # Panics
/// Panics if the slices differ in length, if `from > to`, if `to > bits`,
/// or if the buffers are shorter than [`bytes_for`]`(bits)`.
#[inline]
pub fn xor_range(dst: &mut [u8], src: &[u8], bits: usize, from: usize, to: usize) {
    check_pair("bits::xor_range", "dst", dst.len(), "src", src.len());
    check_range("bits::xor_range", bits, from, to);
    check_holds("bits::xor_range", "dst", dst.len(), bits);
    xor_range_with(dst, src, &RangeXor::new(bits, from, to));
}

/// A prepared bit range for repeated `dst ^= src`: the window, end masks,
/// and XOR backend of `[from, to)` derived once, applied to many buffer pairs.
///
/// The prepared form of [`xor_range`], in the same split [`crate::ops`]
/// uses for prepared coefficients: an elimination that XORs the same
/// column range into many rows pays the range checks, mask arithmetic, and
/// backend resolution once per range instead of once per row. An empty range
/// is valid and [`xor_range_with`] applies it as a no-op.
///
/// The plan is independent of the buffers it is applied to: buffers may be
/// longer than the planned window (the surplus is padding like any other),
/// and bits outside the window are untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeXor {
    window: kernel::gf2::Window,
    backend: kernel::Backend,
}

impl RangeXor {
    /// Prepares the bit range `[from, to)` of a vector `bits` long.
    ///
    /// # Panics
    ///
    /// Panics if `from > to` or `to > bits`.
    #[must_use]
    pub fn new(bits: usize, from: usize, to: usize) -> Self {
        check_range("RangeXor::new", bits, from, to);
        Self {
            window: kernel::gf2::window(from, to),
            backend: kernel::backend(),
        }
    }
}

/// `dst ^= src` over a prepared [`RangeXor`]: the per-row apply of the
/// split form of [`xor_range`].
///
/// Bits outside the planned window — including padding — are untouched.
/// `dst` and `src` must not alias. The buffers may differ in length; each
/// must cover the planned window.
///
/// # Panics
///
/// Panics if either buffer is shorter than the planned window.
#[inline]
pub fn xor_range_with(dst: &mut [u8], src: &[u8], range: &RangeXor) {
    let end = range.window.end;
    if end == 0 {
        return;
    }
    assert!(
        end <= dst.len() && end <= src.len(),
        "bits::xor_range_with: the planned range needs {end} bytes but dst holds {} and src holds {}",
        dst.len(),
        src.len()
    );
    kernel::gf2::xor_window(dst, src, &range.window, range.backend);
}

/// Legacy prepared-range apply retained for internal A/B benchmarks.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub fn benchmark_xor_range_with_peeled(dst: &mut [u8], src: &[u8], range: &RangeXor) {
    let end = range.window.end;
    if end == 0 {
        return;
    }
    assert!(
        end <= dst.len() && end <= src.len(),
        "bits::benchmark_xor_range_with_peeled: the planned range needs {end} bytes but dst holds {} and src holds {}",
        dst.len(),
        src.len()
    );
    kernel::gf2::xor_window_peeled(dst, src, &range.window);
}

/// `dst = a & b`: elementwise GF(2) multiplication.
///
/// AND of zero padding stays zero, so no bit count is needed. The output
/// buffers may alias neither input.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn and_into(dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_pair("bits::and_into", "dst", dst.len(), "a", a.len());
    check_pair("bits::and_into", "a", a.len(), "b", b.len());
    kernel::gf2::and_into(dst, a, b);
}

/// `dst &= src`: in-place elementwise GF(2) multiplication.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn and_assign(dst: &mut [u8], src: &[u8]) {
    check_pair("bits::and_assign", "dst", dst.len(), "src", src.len());
    kernel::gf2::and_assign(dst, src);
}

/// `dst &= !mask`: clears every bit of `dst` where `mask` has a 1.
///
/// The mask's padding is irrelevant; zero bits of `dst` stay zero, so the
/// padding invariant holds regardless.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn andnot_assign(dst: &mut [u8], mask: &[u8]) {
    check_pair("bits::andnot_assign", "dst", dst.len(), "mask", mask.len());
    kernel::gf2::andnot_assign(dst, mask);
}

/// Zeros the bit range `[from, to)` of `dst`.
///
/// # Panics
/// Panics if `from > to`, if `to > bits`, or if `dst` is shorter than
/// [`bytes_for`]`(bits)`.
#[inline]
pub fn clear_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    check_range("bits::clear_range", bits, from, to);
    check_holds("bits::clear_range", "dst", dst.len(), bits);
    kernel::gf2::clear_range(dst, from, to);
}

/// Sets the bit range `[from, to)` of `dst` to one.
///
/// # Panics
/// Panics if `from > to`, if `to > bits`, or if `dst` is shorter than
/// [`bytes_for`]`(bits)`.
pub fn set_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    check_range("bits::set_range", bits, from, to);
    check_holds("bits::set_range", "dst", dst.len(), bits);
    kernel::gf2::set_range(dst, from, to);
}

/// Hamming weight of the live bits `[0, bits)`.
///
/// Padding bits never count, even when the buffer holds more bytes than
/// [`bytes_for`]`(bits)`.
///
/// # Panics
/// Panics if `buf` is shorter than [`bytes_for`]`(bits)`.
#[must_use]
pub fn weight(buf: &[u8], bits: usize) -> u32 {
    check_holds("bits::weight", "buf", buf.len(), bits);
    kernel::gf2::weight(buf, bits)
}

/// The GF(2) inner product of two packed vectors: the parity of
/// `popcount(a & b)` over the live bits.
///
/// Returned as a [`gf2::Elem`]; use
/// [`is_one`](crate::field::Elem::is_one) to test it. Padding bits
/// contribute nothing.
///
/// # Panics
/// Panics if the slices differ in length or are shorter than
/// [`bytes_for`]`(bits)`.
#[must_use]
pub fn parity_dot(a: &[u8], b: &[u8], bits: usize) -> gf2::Elem {
    check_pair("bits::parity_dot", "a", a.len(), "b", b.len());
    check_holds("bits::parity_dot", "a", a.len(), bits);
    #[allow(clippy::cast_possible_truncation)] // parity is exactly 0 or 1
    gf2::Elem::from_raw(kernel::gf2::parity(a, b, bits) as u8)
}

/// `dst ^= srcs[i]` for every set bit `i` of `selector`: the combination
/// step of subset-XOR over a dense row set.
///
/// An empty selector is a no-op. `dst` must not alias any source.
///
/// # Panics
/// Panics if any source differs in length from `dst`, if `selector` names a
/// source at or beyond `srcs.len()`, or if `dst` is shorter than
/// [`bytes_for`]`(bits)`.
pub fn xor_gather(dst: &mut [u8], srcs: &[&[u8]], selector: u64, bits: usize) {
    check_holds("bits::xor_gather", "dst", dst.len(), bits);
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            dst.len(),
            "bits::xor_gather: source {index} is {} bytes but dst is {} bytes",
            src.len(),
            dst.len()
        );
    }
    if srcs.len() < 64 {
        let excess = selector >> srcs.len();
        let count = srcs.len();
        assert!(
            excess == 0,
            "bits::xor_gather: selector names source {} but only {count} sources exist",
            count + excess.trailing_zeros() as usize
        );
    }
    kernel::gf2::xor_gather(dst, srcs, selector);
}
