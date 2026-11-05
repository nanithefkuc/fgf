//! Masked bit ranges and their prepared window.

use super::geometry::{check_holds, check_pair};
use crate::kernel::gf1;

/// Asserts a range stays inside the live bits and runs one direction.
#[inline]
fn check_range(name: &str, bits: usize, from: usize, to: usize) {
    assert!(from <= to, "{name}: range start {from} exceeds end {to}");
    assert!(
        to <= bits,
        "{name}: range end {to} exceeds the {bits} live bits"
    );
}

/// `dst ^= src` over the bit range `[from, to)`.
///
/// Bits outside the range — including padding — are untouched. `dst` and
/// `src` must not alias.
///
/// The checked one-shot form of [`xor_range_with`]: an operation applied
/// once pays one construction; an operation applied to many buffer pairs
/// over the same range should prepare a [`XorRange`] instead.
///
/// # Panics
/// Panics if the slices differ in length, if `from > to`, if `to > bits`,
/// or if the buffers are shorter than [`bytes_for`](super::bytes_for)`(bits)`.
#[inline]
pub fn xor_range(dst: &mut [u8], src: &[u8], bits: usize, from: usize, to: usize) {
    check_pair("bits::xor_range", "dst", dst.len(), "src", src.len());
    check_range("bits::xor_range", bits, from, to);
    check_holds("bits::xor_range", "dst", dst.len(), bits);
    xor_range_with(dst, src, &XorRange::new(bits, from, to));
}

/// A prepared bit range for repeated `dst ^= src`: the window and end
/// masks of `[from, to)` derived once, applied to many buffer pairs.
///
/// The prepared form of [`xor_range`], in the same split [`crate::ops`]
/// uses for prepared coefficients: an elimination that XORs the same
/// column range into many rows pays the range checks and mask arithmetic
/// once per range instead of once per row. An empty range is valid and
/// [`xor_range_with`] applies it as a no-op.
/// The plan is independent of the buffers it is applied to: buffers may be
/// longer than the planned window (the surplus is padding like any other),
/// and bits outside the window are untouched.
///
/// That is the flexibility the prepared form buys over the one-shot
/// [`xor_range`]: the one-shot validates `from <= to <= bits` against the
/// logical bit count of every call, while a prepared range is validated
/// once — against its own `[from, to)` at construction — and then applies
/// to any buffer pair covering its window, whatever each buffer's logical
/// length. A caller folding the same column range into many equal-length
/// rows derives the window and masks exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorRange {
    window: gf1::Window,
}

impl XorRange {
    /// Prepares the bit range `[from, to)` of a vector `bits` long.
    ///
    /// # Panics
    ///
    /// Panics if `from > to` or `to > bits`.
    #[must_use]
    pub fn new(bits: usize, from: usize, to: usize) -> Self {
        check_range("XorRange::new", bits, from, to);
        Self {
            window: gf1::window(from, to),
        }
    }
}

/// `dst ^= src` over a prepared [`XorRange`]: the per-row apply of the
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
pub fn xor_range_with(dst: &mut [u8], src: &[u8], range: &XorRange) {
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
    gf1::xor_window(dst, src, &range.window);
}

/// Zeros the bit range `[from, to)` of `dst`.
///
/// # Panics
/// Panics if `from > to`, if `to > bits`, or if `dst` is shorter than
/// [`bytes_for`](super::bytes_for)`(bits)`.
#[inline]
pub fn clear_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    check_range("bits::clear_range", bits, from, to);
    check_holds("bits::clear_range", "dst", dst.len(), bits);
    gf1::clear_range(dst, from, to);
}

/// Sets the bit range `[from, to)` of `dst` to one.
///
/// # Panics
/// Panics if `from > to`, if `to > bits`, or if `dst` is shorter than
/// [`bytes_for`](super::bytes_for)`(bits)`.
pub fn set_range(dst: &mut [u8], bits: usize, from: usize, to: usize) {
    check_range("bits::set_range", bits, from, to);
    check_holds("bits::set_range", "dst", dst.len(), bits);
    gf1::set_range(dst, from, to);
}
