//! Whole-buffer Boolean algebra: XOR addition and AND multiplication.

use super::geometry::check_pair;
use crate::kernel::gf1;

/// `dst ^= src`: GF(2) addition over whole buffers.
///
/// XOR of zero padding stays zero, so no bit count is needed. Buffers must
/// not alias.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_assign(dst: &mut [u8], src: &[u8]) {
    check_pair("bits::xor_assign", "dst", dst.len(), "src", src.len());
    gf1::xor(dst, src);
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
    gf1::and_into(dst, a, b);
}

/// `dst &= src`: in-place elementwise GF(2) multiplication.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn and_assign(dst: &mut [u8], src: &[u8]) {
    check_pair("bits::and_assign", "dst", dst.len(), "src", src.len());
    gf1::and_assign(dst, src);
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
    gf1::andnot_assign(dst, mask);
}
