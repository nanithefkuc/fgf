//! Portable scalar kernels.
//!
//! These are field-generic and elementwise: correct everywhere, dependent on
//! nothing but [`Field`]. They provide the fallback on targets without a
//! supported vector unit and process sub-lane tails from vector loops. Backend
//! tests use them as the scalar reference except for the Fan–Paar family, whose
//! optimized element arithmetic is checked against the independent
//! [`crate::field::binary::tower`] recurrence.
//!
//! Hot scalar paths that beat the generic form live in the per-field dispatch
//! modules rather than here. The scaled-row loops are the shared generator in
//! [`crate::kernel`]: this convention's unit shortcut is XOR, so a unit
//! coefficient folds sources with the eight-byte [`xor`] and a zero
//! coefficient leaves the destination untouched.

use crate::field::FieldElem;

portable_scaled_rows!(
    "`dst ^= coeff * src`, elementwise.",
    "`dst *= coeff`, elementwise, in place.",
    xor,
    mul_elementwise_assign,
);

/// `dst ^= src`, eight bytes at a time.
///
/// Inlining lets short-buffer callers avoid an out-of-line kernel boundary.
///
/// # Panics
/// Panics if the slices differ in length.
#[inline]
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "xor: length mismatch");

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mixed =
            u64::from_ne_bytes(d.try_into().unwrap()) ^ u64::from_ne_bytes(s.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= s;
    }
}

/// [`super::bytes::xor_broadcast`]'s portable body: XOR the cyclic `value_bytes`
/// pattern over the whole buffer, regardless of element boundaries.
///
/// The vector broadcast kernels peel whole lanes and hand every remaining
/// byte here, and the scalar arm of the dispatch runs it over the whole
/// destination; lane boundaries sit on element boundaries, so the phase of
/// `value_bytes` matches the destination's element phase throughout.
///
/// # Panics
/// Panics if `value_bytes` is empty.
#[inline]
pub(crate) fn xor_broadcast_bytes(dst: &mut [u8], value_bytes: &[u8]) {
    assert!(!value_bytes.is_empty(), "xor_broadcast_bytes: empty value");
    for (d, v) in dst.iter_mut().zip(value_bytes.iter().cycle()) {
        *d ^= v;
    }
}
