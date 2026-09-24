//! Field-independent byte kernels.

use crate::kernel::scalar;

/// `dst ^= src` over 16-byte NEON lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_neon(_token: archmage::NeonToken, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        vst1q_u8(d, veorq_u8(vld1q_u8(&*d), vld1q_u8(s)));
    }
    scalar::xor(dst_tail, src_tail);
}

/// `dst[i] ^= value_bytes[i % value_bytes.len()]` for every byte — one
/// element broadcast across every lane, the binary fields'
/// `add_assign_scalar`.
///
/// `value_bytes` holds one element's little-endian encoding, 1 to 16 bytes.
/// One 16-byte pattern cycled from it is built on the stack, loaded once,
/// and XORed into the destination through the lane-tail walk of
/// [`xor_neon`]. Every lane boundary is a multiple of the element width, so
/// the pattern's phase matches the destination's element phase throughout.
///
/// # Panics
/// Panics if `value_bytes` is empty or longer than 16 bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_broadcast_neon(_token: archmage::NeonToken, dst: &mut [u8], value_bytes: &[u8]) {
    assert!(
        !value_bytes.is_empty() && value_bytes.len() <= 16,
        "xor_broadcast_neon: value is {} bytes, expected 1..=16",
        value_bytes.len(),
    );
    // A lane-wide pattern repeats only if the value divides it; other widths
    // fall to the portable byte-cyclic form rather than misalign the phase.
    if 16 % value_bytes.len() != 0 {
        scalar::xor_broadcast_bytes(dst, value_bytes);
        return;
    }
    let mut pattern = [0u8; 16];
    for (byte, slot) in pattern.iter_mut().zip(value_bytes.iter().cycle()) {
        *byte = *slot;
    }
    let broadcast = vld1q_u8(&pattern);

    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for lane in dst_lanes {
        vst1q_u8(lane, veorq_u8(vld1q_u8(&*lane), broadcast));
    }
    scalar::xor_broadcast_bytes(dst_tail, value_bytes);
}
