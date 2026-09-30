//! SSE2 byte kernels over 16-byte lanes.

use crate::kernel::scalar;

/// `dst ^= src` using 16-byte SSE2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_sse2(_token: archmage::X64V1Token, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let (dst_tiles, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_tail) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = _mm_loadu_si128(&*dst_lane);
            let s = _mm_loadu_si128(src_lane);
            _mm_storeu_si128(dst_lane, _mm_xor_si128(d, s));
        }
    }

    let (dst_lanes, dst_tail) = dst_tail.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_tail.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = _mm_loadu_si128(&*dst_lane);
        let s = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, s));
    }
    scalar::xor(dst_tail, src_tail);
}

/// [`xor_broadcast_avx2`](super::xor_broadcast_avx2) over 16-byte SSE2 lanes, for the `V2` tier.
///
/// # Panics
/// Panics if `value_bytes` is empty or longer than 16 bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_broadcast_sse2(_token: archmage::X64V1Token, dst: &mut [u8], value_bytes: &[u8]) {
    assert!(
        !value_bytes.is_empty() && value_bytes.len() <= 16,
        "xor_broadcast_sse2: value is {} bytes, expected 1..=16",
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
    let broadcast = _mm_loadu_si128(&pattern);

    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for lane in dst_lanes {
        let d = _mm_loadu_si128(&*lane);
        _mm_storeu_si128(lane, _mm_xor_si128(d, broadcast));
    }
    scalar::xor_broadcast_bytes(dst_tail, value_bytes);
}
