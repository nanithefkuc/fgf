//! Field-independent byte kernels.
//!
//! XOR is the additive operation of every binary field in the crate, so it
//! reads no field layout and needs no coefficient. Every entry is a safe
//! [`archmage`] capability-token function over reference-based intrinsics;
//! the row-interleaved candidates walk their groups through `#[rite]` tier
//! helpers.

use super::HALF_LANE_PEEL_MIN;
use crate::kernel::scalar;

/// `dst ^= src` using 32-byte AVX2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // A destination 16 bytes into a 32-byte lane splits every store across
    // cache lines; one 16-byte head aligns the whole remaining body. The
    // floor is [`HALF_LANE_PEEL_MIN`] ("v3 half-lane peels and
    // overwrite-gather dispatch", `BENCHMARKS.md`). The aligned path passes
    // its slices through untouched so the peel check is its only cost.
    if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
        let (dst16, _) = dst[..16].as_chunks_mut::<16>();
        let (src16, _) = src[..16].as_chunks::<16>();
        let d = _mm_loadu_si128(&dst16[0]);
        let s = _mm_loadu_si128(&src16[0]);
        _mm_storeu_si128(&mut dst16[0], _mm_xor_si128(d, s));
        xor_avx2_body(&mut dst[16..], &src[16..]);
    } else {
        xor_avx2_body(dst, src);
    }
}

/// The `xor_avx2` lanes over the slices the entry hands them: 128-byte
/// tiles, single 32-byte lanes, a 16-byte step, then the scalar tail.
#[archmage::rite(v3, import_intrinsics)]
fn xor_avx2_body(dst: &mut [u8], src: &[u8]) {
    let (dst_tiles, dst_tail) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_tail) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = _mm256_loadu_si256(&*dst_lane);
            let s = _mm256_loadu_si256(src_lane);
            _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, s));
        }
    }

    let (dst_lanes, dst_tail) = dst_tail.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src_tail.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = _mm256_loadu_si256(dst_lane);
        let s = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, s));
    }

    let (dst_lanes, dst_tail) = dst_tail.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_tail.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = _mm_loadu_si128(dst_lane);
        let s = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, s));
    }
    scalar::xor(dst_tail, src_tail);
}

/// `dst ^= sum(srcs[i])` holding the destination in registers across the
/// whole source list.
///
/// The gather shape of back-substitution: many sources fold into one short
/// row, so the destination is read and written once per 32-byte lane, not
/// once per source. Destinations wider than eight vectors (256 bytes) or
/// with a sub-vector tail fall back to a per-source [`xor_avx2`], which
/// owns the tail handling; the consumer this kernel exists for writes
/// 64-byte rows.
///
/// # Panics
/// Panics if any offset plus `dst.len()` exceeds the region length.
#[archmage::arcane(import_intrinsics)]
pub fn xor_gather_avx2(
    token: archmage::X64V3Token,
    region: &[u8],
    dst: &mut [u8],
    offsets: &[u32],
) {
    let whole = dst.len() & !31;
    let tail = dst.len() - whole;
    if whole == 0 || whole > 8 * 32 || tail != 0 {
        for &start in offsets {
            let start = start as usize;
            assert!(
                start + dst.len() <= region.len(),
                "xor_gather_avx2: source out of region bounds",
            );
            xor_avx2(token, dst, &region[start..start + dst.len()]);
        }
        return;
    }
    for (index, &start) in offsets.iter().enumerate() {
        let start = start as usize;
        assert!(
            start + dst.len() <= region.len(),
            "xor_gather_avx2: offset {index} ({start}) + {} exceeds region of {} bytes",
            dst.len(),
            region.len(),
        );
    }
    let (dst_lanes, _) = dst.as_chunks_mut::<32>();
    for (lane, dst_lane) in dst_lanes.iter_mut().enumerate() {
        let mut acc = _mm256_loadu_si256(&*dst_lane);
        for &start in offsets {
            let base = start as usize + lane * 32;
            let src_lane = region[base..base + 32].first_chunk().unwrap();
            acc = _mm256_xor_si256(acc, _mm256_loadu_si256(src_lane));
        }
        _mm256_storeu_si256(dst_lane, acc);
    }
}

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

/// `dst[i] ^= value_bytes[i % value_bytes.len()]` for every byte — one
/// element broadcast across every lane, the binary fields'
/// `add_assign_scalar`.
///
/// `value_bytes` holds one element's little-endian encoding, 1 to 16 bytes.
/// One 32-byte pattern cycled from it is built on the stack, loaded once,
/// and exclusive-ORed into the destination through the lane walk of
/// [`xor_avx2`].
/// Every lane boundary is a multiple of the element width, so the pattern's
/// phase matches the destination's element phase throughout. Value widths
/// that do not divide the lane fall back to the portable byte-cyclic form.
///
/// # Panics
/// Panics if `value_bytes` is empty or longer than 16 bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_broadcast_avx2(_token: archmage::X64V3Token, dst: &mut [u8], value_bytes: &[u8]) {
    assert!(
        !value_bytes.is_empty() && value_bytes.len() <= 16,
        "xor_broadcast_avx2: value is {} bytes, expected 1..=16",
        value_bytes.len(),
    );
    // A lane-wide pattern repeats only if the value divides it; other widths
    // fall to the portable byte-cyclic form rather than misalign the phase.
    if 32 % value_bytes.len() != 0 {
        scalar::xor_broadcast_bytes(dst, value_bytes);
        return;
    }
    let mut pattern = [0u8; 32];
    for (byte, slot) in pattern.iter_mut().zip(value_bytes.iter().cycle()) {
        *byte = *slot;
    }
    let broadcast = _mm256_loadu_si256(&pattern);
    // The low half is the same cyclic pattern at phase zero, which is what
    // every 16-byte lane below needs: lanes start on 32-byte boundaries.
    let broadcast16 = _mm256_castsi256_si128(broadcast);

    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for lane in dst_lanes {
        let d = _mm256_loadu_si256(&*lane);
        _mm256_storeu_si256(lane, _mm256_xor_si256(d, broadcast));
    }
    let (dst_lanes, dst_tail) = dst_tail.as_chunks_mut::<16>();
    for lane in dst_lanes {
        let d = _mm_loadu_si128(&*lane);
        _mm_storeu_si128(lane, _mm_xor_si128(d, broadcast16));
    }
    scalar::xor_broadcast_bytes(dst_tail, value_bytes);
}

/// [`xor_broadcast_avx2`] over 16-byte SSE2 lanes, for the `V2` tier.
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
