//! Field-independent 64-byte AVX-512 byte kernels.
//!
//! The `V4x` forms of the byte operations in [`super::bytes`]: whole-buffer
//! XOR, the additive operation of every binary field, and the population
//! count behind bit-packed GF(2) weights. Every entry is a safe
//! [`archmage`] capability-token function over reference-based intrinsics
//! and takes the exact token its instructions need. Complete 64-byte lanes
//! run here; the remainder descends through the AVX2 entry or a scalar
//! count.

use super::bytes::xor_avx2;

/// Shortest XOR buffer the 64-byte kernel takes; shorter buffers stay on
/// [`xor_avx2`], which covers them in one out-of-line call.
///
/// Set by the length sweep in `BENCHMARKS.md` ("AVX-512 byte, popcount, and
/// prime-field kernels").
pub(crate) const XOR512_MIN: usize = 256;

/// Shortest XOR destination that peels its head to a 64-byte boundary.
///
/// The peel is one out-of-line AVX2 call, which short buffers do not repay.
/// Set by the same sweep as [`XOR512_MIN`].
pub(crate) const XOR_PEEL_MIN: usize = 512;

/// `dst ^= src` using 64-byte AVX-512 lanes.
///
/// From `XOR_PEEL_MIN` bytes, head bytes up to the destination's first
/// 64-byte boundary run through [`xor_avx2`], so every 64-byte access on the
/// destination stays within one cache line; the source shares that alignment
/// in every same-offset shape.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn xor512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "xor512: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let head = if dst.len() < XOR_PEEL_MIN {
        0
    } else {
        dst.as_ptr().align_offset(64).min(dst.len())
    };
    let (dst_head, dst) = dst.split_at_mut(head);
    let (src_head, src) = src.split_at(head);
    if head > 0 {
        xor_avx2(token.v3(), dst_head, src_head);
    }

    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<256>();
    let (src_tiles, src_rest) = src.as_chunks::<256>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<64>();
        let (src_lanes, _) = src_tile.as_chunks::<64>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = _mm512_loadu_si512(&*dst_lane);
            let s = _mm512_loadu_si512(src_lane);
            _mm512_storeu_si512(dst_lane, _mm512_xor_si512(d, s));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = _mm512_loadu_si512(&*dst_lane);
        let s = _mm512_loadu_si512(src_lane);
        _mm512_storeu_si512(dst_lane, _mm512_xor_si512(d, s));
    }
    if !dst_tail.is_empty() {
        xor_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// Number of set bits in `buf`, using 64-byte `VPOPCNTQ` lanes.
///
/// Four independent accumulators cover the instruction latency; each 64-bit
/// lane sums per-word counts, which no buffer can overflow.
#[allow(clippy::cast_sign_loss, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn count_ones512(_token: archmage::X64V4xToken, buf: &[u8]) -> u64 {
    let (tiles, rest) = buf.as_chunks::<256>();
    let mut acc = [_mm512_setzero_si512(); 4];
    for tile in tiles {
        let (lanes, _) = tile.as_chunks::<64>();
        for (slot, lane) in acc.iter_mut().zip(lanes) {
            *slot = _mm512_add_epi64(*slot, _mm512_popcnt_epi64(_mm512_loadu_si512(lane)));
        }
    }
    let (lanes, tail) = rest.as_chunks::<64>();
    for lane in lanes {
        acc[0] = _mm512_add_epi64(acc[0], _mm512_popcnt_epi64(_mm512_loadu_si512(lane)));
    }
    let sum = _mm512_add_epi64(
        _mm512_add_epi64(acc[0], acc[1]),
        _mm512_add_epi64(acc[2], acc[3]),
    );
    // Every lane holds a count of at most `8 * buf.len()` bits: non-negative.
    let vector = _mm512_reduce_add_epi64(sum) as u64;
    vector
        + tail
            .iter()
            .map(|byte| u64::from(byte.count_ones()))
            .sum::<u64>()
}
