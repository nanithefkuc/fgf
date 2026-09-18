//! Field-independent byte kernels.
//!
//! XOR is the additive operation of every binary field in the crate, so it
//! reads no field layout and needs no coefficient. Every entry is a safe
//! [`archmage`] capability-token function over reference-based intrinsics;
//! the row-interleaved candidates walk their groups through `#[rite]` tier
//! helpers.

use crate::kernel::scalar;

/// `dst ^= src` using 32-byte AVX2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn xor_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
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
        let d = _mm256_loadu_si256(&*dst_lane);
        let s = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, s));
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

// ---------------------------------------------------------------------------
// Row-interleaved XOR
// ---------------------------------------------------------------------------

/// Bytes of one row covered by a fully-unrolled AVX2 tile iteration: four
/// 32-byte vectors per stream, the leopard `xor_mem4` unroll width.
#[allow(dead_code)]
const AVX2_ROW_TILE: usize = 4 * 32;

/// Bytes of one row covered by a fully-unrolled SSE2 tile iteration: four
/// 16-byte vectors per stream.
#[allow(dead_code)]
const SSE2_ROW_TILE: usize = 4 * 16;

/// `dst ^= src` over contiguous `row_len`-byte rows with four interleaved
/// streams.
///
/// Both buffers hold the same whole number of rows (validated here); every
/// group of four row pairs walks `row_len` in tiles, each stream issuing its
/// own load/xor/store chain per tile, and a trailing odd row reuses
/// [`xor_avx2`], which owns the sub-32-byte tail handling. The four
/// independent streams keep the core's line-fill buffers busy where one
/// sequential stream cannot — the shape of leopard's `xor_mem4`.
///
/// This is an *unwired candidate*: on the reference host it measured at
/// parity with the flat XOR from L1 to DRAM across the whole benchmark
/// matrix, so dispatch keeps routing `FieldKernels::add_assign_rows` to
/// [`xor_avx2`] through the portable default. Kept for future hosts where
/// single-stream throughput falls short of the memory ceiling; see
/// BENCHMARKS.md, "Row-interleaved XOR".
///
/// # Panics
/// Panics if the slices differ in length or their length is not a whole
/// number of rows.
#[archmage::arcane]
pub fn xor_rows_avx2(token: archmage::X64V3Token, dst: &mut [u8], src: &[u8], row_len: usize) {
    assert_eq!(dst.len(), src.len());
    assert!(dst.len().is_multiple_of(row_len));
    debug_assert_ne!(row_len, 0);
    let rows = dst.len() / row_len;
    let four_span = rows / 4 * 4 * row_len;
    let two_span = rows % 4 / 2 * 2 * row_len;
    xor_streams_avx2::<4>(&mut dst[..four_span], &src[..four_span], row_len);
    xor_streams_avx2::<2>(
        &mut dst[four_span..four_span + two_span],
        &src[four_span..four_span + two_span],
        row_len,
    );
    if rows % 2 == 1 {
        let odd = four_span + two_span;
        xor_avx2(token, &mut dst[odd..], &src[odd..]);
    }
}

/// `dst ^= src` over whole `W`-row groups of consecutive `row_len`-byte
/// rows, `W` streams interleaved, until both slices are consumed.
///
/// Each pass takes one group: fully-unrolled [`AVX2_ROW_TILE`] tiles across the
/// row — one tile covers [`AVX2_ROW_TILE`] bytes of *every* stream at the same
/// offset — then single vectors over the remaining tail, and scalar bytes at
/// the end. The inner loops run to the const bound `W`, so LLVM unrolls them
/// away and the body issues one independent load/xor/store chain per stream
/// per tile without index arithmetic.
#[archmage::rite(v3, import_intrinsics)]
fn xor_streams_avx2<const W: usize>(dst: &mut [u8], src: &[u8], row_len: usize) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert!(row_len != 0 && dst.len().is_multiple_of(W * row_len));
    let group_span = W * row_len;
    for (rows_dst, rows_src) in dst
        .chunks_exact_mut(group_span)
        .zip(src.chunks_exact(group_span))
    {
        let mut offset = 0;
        while offset + AVX2_ROW_TILE <= row_len {
            for r in 0..W {
                let base = r * row_len + offset;
                let (d_lanes, _) = rows_dst[base..base + AVX2_ROW_TILE].as_chunks_mut::<32>();
                let (s_lanes, _) = rows_src[base..base + AVX2_ROW_TILE].as_chunks::<32>();
                for (d_lane, s_lane) in d_lanes.iter_mut().zip(s_lanes) {
                    let d = _mm256_loadu_si256(&*d_lane);
                    let s = _mm256_loadu_si256(s_lane);
                    _mm256_storeu_si256(d_lane, _mm256_xor_si256(d, s));
                }
            }
            offset += AVX2_ROW_TILE;
        }
        while offset + 32 <= row_len {
            for r in 0..W {
                let base = r * row_len + offset;
                let d_lane = rows_dst[base..base + 32].first_chunk_mut().unwrap();
                let s_lane = rows_src[base..base + 32].first_chunk().unwrap();
                let d = _mm256_loadu_si256(&*d_lane);
                let s = _mm256_loadu_si256(s_lane);
                _mm256_storeu_si256(d_lane, _mm256_xor_si256(d, s));
            }
            offset += 32;
        }
        if offset < row_len {
            for r in 0..W {
                let row = r * row_len;
                let (d_row, s_row) = (
                    &mut rows_dst[row..row + row_len],
                    &rows_src[row..row + row_len],
                );
                scalar::xor(&mut d_row[offset..], &s_row[offset..]);
            }
        }
    }
}

/// [`xor_rows_avx2`] over 16-byte SSE2 lanes, for the `V2` tier.
///
/// # Panics
/// Panics if the slices differ in length or their length is not a whole
/// number of rows.
#[archmage::arcane]
pub fn xor_rows_sse2(token: archmage::X64V1Token, dst: &mut [u8], src: &[u8], row_len: usize) {
    assert_eq!(dst.len(), src.len());
    assert!(dst.len().is_multiple_of(row_len));
    debug_assert_ne!(row_len, 0);
    let rows = dst.len() / row_len;
    let four_span = rows / 4 * 4 * row_len;
    let two_span = rows % 4 / 2 * 2 * row_len;
    xor_streams_sse2::<4>(&mut dst[..four_span], &src[..four_span], row_len);
    xor_streams_sse2::<2>(
        &mut dst[four_span..four_span + two_span],
        &src[four_span..four_span + two_span],
        row_len,
    );
    if rows % 2 == 1 {
        let odd = four_span + two_span;
        xor_sse2(token, &mut dst[odd..], &src[odd..]);
    }
}

/// SSE2 twin of [`xor_streams_avx2`]; see that function for the layout.
#[archmage::rite(v1, import_intrinsics)]
fn xor_streams_sse2<const W: usize>(dst: &mut [u8], src: &[u8], row_len: usize) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert!(row_len != 0 && dst.len().is_multiple_of(W * row_len));
    let group_span = W * row_len;
    for (rows_dst, rows_src) in dst
        .chunks_exact_mut(group_span)
        .zip(src.chunks_exact(group_span))
    {
        let mut offset = 0;
        while offset + SSE2_ROW_TILE <= row_len {
            for r in 0..W {
                let base = r * row_len + offset;
                let (d_lanes, _) = rows_dst[base..base + SSE2_ROW_TILE].as_chunks_mut::<16>();
                let (s_lanes, _) = rows_src[base..base + SSE2_ROW_TILE].as_chunks::<16>();
                for (d_lane, s_lane) in d_lanes.iter_mut().zip(s_lanes) {
                    let d = _mm_loadu_si128(&*d_lane);
                    let s = _mm_loadu_si128(s_lane);
                    _mm_storeu_si128(d_lane, _mm_xor_si128(d, s));
                }
            }
            offset += SSE2_ROW_TILE;
        }
        while offset + 16 <= row_len {
            for r in 0..W {
                let base = r * row_len + offset;
                let d_lane = rows_dst[base..base + 16].first_chunk_mut().unwrap();
                let s_lane = rows_src[base..base + 16].first_chunk().unwrap();
                let d = _mm_loadu_si128(&*d_lane);
                let s = _mm_loadu_si128(s_lane);
                _mm_storeu_si128(d_lane, _mm_xor_si128(d, s));
            }
            offset += 16;
        }
        if offset < row_len {
            for r in 0..W {
                let row = r * row_len;
                let (d_row, s_row) = (
                    &mut rows_dst[row..row + row_len],
                    &rows_src[row..row + row_len],
                );
                scalar::xor(&mut d_row[offset..], &s_row[offset..]);
            }
        }
    }
}
