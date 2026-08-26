//! x86 / `x86_64` SIMD kernels.
//!
//! Everything `unsafe` in this crate for this architecture lives under here.
//! The safe `pub(crate)` wrappers in the submodules are the boundary: the
//! caller has already selected a matching [`Backend`](crate::kernel::Backend),
//! so each wrapper re-establishes the CPU-feature proof next to the
//! intrinsic call it guards.
//!
//! Two multiply strategies:
//!
//! - **GFNI.** `GF2P8MULB` multiplies bytes in `GF(2)[x] / 0x11B` — exactly
//!   this crate's GF(2^8) — 32 lanes per instruction, no table, no shuffle
//!   port pressure. This is why the field uses the AES polynomial.
//! - **Nibble shuffle.** Without GFNI, `PSHUFB` performs a 16-entry lookup
//!   per lane, so `c * x` becomes two shuffles and an XOR against the
//!   precomputed [`ScaleTable`](crate::kernel::tables::ScaleTable).

#![allow(unsafe_code)]
#![allow(clippy::incompatible_msrv)]

// The 64-byte AVX-512 kernels are the deferred V4x tier (cross-compile-only
// today; not in the shared ladder until validated on executing hardware).
// They compile only for `internals` experiments, where they are reachable
// (and differentially tested on a host that has AVX-512).
#[cfg(feature = "internals")]
pub mod avx512;
pub mod fan_paar;
pub mod gf16;
pub mod gf32;
pub mod gf64;
pub mod gf8;
pub mod prime;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar;

/// Smallest destination for which non-temporal stores pay for themselves.
///
/// `mul_into` writes a destination it never reads, so an ordinary store pays
/// a read-for-ownership fetch of every line it is about to overwrite
/// completely. `vmovntdq` skips that fetch and the allocation, at the price
/// of evicting the destination from cache.
///
/// The threshold is set by the workload this pessimizes — encode, then read
/// the destination back — because that is the case where the eviction costs
/// something. Below it the read-back loop loses; at 2 MiB it breaks even while
/// the write-only case is already several times ahead. `mul_add` and
/// `mul_assign` read their destination anyway and keep ordinary stores. The
/// measurements are under "Crossover and dispatch decisions" in
/// BENCHMARKS.md.
pub(super) const NT_STORE_MIN: usize = 2 << 20;

/// Head bytes to store normally so that a non-temporal body starts on a
/// 32-byte boundary, or `None` when this destination should stay temporal.
///
/// `None` covers a destination too small to repay the eviction, and the case
/// where the alignment peel would not be a whole number of `elem_bytes`
/// elements — a kernel may only split a buffer on an element boundary.
#[inline]
pub(super) fn nt_split(dst: &[u8], elem_bytes: usize) -> Option<usize> {
    if dst.len() < NT_STORE_MIN {
        return None;
    }
    let peel = dst.as_ptr().align_offset(32);
    (peel != usize::MAX && peel.is_multiple_of(elem_bytes)).then_some(peel)
}

/// Bytes of lead-in that put a row group's vector accesses on a 32-byte
/// boundary, or `0` when the row is too short to repay it.
///
/// A 32-byte `vmovdqu` at an odd multiple of 32 straddles two cache lines,
/// and a multi-row body issues one load and one store per row per vector: a
/// misaligned destination therefore doubles the line traffic of every access
/// but the shared source load. Peeling at most 31 bytes per row group buys
/// the aligned body for the rest of the pass; the measurements, and the
/// shapes where the same peel was tried and rejected, are in BENCHMARKS.md.
/// Rows of a group sit `row_len` apart, so aligning one aligns them all
/// whenever `row_len` is a multiple of 32 — the usual case. The source is
/// left where it falls: it is one access against many, and only the
/// destination is ours to choose.
///
/// The peel is rounded up to a whole number of `elem_bytes` elements, because
/// a kernel may only split a buffer on an element boundary. For an odd
/// destination pointer in a 2-byte field that is unreachable, and the peel is
/// skipped.
#[inline]
pub(super) fn peel_to_align(ptr: *const u8, len: usize, elem_bytes: usize) -> usize {
    // The peel runs the narrow single-row kernels, which descend to 128-bit
    // lanes and then scalar code. Keep the conservative crossover measured for
    // the former all-scalar GF(2^8) peel: it won from about 2 KiB up and lost
    // badly below 1 KiB. Sixteen turns of the 128-byte body remains the floor.
    const FLOOR: usize = 16 * 128;
    let head = ptr.align_offset(32);
    // `align_offset` reports `usize::MAX` only for an unreachable alignment,
    // which cannot happen for a byte pointer; `head >= len` also covers the
    // degenerate case of a peel longer than the row.
    if len < FLOOR || head >= len || !head.is_multiple_of(elem_bytes) {
        0
    } else {
        head
    }
}

/// Store 32 bytes, non-temporally when `NT`.
///
/// # Safety
/// `ptr` must be writable for 32 bytes, and when `NT` it must additionally be
/// 32-byte aligned. A `NT` caller must `_mm_sfence()` before any other thread
/// observes the stores.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn store256<const NT: bool>(ptr: *mut u8, value: __m256i) {
    // SAFETY: the caller guarantees writability, and alignment when `NT`.
    unsafe {
        if NT {
            _mm256_stream_si256(ptr.cast(), value);
        } else {
            _mm256_storeu_si256(ptr.cast(), value);
        }
    }
}

/// Store 16 bytes, non-temporally when `NT`.
///
/// # Safety
/// As [`store256`], for 16 bytes and 16-byte alignment.
#[inline]
#[target_feature(enable = "sse2")]
pub(super) unsafe fn store128<const NT: bool>(ptr: *mut u8, value: __m128i) {
    // SAFETY: the caller guarantees writability, and alignment when `NT`.
    unsafe {
        if NT {
            _mm_stream_si128(ptr.cast(), value);
        } else {
            _mm_storeu_si128(ptr.cast(), value);
        }
    }
}

/// `dst ^= src` using 32-byte AVX2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_avx2(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected an AVX2-capable backend, and the slices are
    // equal-length and independently borrowed.
    unsafe { xor_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let s = _mm256_loadu_si256(src_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(d, s));
        }
        offset += 32;
    }
    let mut offset = len;
    if offset + 16 <= dst.len() {
        // SAFETY: `offset + 16 <= dst.len() == src.len()`, and AVX2 implies
        // SSE2 for the single narrow tail lane.
        unsafe {
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let s = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[offset..], &src[offset..]);
}

/// `dst ^= src` using 16-byte SSE2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_sse2(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: SSE2 is baseline on x86_64 and implied by the SSSE3 backend on
    // x86; the slices are equal-length and independently borrowed.
    unsafe { xor_sse2_impl(dst, src) }
}

#[target_feature(enable = "sse2")]
unsafe fn xor_sse2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let s = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[len..], &src[len..]);
}

// ---------------------------------------------------------------------------
// Row-interleaved XOR
// ---------------------------------------------------------------------------

/// Bytes of one row covered by a fully-unrolled AVX2 tile iteration: four
/// 32-byte vectors per stream, the leopard `xor_mem4` unroll width.
const AVX2_ROW_TILE: usize = 4 * 32;

/// Bytes of one row covered by a fully-unrolled SSE2 tile iteration: four
/// 16-byte vectors per stream.
const SSE2_ROW_TILE: usize = 4 * 16;

/// `dst ^= src` over contiguous `row_len`-byte rows with four interleaved
/// streams.
///
/// Both buffers hold the same whole number of rows (validated here); every
/// group of four row pairs walks `row_len` in tiles, each stream issuing its
/// own load/xor/store chain per tile. The four independent streams keep the
/// core's line-fill buffers busy where one sequential stream cannot — the
/// shape of leopard's `xor_mem4`.
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn xor_rows_avx2(dst: &mut [u8], src: &[u8], row_len: usize) {
    assert_eq!(dst.len(), src.len());
    assert!(dst.len().is_multiple_of(row_len));
    debug_assert_ne!(row_len, 0);
    // SAFETY: an AVX2-capable backend was selected, the slices are
    // equal-length and independently borrowed, and `row_len` divides their
    // length.
    unsafe { xor_rows_avx2_impl(dst.as_mut_ptr(), src.as_ptr(), dst.len() / row_len, row_len) }
}

/// Walk `groups` four-row groups from `dst`/`src`, then the two-row and
/// one-row remainders through narrower bodies.
///
/// Rows are consecutive in both buffers, so group and remainder regions are
/// plain pointer arithmetic over validated lengths; the final single row
/// reuses the flat AVX2 body, which also owns the sub-32-byte tail handling.
///
/// # Safety
/// `dst` must be writable and `src` readable for `rows * row_len` bytes,
/// both derived from live, independently borrowed slices; `row_len != 0`;
/// `rows >= 1`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn xor_rows_avx2_impl(dst: *mut u8, src: *const u8, rows: usize, row_len: usize) {
    let fours = rows / 4;
    let twos = (rows % 4) / 2;
    // SAFETY: all pointers stay within the `rows * row_len` bytes the caller
    // guarantees for both buffers; each region is disjoint by construction.
    unsafe {
        let (mut d, mut s) = (dst, src);
        if fours > 0 {
            xor_streams_avx2::<4>(d, s, fours, row_len);
            d = d.add(fours * 4 * row_len);
            s = s.add(fours * 4 * row_len);
        }
        if twos > 0 {
            xor_streams_avx2::<2>(d, s, twos, row_len);
            d = d.add(2 * row_len);
            s = s.add(2 * row_len);
        }
        if rows % 2 == 1 {
            let tail = core::slice::from_raw_parts_mut(d, row_len);
            let tail_src = core::slice::from_raw_parts(s, row_len);
            xor_avx2_impl(tail, tail_src);
        }
    }
}

/// XOR `groups` groups of `W` consecutive `row_len`-byte rows, `W` streams
/// interleaved.
///
/// Each pass takes one group: fully-unrolled [`AVX2_ROW_TILE`] tiles across the
/// row — one tile covers [`AVX2_ROW_TILE`] bytes of *every* stream at the same
/// offset — then single vectors over the remaining tail, and scalar bytes at
/// the end. The inner loops run to the const bound `W`, so LLVM unrolls them
/// away and the body issues one independent load/xor/store chain per stream
/// per tile without index arithmetic.
///
/// # Safety
/// Same contract as [`xor_rows_avx2_impl`], with `rows == groups * W >= W`
/// so every full-width stream below stays in bounds.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn xor_streams_avx2<const W: usize>(
    dst: *mut u8,
    src: *const u8,
    groups: usize,
    row_len: usize,
) {
    for group in 0..groups {
        // SAFETY: the group spans `[base, base + W * row_len)` bytes of both
        // buffers, inside the region the caller guarantees.
        let (base_d, base_s) =
            unsafe { (dst.add(group * W * row_len), src.add(group * W * row_len)) };
        let mut offset = 0;
        while offset + AVX2_ROW_TILE <= row_len {
            for r in 0..W {
                // SAFETY: row `r` of this group spans
                // `[base + r * row_len, base + r * row_len + row_len)` and
                // `offset` plus the unrolled vectors stay inside it.
                unsafe {
                    let d = base_d.add(r * row_len + offset);
                    let s = base_s.add(r * row_len + offset);
                    for v in 0..AVX2_ROW_TILE / 32 {
                        let dv = _mm256_loadu_si256(d.add(v * 32).cast());
                        let sv = _mm256_loadu_si256(s.add(v * 32).cast());
                        _mm256_storeu_si256(d.add(v * 32).cast(), _mm256_xor_si256(dv, sv));
                    }
                }
            }
            offset += AVX2_ROW_TILE;
        }
        while offset + 32 <= row_len {
            for r in 0..W {
                // SAFETY: `offset + 32 <= row_len`, so the vector sits
                // inside row `r` of this group.
                unsafe {
                    let d = base_d.add(r * row_len + offset).cast();
                    let s = base_s.add(r * row_len + offset).cast();
                    _mm256_storeu_si256(
                        d,
                        _mm256_xor_si256(_mm256_loadu_si256(d), _mm256_loadu_si256(s)),
                    );
                }
            }
            offset += 32;
        }
        if offset < row_len {
            for r in 0..W {
                // SAFETY: `[base + r * row_len; row_len]` is inside this
                // group's region, and so is every index below it.
                let tail =
                    unsafe { core::slice::from_raw_parts_mut(base_d.add(r * row_len), row_len) };
                let tail_src =
                    unsafe { core::slice::from_raw_parts(base_s.add(r * row_len), row_len) };
                scalar::xor(&mut tail[offset..], &tail_src[offset..]);
            }
        }
    }
}

/// [`xor_rows_avx2`] over 16-byte SSE2 lanes, for the `V2` tier.
///
/// # Panics
/// Panics if the slices differ in length or their length is not a whole
/// number of rows.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn xor_rows_sse2(dst: &mut [u8], src: &[u8], row_len: usize) {
    assert_eq!(dst.len(), src.len());
    assert!(dst.len().is_multiple_of(row_len));
    debug_assert_ne!(row_len, 0);
    // SAFETY: SSE2 is baseline on x86_64 and implied by the SSSE3 backend on
    // x86; the slices are equal-length, independently borrowed, and whole
    // numbers of rows.
    unsafe { xor_rows_sse2_impl(dst.as_mut_ptr(), src.as_ptr(), dst.len() / row_len, row_len) }
}

/// SSE2 twin of [`xor_rows_avx2_impl`]; see that function for the layout.
///
/// # Safety
/// Same contract as [`xor_rows_avx2_impl`].
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn xor_rows_sse2_impl(dst: *mut u8, src: *const u8, rows: usize, row_len: usize) {
    let fours = rows / 4;
    let twos = (rows % 4) / 2;
    // SAFETY: all pointers stay within the `rows * row_len` bytes the caller
    // guarantees for both buffers; each region is disjoint by construction.
    unsafe {
        let (mut d, mut s) = (dst, src);
        if fours > 0 {
            xor_streams_sse2::<4>(d, s, fours, row_len);
            d = d.add(fours * 4 * row_len);
            s = s.add(fours * 4 * row_len);
        }
        if twos > 0 {
            xor_streams_sse2::<2>(d, s, twos, row_len);
            d = d.add(2 * row_len);
            s = s.add(2 * row_len);
        }
        if rows % 2 == 1 {
            let tail = core::slice::from_raw_parts_mut(d, row_len);
            let tail_src = core::slice::from_raw_parts(s, row_len);
            xor_sse2_impl(tail, tail_src);
        }
    }
}

/// SSE2 twin of [`xor_streams_avx2`]; see that function for the layout.
///
/// # Safety
/// Same contract as [`xor_streams_avx2`].
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn xor_streams_sse2<const W: usize>(
    dst: *mut u8,
    src: *const u8,
    groups: usize,
    row_len: usize,
) {
    for group in 0..groups {
        // SAFETY: the group spans `[base, base + W * row_len)` bytes of both
        // buffers, inside the region the caller guarantees.
        let (base_d, base_s) =
            unsafe { (dst.add(group * W * row_len), src.add(group * W * row_len)) };
        let mut offset = 0;
        while offset + SSE2_ROW_TILE <= row_len {
            for r in 0..W {
                // SAFETY: row `r` of this group spans
                // `[base + r * row_len, base + r * row_len + row_len)` and
                // `offset` plus the unrolled vectors stay inside it.
                unsafe {
                    let d = base_d.add(r * row_len + offset);
                    let s = base_s.add(r * row_len + offset);
                    for v in 0..SSE2_ROW_TILE / 16 {
                        let dv = _mm_loadu_si128(d.add(v * 16).cast());
                        let sv = _mm_loadu_si128(s.add(v * 16).cast());
                        _mm_storeu_si128(d.add(v * 16).cast(), _mm_xor_si128(dv, sv));
                    }
                }
            }
            offset += SSE2_ROW_TILE;
        }
        while offset + 16 <= row_len {
            for r in 0..W {
                // SAFETY: `offset + 16 <= row_len`, so the vector sits
                // inside row `r` of this group.
                unsafe {
                    let d = base_d.add(r * row_len + offset).cast();
                    let s = base_s.add(r * row_len + offset).cast();
                    _mm_storeu_si128(d, _mm_xor_si128(_mm_loadu_si128(d), _mm_loadu_si128(s)));
                }
            }
            offset += 16;
        }
        if offset < row_len {
            for r in 0..W {
                // SAFETY: `[base + r * row_len; row_len]` is inside this
                // group's region, and so is every index below it.
                let tail =
                    unsafe { core::slice::from_raw_parts_mut(base_d.add(r * row_len), row_len) };
                let tail_src =
                    unsafe { core::slice::from_raw_parts(base_s.add(r * row_len), row_len) };
                scalar::xor(&mut tail[offset..], &tail_src[offset..]);
            }
        }
    }
}
