//! `AArch64` NEON kernels.
//!
//! Everything `unsafe` in this crate for this architecture lives under here.
//! The safe `pub(crate)` wrappers in the submodules are the boundary.
//!
//! NEON has no byte-wide field multiply, so both fields use the split-nibble
//! strategy: `VTBL`/`vqtbl1q_u8` performs a 16-entry lookup across all 16
//! lanes, and `c * x` is two lookups and an XOR against the precomputed
//! [`ScaleTable`](crate::kernel::tables::ScaleTable).

#![allow(unsafe_code)]

pub mod gf16;
pub mod gf8;

use core::arch::aarch64::*;

use crate::kernel::scalar;

/// `dst ^= src` using 16-byte NEON lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_neon(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: NEON is baseline on AArch64; the slices are equal-length and
    // independently borrowed.
    unsafe { xor_neon_impl(dst, src) }
}

#[target_feature(enable = "neon")]
unsafe fn xor_neon_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = vld1q_u8(dst_ptr.add(offset));
            let s = vld1q_u8(src_ptr.add(offset));
            vst1q_u8(dst_ptr.add(offset), veorq_u8(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[len..], &src[len..]);
}

/// Bytes of one row covered by a fully-unrolled tile iteration: four 16-byte
/// vectors per stream. Shared with the x86 row kernels' tile constant by
/// value; NEON keeps its own copy because the architecture modules are
/// compiled independently.
const NEON_ROW_TILE: usize = 4 * 16;

/// `dst ^= src` over contiguous `row_len`-byte rows with four interleaved
/// streams.
///
/// The same shape as the x86 AVX2 row kernel: groups of four row pairs
/// walk the row in tiles with one independent load/xor/store chain per
/// stream. Like there, this is an *unwired candidate* — never measured on
/// executing AArch64 hardware, so dispatch keeps the flat body until a
/// repeatable win is recorded (BENCHMARKS.md, "Row-interleaved XOR").
///
/// # Panics
/// Panics if the slices differ in length or their length is not a whole
/// number of rows.
pub fn xor_rows_neon(dst: &mut [u8], src: &[u8], row_len: usize) {
    assert_eq!(dst.len(), src.len());
    assert!(dst.len().is_multiple_of(row_len));
    debug_assert_ne!(row_len, 0);
    // SAFETY: NEON is baseline on AArch64; the slices are equal-length,
    // independently borrowed, and whole numbers of rows.
    unsafe { xor_rows_neon_impl(dst.as_mut_ptr(), src.as_ptr(), dst.len() / row_len, row_len) }
}

/// Walk `groups` four-row groups from `dst`/`src`, then the two-row and
/// one-row remainders through narrower bodies.
///
/// # Safety
/// `dst` must be writable and `src` readable for `rows * row_len` bytes,
/// both derived from live, independently borrowed slices; `row_len != 0`;
/// `rows >= 1`.
#[target_feature(enable = "neon")]
unsafe fn xor_rows_neon_impl(dst: *mut u8, src: *const u8, rows: usize, row_len: usize) {
    let fours = rows / 4;
    let twos = (rows % 4) / 2;
    // SAFETY: all pointers stay within the `rows * row_len` bytes the caller
    // guarantees for both buffers; each region is disjoint by construction.
    unsafe {
        let (mut d, mut s) = (dst, src);
        if fours > 0 {
            xor_streams_neon::<4>(d, s, fours, row_len);
            d = d.add(fours * 4 * row_len);
            s = s.add(fours * 4 * row_len);
        }
        if twos > 0 {
            xor_streams_neon::<2>(d, s, twos, row_len);
            d = d.add(2 * row_len);
            s = s.add(2 * row_len);
        }
        if rows % 2 == 1 {
            let tail = core::slice::from_raw_parts_mut(d, row_len);
            let tail_src = core::slice::from_raw_parts(s, row_len);
            xor_neon_impl(tail, tail_src);
        }
    }
}

/// XOR `groups` groups of `W` consecutive `row_len`-byte rows, `W` streams
/// interleaved.
///
/// # Safety
/// Same contract as [`xor_rows_neon_impl`], with `rows == groups * W >= W`
/// so every full-width stream below stays in bounds.
#[target_feature(enable = "neon")]
unsafe fn xor_streams_neon<const W: usize>(
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
        while offset + NEON_ROW_TILE <= row_len {
            for r in 0..W {
                // SAFETY: row `r` of this group spans
                // `[base + r * row_len, base + r * row_len + row_len)` and
                // `offset` plus the unrolled vectors stay inside it.
                unsafe {
                    let d = base_d.add(r * row_len + offset);
                    let s = base_s.add(r * row_len + offset);
                    for v in 0..NEON_ROW_TILE / 16 {
                        let dv = vld1q_u8(d.add(v * 16));
                        let sv = vld1q_u8(s.add(v * 16));
                        vst1q_u8(d.add(v * 16), veorq_u8(dv, sv));
                    }
                }
            }
            offset += NEON_ROW_TILE;
        }
        while offset + 16 <= row_len {
            for r in 0..W {
                // SAFETY: `offset + 16 <= row_len`, so the vector sits
                // inside row `r` of this group.
                unsafe {
                    let d = base_d.add(r * row_len + offset);
                    let s = base_s.add(r * row_len + offset);
                    vst1q_u8(d, veorq_u8(vld1q_u8(d), vld1q_u8(s)));
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
