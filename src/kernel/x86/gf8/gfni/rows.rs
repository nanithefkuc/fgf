//! Register-blocked row-group bodies behind the matrix kernels.
//!
//! A group of one, two or four rows resolves its coefficients into one stack
//! array, runs the tile loops over that chunk, and finishes the sub-lane
//! remainder from the terms themselves. The scattered and contiguous matrix
//! walks share these bodies.
//!
//! The bodies are safe [`archmage::rite`] helpers. What unsafe remains is
//! residue, per item: the rows arrive as runtime offsets into one region —
//! staged by the callers, which assert the row geometry — and the resolve
//! scratch is a `MaybeUninit` staging array whose occupied prefix is written
//! before it is read.

use super::super::RESOLVE_CHUNK;
use super::{Blocked, bmul_gfni, brem_gfni, factor_word_gfni, wfactor_gfni};
use crate::kernel::Matrix;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// How far ahead of the tile being multiplied the matrix tile loop prefetches
/// each source, in bytes.
///
/// The value bounds the gap between the prefetch window and the loads the
/// tile loop is about to issue. It is a measured parameter.
const MATRIX_PREFETCH_DISTANCE: usize = 1024;

/// Shortest vector body for which the matrix tile loop prefetches its
/// sources.
///
/// Below this length the loop issues too few loads for a software prefetch
/// one distance ahead to pay for itself, so the body runs without prefetch.
/// It is a measured parameter.
pub(crate) const MATRIX_PREFETCH_MIN: usize = 8192;

/// Fold one chunk of resolved terms into `ROWS` rows, `LANES` 32-byte lanes
/// per row per iteration, then whole 32-byte lanes.
///
/// `maps[term][row]` is the resolved factor word, so the loop walks the
/// factor array and the source array in lockstep and indexes neither. When
/// `PREFETCH` is set, the tile loop issues a software prefetch for every
/// source one [`MATRIX_PREFETCH_DISTANCE`] ahead of the tile.
///
/// The rows are addressed as runtime offsets into one region; the entry that
/// staged them asserted `coeffs.len() * row_len` (or the scattered-row
/// equivalent) against the region length, and `rows_resolved` asserted every
/// staged source spans the `vector_len` the caller passes.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn rows_body<
    S: Blocked,
    const ROWS: usize,
    const LANES: usize,
    const OVERWRITE: bool,
    const PREFETCH: bool,
>(
    ptrs: [*mut u8; ROWS],
    vector_len: usize,
    maps: &[[u64; ROWS]],
    srcs: &[&[u8]],
) {
    let step = 32 * LANES;
    let mut tile = 0;
    while tile + step <= vector_len {
        // Seeding from zero must stay a plain array initializer: a seeding
        // loop over the accumulator array keeps LLVM from promoting it out of
        // memory, and the overwrite path then spills a tile per source.
        let mut acc = [[_mm256_setzero_si256(); LANES]; ROWS];
        if !OVERWRITE {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + step <= vector_len` and every row spans at
            //        least `vector_len` bytes, so `tile + 32 * lane` lies
            //        inside each row for every lane below.
            // THUS: each 32-byte load stays within its originating row.
            //
            // ALIASING
            // SINCE: the caller staged `ROWS` pairwise-disjoint row windows,
            //        each inside its own checked row, and only one window is
            //        addressed at a time against accumulators this body owns.
            // THUS: the loads do not race any store this body issues.
            unsafe {
                for (&ptr, slots) in ptrs.iter().zip(acc.iter_mut()) {
                    for (lane, slot) in slots.iter_mut().enumerate() {
                        *slot = _mm256_loadu_si256(ptr.add(tile + 32 * lane).cast());
                    }
                }
            }
        }
        for (map, src) in maps.iter().zip(srcs) {
            let mut factor = [_mm256_setzero_si256(); ROWS];
            for (slot, &word) in factor.iter_mut().zip(map) {
                *slot = wfactor_gfni::<S>(word);
            }
            if PREFETCH {
                // A prefetch never faults, so the window may point past the
                // source without a bounds proof.
                let ahead = src.as_ptr().wrapping_add(tile + MATRIX_PREFETCH_DISTANCE);
                // One prefetch per 64-byte cache line of the tile window.
                for line in (0..step).step_by(64) {
                    _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(line).cast());
                }
            }
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: every source spans at least `vector_len` bytes, so
            //        `tile + 32 * lane` lies inside `src` for every lane.
            // THUS: each 32-byte load stays within its originating source.
            unsafe {
                let sp = src.as_ptr().add(tile);
                let mut x = [_mm256_setzero_si256(); LANES];
                for (lane, slot) in x.iter_mut().enumerate() {
                    *slot = _mm256_loadu_si256(sp.add(32 * lane).cast());
                }
                for (slots, &f) in acc.iter_mut().zip(&factor) {
                    for (slot, &value) in slots.iter_mut().zip(&x) {
                        *slot = _mm256_xor_si256(*slot, bmul_gfni::<S>(value, f));
                    }
                }
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the same `tile + 32 * lane < vector_len` relation that
        //        bounded the loads bounds every store below.
        // THUS: each 32-byte store stays within its originating row.
        //
        // ALIASING
        // SINCE: the staged row windows are pairwise disjoint, as above.
        // THUS: no store conflicts with another row's window.
        unsafe {
            for (&ptr, slots) in ptrs.iter().zip(&acc) {
                for (lane, &value) in slots.iter().enumerate() {
                    _mm256_storeu_si256(ptr.add(tile + 32 * lane).cast(), value);
                }
            }
        }
        tile += step;
    }
    while tile + 32 <= vector_len {
        let mut acc = [_mm256_setzero_si256(); ROWS];
        if !OVERWRITE {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 32 <= vector_len` and every row spans at least
            //        `vector_len` bytes.
            // THUS: the 32-byte load stays within its originating row.
            unsafe {
                for (&ptr, slot) in ptrs.iter().zip(acc.iter_mut()) {
                    *slot = _mm256_loadu_si256(ptr.add(tile).cast());
                }
            }
        }
        for (map, src) in maps.iter().zip(srcs) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: every source spans at least `vector_len` bytes and
            //        `tile + 32 <= vector_len`.
            // THUS: the 32-byte load stays within its originating source.
            unsafe {
                let x = _mm256_loadu_si256(src.as_ptr().add(tile).cast());
                for (slot, &word) in acc.iter_mut().zip(map) {
                    *slot = _mm256_xor_si256(*slot, bmul_gfni::<S>(x, wfactor_gfni::<S>(word)));
                }
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 32 <= vector_len`, the same relation that bounded
        //        the loads, and the row windows are disjoint.
        // THUS: the 32-byte store stays within its originating row and does
        //       not conflict with another row's window.
        unsafe {
            for (&ptr, &value) in ptrs.iter().zip(&acc) {
                _mm256_storeu_si256(ptr.add(tile).cast(), value);
            }
        }
        tile += 32;
    }
}

/// Resolve a row group's coefficients chunk by chunk, run the tile loops over
/// each chunk, then finish the sub-lane remainder from the terms themselves.
///
/// The first chunk carries the caller's overwrite policy; every later chunk
/// accumulates into what the earlier ones wrote. A vector body at or above
/// [`MATRIX_PREFETCH_MIN`] whose terms fit in one chunk runs those loops with
/// source prefetch; more terms than that already re-stream the destination
/// per chunk, and the column-blocked walk serves them instead.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn rows_resolved<
    S: Blocked,
    M: Matrix<S::Coeff> + ?Sized,
    const ROWS: usize,
    const LANES: usize,
    const OVERWRITE: bool,
>(
    ptrs: [*mut u8; ROWS],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    let vector_len = row_len & !31;
    let count = terms.len();
    let prefetch = vector_len >= MATRIX_PREFETCH_MIN && count <= RESOLVE_CHUNK;
    let mut start = 0;
    let mut first = true;
    loop {
        let taken = (count - start).min(RESOLVE_CHUNK);
        // Declaring the scratch as `MaybeUninit` keeps the unused tail
        // unwritten: a plain declaration initializes every slot, which costs a
        // fixed per-chunk store burst even when `taken` is small.
        //
        // SAFETY:
        // INITIALIZATION
        // SINCE: `MaybeUninit` has no validity invariant, so an array of it
        //        may be assumed initialized, and exactly `taken` slots are
        //        written below before any slot is read.
        // THUS: forming the uninitialized staging arrays reads no invalid
        //       value.
        let mut maps: [core::mem::MaybeUninit<[u64; ROWS]>; RESOLVE_CHUNK] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        let mut srcs: [core::mem::MaybeUninit<&[u8]>; RESOLVE_CHUNK] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        for offset in 0..taken {
            let term = start + offset;
            let mut words = [0u64; ROWS];
            for (row, word) in words.iter_mut().enumerate() {
                *word = factor_word_gfni::<S>(*terms.coefficient(term, g + row));
            }
            maps[offset].write(words);
            // The body reads the staged slice without bounds checks, so the
            // captured slice itself must cover the span; the provider may
            // return a different slice than the one the entry validated.
            let src = terms.source(term);
            assert!(
                src.len() >= vector_len,
                "matrix kernel: term {term} source is shorter than the row span"
            );
            srcs[offset].write(src);
        }
        // SAFETY:
        // INITIALIZATION
        // SINCE: exactly `taken` leading slots were written above, and the
        //        reinterpretation covers only those slots.
        // THUS: the slices expose no unwritten slot.
        let (maps, srcs): (&[[u64; ROWS]], &[&[u8]]) = unsafe {
            (
                core::slice::from_raw_parts(maps.as_ptr().cast::<[u64; ROWS]>(), taken),
                core::slice::from_raw_parts(srcs.as_ptr().cast::<&[u8]>(), taken),
            )
        };
        if OVERWRITE && first {
            if prefetch {
                rows_body::<S, ROWS, LANES, true, true>(ptrs, vector_len, maps, srcs);
            } else {
                rows_body::<S, ROWS, LANES, true, false>(ptrs, vector_len, maps, srcs);
            }
        } else if prefetch {
            rows_body::<S, ROWS, LANES, false, true>(ptrs, vector_len, maps, srcs);
        } else {
            rows_body::<S, ROWS, LANES, false, false>(ptrs, vector_len, maps, srcs);
        }
        first = false;
        start += taken;
        if start >= count {
            break;
        }
    }
    matrix_tail::<S, M, OVERWRITE>(&ptrs, row_len, g, vector_len, terms);
}

/// Fold every term into four rows, 64 bytes of each row at a time.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn matrix_rows4<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    rows_resolved::<S, M, 4, 2, OVERWRITE>(ptrs, row_len, g, terms);
}

/// Fold every term into two rows, 128 bytes of each row at a time.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn matrix_rows2<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 2],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    rows_resolved::<S, M, 2, 4, OVERWRITE>(ptrs, row_len, g, terms);
}

/// Fold every term into one row, 128 bytes at a time.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn matrix_rows1<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    ptr: *mut u8,
    row_len: usize,
    g: usize,
    terms: &M,
) {
    rows_resolved::<S, M, 1, 4, OVERWRITE>([ptr], row_len, g, terms);
}

/// Hierarchical remainder shared by the matrix row groups.
///
/// Complete 32- and 16-byte lanes stay on the one-instruction multiply through
/// the field's single-row remainder; only the final sub-XMM bytes use scalar
/// arithmetic.
///
/// Each row's tail slice is formed from the row's runtime offset; the entry
/// that staged the rows asserted the row geometry, `tile` is the vector-covered
/// prefix, and only one tail borrow is live at a time.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn matrix_tail<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    ptrs: &[*mut u8],
    row_len: usize,
    g: usize,
    tile: usize,
    terms: &M,
) {
    if tile == row_len {
        return;
    }
    let remaining = row_len - tile;
    for (slot, &row) in ptrs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + remaining == row_len`, and the entry that staged the
        //        rows asserted each row spans `row_len` bytes inside one
        //        region, so this is the row's own final span.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the staged row windows are pairwise disjoint, each lying
        //        inside its own checked row, and the borrow is released
        //        before the next row's tail is formed.
        // THUS: the mutable borrow aliases no other live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(row.add(tile), remaining) };
        // Overwrite: start the tail at zero so the accumulating brem_gfni below
        // yields `sum(terms)` without reading prior destination bytes.
        if OVERWRITE {
            tail.fill(0);
        }
        for term in 0..terms.len() {
            let src = terms.source(term);
            let coeff = *terms.coefficient(term, g + slot);
            if S::byte(coeff) != 0 {
                brem_gfni::<S>(tail, coeff, &src[tile..]);
            }
        }
    }
}
