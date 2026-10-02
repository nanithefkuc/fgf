//! Register-blocked row-group bodies behind the matrix kernels.
//!
//! A group of one to [`MAX_GROUP_ROWS`] rows resolves its coefficients into
//! one stack array, runs the tile loops over that chunk, and finishes the
//! sub-lane remainder from the terms themselves. The scattered and contiguous
//! matrix walks share these bodies and the [`group_rows`] split.
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

/// Fold one term's source lanes `x` into a `ROWS` x `LANES` accumulator tile:
/// `acc[row][lane] ^= map[row] * x[lane]`, or `=` when `first` is set.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
fn fold_term<S: Blocked, const ROWS: usize, const LANES: usize>(
    acc: &mut [[__m256i; LANES]; ROWS],
    map: &[u64; ROWS],
    x: &[__m256i; LANES],
    first: bool,
) {
    for (slots, &word) in acc.iter_mut().zip(map) {
        let factor = wfactor_gfni::<S>(word);
        for (slot, &value) in slots.iter_mut().zip(x) {
            let product = bmul_gfni::<S>(value, factor);
            *slot = if first {
                product
            } else {
                _mm256_xor_si256(*slot, product)
            };
        }
    }
}

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
        // One term's source lanes at this tile, with the window one
        // prefetch distance ahead requested when `PREFETCH` is set.
        let load = |src: &[u8]| -> [__m256i; LANES] {
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
            // SINCE: every source spans at least `vector_len` bytes and
            //        `tile + step <= vector_len`, so `tile + 32 * lane` lies
            //        inside `src` for every lane.
            // THUS: each 32-byte load stays within its originating source.
            unsafe {
                let sp = src.as_ptr().add(tile);
                core::array::from_fn(|lane| _mm256_loadu_si256(sp.add(32 * lane).cast()))
            }
        };
        // Overwrite seeds a multi-row group's accumulators from the first
        // product instead of folding it into zero; the one-row group keeps the
        // zero seed. The three-lane tile folds its terms two at a time. Both
        // choices are measured parameters.
        let mut first = OVERWRITE && ROWS > 1;
        if LANES == 3 {
            let (map_pairs, map_rest) = maps.as_chunks::<2>();
            let (src_pairs, src_rest) = srcs.as_chunks::<2>();
            for ([map0, map1], [src0, src1]) in map_pairs.iter().zip(src_pairs) {
                fold_term::<S, ROWS, LANES>(&mut acc, map0, &load(src0), first);
                fold_term::<S, ROWS, LANES>(&mut acc, map1, &load(src1), false);
                first = false;
            }
            if let ([map], [src]) = (map_rest, src_rest) {
                fold_term::<S, ROWS, LANES>(&mut acc, map, &load(src), first);
            }
        } else {
            for (map, src) in maps.iter().zip(srcs) {
                fold_term::<S, ROWS, LANES>(&mut acc, map, &load(src), first);
                first = false;
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

/// The widest row group a matrix walk runs in one pass over its sources.
///
/// Six rows of two 32-byte lanes hold twelve accumulators, leaving the two
/// source lanes and one factor inside the sixteen `ymm` registers. The group
/// widths are measured parameters.
pub(super) const MAX_GROUP_ROWS: usize = 6;

/// Term counts at or below this bound cap row groups at four rows.
///
/// With one or two sources the destination streams outnumber the source
/// streams, so a wider group saves little source traffic and costs the store
/// pressure of more rows per tile. The bound is a measured parameter.
const FEW_TERMS: usize = 2;

/// Fewest terms for which a three-row group runs three 32-byte lanes per row.
///
/// Below it the group keeps two lanes, which leave more of the loop to the
/// term-independent loads and stores. The bound is a measured parameter.
const THREE_LANE_MIN_TERMS: usize = 6;

/// Shortest row, in bytes, for which a three-row group runs three 32-byte
/// lanes per row.
///
/// Shorter rows leave a larger share of whole lanes to the one-lane remainder
/// loop, which the two-lane tile covers instead. The bound is a measured
/// parameter.
const THREE_LANE_MIN_ROW: usize = 512;

/// Longest row, in bytes, for which a three-row group runs three 32-byte lanes
/// per row.
///
/// Longer rows stream from beyond the core's caches, where the two-lane group
/// keeps fewer streams in flight. The bound is a measured parameter.
const THREE_LANE_MAX_ROW: usize = 128 * 1024;

/// Whether a three-row group over `nterms` terms and `row_len`-byte rows runs
/// three 32-byte lanes per row.
///
/// Term counts past one resolve chunk keep two lanes: the column-blocked walk
/// serves them with short blocks, where three lanes lose on long rows.
const fn three_lanes(nterms: usize, row_len: usize) -> bool {
    nterms >= THREE_LANE_MIN_TERMS
        && nterms <= RESOLVE_CHUNK
        && row_len >= THREE_LANE_MIN_ROW
        && row_len <= THREE_LANE_MAX_ROW
}

/// The row count of the group that starts `done` rows into an `nrows`-row
/// walk over `nterms` terms.
///
/// Every group streams every source once, so the walk runs the fewest groups
/// that fit in [`MAX_GROUP_ROWS`] (four rows up to [`FEW_TERMS`] terms),
/// balanced so that no group is more than one row wider than another.
///
/// # Panics
/// Panics if `done >= nrows`.
pub(super) fn group_rows(nrows: usize, done: usize, nterms: usize) -> usize {
    let left = nrows - done;
    let widest = if nterms <= FEW_TERMS {
        4
    } else {
        MAX_GROUP_ROWS
    };
    left.div_ceil(left.div_ceil(widest))
}

/// Fold every term into the staged rows of one group: one or two rows with
/// four 32-byte lanes per row, four to six rows with two, and three rows with
/// three lanes where [`three_lanes`] selects them, two otherwise.
///
/// The caller stages one to [`MAX_GROUP_ROWS`] pairwise-disjoint row windows of
/// `row_len` bytes each, the first being row `g` of the matrix.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn matrix_group<S: Blocked, M: Matrix<S::Coeff> + ?Sized, const OVERWRITE: bool>(
    rows: &[*mut u8],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    match *rows {
        [a] => rows_resolved::<S, M, 1, 4, OVERWRITE>([a], row_len, g, terms),
        [a, b] => rows_resolved::<S, M, 2, 4, OVERWRITE>([a, b], row_len, g, terms),
        [a, b, c] if three_lanes(terms.len(), row_len) => {
            rows_resolved::<S, M, 3, 3, OVERWRITE>([a, b, c], row_len, g, terms);
        }
        [a, b, c] => rows_resolved::<S, M, 3, 2, OVERWRITE>([a, b, c], row_len, g, terms),
        [a, b, c, d] => rows_resolved::<S, M, 4, 2, OVERWRITE>([a, b, c, d], row_len, g, terms),
        [a, b, c, d, e] => {
            rows_resolved::<S, M, 5, 2, OVERWRITE>([a, b, c, d, e], row_len, g, terms);
        }
        [a, b, c, d, e, f] => {
            rows_resolved::<S, M, 6, 2, OVERWRITE>([a, b, c, d, e, f], row_len, g, terms);
        }
        _ => unreachable!("matrix row group holds one to {MAX_GROUP_ROWS} rows"),
    }
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
