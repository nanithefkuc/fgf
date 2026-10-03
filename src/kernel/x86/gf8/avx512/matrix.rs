//! AVX-512 affine GF(2^8) matrix kernels (`VGF2P8AFFINEQB`, 64-byte lanes).
//!
//! Many sources into many rows, mirroring the AVX2 affine blocked shapes at
//! 64-byte lanes: four-row groups over 256-byte tiles. A row group resolves
//! its coefficients into a stack array of map words chunk by chunk, keeps its
//! accumulators in registers across each chunk, and finishes the sub-lane
//! remainder from the terms themselves. The immediate is the XOR constant,
//! so `<0>` selects the pure linear map; each factor replicates one 64-bit
//! map qword to all eight 64-bit lanes.
//!
//! The kernels dispatch on the `V4x` tier under the `simd512` feature and stay
//! reachable through the `internals` facade for differential tests and
//! measurement on AVX-512 hardware.

use super::super::{ColumnWindow, RESOLVE_CHUNK, check_scattered, column_blocked, column_blocks};
use super::{Blocked, bfactor_avx512, bmul_avx512, brem_half_avx512};
use crate::kernel::Matrix;

/// Shortest matrix row that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per row per term, so it repays only once
/// the tile body is long enough.
pub(crate) const MATRIX_PEEL_MIN: usize = 8192;

/// Shortest row using chunk-resolved matrix coefficients.
///
/// Shorter rows read coefficients directly in each tile or lane. The crossover
/// is a measured parameter; public matrix shapes appear in `benchmarks/gf8.md`.
const MATRIX_RESOLVE_MIN: usize = 512;

/// Many sources into many rows, four rows per group over 256-byte tiles.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes and
/// every term supplies `nrows` coefficients and a `row_len`-byte source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512<C: Blocked>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[C], &[u8])],
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_matrix_avx512: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for (t, (coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_avx512: term {t} needs {nrows} coefficients"
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_avx512: term {t} source must span one row"
        );
    }
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_impl::<C, [(&[C], &[u8])], false>(rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_avx512`] with overwrite semantics: `rows[j] = sum_t
/// coeffs[t][j] * src[t]`.
///
/// Accumulators are seeded from zero in registers instead of the
/// destination, so prior destination contents are ignored; term counts above
/// one resolve chunk accumulate later chunks into what the first wrote.
///
/// # Panics
/// As [`mul_add_matrix_avx512`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_avx512<C: Blocked>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[C], &[u8])],
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_into_matrix_avx512: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for (t, (coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix_avx512: term {t} needs {nrows} coefficients"
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix_avx512: term {t} source must span one row"
        );
    }
    // Overwrite seeds accumulators from zero, so no destination bytes are
    // read; an empty term list still writes the zeroed sum.
    if terms.is_empty() {
        rows[..nrows * row_len].fill(0);
        return;
    }
    if row_len == 0 || nrows == 0 {
        return;
    }
    mul_into_matrix_avx512_with(_token, rows, row_len, nrows, terms);
}

/// [`mul_add_matrix_avx512`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_avx512`], with the coefficient count per term left
/// to the provider contract.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512_with<C: Blocked, M: Matrix<C> + ?Sized>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_matrix_avx512: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(
            terms.source(term).len(),
            row_len,
            "mul_add_matrix_avx512: term {term} source must span one row"
        );
    }
    if terms.len() == 0 || row_len == 0 || nrows == 0 {
        return;
    }
    mul_add_matrix_impl::<C, M, false>(rows, row_len, nrows, terms);
}

/// [`mul_into_matrix_avx512`] over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_avx512_with`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_avx512_with<C: Blocked, M: Matrix<C> + ?Sized>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_into_matrix_avx512: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(
            terms.source(term).len(),
            row_len,
            "mul_into_matrix_avx512: term {term} source must span one row"
        );
    }
    // Overwrite seeds accumulators from zero, so no destination bytes are
    // read; an empty term list still writes the zeroed sum.
    if terms.len() == 0 {
        if row_len != 0 && nrows != 0 {
            rows[..nrows * row_len].fill(0);
        }
        return;
    }
    if row_len == 0 || nrows == 0 {
        return;
    }
    mul_add_matrix_impl::<C, M, true>(rows, row_len, nrows, terms);
}

/// Provider-generic matrix walk shared by the 512-bit matrix entries.
///
/// With more than one row group over long rows, the walk splits the rows into
/// column blocks and runs every row group over one block before the next, so
/// each source block is reused across the groups while it is cache-resident.
/// The first block ends where row 0 reaches a 64-byte boundary, so later
/// blocks start aligned in every row whose pitch is a multiple of 64.
/// Otherwise the row groups run over whole rows.
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_matrix_impl<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    if row_len < MATRIX_RESOLVE_MIN {
        matrix_block::<C, M, OVERWRITE, true>(rows, row_len, 0, row_len, nrows, terms);
        return;
    }
    matrix_long::<C, M, OVERWRITE>(rows, row_len, nrows, terms);
}

/// Column-blocked or whole-row resolved matrix walk.
#[archmage::rite(v4x, import_intrinsics)]
#[inline(never)]
fn matrix_long<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    if !column_blocked(nrows, row_len) {
        matrix_block::<C, M, OVERWRITE, false>(rows, row_len, 0, row_len, nrows, terms);
        return;
    }
    let head = aligned_head(rows.as_ptr(), row_len);
    for (start, len) in column_blocks(row_len, head) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_block::<C, _, OVERWRITE, false>(rows, row_len, start, len, nrows, &window);
    }
}

/// Bytes from `first` to its next 64-byte boundary when rows `pitch` bytes
/// apart share that alignment, zero otherwise: the same condition under which
/// the row groups peel a head.
fn aligned_head(first: *const u8, pitch: usize) -> usize {
    if pitch.is_multiple_of(64) {
        (first as usize).wrapping_neg() & 63
    } else {
        0
    }
}

/// Provider-generic row-group walk over one column block: the
/// `block_start..block_start + block_len` span of each of `nrows` rows that
/// sit `pitch` bytes apart, against sources that span the block.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_block<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool, const DIRECT: bool>(
    rows: &mut [u8],
    pitch: usize,
    block_start: usize,
    block_len: usize,
    nrows: usize,
    terms: &M,
) {
    debug_assert!(block_start + block_len <= pitch);
    let base = rows.as_mut_ptr();
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: row `j`'s block starts at `base + j * pitch + block_start`
        //        and spans `block_len` bytes, `block_start + block_len <=
        //        pitch`, and the entry asserted `nrows * pitch <=
        //        rows.len()`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: distinct rows sit `pitch` apart in one region and each block
        //        lies inside its own row, so the four windows are pairwise
        //        disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(g * pitch + block_start),
                base.add((g + 1) * pitch + block_start),
                base.add((g + 2) * pitch + block_start),
                base.add((g + 3) * pitch + block_start),
            ]
        };
        matrix_rows4::<C, M, OVERWRITE, DIRECT>(ptrs, block_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as for the four-row group above, the blocks of rows `g` and
        //        `g + 1` lie wholly inside `rows`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: distinct rows sit `pitch` apart and each block lies inside
        //        its own row, so the two windows are disjoint.
        // THUS: no row's store conflicts with the other row's window.
        let ptrs = unsafe {
            [
                base.add(g * pitch + block_start),
                base.add((g + 1) * pitch + block_start),
            ]
        };
        matrix_rows2::<C, M, OVERWRITE, DIRECT>(ptrs, block_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, the block of row `g` lies wholly inside `rows`.
        // THUS: the staged pointer addresses an in-bounds, whole block of its
        //       row.
        let ptr = unsafe { base.add(g * pitch + block_start) };
        matrix_rows1::<C, M, OVERWRITE, DIRECT>(ptr, block_len, g, terms);
    }
}

/// Provider-generic four-row fold.
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows4<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool, const DIRECT: bool>(
    ptrs: [*mut u8; 4],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    // Head bytes run through the hierarchical tail until the first row is
    // 64-byte aligned; aligning one aligns every row when the row pitch and
    // this span's length are multiples of 64.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if row_len >= MATRIX_PEEL_MIN && head < row_len && row_len.is_multiple_of(64) {
        head
    } else {
        0
    };
    if head > 0 {
        matrix_tail::<C, M, OVERWRITE>(&ptrs, head, 0, g, terms);
    }
    let vector_len = row_len & !63;
    let tile = if DIRECT {
        let mut tile = head;
        while tile + 256 <= vector_len {
            rows_tile4::<C, M, OVERWRITE>(ptrs, tile, g, terms);
            tile += 256;
        }
        while tile + 64 <= vector_len {
            rows_lane4::<C, M, OVERWRITE>(ptrs, tile, g, terms);
            tile += 64;
        }
        tile
    } else {
        rows_resolved::<C, M, 4, OVERWRITE>(ptrs, head, vector_len, g, terms)
    };
    matrix_tail::<C, M, OVERWRITE>(&ptrs, row_len, tile, g, terms);
}

/// Provider-generic two-row fold.
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows2<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool, const DIRECT: bool>(
    ptrs: [*mut u8; 2],
    row_len: usize,
    g: usize,
    terms: &M,
) {
    // Head bytes run through the hierarchical tail until the first row is
    // 64-byte aligned; rows sit `row_len` apart, so aligning one aligns
    // every row whose pitch is a multiple of 64.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if row_len >= MATRIX_PEEL_MIN && head < row_len && row_len.is_multiple_of(64) {
        head
    } else {
        0
    };
    if head > 0 {
        matrix_tail::<C, M, OVERWRITE>(&ptrs, head, 0, g, terms);
    }
    let vector_len = row_len & !63;
    let tile = if DIRECT {
        let mut tile = head;
        while tile + 256 <= vector_len {
            rows_tile2::<C, M, OVERWRITE>(ptrs, tile, g, terms);
            tile += 256;
        }
        while tile + 64 <= vector_len {
            rows_lane2::<C, M, OVERWRITE>(ptrs, tile, g, terms);
            tile += 64;
        }
        tile
    } else {
        rows_resolved::<C, M, 2, OVERWRITE>(ptrs, head, vector_len, g, terms)
    };
    matrix_tail::<C, M, OVERWRITE>(&ptrs, row_len, tile, g, terms);
}

/// Provider-generic one-row fold.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows1<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool, const DIRECT: bool>(
    ptr: *mut u8,
    row_len: usize,
    g: usize,
    terms: &M,
) {
    let head = (ptr as usize).wrapping_neg() & 63;
    let head = if row_len >= MATRIX_PEEL_MIN && head < row_len && row_len.is_multiple_of(64) {
        head
    } else {
        0
    };
    if head > 0 {
        matrix_tail::<C, M, OVERWRITE>(&[ptr], head, 0, g, terms);
    }
    let vector_len = row_len & !63;
    let tile = if DIRECT {
        let mut tile = head;
        while tile + 256 <= vector_len {
            rows_tile1::<C, M, OVERWRITE>(ptr, tile, g, terms);
            tile += 256;
        }
        while tile + 64 <= vector_len {
            rows_lane1::<C, M, OVERWRITE>(ptr, tile, g, terms);
            tile += 64;
        }
        tile
    } else {
        rows_resolved::<C, M, 1, OVERWRITE>([ptr], head, vector_len, g, terms)
    };
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the entry staged this row in-bounds at `row_len` bytes, and
    //        `tile` is the vector-covered prefix, so `tile..row_len` is the
    //        row's own final span.
    // THUS: the tail slice lies wholly within its originating row.
    let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), row_len - tile) };
    if OVERWRITE {
        tail.fill(0);
    }
    for term in 0..terms.len() {
        let coeff = *terms.coefficient(term, g);
        if !C::is_zero(coeff) {
            brem_half_avx512(tail, coeff, &terms.source(term)[tile..]);
        }
    }
}

/// Provider-generic four-row tile.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_tile4<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: every 64-byte load and store below stays inside its slice.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = [[_mm512_setzero_si512(); 4]; 4];
        if !OVERWRITE {
            for (&ptr, slots) in ptrs.iter().zip(acc.iter_mut()) {
                for (lane, slot) in slots.iter_mut().enumerate() {
                    {
                        let (rc, _) = core::slice::from_raw_parts(ptr.add(tile + 64 * lane), 64)
                            .as_chunks::<64>();
                        *slot = _mm512_loadu_si512(&rc[0]);
                    }
                }
            }
        }
        for term in 0..terms.len() {
            let maps = [
                C::map(*terms.coefficient(term, g)),
                C::map(*terms.coefficient(term, g + 1)),
                C::map(*terms.coefficient(term, g + 2)),
                C::map(*terms.coefficient(term, g + 3)),
            ];
            let f = [
                bfactor_avx512(maps[0]),
                bfactor_avx512(maps[1]),
                bfactor_avx512(maps[2]),
                bfactor_avx512(maps[3]),
            ];
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 256].as_chunks::<64>();
            let x = [
                _mm512_loadu_si512(&s[0]),
                _mm512_loadu_si512(&s[1]),
                _mm512_loadu_si512(&s[2]),
                _mm512_loadu_si512(&s[3]),
            ];
            for (slots, &factor) in acc.iter_mut().zip(&f) {
                for (slot, &value) in slots.iter_mut().zip(&x) {
                    *slot = _mm512_xor_si512(*slot, bmul_avx512(value, factor));
                }
            }
        }
        for (&ptr, slots) in ptrs.iter().zip(&acc) {
            for (lane, &value) in slots.iter().enumerate() {
                {
                    let (rc, _) = core::slice::from_raw_parts_mut(ptr.add(tile + 64 * lane), 64)
                        .as_chunks_mut::<64>();
                    _mm512_storeu_si512(&mut rc[0], value);
                }
            }
        }
    }
}

/// Provider-generic four-row lane.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_lane4<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: the 64-byte load and each store stay inside their slices.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = [_mm512_setzero_si512(); 4];
        if !OVERWRITE {
            for (&ptr, slot) in ptrs.iter().zip(acc.iter_mut()) {
                {
                    let (rc, _) = core::slice::from_raw_parts(ptr.add(tile), 64).as_chunks::<64>();
                    *slot = _mm512_loadu_si512(&rc[0]);
                }
            }
        }
        for term in 0..terms.len() {
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 64].as_chunks::<64>();
            let x = _mm512_loadu_si512(&s[0]);
            for (row, slot) in acc.iter_mut().enumerate() {
                let map = C::map(*terms.coefficient(term, g + row));
                *slot = _mm512_xor_si512(*slot, bmul_avx512(x, bfactor_avx512(map)));
            }
        }
        for (&ptr, &value) in ptrs.iter().zip(&acc) {
            {
                let (rc, _) =
                    core::slice::from_raw_parts_mut(ptr.add(tile), 64).as_chunks_mut::<64>();
                _mm512_storeu_si512(&mut rc[0], value);
            }
        }
    }
}

/// Provider-generic two-row tile.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_tile2<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 2],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: every 64-byte load and store below stays inside its slice.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = [[_mm512_setzero_si512(); 4]; 2];
        if !OVERWRITE {
            for (&ptr, slots) in ptrs.iter().zip(acc.iter_mut()) {
                for (lane, slot) in slots.iter_mut().enumerate() {
                    {
                        let (rc, _) = core::slice::from_raw_parts(ptr.add(tile + 64 * lane), 64)
                            .as_chunks::<64>();
                        *slot = _mm512_loadu_si512(&rc[0]);
                    }
                }
            }
        }
        for term in 0..terms.len() {
            let maps = [
                C::map(*terms.coefficient(term, g)),
                C::map(*terms.coefficient(term, g + 1)),
            ];
            let f = [bfactor_avx512(maps[0]), bfactor_avx512(maps[1])];
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 256].as_chunks::<64>();
            let x = [
                _mm512_loadu_si512(&s[0]),
                _mm512_loadu_si512(&s[1]),
                _mm512_loadu_si512(&s[2]),
                _mm512_loadu_si512(&s[3]),
            ];
            for (slots, &factor) in acc.iter_mut().zip(&f) {
                for (slot, &value) in slots.iter_mut().zip(&x) {
                    *slot = _mm512_xor_si512(*slot, bmul_avx512(value, factor));
                }
            }
        }
        for (&ptr, slots) in ptrs.iter().zip(&acc) {
            for (lane, &value) in slots.iter().enumerate() {
                {
                    let (rc, _) = core::slice::from_raw_parts_mut(ptr.add(tile + 64 * lane), 64)
                        .as_chunks_mut::<64>();
                    _mm512_storeu_si512(&mut rc[0], value);
                }
            }
        }
    }
}

/// Provider-generic two-row lane.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_lane2<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 2],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: the 64-byte load and each store stay inside their slices.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = [_mm512_setzero_si512(); 2];
        if !OVERWRITE {
            for (&ptr, slot) in ptrs.iter().zip(acc.iter_mut()) {
                {
                    let (rc, _) = core::slice::from_raw_parts(ptr.add(tile), 64).as_chunks::<64>();
                    *slot = _mm512_loadu_si512(&rc[0]);
                }
            }
        }
        for term in 0..terms.len() {
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 64].as_chunks::<64>();
            let x = _mm512_loadu_si512(&s[0]);
            for (row, slot) in acc.iter_mut().enumerate() {
                let map = C::map(*terms.coefficient(term, g + row));
                *slot = _mm512_xor_si512(*slot, bmul_avx512(x, bfactor_avx512(map)));
            }
        }
        for (&ptr, &value) in ptrs.iter().zip(&acc) {
            {
                let (rc, _) =
                    core::slice::from_raw_parts_mut(ptr.add(tile), 64).as_chunks_mut::<64>();
                _mm512_storeu_si512(&mut rc[0], value);
            }
        }
    }
}

/// One row by one 256-byte tile.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_tile1<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptr: *mut u8,
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: every 64-byte load and store below stays inside its slice.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = [_mm512_setzero_si512(); 4];
        if !OVERWRITE {
            for (lane, slot) in acc.iter_mut().enumerate() {
                {
                    let (rc, _) = core::slice::from_raw_parts(ptr.add(tile + 64 * lane), 64)
                        .as_chunks::<64>();
                    *slot = _mm512_loadu_si512(&rc[0]);
                }
            }
        }
        for term in 0..terms.len() {
            let factor = bfactor_avx512(C::map(*terms.coefficient(term, g)));
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 256].as_chunks::<64>();
            acc[0] = _mm512_xor_si512(acc[0], bmul_avx512(_mm512_loadu_si512(&s[0]), factor));
            acc[1] = _mm512_xor_si512(acc[1], bmul_avx512(_mm512_loadu_si512(&s[1]), factor));
            acc[2] = _mm512_xor_si512(acc[2], bmul_avx512(_mm512_loadu_si512(&s[2]), factor));
            acc[3] = _mm512_xor_si512(acc[3], bmul_avx512(_mm512_loadu_si512(&s[3]), factor));
        }
        for (lane, &value) in acc.iter().enumerate() {
            {
                let (rc, _) = core::slice::from_raw_parts_mut(ptr.add(tile + 64 * lane), 64)
                    .as_chunks_mut::<64>();
                _mm512_storeu_si512(&mut rc[0], value);
            }
        }
    }
}

/// One row by one 64-byte lane.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_lane1<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptr: *mut u8,
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        so each destination window lies inside its row. Source windows
    //        use checked slice indexing on the captured provider slice.
    // THUS: the 64-byte load and each store stay inside their slices.
    //
    // ALIASING
    // SINCE: the rows are pairwise disjoint, as staged by the caller.
    // THUS: no row's store conflicts with another row's window.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        if !OVERWRITE {
            {
                let (rc, _) = core::slice::from_raw_parts(ptr.add(tile), 64).as_chunks::<64>();
                acc = _mm512_loadu_si512(&rc[0]);
            }
        }
        for term in 0..terms.len() {
            let factor = bfactor_avx512(C::map(*terms.coefficient(term, g)));
            let src = terms.source(term);
            let (s, _) = src[tile..tile + 64].as_chunks::<64>();
            acc = _mm512_xor_si512(acc, bmul_avx512(_mm512_loadu_si512(&s[0]), factor));
        }
        {
            let (rc, _) = core::slice::from_raw_parts_mut(ptr.add(tile), 64).as_chunks_mut::<64>();
            _mm512_storeu_si512(&mut rc[0], acc);
        }
    }
}

/// Resolve a row group's coefficients chunk by chunk and run the 256-byte
/// tile and 64-byte lane loops over each chunk, from `head` up to the last
/// whole lane below `vector_len`.
///
/// Returns the end of the vector-covered span; the caller finishes the rest
/// from the terms themselves. The first chunk carries the caller's overwrite
/// policy; every later chunk accumulates into what the earlier ones wrote.
/// `head <= vector_len`.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_resolved<C: Blocked, M: Matrix<C> + ?Sized, const ROWS: usize, const OVERWRITE: bool>(
    ptrs: [*mut u8; ROWS],
    head: usize,
    vector_len: usize,
    g: usize,
    terms: &M,
) -> usize {
    let end = head + ((vector_len - head) & !63);
    if end == head {
        return head;
    }
    let count = terms.len();
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
                *word = C::map(*terms.coefficient(term, g + row));
            }
            maps[offset].write(words);
            // The body reads the staged slice without bounds checks, so the
            // captured slice itself must cover the span; the provider may
            // return a different slice than the one the entry validated.
            let src = terms.source(term);
            assert!(
                src.len() >= end,
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
            rows_body::<ROWS, true>(ptrs, head, end, maps, srcs);
        } else {
            rows_body::<ROWS, false>(ptrs, head, end, maps, srcs);
        }
        first = false;
        start += taken;
        if start >= count {
            break;
        }
    }
    end
}

/// Fold one chunk of resolved terms into `ROWS` rows over `head..end`: four
/// 64-byte lanes per row per tile, then whole 64-byte lanes.
///
/// `maps[term][row]` is the resolved map word, so the loops walk the map
/// array and the source array in lockstep and index neither. Terms fold two
/// at a time, so each accumulator takes one three-input XOR per pair; an odd
/// final term folds alone. `end - head` is a multiple of 64.
///
/// The rows are addressed as runtime offsets into one region; the entry that
/// staged them asserted each row in-bounds and pairwise disjoint, and
/// `rows_resolved` asserted every staged source spans at least `end` bytes.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_body<const ROWS: usize, const OVERWRITE: bool>(
    ptrs: [*mut u8; ROWS],
    head: usize,
    end: usize,
    maps: &[[u64; ROWS]],
    srcs: &[&[u8]],
) {
    let (map_pairs, map_odd) = maps.as_chunks::<2>();
    let (src_pairs, src_odd) = srcs.as_chunks::<2>();
    let mut tile = head;
    while tile + 256 <= end {
        // Seeding from zero must stay a plain array initializer: a seeding
        // loop over the accumulator array keeps LLVM from promoting it out of
        // memory, and the overwrite path then spills a tile per source.
        let mut acc = [[_mm512_setzero_si512(); 4]; ROWS];
        if !OVERWRITE {
            for (&ptr, slots) in ptrs.iter().zip(acc.iter_mut()) {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `tile + 256 <= end` and every row spans at least
                //        `end` bytes.
                // THUS: the 256-byte window lies inside its originating row.
                //
                // ALIASING
                // SINCE: the rows are pairwise disjoint, as staged by the
                //        caller, and the shared window is released before
                //        any store below.
                // THUS: the window aliases no live mutable borrow.
                let (rc, _) =
                    unsafe { core::slice::from_raw_parts(ptr.add(tile), 256) }.as_chunks::<64>();
                for (slot, chunk) in slots.iter_mut().zip(rc) {
                    *slot = _mm512_loadu_si512(chunk);
                }
            }
        }
        for ([map0, map1], [src0, src1]) in map_pairs.iter().zip(src_pairs) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 256 <= end` and every source spans at least
            //        `end` bytes.
            // THUS: each 256-byte window lies inside its originating source.
            let (s0, s1) = unsafe {
                (
                    core::slice::from_raw_parts(src0.as_ptr().add(tile), 256),
                    core::slice::from_raw_parts(src1.as_ptr().add(tile), 256),
                )
            };
            let (s0, _) = s0.as_chunks::<64>();
            let (s1, _) = s1.as_chunks::<64>();
            let x0 = [
                _mm512_loadu_si512(&s0[0]),
                _mm512_loadu_si512(&s0[1]),
                _mm512_loadu_si512(&s0[2]),
                _mm512_loadu_si512(&s0[3]),
            ];
            let x1 = [
                _mm512_loadu_si512(&s1[0]),
                _mm512_loadu_si512(&s1[1]),
                _mm512_loadu_si512(&s1[2]),
                _mm512_loadu_si512(&s1[3]),
            ];
            for ((slots, &word0), &word1) in acc.iter_mut().zip(map0).zip(map1) {
                let (f0, f1) = (bfactor_avx512(word0), bfactor_avx512(word1));
                for ((slot, &a), &b) in slots.iter_mut().zip(&x0).zip(&x1) {
                    let pair = _mm512_xor_si512(bmul_avx512(a, f0), bmul_avx512(b, f1));
                    *slot = _mm512_xor_si512(*slot, pair);
                }
            }
        }
        for (map, src) in map_odd.iter().zip(src_odd) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 256 <= end` and every source spans at least
            //        `end` bytes.
            // THUS: the 256-byte window lies inside its originating source.
            let (s, _) = unsafe { core::slice::from_raw_parts(src.as_ptr().add(tile), 256) }
                .as_chunks::<64>();
            let x = [
                _mm512_loadu_si512(&s[0]),
                _mm512_loadu_si512(&s[1]),
                _mm512_loadu_si512(&s[2]),
                _mm512_loadu_si512(&s[3]),
            ];
            for (slots, &word) in acc.iter_mut().zip(map) {
                let f = bfactor_avx512(word);
                for (slot, &value) in slots.iter_mut().zip(&x) {
                    *slot = _mm512_xor_si512(*slot, bmul_avx512(value, f));
                }
            }
        }
        for (&ptr, slots) in ptrs.iter().zip(&acc) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the same `tile + 256 <= end` relation that bounded the
            //        loads bounds this window.
            // THUS: the 256-byte window lies inside its originating row.
            //
            // ALIASING
            // SINCE: the rows are pairwise disjoint, and the borrow is
            //        released before the next row's window is formed.
            // THUS: the mutable window aliases no other live row window.
            let (rc, _) = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), 256) }
                .as_chunks_mut::<64>();
            for (chunk, &value) in rc.iter_mut().zip(slots) {
                _mm512_storeu_si512(chunk, value);
            }
        }
        tile += 256;
    }
    while tile + 64 <= end {
        let mut acc = [_mm512_setzero_si512(); ROWS];
        if !OVERWRITE {
            for (&ptr, slot) in ptrs.iter().zip(acc.iter_mut()) {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `tile + 64 <= end` and every row spans at least
                //        `end` bytes.
                // THUS: the 64-byte window lies inside its originating row.
                //
                // ALIASING
                // SINCE: the rows are pairwise disjoint, and the shared
                //        window is released before any store below.
                // THUS: the window aliases no live mutable borrow.
                let (rc, _) =
                    unsafe { core::slice::from_raw_parts(ptr.add(tile), 64) }.as_chunks::<64>();
                *slot = _mm512_loadu_si512(&rc[0]);
            }
        }
        for ([map0, map1], [src0, src1]) in map_pairs.iter().zip(src_pairs) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 64 <= end` and every source spans at least
            //        `end` bytes.
            // THUS: each 64-byte window lies inside its originating source.
            let (s0, s1) = unsafe {
                (
                    core::slice::from_raw_parts(src0.as_ptr().add(tile), 64),
                    core::slice::from_raw_parts(src1.as_ptr().add(tile), 64),
                )
            };
            let x0 = _mm512_loadu_si512(&s0.as_chunks::<64>().0[0]);
            let x1 = _mm512_loadu_si512(&s1.as_chunks::<64>().0[0]);
            for ((slot, &word0), &word1) in acc.iter_mut().zip(map0).zip(map1) {
                let pair = _mm512_xor_si512(
                    bmul_avx512(x0, bfactor_avx512(word0)),
                    bmul_avx512(x1, bfactor_avx512(word1)),
                );
                *slot = _mm512_xor_si512(*slot, pair);
            }
        }
        for (map, src) in map_odd.iter().zip(src_odd) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 64 <= end` and every source spans at least
            //        `end` bytes.
            // THUS: the 64-byte window lies inside its originating source.
            let (s, _) = unsafe { core::slice::from_raw_parts(src.as_ptr().add(tile), 64) }
                .as_chunks::<64>();
            let x = _mm512_loadu_si512(&s[0]);
            for (slot, &word) in acc.iter_mut().zip(map) {
                *slot = _mm512_xor_si512(*slot, bmul_avx512(x, bfactor_avx512(word)));
            }
        }
        for (&ptr, &value) in ptrs.iter().zip(&acc) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 64 <= end`, the same relation that bounded the
            //        loads.
            // THUS: the 64-byte window lies inside its originating row.
            //
            // ALIASING
            // SINCE: the rows are pairwise disjoint, and the borrow is
            //        released before the next row's window is formed.
            // THUS: the mutable window aliases no other live row window.
            let (rc, _) =
                unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), 64) }.as_chunks_mut::<64>();
            _mm512_storeu_si512(&mut rc[0], value);
        }
        tile += 64;
    }
}

/// Hierarchical remainder shared by the 512-bit matrix row groups.
///
/// Complete 64-byte lanes stay on the one-instruction multiply; the rest
/// descends through the AVX2 ladder to the scalar nibble kernel.
///
/// Each row's tail slice is formed from the row's runtime offset; the entry
/// that staged the rows asserted each row in-bounds and pairwise disjoint
/// (`check_scattered` for `_at` rows, `nrows * row_len` plus `row_len` spacing
/// for contiguous rows), `tile` is the vector-covered prefix, and only one
/// tail borrow is live at a time.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_tail<C: Blocked, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: &[*mut u8],
    row_len: usize,
    tile: usize,
    g: usize,
    terms: &M,
) {
    if tile == row_len {
        return;
    }
    for (slot, &row) in ptrs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + (row_len - tile) == row_len`, and the entry that
        //        staged the rows asserted each row spans `row_len` bytes
        //        inside one region, so this is the row's own final span.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the staged row windows are pairwise disjoint, each lying
        //        inside its own checked row, and the borrow is released
        //        before the next row's tail is formed.
        // THUS: the mutable borrow aliases no other live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(row.add(tile), row_len - tile) };
        if OVERWRITE {
            tail.fill(0);
        }
        for term in 0..terms.len() {
            let coeff = *terms.coefficient(term, g + slot);
            if !C::is_zero(coeff) {
                brem_half_avx512(tail, coeff, &terms.source(term)[tile..row_len]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scattered rows (`_at`): destination rows at explicit offsets.
// ---------------------------------------------------------------------------

/// [`mul_add_matrix_avx512`] with destination rows at explicit offsets.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, the rows are
/// pairwise disjoint, and every term supplies `row_starts.len()`
/// coefficients for a `row_len`-byte source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_at_avx512<C: Blocked>(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[C], &[u8])],
) {
    check_scattered("mul_add_matrix_at_avx512", dst.len(), row_len, row_starts);
    for (t, (coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            row_starts.len(),
            "mul_add_matrix_at_avx512: term {t} needs {} coefficients",
            row_starts.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_at_avx512: term {t} source must span one row"
        );
    }
    if terms.is_empty() || row_starts.is_empty() {
        return;
    }
    mul_add_matrix_at_impl(dst, row_len, row_starts, terms);
}

/// Scattered matrix walk: the same column blocking as
/// [`mul_add_matrix_impl`], with each row starting at an arbitrary disjoint
/// offset and the first block aligning the first row.
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_matrix_at_impl<C: Blocked>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[C], &[u8])],
) {
    if row_len < MATRIX_RESOLVE_MIN {
        matrix_at_block::<C, [(&[C], &[u8])], true>(dst, row_len, 0, row_len, row_starts, terms);
        return;
    }
    if !column_blocked(row_starts.len(), row_len) {
        matrix_at_block::<C, [(&[C], &[u8])], false>(dst, row_len, 0, row_len, row_starts, terms);
        return;
    }
    let head = aligned_head(dst.as_ptr().wrapping_add(row_starts[0]), row_len);
    for (start, len) in column_blocks(row_len, head) {
        let window = ColumnWindow::new(terms, start, len);
        matrix_at_block::<C, _, false>(dst, row_len, start, len, row_starts, &window);
    }
}

/// Scattered row-group walk over one column block: the
/// `block_start..block_start + block_len` span of each `row_len`-byte row
/// starting at `row_starts[j]`, against sources that span the block.
///
/// Residue: the rows are addressed at runtime offsets the caller proved
/// in-bounds and pairwise disjoint, which no safe primitive expresses.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_at_block<C: Blocked, M: Matrix<C> + ?Sized, const DIRECT: bool>(
    dst: &mut [u8],
    row_len: usize,
    block_start: usize,
    block_len: usize,
    row_starts: &[usize],
    terms: &M,
) {
    debug_assert!(block_start + block_len <= row_len);
    let base = dst.as_mut_ptr();
    let nrows = row_starts.len();
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: each `row_starts[j] + row_len <= dst.len()` was asserted by
        //        the entry, and `block_start + block_len <= row_len`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint, and each
        //        block lies inside its own row.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(row_starts[g] + block_start),
                base.add(row_starts[g + 1] + block_start),
                base.add(row_starts[g + 2] + block_start),
                base.add(row_starts[g + 3] + block_start),
            ]
        };
        matrix_rows4::<C, M, false, DIRECT>(ptrs, block_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as for the four-row group above, the blocks of rows `g` and
        //        `g + 1` lie wholly inside `dst`.
        // THUS: each staged pointer addresses an in-bounds, whole block of
        //       its row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint, and each
        //        block lies inside its own row.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(row_starts[g] + block_start),
                base.add(row_starts[g + 1] + block_start),
            ]
        };
        matrix_rows2::<C, M, false, DIRECT>(ptrs, block_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, the block of row `g` lies wholly inside `dst`.
        // THUS: the staged pointer addresses an in-bounds, whole block of its
        //       row.
        let ptr = unsafe { base.add(row_starts[g] + block_start) };
        matrix_rows1::<C, M, false, DIRECT>(ptr, block_len, g, terms);
    }
}
