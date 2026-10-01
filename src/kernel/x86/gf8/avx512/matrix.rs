//! `Gf8D` AVX-512 affine matrix kernels (`VGF2P8AFFINEQB`, 64-byte lanes).
//!
//! Many sources into many rows for the Reed–Solomon field GF(2^8)/`0x11D`,
//! mirroring the AVX2 affine blocked shapes at 64-byte lanes: four-row
//! groups over 256-byte tiles, accumulators in registers across all terms.
//! The immediate is the XOR constant, so `<0>` selects the pure linear map;
//! each factor replicates one 64-bit map qword to all eight 64-bit lanes.
//!
//! The kernels dispatch on the `V4x` tier under the `simd512` feature and stay
//! reachable through the `internals` facade for differential tests and
//! measurement on AVX-512 hardware.

use super::super::check_scattered;
use super::super::gfni::mul_add_gfni_8d_impl;
use super::{MapCoeff, bfactor_avx512, bmul_avx512};
use crate::kernel::Matrix;

/// Shortest matrix row that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per row per term, so it repays only once
/// the tile body is long enough.
pub(crate) const MATRIX_PEEL_MIN: usize = 8192;

/// Many sources into many rows, four rows per group over 256-byte tiles.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes and
/// every term supplies `nrows` coefficients and a `row_len`-byte source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512<C: MapCoeff>(
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
/// destination, so the destination is written once with no read.
///
/// # Panics
/// As [`mul_add_matrix_avx512`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_avx512<C: MapCoeff>(
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
pub fn mul_add_matrix_avx512_with<C: MapCoeff, M: Matrix<C> + ?Sized>(
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
pub fn mul_into_matrix_avx512_with<C: MapCoeff, M: Matrix<C> + ?Sized>(
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

/// Provider-generic row-group walk shared by the 512-bit matrix entries.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_matrix_impl<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    let base = rows.as_mut_ptr();
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: row `j` starts at `base + j * row_len` and spans `row_len`
        //        bytes, and the entry asserted `nrows * row_len <=
        //        rows.len()`.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: distinct rows sit `row_len` apart in one region, so the four
        //        windows are pairwise disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(g * row_len),
                base.add((g + 1) * row_len),
                base.add((g + 2) * row_len),
                base.add((g + 3) * row_len),
            ]
        };
        matrix_rows4::<C, M, OVERWRITE>(ptrs, row_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as for the four-row group above, rows `g` and `g + 1` lie
        //        wholly inside `rows`.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: distinct rows sit `row_len` apart, so the two windows are
        //        disjoint.
        // THUS: no row's store conflicts with the other row's window.
        let ptrs = unsafe { [base.add(g * row_len), base.add((g + 1) * row_len)] };
        matrix_rows2::<C, M, OVERWRITE>(ptrs, row_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, row `g` lies wholly inside `rows`.
        // THUS: the staged pointer addresses an in-bounds, whole row.
        let ptr = unsafe { base.add(g * row_len) };
        matrix_rows1::<C, M, OVERWRITE>(ptr, row_len, g, terms);
    }
}

/// Provider-generic four-row fold.
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows4<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
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
    let mut tile = head;
    while tile + 256 <= vector_len {
        rows_tile4::<C, M, OVERWRITE>(ptrs, tile, g, terms);
        tile += 256;
    }
    while tile + 64 <= vector_len {
        rows_lane4::<C, M, OVERWRITE>(ptrs, tile, g, terms);
        tile += 64;
    }
    matrix_tail::<C, M, OVERWRITE>(&ptrs, row_len, tile, g, terms);
}

/// Provider-generic two-row fold.
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows2<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
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
    let mut tile = head;
    while tile + 256 <= vector_len {
        rows_tile2::<C, M, OVERWRITE>(ptrs, tile, g, terms);
        tile += 256;
    }
    while tile + 64 <= vector_len {
        rows_lane2::<C, M, OVERWRITE>(ptrs, tile, g, terms);
        tile += 64;
    }
    matrix_tail::<C, M, OVERWRITE>(&ptrs, row_len, tile, g, terms);
}

/// Provider-generic one-row fold.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn matrix_rows1<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
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
    let mut tile = head;

    while tile + 256 <= vector_len {
        rows_tile1::<C, M, OVERWRITE>(ptr, tile, g, terms);
        tile += 256;
    }
    while tile + 64 <= vector_len {
        rows_lane1::<C, M, OVERWRITE>(ptr, tile, g, terms);
        tile += 64;
    }
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
            mul_add_gfni_8d_impl(
                tail,
                C::map(coeff),
                C::table(coeff),
                &terms.source(term)[tile..],
            );
        }
    }
}

/// Provider-generic four-row tile.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn rows_tile4<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 256`
    //        lies inside each row and each source.
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
fn rows_lane4<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 4],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 64`
    //        lies inside each row and each source.
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
fn rows_tile2<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 2],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 256`
    //        lies inside each row and each source.
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
fn rows_lane2<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptrs: [*mut u8; 2],
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 64`
    //        lies inside each row and each source.
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
fn rows_tile1<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptr: *mut u8,
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 256 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 256`
    //        lies inside each row and each source.
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
fn rows_lane1<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
    ptr: *mut u8,
    tile: usize,
    g: usize,
    terms: &M,
) {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `tile + 64 <= vector_len <= row_len` bounds every row window,
    //        and every source spans `row_len` bytes, so `tile..tile + 64`
    //        lies inside each row and each source.
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
fn matrix_tail<C: MapCoeff, M: Matrix<C> + ?Sized, const OVERWRITE: bool>(
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
        // SINCE: the rows are `row_len` apart and disjoint, and the borrow is
        //        released before the next row's tail is formed.
        // THUS: the mutable borrow aliases no other live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(row.add(tile), row_len - tile) };
        if OVERWRITE {
            tail.fill(0);
        }
        for term in 0..terms.len() {
            let coeff = *terms.coefficient(term, g + slot);
            if !C::is_zero(coeff) {
                mul_add_gfni_8d_impl(
                    tail,
                    C::map(coeff),
                    C::table(coeff),
                    &terms.source(term)[tile..row_len],
                );
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
pub fn mul_add_matrix_at_avx512<C: MapCoeff>(
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

/// Scattered row-group walk: the same blocking as [`mul_add_matrix_impl`],
/// with each row starting at an arbitrary disjoint offset.
///
/// Residue: the rows are addressed at runtime offsets the caller proved
/// in-bounds and pairwise disjoint, which no safe primitive expresses.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_matrix_at_impl<C: MapCoeff>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[C], &[u8])],
) {
    let base = dst.as_mut_ptr();
    let nrows = row_starts.len();
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: each `row_starts[j] + row_len <= dst.len()` was asserted by
        //        the entry.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe {
            [
                base.add(row_starts[g]),
                base.add(row_starts[g + 1]),
                base.add(row_starts[g + 2]),
                base.add(row_starts[g + 3]),
            ]
        };
        matrix_rows4::<C, [(&[C], &[u8])], false>(ptrs, row_len, g, terms);
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as for the four-row group above, rows `g` and `g + 1` lie
        //        wholly inside `dst`.
        // THUS: each staged pointer addresses an in-bounds, whole row.
        //
        // ALIASING
        // SINCE: the entry asserted every pair of rows disjoint.
        // THUS: no row's store conflicts with another row's window.
        let ptrs = unsafe { [base.add(row_starts[g]), base.add(row_starts[g + 1])] };
        matrix_rows2::<C, [(&[C], &[u8])], false>(ptrs, row_len, g, terms);
        g += 2;
    }
    if g < nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, row `g` lies wholly inside `dst`.
        // THUS: the staged pointer addresses an in-bounds, whole row.
        let ptr = unsafe { base.add(row_starts[g]) };
        matrix_rows1::<C, [(&[C], &[u8])], false>(ptr, row_len, g, terms);
    }
}

/// A row-major coefficient matrix over already-prepared coefficients, in the
/// same term-major order a [`crate::ops::CoeffMatrix`] stores: index
/// `term * nrows + row`.
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
struct PreparedMatrix<'a> {
    /// Prepared coefficients, `terms * nrows` entries.
    prepared: &'a [crate::kernel::gf8::Prepared8D],
    /// Destination row count.
    nrows: usize,
    /// Source buffer of each term.
    sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
impl crate::kernel::Matrix<crate::kernel::gf8::Prepared8D> for PreparedMatrix<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }
    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &crate::kernel::gf8::Prepared8D {
        &self.prepared[term * self.nrows + row]
    }
    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

/// [`mul_add_matrix_gfni_8d_with`](super::super::mul_add_matrix_gfni_8d_with) over prepared
/// coefficients: the tile loop reads each term's stored affine map instead of
/// re-deriving it per tile.
///
/// # Panics
/// As [`mul_add_matrix_gfni_8d_with`](super::super::mul_add_matrix_gfni_8d_with).
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512_8d_with(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    prepared: &[crate::kernel::gf8::Prepared8D],
    srcs: &[&[u8]],
) {
    let terms = PreparedMatrix {
        prepared,
        nrows,
        sources: srcs,
    };
    mul_add_matrix_avx512_with(_token, rows, row_len, nrows, &terms);
}

/// [`mul_add_matrix_avx512_8d_with`] with overwrite semantics.
///
/// # Panics
/// As [`mul_add_matrix_avx512_8d_with`].
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_avx512_8d_with(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    prepared: &[crate::kernel::gf8::Prepared8D],
    srcs: &[&[u8]],
) {
    let terms = PreparedMatrix {
        prepared,
        nrows,
        sources: srcs,
    };
    mul_into_matrix_avx512_with(_token, rows, row_len, nrows, &terms);
}
