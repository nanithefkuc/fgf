//! Multi-row crossed panel: resolved and externally-resolved, chunked.
//!
//! Measurement scaffolding. These extend the one-output resolve contrasts in
//! `resolve` to every output count, over the production row-group walk and
//! its group-major map order.

use super::super::Affine8D;
use super::super::rows::rows_resolved_chunked;
use super::rows_external_chunked;
use crate::field::gf8d;

/// Experimental multi-row overwrite matrix on the production resolved path
/// with a selectable resolve-chunk width.
///
/// Same grouping (four rows at a 64-byte tile, then a pair and a single row at
/// 128 bytes), resolution and remainder as the production `matrix_overwrite`
/// path; `chunk` 32 is production by construction. Paired against
/// [`mul_into_matrix_external_grouped_8d`] it extends the one-output crossed
/// panel to every output count. Benchmark evidence only; production dispatch
/// is unchanged.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes, every term
/// supplies `nrows` coefficients over a `row_len`-byte source, and `chunk` is
/// 32, 64 or 96.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_chunk_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
    chunk: usize,
) {
    type Terms<'a> = [(&'a [gf8d::Elem], &'a [u8])];
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|used| used <= rows.len()),
        "matrix_overwrite_chunk: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for (coeffs, src) in terms {
        assert_eq!(src.len(), row_len);
        assert_eq!(
            coeffs.len(),
            nrows,
            "matrix_overwrite_chunk: term needs {nrows} coefficients"
        );
    }
    assert!(
        matches!(chunk, 32 | 64 | 96),
        "matrix_overwrite_chunk: chunk must be 32, 64 or 96"
    );
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the asserts above bound `nrows * row_len <= rows.len()`, so
    //        every staged row pointer addresses a whole in-bounds row.
    // THUS: each group walk stays inside `rows`.
    //
    // ALIASING
    // SINCE: distinct rows sit `row_len` apart in one region, and the row
    //        span was asserted.
    // THUS: the grouped row windows are disjoint.
    unsafe {
        let mut g = 0;
        while g + 4 <= nrows {
            let ptrs = [
                base.add(g * row_len),
                base.add((g + 1) * row_len),
                base.add((g + 2) * row_len),
                base.add((g + 3) * row_len),
            ];
            match chunk {
                32 => rows_resolved_chunked::<Affine8D, Terms, 4, 2, true, 32, false>(
                    ptrs, row_len, g, terms,
                ),
                64 => rows_resolved_chunked::<Affine8D, Terms, 4, 2, true, 64, false>(
                    ptrs, row_len, g, terms,
                ),
                _ => rows_resolved_chunked::<Affine8D, Terms, 4, 2, true, 96, false>(
                    ptrs, row_len, g, terms,
                ),
            }
            g += 4;
        }
        if g + 2 <= nrows {
            let ptrs = [base.add(g * row_len), base.add((g + 1) * row_len)];
            match chunk {
                32 => rows_resolved_chunked::<Affine8D, Terms, 2, 4, true, 32, false>(
                    ptrs, row_len, g, terms,
                ),
                64 => rows_resolved_chunked::<Affine8D, Terms, 2, 4, true, 64, false>(
                    ptrs, row_len, g, terms,
                ),
                _ => rows_resolved_chunked::<Affine8D, Terms, 2, 4, true, 96, false>(
                    ptrs, row_len, g, terms,
                ),
            }
            g += 2;
        }
        if g < nrows {
            let ptrs = [base.add(g * row_len)];
            match chunk {
                32 => rows_resolved_chunked::<Affine8D, Terms, 1, 4, true, 32, false>(
                    ptrs, row_len, g, terms,
                ),
                64 => rows_resolved_chunked::<Affine8D, Terms, 1, 4, true, 64, false>(
                    ptrs, row_len, g, terms,
                ),
                _ => rows_resolved_chunked::<Affine8D, Terms, 1, 4, true, 96, false>(
                    ptrs, row_len, g, terms,
                ),
            }
        }
    }
}

/// Experimental multi-row overwrite matrix running the production row-group
/// bodies over externally resolved map words, chunked at `chunk` terms.
///
/// `maps` is group-major: every four-row group's words first, then the
/// pair's, then the single row's, each group term-major (`maps[term * rows +
/// row within group]`). `chunk` 0 folds all terms in one pass. This is the
/// resolution-free half of the multi-row crossed panel.
/// Benchmark evidence only; production dispatch is unchanged.
///
/// # Panics
/// Panics unless `row_len` is a 32-byte multiple, `rows` holds `nrows` rows
/// of `row_len` bytes, `maps` holds `nrows * srcs.len()` words in group-major
/// order, every source spans `row_len` bytes, and `chunk` is 0, 32, 64 or 96.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_external_grouped_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    maps: &[u64],
    srcs: &[&[u8]],
    chunk: usize,
) {
    assert_eq!(
        row_len % 32,
        0,
        "matrix_overwrite_external_grouped: needs a 32-byte-multiple row"
    );
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|used| used <= rows.len()),
        "matrix_overwrite_external_grouped: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    assert_eq!(
        maps.len(),
        nrows * srcs.len(),
        "matrix_overwrite_external_grouped: one map word per (term, row)"
    );
    for src in srcs {
        assert_eq!(src.len(), row_len);
    }
    assert!(
        matches!(chunk, 0 | 32 | 64 | 96),
        "matrix_overwrite_external_grouped: chunk must be 0, 32, 64 or 96"
    );
    let terms = srcs.len();
    if terms == 0 {
        // The empty overwrite sum is zero, as the resolved path's tile loops
        // produce.
        rows[..nrows * row_len].fill(0);
        return;
    }
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the asserts above bound `nrows * row_len <= rows.len()`, so
    //        every staged row pointer addresses a whole in-bounds row, and
    //        `maps` holds one word per (term, row) in the exact order each
    //        group walk reads.
    // THUS: each group walk stays inside `rows` and every map read is
    //       in-bounds.
    //
    // ALIASING
    // SINCE: distinct rows sit `row_len` apart in one region, and the row
    //        span was asserted.
    // THUS: the grouped row windows are disjoint.
    unsafe {
        let mut cursor = 0;
        let mut g = 0;
        while g + 4 <= nrows {
            let (group, rest) = maps[cursor..cursor + terms * 4].as_chunks::<4>();
            debug_assert!(rest.is_empty());
            rows_external_chunked::<4, 2>(
                [
                    base.add(g * row_len),
                    base.add((g + 1) * row_len),
                    base.add((g + 2) * row_len),
                    base.add((g + 3) * row_len),
                ],
                row_len,
                group,
                srcs,
                chunk,
            );
            cursor += terms * 4;
            g += 4;
        }
        if g + 2 <= nrows {
            let (group, rest) = maps[cursor..cursor + terms * 2].as_chunks::<2>();
            debug_assert!(rest.is_empty());
            rows_external_chunked::<2, 4>(
                [base.add(g * row_len), base.add((g + 1) * row_len)],
                row_len,
                group,
                srcs,
                chunk,
            );
            cursor += terms * 2;
            g += 2;
        }
        if g < nrows {
            let (group, rest) = maps[cursor..cursor + terms].as_chunks::<1>();
            debug_assert!(rest.is_empty());
            rows_external_chunked::<1, 4>([base.add(g * row_len)], row_len, group, srcs, chunk);
        }
    }
}
