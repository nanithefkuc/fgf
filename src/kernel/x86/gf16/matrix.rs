//! Every source into every destination row.
//!
//! The register-blocked shape: a destination tile is loaded into
//! accumulators once, every term of a block is folded in, and the tile is
//! stored once, so destination traffic is one read/write per tile per
//! *block* rather than per term. How many terms a block holds is what the
//! coefficient state costs — two broadcast words under GFNI, four nibble
//! tables under the shuffle strategies.
//!
//! The group bodies are the sanctioned offset-addressed-rows residue: the
//! borrow checker cannot see that `base + k * row_len` windows of one region
//! are disjoint, so each body is an `unsafe fn` taking the geometry the safe
//! entrypoint established, with a per-item `#[allow(unsafe_code)]`.

use crate::field::gf16::Elem;
use crate::kernel::Matrix;
use crate::kernel::gf16::{factor_tables, mul_add_scalar};
use crate::kernel::tables::{ScaleTable, TowerCoeff};

use super::{
    LaneAvx2, NibbleSsse3, TERM_TILE, TableCoefficient, broadcast_words, lane_avx2, nibble_ssse3,
    scale_gfni, scale_split_avx2, scale_ssse3, split_source_avx2, swap_mask256,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// How many terms [`mul_add_matrix_gfni`] folds into a destination tile at once.
///
/// A GF(2^16) coefficient must be turned into two broadcast words before it
/// can be used, which — unlike the GF(2^8) kernel's single byte splat — is
/// too expensive to redo for every tile. So a block of terms is derived once
/// per row group and then folded into every tile of that group. Destination
/// traffic is one read/write per tile per *block*, not per term, and every
/// coefficient is derived exactly once. At or below this many terms the
/// destination is touched exactly once, which is the case that matters.
const TERM_BLOCK: usize = 16;

/// Apply every `(coeffs, src)` term to all `nrows` rows, with `GF2P8MULB`.
///
/// Register-blocked: a destination tile is loaded into accumulators once,
/// every term of a block is folded in, and the tile is stored once. The
/// non-blocked shape re-streams each destination row per term, which is what
/// dominates once `nrows * row_len` leaves L1.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
/// every term supplies `nrows` coefficients, and every source is `row_len`
/// bytes.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni(
    token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    super::check_elements("gf16::mul_add_matrix_gfni", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_gfni: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_gfni: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for (t, &(coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "gf16::mul_add_matrix_gfni: term {t} coefficients is {} but rows is {nrows}",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "gf16::mul_add_matrix_gfni: term {t} source is {} bytes but row_len is {row_len} bytes",
            src.len(),
        );
    }
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_gfni_with(token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, GFNI backend, over a generic matrix source.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics when row geometry overflows or the destination cannot hold the
/// requested rows.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_gfni_with<M: Matrix<Elem> + ?Sized>(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    super::check_elements("gf16::mul_add_matrix_gfni_with", row_len);
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        return;
    }
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_gfni_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_gfni_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    let nrows = nrows.min(rows.len() / row_len);
    let mut span = row_len;
    for term in 0..terms.len() {
        span = span.min(terms.source(term).len());
    }
    if nrows == 0 || span == 0 {
        return;
    }
    let base = rows.as_mut_ptr();
    let swap = swap_mask256();
    let mut j = 0;
    while j + 4 <= nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: this entrypoint holds an `X64V3GfniCryptoToken` (the
        //        `_token` parameter), whose invariant guarantees AVX2 and
        //        GFNI on the executing CPU.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: `rows.len() >= nrows * row_len` was asserted above and
        //        `nrows` only shrank, so the four rows at
        //        `base + k * row_len` lie inside `rows`; `span <= row_len`
        //        and `span <= every source length` bound every load.
        // THUS: every row and source access in the callee stays within its
        //       originating slice.
        //
        // ALIASING
        // SINCE: the four rows are `row_len` bytes apart with `span <=
        //        row_len` windows, so they are pairwise disjoint.
        // THUS: no two row writes in the callee conflict.
        unsafe { matrix_group_gfni::<4, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
        j += 4;
    }
    if j + 2 <= nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: as the four-row call above, the same `_token` is held.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: as the four-row call above, the asserted row span covers
        //        rows `j` and `j + 1`, and `span` bounds every source.
        // THUS: every access in the callee stays within its slice.
        //
        // ALIASING
        // SINCE: as the four-row call above, the two rows are disjoint.
        // THUS: the callee's row writes do not conflict.
        unsafe { matrix_group_gfni::<2, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
        j += 2;
    }
    if j < nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: as the four-row call above, the same `_token` is held.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: as the four-row call above, the asserted row span covers
        //        the final row, and `span` bounds every source.
        // THUS: every access in the callee stays within its slice.
        //
        // ALIASING
        // SINCE: a single row has no peer to conflict with.
        // THUS: the callee's row writes do not conflict.
        unsafe { matrix_group_gfni::<1, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
    }
}

/// Fold every term into `N` consecutive rows starting at `base`, which is row
/// `first` of the destination.
///
/// No destination alignment peel, unlike the scatter kernels. The peel pays
/// there because a scatter loads and stores every row once per 32-byte source
/// window; here the row tile is loaded and stored once per *term block*, so a
/// straddling access is amortized over eight multiplies and the loop is
/// compute-bound rather than traffic-bound. Adding the peel here measured as
/// noise at best (BENCHMARKS.md).
///
/// # Safety
/// The executing CPU must provide AVX2 and GFNI, the `N` rows at
/// `base + k * row_len` must be readable and writable for `span` bytes within
/// one live region, every term must supply more than `first + N - 1`
/// coefficients, and `span` must not exceed any term's source length.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
unsafe fn matrix_group_gfni<const N: usize, M: Matrix<Elem> + ?Sized>(
    base: *mut u8,
    row_len: usize,
    span: usize,
    first: usize,
    terms: &M,
    swap: __m256i,
) {
    let mut rows = [core::ptr::null_mut::<u8>(); N];
    for (k, row) in rows.iter_mut().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract places the `N` rows at
        //        `base + k * row_len` inside one live region.
        // THUS: forming each row pointer stays within that region.
        *row = unsafe { base.add(k * row_len) };
    }

    for block_start in (0..terms.len()).step_by(TERM_BLOCK) {
        let block_len = (terms.len() - block_start).min(TERM_BLOCK);
        // Every coefficient of the block is derived exactly once, here,
        // outside the byte loop. Kept as raw broadcast words because
        // `vpbroadcastw` can read them directly from memory.
        let mut words = [[(0i16, 0i16); N]; TERM_BLOCK];
        for (t, row_words) in words.iter_mut().take(block_len).enumerate() {
            for (k, slot) in row_words.iter_mut().enumerate() {
                let coeff = *terms.coefficient(block_start + t, first + k);
                *slot = broadcast_words(TowerCoeff::new(coeff));
            }
        }
        let mut offset = 0;
        while offset + 32 <= span {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 32 <= span`, the function contract bounds
            //        every row by `span` bytes, and `span` does not exceed
            //        any term's source length.
            // THUS: each row's load and store and each source load land
            //       inside their originating slices.
            //
            // ALIASING
            // SINCE: distinct `k` address disjoint rows of one region.
            // THUS: a row store conflicts with no other live access.
            unsafe {
                let mut acc = [_mm256_setzero_si256(); N];
                for (k, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_loadu_si256(rows[k].add(offset).cast());
                }
                for (t, row_words) in words.iter().take(block_len).enumerate() {
                    let src = terms.source(block_start + t);
                    let x = _mm256_loadu_si256(src.as_ptr().add(offset).cast());
                    let swapped = _mm256_shuffle_epi8(x, swap);
                    for (k, a) in acc.iter_mut().enumerate() {
                        let (same, cross) = row_words[k];
                        let scaled = scale_gfni(
                            x,
                            swapped,
                            _mm256_set1_epi16(same),
                            _mm256_set1_epi16(cross),
                        );
                        *a = _mm256_xor_si256(*a, scaled);
                    }
                }
                for (k, &a) in acc.iter().enumerate() {
                    _mm256_storeu_si256(rows[k].add(offset).cast(), a);
                }
            }
            offset += 32;
        }

        if offset < span {
            for (k, &row) in rows.iter().enumerate() {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: the loop above advanced `offset` past every whole
                //        32-byte window, so `offset..span` is the untouched
                //        remainder of row `k`, and 32-byte steps leave it
                //        starting on an element boundary.
                // THUS: the tail slice lies inside row `k`, and each
                //       `&terms.source(term)[offset..span]` inside its
                //       source.
                let tail =
                    unsafe { core::slice::from_raw_parts_mut(row.add(offset), span - offset) };
                for term in block_start..block_start + block_len {
                    let coeff = *terms.coefficient(term, first + k);
                    mul_add_scalar(tail, coeff, &terms.source(term)[offset..span]);
                }
            }
        }
    }
}

/// Many sources into many rows using AVX2 nibble shuffles.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[cfg_attr(not(test), allow(dead_code))]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx2(
    token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    super::check_elements("gf16::mul_add_matrix_avx2", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_avx2: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_avx2: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for (t, &(coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "gf16::mul_add_matrix_avx2: term {t} coefficients is {} but rows is {nrows}",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "gf16::mul_add_matrix_avx2: term {t} source is {} bytes but row_len is {row_len} bytes",
            src.len(),
        );
    }
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_avx2_with(token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, AVX2 backend, over a generic matrix source.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics when row geometry overflows or the destination cannot hold the
/// requested rows.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx2_with<C: TableCoefficient, M: Matrix<C> + ?Sized>(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    super::check_elements("gf16::mul_add_matrix_avx2_with", row_len);
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        return;
    }
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_avx2_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_avx2_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    let nrows = nrows.min(rows.len() / row_len);
    let mut span = row_len;
    for term in 0..terms.len() {
        span = span.min(terms.source(term).len());
    }
    if nrows == 0 || span == 0 {
        return;
    }
    let base = rows.as_mut_ptr();
    let lanes = lane_avx2();
    let mut j = 0;
    while j + 4 <= nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: this entrypoint holds an `X64V3Token` (the `_token`
        //        parameter), whose invariant guarantees AVX2 on the
        //        executing CPU.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: `rows.len() >= nrows * row_len` was asserted above and
        //        `nrows` only shrank, so the four rows at
        //        `base + k * row_len` lie inside `rows`; `span <= row_len`
        //        and `span <= every source length` bound every load.
        // THUS: every row and source access in the callee stays within its
        //       originating slice.
        //
        // ALIASING
        // SINCE: the four rows are `row_len` bytes apart with `span <=
        //        row_len` windows, so they are pairwise disjoint.
        // THUS: no two row writes in the callee conflict.
        unsafe {
            matrix_group_avx2::<4, C, M>(base.add(j * row_len), row_len, span, j, terms, &lanes);
        };
        j += 4;
    }
    if j + 2 <= nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: as the four-row call above, the same `_token` is held.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: as the four-row call above, the asserted row span covers
        //        rows `j` and `j + 1`, and `span` bounds every source.
        // THUS: every access in the callee stays within its slice.
        //
        // ALIASING
        // SINCE: as the four-row call above, the two rows are disjoint.
        // THUS: the callee's row writes do not conflict.
        unsafe {
            matrix_group_avx2::<2, C, M>(base.add(j * row_len), row_len, span, j, terms, &lanes);
        };
        j += 2;
    }
    if j < nrows {
        // SAFETY:
        // CPU FEATURES
        // SINCE: as the four-row call above, the same `_token` is held.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: as the four-row call above, the asserted row span covers
        //        the final row, and `span` bounds every source.
        // THUS: every access in the callee stays within its slice.
        //
        // ALIASING
        // SINCE: a single row has no peer to conflict with.
        // THUS: the callee's row writes do not conflict.
        unsafe {
            matrix_group_avx2::<1, C, M>(base.add(j * row_len), row_len, span, j, terms, &lanes);
        };
    }
}

/// Fold every term into `N` consecutive rows starting at `base`, which is row
/// `first` of the destination.
///
/// The tile's source is exchanged and split once per term and all `N` rows
/// consume the same index vectors. That prologue is the only part of the
/// multiply independent of the coefficient, and a four-row group used to
/// repeat it four times. What still varies per row is four borrows into the
/// shared bank, not the eleven live vectors a `NibbleAvx2` per (term, row)
/// demanded — that array was the spill.
///
/// # Safety
/// The executing CPU must provide AVX2, the `N` rows at `base + k * row_len`
/// must be readable and writable for `span` bytes within one live region,
/// every term must supply more than `first + N - 1` coefficients, and `span`
/// must not exceed any term's source length.
#[allow(unsafe_code)]
#[archmage::rite(v3)]
unsafe fn matrix_group_avx2<const N: usize, C: TableCoefficient, M: Matrix<C> + ?Sized>(
    base: *mut u8,
    row_len: usize,
    span: usize,
    first: usize,
    terms: &M,
    lanes: &LaneAvx2,
) {
    let vector_len = span & !31;
    let mut rows = [core::ptr::null_mut::<u8>(); N];
    for (k, row) in rows.iter_mut().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract places the `N` rows at
        //        `base + k * row_len` inside one live region.
        // THUS: forming each row pointer stays within that region.
        *row = unsafe { base.add(k * row_len) };
    }

    for block in (0..terms.len()).step_by(TERM_TILE) {
        let block_len = (terms.len() - block).min(TERM_TILE);
        // Every coefficient of the block is resolved exactly once, here,
        // outside the byte loop — and to four pointers, so the block's live
        // state is bytes rather than a stack frame of widened tables.
        let factors: [[[&'static ScaleTable; 4]; N]; TERM_TILE] = core::array::from_fn(|t| {
            core::array::from_fn(|k| {
                let coeff = terms.coefficient(block + t.min(block_len - 1), first + k);
                factor_tables(coeff.coefficient())
            })
        });

        let mut offset = 0;
        while offset < vector_len {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 32 <= vector_len <= span`, the function
            //        contract bounds every row by `span` bytes, and `span`
            //        does not exceed any term's source length.
            // THUS: each row's load and store and each source load land
            //       inside their originating slices.
            //
            // ALIASING
            // SINCE: distinct `k` address disjoint rows of one region.
            // THUS: a row store conflicts with no other live access.
            unsafe {
                let mut acc = [_mm256_setzero_si256(); N];
                for (k, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_loadu_si256(rows[k].add(offset).cast());
                }
                for (t, row_factors) in factors.iter().take(block_len).enumerate() {
                    let source =
                        _mm256_loadu_si256(terms.source(block + t).as_ptr().add(offset).cast());
                    let split = split_source_avx2(source, lanes);
                    for (a, factor) in acc.iter_mut().zip(row_factors) {
                        *a = _mm256_xor_si256(*a, scale_split_avx2(&split, factor, lanes.even));
                    }
                }
                for (k, &a) in acc.iter().enumerate() {
                    _mm256_storeu_si256(rows[k].add(offset).cast(), a);
                }
            }
            offset += 32;
        }

        if vector_len < span {
            for (k, &row) in rows.iter().enumerate() {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `vector_len..span` is the untouched remainder of
                //        row `k` (the loop above covered every whole 32-byte
                //        window), 32-byte steps leave it starting on an
                //        element boundary, and `span` does not exceed any
                //        term's source length.
                // THUS: the tail slice lies inside row `k`, and each
                //       `&terms.source(term)[vector_len..]` inside its
                //       source.
                let tail = unsafe {
                    core::slice::from_raw_parts_mut(row.add(vector_len), span - vector_len)
                };
                for term in block..block + block_len {
                    mul_add_scalar(
                        tail,
                        terms.coefficient(term, first + k).coefficient(),
                        &terms.source(term)[vector_len..],
                    );
                }
            }
        }
    }
}

/// Many sources into many rows using SSSE3 nibble shuffles.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// As [`mul_add_matrix_gfni`].
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3(
    token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    super::check_elements("gf16::mul_add_matrix_ssse3", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_ssse3: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_ssse3: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for (t, &(coeffs, src)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "gf16::mul_add_matrix_ssse3: term {t} coefficients is {} but rows is {nrows}",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "gf16::mul_add_matrix_ssse3: term {t} source is {} bytes but row_len is {row_len} bytes",
            src.len(),
        );
    }
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_ssse3_with(token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, SSSE3 backend, over a generic matrix source.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics when row geometry overflows or the destination cannot hold the
/// requested rows.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3_with<C: TableCoefficient, M: Matrix<C> + ?Sized>(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    super::check_elements("gf16::mul_add_matrix_ssse3_with", row_len);
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        return;
    }
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_ssse3_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_ssse3_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    let nrows = nrows.min(rows.len() / row_len);
    let mut span = row_len;
    for term in 0..terms.len() {
        span = span.min(terms.source(term).len());
    }
    if nrows == 0 || span == 0 {
        return;
    }
    // SAFETY:
    // CPU FEATURES
    // SINCE: this entrypoint holds an `X64V2Token` (the `_token`
    //        parameter), whose invariant guarantees SSSE3 on the executing
    //        CPU.
    // THUS: the callee's CPU-feature precondition holds.
    //
    // MEMORY VALIDITY
    // SINCE: `rows.len() >= nrows * row_len` was asserted above and `nrows`
    //        only shrank, so every row at `base + j * row_len` lies inside
    //        `rows`; `span <= row_len` and `span <= every source length`
    //        bound every load, and every term supplies `nrows` coefficients
    //        by the [`Matrix`] contract.
    // THUS: every row, coefficient and source access in the callee stays
    //       within its originating slice.
    //
    // ALIASING
    // SINCE: the rows are `row_len` bytes apart with `span <= row_len`
    //        windows, so they are pairwise disjoint.
    // THUS: no two row writes in the callee conflict.
    unsafe { matrix_rows_ssse3(rows, row_len, span, nrows, terms) };
}

/// The SSSE3 matrix body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide SSSE3, `rows.len()` must be at least
/// `nrows * row_len`, every term must supply `nrows` coefficients, and
/// `span` must be at most `row_len` and at most every term's source length.
#[allow(unsafe_code)]
#[archmage::rite(v2)]
unsafe fn matrix_rows_ssse3<C: TableCoefficient, M: Matrix<C> + ?Sized>(
    rows: &mut [u8],
    row_len: usize,
    span: usize,
    nrows: usize,
    terms: &M,
) {
    let vector_len = span & !15;
    let base = rows.as_mut_ptr();
    for group in (0..nrows).step_by(4) {
        let row_count = (nrows - group).min(4);
        for block in (0..terms.len()).step_by(TERM_TILE) {
            let term_count = (terms.len() - block).min(TERM_TILE);
            let vectors: [[NibbleSsse3; 4]; TERM_TILE] = core::array::from_fn(|t| {
                core::array::from_fn(|r| {
                    terms
                        .coefficient(block + t.min(term_count - 1), group + r.min(row_count - 1))
                        .with_tables(|tables| nibble_ssse3(tables))
                })
            });
            let mut offset = 0;
            while offset < vector_len {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `offset + 16 <= vector_len <= span`, the function
                //        contract bounds every selected row by `span` bytes,
                //        and `span` does not exceed any term's source
                //        length.
                // THUS: each row's load and store and each source load land
                //       inside their originating slices.
                //
                // ALIASING
                // SINCE: the selected rows are `row_len` bytes apart with
                //        `span <= row_len` windows, so they are pairwise
                //        disjoint.
                // THUS: a row store conflicts with no other live access.
                unsafe {
                    let mut acc = [_mm_setzero_si128(); 4];
                    for (r, slot) in acc.iter_mut().take(row_count).enumerate() {
                        *slot = _mm_loadu_si128(base.add((group + r) * row_len + offset).cast());
                    }
                    for (t, vector) in vectors.iter().take(term_count).enumerate() {
                        let source =
                            _mm_loadu_si128(terms.source(block + t).as_ptr().add(offset).cast());
                        for (value, scale) in acc.iter_mut().zip(vector).take(row_count) {
                            *value = _mm_xor_si128(*value, scale_ssse3(source, scale));
                        }
                    }
                    for (r, &slot) in acc.iter().take(row_count).enumerate() {
                        _mm_storeu_si128(base.add((group + r) * row_len + offset).cast(), slot);
                    }
                }
                offset += 16;
            }
            for r in 0..row_count {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `vector_len..span` is the untouched remainder of
                //        row `group + r` (the loop above covered every whole
                //        16-byte window), 16-byte steps leave it starting on
                //        an element boundary, and `span` does not exceed any
                //        term's source length.
                // THUS: the tail slice lies inside that row, and each
                //       `&terms.source(term)[vector_len..]` inside its
                //       source.
                let tail = unsafe {
                    core::slice::from_raw_parts_mut(
                        base.add((group + r) * row_len + vector_len),
                        span - vector_len,
                    )
                };
                for term in block..block + term_count {
                    mul_add_scalar(
                        tail,
                        terms.coefficient(term, group + r).coefficient(),
                        &terms.source(term)[vector_len..],
                    );
                }
            }
        }
    }
}
