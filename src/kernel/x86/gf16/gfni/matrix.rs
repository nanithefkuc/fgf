//! Every source into every destination row.
//!
//! A destination tile is loaded into accumulators once, every term of a block
//! is folded in, and the tile is stored once, so destination traffic is one
//! read/write per tile per block of [`TERM_BLOCK`]
//! terms. The group body is the sanctioned offset-addressed-rows residue.

use super::super::{broadcast_words, check_elements, swap_mask_avx2};
use crate::field::gf16::Elem;
use crate::kernel::Matrix;
use crate::kernel::gf16::mul_add_scalar;
use crate::kernel::tables::TowerCoeff;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::scale_gfni;

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
    check_elements("gf16::mul_add_matrix_gfni", row_len);
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
    check_elements("gf16::mul_add_matrix_gfni_with", row_len);
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
    let swap = swap_mask_avx2();
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
        unsafe { matrix_group::<4, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
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
        unsafe { matrix_group::<2, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
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
        unsafe { matrix_group::<1, M>(base.add(j * row_len), row_len, span, j, terms, swap) };
    }
}

/// Fold every term into `N` consecutive rows starting at `base`, which is row
/// `first` of the destination.
///
/// No destination alignment peel, unlike the scatter kernels. The peel pays
/// there because a scatter loads and stores every row once per 32-byte source
/// window; here the row tile is loaded and stored once per *term block*, so a
/// straddling access is amortized over eight multiplies and the loop is
/// compute-bound rather than traffic-bound.
///
/// # Safety
/// The executing CPU must provide AVX2 and GFNI, the `N` rows at
/// `base + k * row_len` must be readable and writable for `span` bytes within
/// one live region, every term must supply more than `first + N - 1`
/// coefficients, and `span` must not exceed any term's source length.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
unsafe fn matrix_group<const N: usize, M: Matrix<Elem> + ?Sized>(
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
