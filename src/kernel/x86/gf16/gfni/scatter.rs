//! One source into many destination rows.
//!
//! Rows are updated in groups, so the source is loaded and exchanged once per
//! group instead of once per row, and every row pays a destination alignment
//! peel. The group body is the sanctioned offset-addressed-rows residue.

use super::super::{broadcast_words, check_elements, swap_mask_avx2};
use crate::kernel::gf16::Coeffs;
use crate::kernel::tables::TowerCoeff;
use crate::kernel::x86::peel_to_align;

use super::scale_gfni;
use super::single::mul_add_gfni;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `rows[j] ^= coeffs[j] * src` for every row, with `GF2P8MULB`.
///
/// Rows are updated in groups of four, then a pair, then one at a time. Each
/// group loads the source once and exchanges its bytes once, so those costs
/// are amortized over the whole group instead of paid per row.
///
/// Coefficients of zero and one need no special case: `TowerCoeff` reduces
/// them to the broadcast pairs `(0, 0)` and `(0x0101, 0)`, which the same
/// multiply turns into a no-op and a plain XOR respectively.
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
/// bytes and `row_len == src.len()`.
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_gfni(
    token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &(impl Coeffs + ?Sized),
    src: &[u8],
) {
    check_elements("gf16::mul_add_scatter_gfni", row_len);
    assert_eq!(
        row_len,
        src.len(),
        "gf16::mul_add_scatter_gfni: row_len is {row_len} bytes but src is {} bytes",
        src.len(),
    );
    let used = coeffs
        .count()
        .checked_mul(row_len)
        .expect("gf16::mul_add_scatter_gfni: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_scatter_gfni: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.count(),
    );
    if row_len == 0 || coeffs.count() == 0 {
        return;
    }
    let base = rows.as_mut_ptr();
    let swap = swap_mask_avx2();
    let mut j = 0;
    while j + 4 <= coeffs.count() {
        let group: [TowerCoeff; 4] = core::array::from_fn(|slot| coeffs.compact(j + slot));
        // SAFETY:
        // CPU FEATURES
        // SINCE: this entrypoint holds an `X64V3GfniCryptoToken` (the
        //        `token` parameter), whose invariant guarantees AVX2 and
        //        GFNI on the executing CPU.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: `rows.len() >= coeffs.len() * row_len` was asserted above,
        //        so the four rows at `base + k * row_len` lie inside `rows`,
        //        and `src.len() == row_len` was asserted above.
        // THUS: every row and source access in the callee stays within its
        //       originating slice.
        //
        // ALIASING
        // SINCE: the four rows are `row_len` bytes apart and `row_len` bytes
        //        long, so they are pairwise disjoint, and `src` is an
        //        independent shared borrow.
        // THUS: no two row writes in the callee conflict.
        unsafe { scatter_group(token, base.add(j * row_len), row_len, group, src, swap) };
        j += 4;
    }
    if j + 2 <= coeffs.count() {
        let group: [TowerCoeff; 2] = core::array::from_fn(|slot| coeffs.compact(j + slot));
        // SAFETY:
        // CPU FEATURES
        // SINCE: as the four-row call above, the same `token` is held.
        // THUS: the callee's CPU-feature precondition holds.
        //
        // MEMORY VALIDITY
        // SINCE: as the four-row call above, the asserted row span covers
        //        rows `j` and `j + 1`, and `src.len() == row_len`.
        // THUS: every access in the callee stays within its slice.
        //
        // ALIASING
        // SINCE: as the four-row call above, the two rows are disjoint.
        // THUS: the callee's row writes do not conflict.
        unsafe { scatter_group(token, base.add(j * row_len), row_len, group, src, swap) };
        j += 2;
    }
    if j < coeffs.count() {
        // The unrolled single-destination kernel is the better shape for the
        // last row.
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `rows.len() >= coeffs.len() * row_len` was asserted above
        //        and `j < coeffs.len()`, so the `row_len` bytes at
        //        `base + j * row_len` lie inside `rows`.
        // THUS: the reconstructed row slice stays within `rows`.
        unsafe {
            let row = core::slice::from_raw_parts_mut(base.add(j * row_len), row_len);
            mul_add_gfni(token, row, coeffs.compact(j), src);
        }
    }
}

/// `row[k] ^= coeffs[k] * src` for `N` consecutive rows starting at `base`.
///
/// # Safety
/// The executing CPU must provide AVX2 and GFNI, the `N` rows at
/// `base + k * row_len` must be readable and writable for `row_len` bytes
/// within one live region, and `src.len()` must equal `row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
unsafe fn scatter_group<const N: usize>(
    token: archmage::X64V3GfniCryptoToken,
    base: *mut u8,
    row_len: usize,
    coeffs: [TowerCoeff; N],
    src: &[u8],
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
    // Derived once per group, never inside the byte loop.
    let mut same = [_mm256_setzero_si256(); N];
    let mut cross = [_mm256_setzero_si256(); N];
    for (k, coeff) in coeffs.iter().enumerate() {
        let (same_word, cross_word) = broadcast_words(*coeff);
        same[k] = _mm256_set1_epi16(same_word);
        cross[k] = _mm256_set1_epi16(cross_word);
    }

    let src_ptr = src.as_ptr();
    // Bring the destinations to a 32-byte boundary; see `peel_to_align`.
    let head = peel_to_align(rows[0], row_len, 2);
    for (k, coeff) in coeffs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract bounds every row by `row_len`,
        //        `head <= row_len` by `peel_to_align`'s contract, and
        //        `src.len() == row_len`.
        // THUS: the `head`-byte lead slice lies inside row `k`, and
        //       `&src[..head]` inside `src`.
        let lead = unsafe { core::slice::from_raw_parts_mut(rows[k], head) };
        mul_add_gfni(token, lead, *coeff, &src[..head]);
    }
    let mut offset = head;
    while offset + 32 <= row_len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= row_len` bounds this loop iteration, every
        //        row is `row_len` bytes by the function contract, and
        //        `src.len() == row_len`.
        // THUS: the 32-byte source load and each row's load and store land
        //       inside `src` and row `k` respectively.
        //
        // ALIASING
        // SINCE: distinct `k` address disjoint `row_len`-byte rows of one
        //        region, and `src` is an independent shared borrow.
        // THUS: a row store conflicts with no other live access.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let swapped = _mm256_shuffle_epi8(x, swap);
            for (k, &row) in rows.iter().enumerate() {
                let p = row.add(offset);
                let d = _mm256_loadu_si256(p.cast());
                let scaled = scale_gfni(x, swapped, same[k], cross[k]);
                _mm256_storeu_si256(p.cast(), _mm256_xor_si256(d, scaled));
            }
        }
        offset += 32;
    }
    if offset < row_len {
        for (k, coeff) in coeffs.iter().enumerate() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the loop above advanced `offset` past every whole 32-byte
            //        window, so `offset..row_len` is the untouched remainder of
            //        row `k`, and 32-byte steps leave it starting on an element
            //        boundary.
            // THUS: the tail slice lies inside row `k`, and `&src[offset..]`
            //       inside `src`.
            let tail =
                unsafe { core::slice::from_raw_parts_mut(rows[k].add(offset), row_len - offset) };
            mul_add_gfni(token, tail, *coeff, &src[offset..]);
        }
    }
}
