//! One source into many destination rows.
//!
//! Rows are updated in groups, so the source is loaded and exchanged once
//! per group instead of once per row. Every row is still read and written
//! once per source window, which is why this shape — unlike the blocked
//! matrix — pays for a destination alignment peel on the 32-byte backends.
//!
//! The group bodies are the sanctioned offset-addressed-rows residue: the
//! borrow checker cannot see that `base + k * row_len` windows of one region
//! are disjoint, so each body is an `unsafe fn` taking the geometry the safe
//! entrypoint asserted, with a per-item `#[allow(unsafe_code)]`.

use crate::field::gf16::Elem;
use crate::kernel::gf16::mul_add_scalar;
use crate::kernel::tables::TowerCoeff;

use super::gfni::mul_add_gfni;
use super::nibble::mul_add_avx2;
use super::{
    NibbleAvx2, NibbleSsse3, TableCoefficient, broadcast_words, nibble_avx2, nibble_ssse3,
    scale_avx2, scale_gfni, scale_ssse3, swap_mask256,
};

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
    coeffs: &[Elem],
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_scatter_gfni", row_len);
    assert_eq!(
        row_len,
        src.len(),
        "gf16::mul_add_scatter_gfni: row_len is {row_len} bytes but src is {} bytes",
        src.len(),
    );
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("gf16::mul_add_scatter_gfni: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_scatter_gfni: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    let base = rows.as_mut_ptr();
    let swap = swap_mask256();
    let mut j = 0;
    while j + 4 <= coeffs.len() {
        let mut group = [Elem(0); 4];
        group.copy_from_slice(&coeffs[j..j + 4]);
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
        unsafe { scatter_group_gfni(token, base.add(j * row_len), row_len, group, src, swap) };
        j += 4;
    }
    if j + 2 <= coeffs.len() {
        let mut group = [Elem(0); 2];
        group.copy_from_slice(&coeffs[j..j + 2]);
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
        unsafe { scatter_group_gfni(token, base.add(j * row_len), row_len, group, src, swap) };
        j += 2;
    }
    if j < coeffs.len() {
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
            mul_add_gfni(token, row, TowerCoeff::new(coeffs[j]), src);
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
unsafe fn scatter_group_gfni<const N: usize>(
    token: archmage::X64V3GfniCryptoToken,
    base: *mut u8,
    row_len: usize,
    coeffs: [Elem; N],
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
    for (k, &coeff) in coeffs.iter().enumerate() {
        let (same_word, cross_word) = broadcast_words(TowerCoeff::new(coeff));
        same[k] = _mm256_set1_epi16(same_word);
        cross[k] = _mm256_set1_epi16(cross_word);
    }

    let src_ptr = src.as_ptr();
    // Bring the destinations to a 32-byte boundary; see `peel_to_align`.
    let head = super::peel_to_align(rows[0], row_len, 2);
    for (k, &coeff) in coeffs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract bounds every row by `row_len`,
        //        `head <= row_len` by `peel_to_align`'s contract, and
        //        `src.len() == row_len`.
        // THUS: the `head`-byte lead slice lies inside row `k`, and
        //       `&src[..head]` inside `src`.
        let lead = unsafe { core::slice::from_raw_parts_mut(rows[k], head) };
        mul_add_gfni(token, lead, TowerCoeff::new(coeff), &src[..head]);
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
    for (k, &coeff) in coeffs.iter().enumerate() {
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
        mul_add_scalar(tail, coeff, &src[offset..]);
    }
}

/// One source into many rows using four AVX2 table sets at a time.
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// As [`mul_add_scatter_gfni`].
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_avx2<C: TableCoefficient>(
    token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_scatter_avx2", row_len);
    assert_eq!(
        row_len,
        src.len(),
        "gf16::mul_add_scatter_avx2: row_len is {row_len} bytes but src is {} bytes",
        src.len(),
    );
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("gf16::mul_add_scatter_avx2: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_scatter_avx2: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY:
    // CPU FEATURES
    // SINCE: this entrypoint holds an `X64V3Token` (the `token` parameter),
    //        whose invariant guarantees AVX2 on the executing CPU.
    // THUS: the callee's CPU-feature precondition holds.
    //
    // MEMORY VALIDITY
    // SINCE: `rows.len() >= coeffs.len() * row_len` was asserted above, so
    //        every row at `base + j * row_len` lies inside `rows`, and
    //        `src.len() == row_len` was asserted above.
    // THUS: every row and source access in the callee stays within its
    //       originating slice.
    //
    // ALIASING
    // SINCE: the rows are `row_len` bytes apart and `row_len` bytes long, so
    //        they are pairwise disjoint, and `src` is an independent shared
    //        borrow.
    // THUS: no two row writes in the callee conflict.
    unsafe { scatter_rows_avx2(token, rows, row_len, coeffs, src) };
}

/// The AVX2 scatter body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide AVX2, `rows.len()` must be at least
/// `coeffs.len() * row_len`, and `src.len()` must equal `row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v3)]
unsafe fn scatter_rows_avx2<C: TableCoefficient>(
    token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    let base = rows.as_mut_ptr();
    let src_ptr = src.as_ptr();
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let vectors: [NibbleAvx2; 4] = core::array::from_fn(|i| {
            coeffs[group + i.min(count - 1)].with_tables(|tables| nibble_avx2(tables))
        });
        // Bring this group's rows to a 32-byte boundary; see
        // `peel_to_align`.
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract places row `group` inside `rows`, and
        //        `peel_to_align` returns at most `row_len`.
        // THUS: reading row `group`'s address stays within `rows`.
        let head = super::peel_to_align(unsafe { base.add(group * row_len) }, row_len, 2);
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the function contract bounds row `group + slot` by
            //        `row_len`, `head <= row_len`, and `src.len() == row_len`.
            // THUS: the `head`-byte lead slice lies inside that row, and
            //       `&src[..head]` inside `src`.
            unsafe {
                let lead =
                    core::slice::from_raw_parts_mut(base.add((group + slot) * row_len), head);
                coeffs[group + slot]
                    .with_tables(|tables| mul_add_avx2(token, lead, tables, &src[..head]));
            }
        }
        let mut offset = head;
        while offset + 32 <= row_len {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 32 <= row_len` bounds this iteration, every
            //        row is `row_len` bytes by the function contract, and
            //        `src.len() == row_len`.
            // THUS: the source load and each row's load and store land
            //       inside `src` and row `group + slot` respectively.
            //
            // ALIASING
            // SINCE: distinct slots address disjoint `row_len`-byte rows of
            //        one region, and `src` is an independent shared borrow.
            // THUS: a row store conflicts with no other live access.
            unsafe {
                let source = _mm256_loadu_si256(src_ptr.add(offset).cast());
                for (slot, vector) in vectors.iter().take(count).enumerate() {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm256_storeu_si256(
                        ptr.cast(),
                        _mm256_xor_si256(
                            _mm256_loadu_si256(ptr.cast()),
                            scale_avx2(source, vector),
                        ),
                    );
                }
            }
            offset += 32;
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the loop above advanced `offset` past every whole
            //        32-byte window, so `offset..row_len` is the untouched
            //        remainder of row `group + slot`, and `src.len() ==
            //        row_len`.
            // THUS: the tail slice lies inside that row, and
            //       `&src[offset..]` inside `src`.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + offset),
                    row_len - offset,
                )
            };
            mul_add_scalar(tail, coeffs[group + slot].coefficient(), &src[offset..]);
        }
    }
}

/// One source into many rows using four SSSE3 table sets at a time.
///
/// No destination alignment peel, unlike [`mul_add_scatter_avx2`]. A 16-byte
/// access only straddles a cache line when it is not 16-byte aligned, and
/// every allocator this runs behind already returns 16-byte-aligned memory,
/// so the peel has nothing to fix and measured as a small loss
/// (BENCHMARKS.md).
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// As [`mul_add_scatter_gfni`].
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_ssse3<C: TableCoefficient>(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_scatter_ssse3", row_len);
    assert_eq!(
        row_len,
        src.len(),
        "gf16::mul_add_scatter_ssse3: row_len is {row_len} bytes but src is {} bytes",
        src.len(),
    );
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("gf16::mul_add_scatter_ssse3: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_scatter_ssse3: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY:
    // CPU FEATURES
    // SINCE: this entrypoint holds an `X64V2Token` (the `_token` parameter),
    //        whose invariant guarantees SSSE3 on the executing CPU.
    // THUS: the callee's CPU-feature precondition holds.
    //
    // MEMORY VALIDITY
    // SINCE: `rows.len() >= coeffs.len() * row_len` was asserted above, so
    //        every row at `base + j * row_len` lies inside `rows`, and
    //        `src.len() == row_len` was asserted above.
    // THUS: every row and source access in the callee stays within its
    //       originating slice.
    //
    // ALIASING
    // SINCE: the rows are `row_len` bytes apart and `row_len` bytes long, so
    //        they are pairwise disjoint, and `src` is an independent shared
    //        borrow.
    // THUS: no two row writes in the callee conflict.
    unsafe { scatter_rows_ssse3(rows, row_len, coeffs, src) };
}

/// The SSSE3 scatter body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide SSSE3, `rows.len()` must be at least
/// `coeffs.len() * row_len`, and `src.len()` must equal `row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v2)]
unsafe fn scatter_rows_ssse3<C: TableCoefficient>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    let base = rows.as_mut_ptr();
    let src_ptr = src.as_ptr();
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let vectors: [NibbleSsse3; 4] = core::array::from_fn(|i| {
            coeffs[group + i.min(count - 1)].with_tables(|tables| nibble_ssse3(tables))
        });
        let mut offset = 0;
        while offset + 16 <= row_len {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 16 <= row_len` bounds this iteration, every
            //        row is `row_len` bytes by the function contract, and
            //        `src.len() == row_len`.
            // THUS: the source load and each row's load and store land
            //       inside `src` and row `group + slot` respectively.
            //
            // ALIASING
            // SINCE: distinct slots address disjoint `row_len`-byte rows of
            //        one region, and `src` is an independent shared borrow.
            // THUS: a row store conflicts with no other live access.
            unsafe {
                let source = _mm_loadu_si128(src_ptr.add(offset).cast());
                for (slot, vector) in vectors.iter().take(count).enumerate() {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm_storeu_si128(
                        ptr.cast(),
                        _mm_xor_si128(_mm_loadu_si128(ptr.cast()), scale_ssse3(source, vector)),
                    );
                }
            }
            offset += 16;
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the loop above advanced `offset` past every whole
            //        16-byte window, so `offset..row_len` is the untouched
            //        remainder of row `group + slot`, and `src.len() ==
            //        row_len`.
            // THUS: the tail slice lies inside that row, and
            //       `&src[offset..]` inside `src`.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + offset),
                    row_len - offset,
                )
            };
            mul_add_scalar(tail, coeffs[group + slot].coefficient(), &src[offset..]);
        }
    }
}
