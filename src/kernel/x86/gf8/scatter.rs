//! One source into many rows: the blocked GFNI scatter family.
//!
//! One source load feeds every row in a group, so the source is read once per
//! group instead of once per row. Zero coefficients drop out before grouping,
//! outside every loop, which is where sparsity belongs.
//!
//! The entries are safe [`archmage`] capability-token functions carrying the
//! geometry contract; the row-group bodies keep the offset-addressed-row
//! residue in [`archmage::rite`] helpers.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{Affine8D, Blocked, Gfni, bfactor, bfactor128, bmul, bmul128, brem};
use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::gf8::mul_add_nibble;

/// `rows[j] ^= coeffs[j] * src` for every row, four rows at a time.
///
/// Rows with a zero coefficient contribute nothing and are dropped before
/// grouping, so a sparse coefficient vector costs no row traffic at all. The
/// surviving rows are batched in fours (then a pair, then a single) so one
/// source load feeds several destinations.
///
/// # Panics
/// Panics unless `src.len() == row_len` and `rows` holds `coeffs.len()` rows
/// of `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
    src: &[u8],
) {
    assert_eq!(row_len, src.len());
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_scatter_gfni: rows buffer does not hold {} rows of {row_len} bytes",
        coeffs.len()
    );
    mul_add_scatter_impl::<Gfni>(rows, row_len, coeffs, src);
}

/// [`mul_add_scatter_gfni`] under `0x11D`: one source into many rows, folding each in
/// with its `VGF2P8AFFINEQB` map instead of `GF2P8MULB`.
///
/// # Panics
/// As [`mul_add_scatter_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_affine(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[gf8d::Elem],
    src: &[u8],
) {
    assert_eq!(row_len, src.len());
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_scatter_affine: rows buffer does not hold {} rows of {row_len} bytes",
        coeffs.len()
    );
    mul_add_scatter_impl::<Affine8D>(rows, row_len, coeffs, src);
}

/// Stage the surviving rows into four-row groups and run the group bodies.
///
/// Row staging addresses disjoint windows of one region at runtime offsets —
/// zero coefficients leave gaps, so the rows are not adjacent — which is the
/// residue this body keeps. Both entries assert `coeffs.len() * row_len <=
/// rows.len()` and `row_len == src.len()` before dispatch.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_add_scatter_impl<S: Blocked>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[S::Coeff],
    src: &[u8],
) {
    let base = rows.as_mut_ptr();
    let mut ptrs = [base; 4];
    let mut group = [S::zero(); 4];
    let mut filled = 0;

    for (j, &coeff) in coeffs.iter().enumerate() {
        if S::byte(coeff) == 0 {
            // `row ^= 0 * src` is the identity: skip the row entirely rather
            // than stream it through the multiplier to add zero.
            continue;
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `j < coeffs.len()` and `coeffs.len() * row_len <=
        //        rows.len()` was asserted by the entry, so row `j` lies
        //        wholly inside `rows`.
        // THUS: the row pointer addresses writable memory of one row's span.
        //
        // ALIASING
        // SINCE: distinct `j` place the rows `row_len` apart in one region,
        //        so the staged rows never overlap.
        // THUS: no two grouped row windows alias.
        ptrs[filled] = unsafe { base.add(j * row_len) };
        group[filled] = coeff;
        filled += 1;
        if filled == 4 {
            // Four in-bounds, pairwise disjoint rows of `src.len()` bytes;
            // every nonzero coefficient is exact under the multiply.
            scatter_rows4::<S>(ptrs, group, src);
            filled = 0;
        }
    }

    // 3 left over is a pair plus a single; 2 a pair; 1 a single.
    if filled >= 2 {
        // As above, for the first two staged rows.
        scatter_rows2::<S>([ptrs[0], ptrs[1]], [group[0], group[1]], src);
    }
    if filled == 1 || filled == 3 {
        let last = filled - 1;
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `ptrs[last]` is a staged in-bounds row of `src.len()`
        //        bytes, as established above.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: no slice into `rows` outlives this statement and the rows
        //        are disjoint.
        // THUS: the mutable borrow conflicts with no live row window.
        let row = unsafe { core::slice::from_raw_parts_mut(ptrs[last], src.len()) };
        brem::<S>(row, group[last], src);
    }
}

/// `rows[i] ^= coeffs[i] * src` for four disjoint rows of `src.len()` bytes.
///
/// Residue: the rows are addressed as runtime offsets into one region, staged
/// and bounded by [`mul_add_scatter_impl`].
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn scatter_rows4<S: Blocked>(ptrs: [*mut u8; 4], coeffs: [S::Coeff; 4], src: &[u8]) {
    let factors = [
        bfactor::<S>(coeffs[0]),
        bfactor::<S>(coeffs[1]),
        bfactor::<S>(coeffs[2]),
        bfactor::<S>(coeffs[3]),
    ];
    let len = src.len();
    let src_ptr = src.as_ptr();

    // Bring the destinations to a 32-byte boundary before the vector body
    // starts; see `peel_to_align`.
    let head = super::super::peel_to_align(ptrs[0], len, 1);
    // `head <= len`, the length shared by `src` and every row, so `0..head`
    // is a sub-range of each; the four rows are distinct and in-bounds, and
    // no slice into them is live here.
    scatter_span::<S>(ptrs.iter().zip(&coeffs), 0, head, src);
    let mut offset = head;

    // One 128-byte source window feeds all four rows, and each row runs four
    // independent multiply chains over it.
    while offset + 128 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 128 <= len == src.len()`, which is also the
        //        length of every staged row.
        // THUS: every load and store below stays inside `src` or its row.
        //
        // ALIASING
        // SINCE: the four rows are pairwise disjoint, as staged.
        // THUS: no row's store conflicts with another row's window.
        unsafe {
            let sp = src_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d0 = _mm256_loadu_si256(rp.cast());
                let d1 = _mm256_loadu_si256(rp.add(32).cast());
                let d2 = _mm256_loadu_si256(rp.add(64).cast());
                let d3 = _mm256_loadu_si256(rp.add(96).cast());
                let r0 = _mm256_xor_si256(d0, bmul::<S>(x0, factor));
                let r1 = _mm256_xor_si256(d1, bmul::<S>(x1, factor));
                let r2 = _mm256_xor_si256(d2, bmul::<S>(x2, factor));
                let r3 = _mm256_xor_si256(d3, bmul::<S>(x3, factor));
                _mm256_storeu_si256(rp.cast(), r0);
                _mm256_storeu_si256(rp.add(32).cast(), r1);
                _mm256_storeu_si256(rp.add(64).cast(), r2);
                _mm256_storeu_si256(rp.add(96).cast(), r3);
            }
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= len`, the common length of `src` and every
        //        staged row.
        // THUS: the load and each store stay inside their slices.
        //
        // ALIASING
        // SINCE: the rows are pairwise disjoint, as staged.
        // THUS: no row's store conflicts with another row's window.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d = _mm256_loadu_si256(rp.cast());
                _mm256_storeu_si256(rp.cast(), _mm256_xor_si256(d, bmul::<S>(x, factor)));
            }
        }
        offset += 32;
    }

    // The four rows are distinct, in-bounds, and `src.len()` bytes long; no
    // slice into them is live here.
    scatter_span::<S>(ptrs.iter().zip(&coeffs), offset, len, src);
}

/// `rows[i] ^= coeffs[i] * src` for two disjoint rows of `src.len()` bytes.
///
/// Residue: as [`scatter_rows4`].
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn scatter_rows2<S: Blocked>(ptrs: [*mut u8; 2], coeffs: [S::Coeff; 2], src: &[u8]) {
    let factors = [bfactor::<S>(coeffs[0]), bfactor::<S>(coeffs[1])];
    let len = src.len();
    let src_ptr = src.as_ptr();

    // As in `scatter_rows4`, with four of every five accesses on the
    // destination side.
    let head = super::super::peel_to_align(ptrs[0], len, 1);
    // `head <= len`, the length shared by `src` and both rows, so `0..head`
    // is a sub-range of each; the rows are distinct, in-bounds, and no slice
    // into them is live here.
    scatter_span::<S>(ptrs.iter().zip(&coeffs), 0, head, src);
    let mut offset = head;

    // With only two rows the 128-byte window leaves room for eight chains.
    while offset + 128 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 128 <= len == src.len()`, which is also the
        //        length of both staged rows.
        // THUS: every load and store below stays inside `src` or its row.
        //
        // ALIASING
        // SINCE: the two rows are disjoint, as staged.
        // THUS: no row's store conflicts with the other row's window.
        unsafe {
            let sp = src_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d0 = _mm256_loadu_si256(rp.cast());
                let d1 = _mm256_loadu_si256(rp.add(32).cast());
                let d2 = _mm256_loadu_si256(rp.add(64).cast());
                let d3 = _mm256_loadu_si256(rp.add(96).cast());
                let r0 = _mm256_xor_si256(d0, bmul::<S>(x0, factor));
                let r1 = _mm256_xor_si256(d1, bmul::<S>(x1, factor));
                let r2 = _mm256_xor_si256(d2, bmul::<S>(x2, factor));
                let r3 = _mm256_xor_si256(d3, bmul::<S>(x3, factor));
                _mm256_storeu_si256(rp.cast(), r0);
                _mm256_storeu_si256(rp.add(32).cast(), r1);
                _mm256_storeu_si256(rp.add(64).cast(), r2);
                _mm256_storeu_si256(rp.add(96).cast(), r3);
            }
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= len`, the common length of `src` and both
        //        staged rows.
        // THUS: the load and each store stay inside their slices.
        //
        // ALIASING
        // SINCE: the two rows are disjoint, as staged.
        // THUS: no row's store conflicts with the other row's window.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d = _mm256_loadu_si256(rp.cast());
                _mm256_storeu_si256(rp.cast(), _mm256_xor_si256(d, bmul::<S>(x, factor)));
            }
        }
        offset += 32;
    }

    // The two rows are distinct, in-bounds, and `src.len()` bytes long; no
    // slice into them is live here.
    scatter_span::<S>(ptrs.iter().zip(&coeffs), offset, len, src);
}

/// Narrow SIMD span shared by the scatter row groups.
///
/// Processes complete 16-byte lanes with the field's one-instruction multiply
/// and leaves fewer than 16 bytes to the scalar nibble kernel.
///
/// Residue: the rows arrive as runtime offsets into one region, staged and
/// bounded by the caller; `start..end` lies within `0..=src.len()`.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn scatter_span<'a, S: Blocked>(
    rows: impl Iterator<Item = (&'a *mut u8, &'a S::Coeff)>,
    start: usize,
    end: usize,
    src: &[u8],
) where
    S::Coeff: 'a,
{
    if start >= end {
        return;
    }
    let vector_end = start + ((end - start) & !15);
    for (&row, &coeff) in rows {
        let factor = bfactor128::<S>(coeff);
        let mut offset = start;
        while offset < vector_end {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 16 <= vector_end <= end <= src.len()` bounds
            //        both the source load and the row window, which spans
            //        `src.len()` bytes.
            // THUS: the load and store stay inside their slices.
            unsafe {
                let x = _mm_loadu_si128(src.as_ptr().add(offset).cast());
                let ptr = row.add(offset);
                let d = _mm_loadu_si128(ptr.cast());
                _mm_storeu_si128(ptr.cast(), _mm_xor_si128(d, bmul128::<S>(x, factor)));
            }
            offset += 16;
        }
        if vector_end < end {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `vector_end..end` is the row's final sub-lane span, and
            //        the row spans `src.len() >= end` bytes.
            // THUS: the tail slice lies wholly within its originating row.
            //
            // ALIASING
            // SINCE: the rows are disjoint and only one mutable slice is
            //        live at a time.
            // THUS: the borrow conflicts with no live row window.
            let dst =
                unsafe { core::slice::from_raw_parts_mut(row.add(vector_end), end - vector_end) };
            mul_add_nibble(dst, S::table(coeff), &src[vector_end..end]);
        }
    }
}
