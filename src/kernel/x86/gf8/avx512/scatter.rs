//! One source into many rows for `Gf8D` over 64-byte lanes.
//!
//! Mirrors the GFNI scatter at 64-byte lanes with `VGF2P8AFFINEQB`: the
//! immediate is the XOR constant, so `<0>` selects the pure linear map, and
//! each factor replicates one 64-bit map qword to all eight 64-bit lanes.

use super::{MapCoeff, bfactor_avx512, bmul_avx512, brem_avx512};

/// Shortest scatter row that peels its head to a 64-byte boundary.
///
/// Below this the scalar/AVX2 head costs more than the split-line loads it
/// removes.
pub(crate) const SCATTER_PEEL_MIN: usize = 512;

// ---------------------------------------------------------------------------
// Scatter: one source into many rows.
// ---------------------------------------------------------------------------

/// `rows[j] ^= coeffs[j] * src` for every row, four rows at a time.
///
/// Rows with a zero coefficient contribute nothing and are dropped before
/// grouping. One source load feeds every row in a group.
///
/// # Panics
/// Panics unless `src.len() == row_len` and `rows` holds `coeffs.len()` rows
/// of `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_avx512<C: MapCoeff>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    assert_eq!(
        src.len(),
        row_len,
        "mul_add_scatter_avx512: src must span one row"
    );
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_scatter_avx512: rows buffer does not hold coefficients rows of {row_len} bytes"
    );
    mul_add_scatter_impl(rows, coeffs, src);
}

/// Stage the surviving rows into four-row groups and run the group bodies.
///
/// Row staging addresses disjoint windows of one region at runtime offsets —
/// zero coefficients leave gaps, so the rows are not adjacent — which is the
/// residue this body keeps.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_scatter_impl<C: MapCoeff>(rows: &mut [u8], coeffs: &[C], src: &[u8]) {
    let base = rows.as_mut_ptr();
    let row_len = src.len();
    let mut ptrs = [base; 4];
    let mut group = [C::zero(); 4];
    let mut filled = 0;

    for (j, &coeff) in coeffs.iter().enumerate() {
        if C::is_zero(coeff) {
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
            scatter_rows4(ptrs, group, src);
            filled = 0;
        }
    }

    // 3 left over is a pair plus a single; 2 a pair; 1 a single.
    if filled >= 2 {
        // As above, for the first two staged rows.
        scatter_rows2([ptrs[0], ptrs[1]], [group[0], group[1]], src);
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
        brem_avx512(row, C::map(group[last]), C::table(group[last]), src);
    }
}

/// `rows[i] ^= coeffs[i] * src` for four disjoint rows of `src.len()` bytes.
///
/// Residue: the rows are addressed as runtime offsets into one region, staged
/// and bounded by [`mul_add_scatter_impl`].
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn scatter_rows4<C: MapCoeff>(ptrs: [*mut u8; 4], coeffs: [C; 4], src: &[u8]) {
    let factors = [
        bfactor_avx512(C::map(coeffs[0])),
        bfactor_avx512(C::map(coeffs[1])),
        bfactor_avx512(C::map(coeffs[2])),
        bfactor_avx512(C::map(coeffs[3])),
    ];
    let len = src.len();

    // The head aligns the first row. A quarter-line row pitch cycles through
    // different line offsets within each group; the peel merely moves the
    // split-line accesses to other rows, so that pitch runs without it.
    // The source shares the rows' alignment in dispatched shapes.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if len >= SCATTER_PEEL_MIN && len % 64 != 16 && head < len {
        head
    } else {
        0
    };
    scatter_span(ptrs.iter().zip(coeffs.iter()), 0, head, src);

    // One 256-byte source window feeds all four rows, and each row runs four
    // independent multiply chains over it.
    let mut offset = head;
    while offset + 256 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 256 <= len == src.len()`, which is also the
        //        length of every staged row.
        // THUS: every load and store below stays inside `src` or its row.
        //
        // ALIASING
        // SINCE: the four rows are pairwise disjoint, as staged.
        // THUS: no row's store conflicts with another row's window.
        unsafe {
            let (s, _) = src[offset..offset + 256].as_chunks::<64>();
            let x0 = _mm512_loadu_si512(&s[0]);
            let x1 = _mm512_loadu_si512(&s[1]);
            let x2 = _mm512_loadu_si512(&s[2]);
            let x3 = _mm512_loadu_si512(&s[3]);
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let (d, _) =
                    core::slice::from_raw_parts_mut(row.add(offset), 256).as_chunks_mut::<64>();
                let d0 = _mm512_loadu_si512(&d[0]);
                let d1 = _mm512_loadu_si512(&d[1]);
                let d2 = _mm512_loadu_si512(&d[2]);
                let d3 = _mm512_loadu_si512(&d[3]);
                _mm512_storeu_si512(&mut d[0], _mm512_xor_si512(d0, bmul_avx512(x0, factor)));
                _mm512_storeu_si512(&mut d[1], _mm512_xor_si512(d1, bmul_avx512(x1, factor)));
                _mm512_storeu_si512(&mut d[2], _mm512_xor_si512(d2, bmul_avx512(x2, factor)));
                _mm512_storeu_si512(&mut d[3], _mm512_xor_si512(d3, bmul_avx512(x3, factor)));
            }
        }
        offset += 256;
    }
    while offset + 64 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 64 <= len`, the common length of `src` and every
        //        staged row.
        // THUS: the load and each store stay inside their slices.
        //
        // ALIASING
        // SINCE: the rows are pairwise disjoint, as staged.
        // THUS: no row's store conflicts with another row's window.
        unsafe {
            let (s64, _) = src[offset..offset + 64].as_chunks::<64>();
            let x = _mm512_loadu_si512(&s64[0]);
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let (d64, _) =
                    core::slice::from_raw_parts_mut(row.add(offset), 64).as_chunks_mut::<64>();
                let d = _mm512_loadu_si512(&d64[0]);
                _mm512_storeu_si512(&mut d64[0], _mm512_xor_si512(d, bmul_avx512(x, factor)));
            }
        }
        offset += 64;
    }

    scatter_span(ptrs.iter().zip(coeffs.iter()), offset, len, src);
}

/// `rows[i] ^= coeffs[i] * src` for two disjoint rows of `src.len()` bytes.
///
/// Residue: as [`scatter_rows4`].
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn scatter_rows2<C: MapCoeff>(ptrs: [*mut u8; 2], coeffs: [C; 2], src: &[u8]) {
    let factors = [
        bfactor_avx512(C::map(coeffs[0])),
        bfactor_avx512(C::map(coeffs[1])),
    ];
    let len = src.len();

    // As in `scatter_rows4`: peel to 64-byte alignment first.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if len >= SCATTER_PEEL_MIN && len % 64 != 16 && head < len {
        head
    } else {
        0
    };
    scatter_span(ptrs.iter().zip(coeffs.iter()), 0, head, src);

    // With only two rows the 256-byte window leaves room for eight chains.
    let mut offset = head;
    while offset + 256 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 256 <= len == src.len()`, which is also the
        //        length of both staged rows.
        // THUS: every load and store below stays inside `src` or its row.
        //
        // ALIASING
        // SINCE: the two rows are disjoint, as staged.
        // THUS: no row's store conflicts with the other row's window.
        unsafe {
            let (s, _) = src[offset..offset + 256].as_chunks::<64>();
            let x0 = _mm512_loadu_si512(&s[0]);
            let x1 = _mm512_loadu_si512(&s[1]);
            let x2 = _mm512_loadu_si512(&s[2]);
            let x3 = _mm512_loadu_si512(&s[3]);
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let (d, _) =
                    core::slice::from_raw_parts_mut(row.add(offset), 256).as_chunks_mut::<64>();
                let d0 = _mm512_loadu_si512(&d[0]);
                let d1 = _mm512_loadu_si512(&d[1]);
                let d2 = _mm512_loadu_si512(&d[2]);
                let d3 = _mm512_loadu_si512(&d[3]);
                _mm512_storeu_si512(&mut d[0], _mm512_xor_si512(d0, bmul_avx512(x0, factor)));
                _mm512_storeu_si512(&mut d[1], _mm512_xor_si512(d1, bmul_avx512(x1, factor)));
                _mm512_storeu_si512(&mut d[2], _mm512_xor_si512(d2, bmul_avx512(x2, factor)));
                _mm512_storeu_si512(&mut d[3], _mm512_xor_si512(d3, bmul_avx512(x3, factor)));
            }
        }
        offset += 256;
    }
    while offset + 64 <= len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 64 <= len`, the common length of `src` and both
        //        staged rows.
        // THUS: the load and each store stay inside their slices.
        //
        // ALIASING
        // SINCE: the two rows are disjoint, as staged.
        // THUS: no row's store conflicts with the other row's window.
        unsafe {
            let (s64, _) = src[offset..offset + 64].as_chunks::<64>();
            let x = _mm512_loadu_si512(&s64[0]);
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let (d64, _) =
                    core::slice::from_raw_parts_mut(row.add(offset), 64).as_chunks_mut::<64>();
                let d = _mm512_loadu_si512(&d64[0]);
                _mm512_storeu_si512(&mut d64[0], _mm512_xor_si512(d, bmul_avx512(x, factor)));
            }
        }
        offset += 64;
    }

    scatter_span(ptrs.iter().zip(coeffs.iter()), offset, len, src);
}

/// Narrow SIMD span shared by the 512-bit scatter row groups.
///
/// Complete 64-byte lanes stay on the one-instruction multiply; the
/// remainder descends through the AVX2 ladder to the scalar nibble kernel.
///
/// Residue: the rows arrive as runtime offsets into one region, staged and
/// bounded by the caller; `start..end` lies within `0..=src.len()`.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn scatter_span<'a, C: MapCoeff + 'a>(
    rows: impl Iterator<Item = (&'a *mut u8, &'a C)>,
    start: usize,
    end: usize,
    src: &[u8],
) {
    // The row-group bodies finish every whole 64-byte lane before calling
    // here, so `end - start < 64` always holds: only the sub-lane tail
    // through the AVX2 ladder remains.
    if start >= end {
        return;
    }
    for (&row, &coeff) in rows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `start..end` is the row's final sub-lane span, and the row
        //        spans `src.len() >= end` bytes.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the rows are disjoint and only one mutable slice is live at
        //        a time.
        // THUS: the borrow conflicts with no live row window.
        let dst = unsafe { core::slice::from_raw_parts_mut(row.add(start), end - start) };
        super::super::gfni::mul_add_gfni_8d_impl(
            dst,
            C::map(coeff),
            C::table(coeff),
            &src[start..end],
        );
    }
}
