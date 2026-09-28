//! `Gf8D` AVX-512 affine scatter/gather kernels (`VGF2P8AFFINEQB`, 64-byte lanes).
//!
//! One source into many rows (scatter) and many sources into one destination
//! (gather) for the Reed–Solomon field GF(2^8)/`0x11D`, mirroring the AVX2
//! affine blocked shapes at 64-byte lanes. The immediate is the XOR constant,
//! so `<0>` selects the pure linear map; each factor replicates one 64-bit
//! map qword to all eight 64-bit lanes.
//!
//! The kernels dispatch on the `V4x` tier under the `simd512` feature and stay
//! reachable through the `internals` facade for differential tests and
//! measurement on AVX-512 hardware.

use super::gather::check_gather;
use super::gf8d512_matrix::MapCoeff;
use super::gf8d512_single::{bfactor512, bmul512, brem512};

/// Shortest scatter row that peels its head to a 64-byte boundary.
///
/// Below this the scalar/AVX2 head costs more than the split-line loads it
/// removes. Set by the misaligned-scatter floor sweep in `BENCHMARKS.md`
/// ("AVX-512 alignment peel floors").
pub(crate) const SCATTER_PEEL_MIN: usize = 512;

/// Shortest gather destination that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per source, so it repays only once the
/// tile body is long enough. Set by the misaligned-gather floor sweep in
/// `BENCHMARKS.md` ("AVX-512 alignment peel floors").
pub(crate) const GATHER_PEEL_MIN: usize = 3072;

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
pub fn mul_add_scatter_affine512<C: MapCoeff>(
    _token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    assert_eq!(
        src.len(),
        row_len,
        "mul_add_scatter_affine512: src must span one row"
    );
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_scatter_affine512: rows buffer does not hold coefficients rows of {row_len} bytes"
    );
    mul_add_scatter512_impl(rows, coeffs, src);
}

/// Stage the surviving rows into four-row groups and run the group bodies.
///
/// Row staging addresses disjoint windows of one region at runtime offsets —
/// zero coefficients leave gaps, so the rows are not adjacent — which is the
/// residue this body keeps.
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn mul_add_scatter512_impl<C: MapCoeff>(rows: &mut [u8], coeffs: &[C], src: &[u8]) {
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
            scatter_rows4_512(ptrs, group, src);
            filled = 0;
        }
    }

    // 3 left over is a pair plus a single; 2 a pair; 1 a single.
    if filled >= 2 {
        // As above, for the first two staged rows.
        scatter_rows2_512([ptrs[0], ptrs[1]], [group[0], group[1]], src);
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
        brem512(row, C::map(group[last]), C::table(group[last]), src);
    }
}

/// `rows[i] ^= coeffs[i] * src` for four disjoint rows of `src.len()` bytes.
///
/// Residue: the rows are addressed as runtime offsets into one region, staged
/// and bounded by [`mul_add_scatter512_impl`].
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn scatter_rows4_512<C: MapCoeff>(ptrs: [*mut u8; 4], coeffs: [C; 4], src: &[u8]) {
    let factors = [
        bfactor512(C::map(coeffs[0])),
        bfactor512(C::map(coeffs[1])),
        bfactor512(C::map(coeffs[2])),
        bfactor512(C::map(coeffs[3])),
    ];
    let len = src.len();

    // Head bytes run through the narrow span until the first row is 64-byte
    // aligned. Rows sit `len` apart, so a row pitch that is a multiple of 64
    // aligns every row with the first; other pitches align the first row
    // only. The span also covers the source head, which shares the rows'
    // alignment in every dispatched shape.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if len >= SCATTER_PEEL_MIN && head < len {
        head
    } else {
        0
    };
    scatter_span512(ptrs.iter().zip(coeffs.iter()), 0, head, src);

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
                _mm512_storeu_si512(&mut d[0], _mm512_xor_si512(d0, bmul512(x0, factor)));
                _mm512_storeu_si512(&mut d[1], _mm512_xor_si512(d1, bmul512(x1, factor)));
                _mm512_storeu_si512(&mut d[2], _mm512_xor_si512(d2, bmul512(x2, factor)));
                _mm512_storeu_si512(&mut d[3], _mm512_xor_si512(d3, bmul512(x3, factor)));
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
                _mm512_storeu_si512(&mut d64[0], _mm512_xor_si512(d, bmul512(x, factor)));
            }
        }
        offset += 64;
    }

    scatter_span512(ptrs.iter().zip(coeffs.iter()), offset, len, src);
}

/// `rows[i] ^= coeffs[i] * src` for two disjoint rows of `src.len()` bytes.
///
/// Residue: as [`scatter_rows4_512`].
#[allow(unsafe_code)]
#[archmage::rite(v4x, import_intrinsics)]
fn scatter_rows2_512<C: MapCoeff>(ptrs: [*mut u8; 2], coeffs: [C; 2], src: &[u8]) {
    let factors = [bfactor512(C::map(coeffs[0])), bfactor512(C::map(coeffs[1]))];
    let len = src.len();

    // As in `scatter_rows4_512`: peel to 64-byte alignment first.
    let head = (ptrs[0] as usize).wrapping_neg() & 63;
    let head = if len >= SCATTER_PEEL_MIN && head < len {
        head
    } else {
        0
    };
    scatter_span512(ptrs.iter().zip(coeffs.iter()), 0, head, src);

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
                _mm512_storeu_si512(&mut d[0], _mm512_xor_si512(d0, bmul512(x0, factor)));
                _mm512_storeu_si512(&mut d[1], _mm512_xor_si512(d1, bmul512(x1, factor)));
                _mm512_storeu_si512(&mut d[2], _mm512_xor_si512(d2, bmul512(x2, factor)));
                _mm512_storeu_si512(&mut d[3], _mm512_xor_si512(d3, bmul512(x3, factor)));
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
                _mm512_storeu_si512(&mut d64[0], _mm512_xor_si512(d, bmul512(x, factor)));
            }
        }
        offset += 64;
    }

    scatter_span512(ptrs.iter().zip(coeffs.iter()), offset, len, src);
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
fn scatter_span512<'a, C: MapCoeff + 'a>(
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
        super::gfni::mul_add_affine_impl(dst, C::map(coeff), C::table(coeff), &src[start..end]);
    }
}

// ---------------------------------------------------------------------------
// Gather: many sources into one destination.
// ---------------------------------------------------------------------------

/// Many sources into one destination, register-blocked over 256-byte tiles.
///
/// The destination tile stays in registers across every source, so it is
/// read and written once per tile, not once per source.
///
/// # Panics
/// Panics unless `srcs.len() == coeffs.len()` and every source matches `dst`
/// in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_affine512<C: MapCoeff>(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    coeffs: &[C],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_affine512", dst, coeffs.len(), srcs);
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    // Head bytes run through the single-source seam until the destination
    // is 64-byte aligned; the sources share the destination's alignment in
    // every dispatched shape.
    let head = dst.as_ptr().align_offset(64);
    let head = if head == usize::MAX || head >= dst.len() || dst.len() < GATHER_PEEL_MIN {
        0
    } else {
        head
    };
    if head > 0 {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            brem512(
                &mut dst[..head],
                C::map(coeff),
                C::table(coeff),
                &src[..head],
            );
        }
    }
    let (dhead, dtile) = dst.split_at_mut(head);
    let _ = dhead;
    gather_tiles512(dtile, coeffs, srcs, head);
    let tile = dtile.len() / 256 * 256;
    gather_remainder512(&mut dtile[tile..], coeffs, srcs, head + tile);
}

/// The main register-blocked tile loop: four 64-byte accumulators over whole
/// tiles, every source folded in before the destination stores.
#[archmage::rite(v4x, import_intrinsics)]
fn gather_tiles512<C: MapCoeff>(dst: &mut [u8], coeffs: &[C], srcs: &[&[u8]], base: usize) {
    let mut offset = base;
    for dtile in dst.chunks_exact_mut(256) {
        let (dl, _) = dtile.as_chunks_mut::<64>();
        // `offset + 256 <= dst.len() <= every src.len()` (asserted by the
        // entry), so each accumulator load stays inside `dst`.
        let mut acc = [
            _mm512_loadu_si512(&dl[0]),
            _mm512_loadu_si512(&dl[1]),
            _mm512_loadu_si512(&dl[2]),
            _mm512_loadu_si512(&dl[3]),
        ];
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = bfactor512(C::map(coeff));
            let (sl, _) = src[offset..offset + 256].as_chunks::<64>();
            let x0 = _mm512_loadu_si512(&sl[0]);
            let x1 = _mm512_loadu_si512(&sl[1]);
            let x2 = _mm512_loadu_si512(&sl[2]);
            let x3 = _mm512_loadu_si512(&sl[3]);
            acc[0] = _mm512_xor_si512(acc[0], bmul512(x0, factor));
            acc[1] = _mm512_xor_si512(acc[1], bmul512(x1, factor));
            acc[2] = _mm512_xor_si512(acc[2], bmul512(x2, factor));
            acc[3] = _mm512_xor_si512(acc[3], bmul512(x3, factor));
        }
        _mm512_storeu_si512(&mut dl[0], acc[0]);
        _mm512_storeu_si512(&mut dl[1], acc[1]);
        _mm512_storeu_si512(&mut dl[2], acc[2]);
        _mm512_storeu_si512(&mut dl[3], acc[3]);
        offset += 256;
    }
}

/// The single-source AXPY remainder over what no tile covered.
#[archmage::rite(v4x, import_intrinsics)]
fn gather_remainder512<C: MapCoeff>(dst: &mut [u8], coeffs: &[C], srcs: &[&[u8]], tail: usize) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        brem512(dst, C::map(coeff), C::table(coeff), &src[tail..]);
    }
}
