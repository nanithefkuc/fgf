//! SSSE3 GF(2^16) kernels over 16-byte lanes, on split-nibble `PSHUFB`
//! lookups.
//!
//! SSSE3 has no field multiply, so each of the four base-field factors
//! becomes a `PSHUFB` pair against the prepared nibble tables and the even and
//! odd byte lanes are selected with a `0x00ff` halfword mask. The
//! register-resident table sets and their `scale_ssse3` core are shared with
//! the Fan–Paar tower kernels and the AVX2 sub-lane tails. Gather and matrix
//! are register-blocked over [`TERM_TILE`](super::TERM_TILE) prepared table
//! sets; lane-wise products borrow the GF(2^8) shift/reduce vector multiply
//! and keep a nibble shuffle only for the constant `DELTA` factor.
//!
//! Every entry is a safe [`archmage`] capability-token function taking an
//! `X64V2Token`. The scatter and matrix group bodies are the sanctioned
//! offset-addressed-rows residue.

use super::{TERM_TILE, TableCoefficient, check_elements, swap_mask_ssse3};
use crate::field::gf16::Elem;
use crate::kernel::Matrix;
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::{NibbleFactors, TowerTables, scale_table};
use crate::kernel::x86::gf8;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Nibble tables and lane masks for one coefficient, held in SSE registers.
pub(crate) struct NibbleSsse3 {
    /// Low-nibble table of factor `i`.
    pub(crate) lo: [__m128i; 4],
    /// High-nibble table of factor `i`.
    pub(crate) hi: [__m128i; 4],
    /// `0x0f` in every byte: the nibble-extraction mask.
    pub(crate) nibble: __m128i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    pub(crate) even: __m128i,
    /// Adjacent-byte exchange control.
    pub(crate) swap: __m128i,
}

/// Load the 16-byte factor tables of `tables` into SSE registers.
///
/// Generic over [`NibbleFactors`]; see [`nibble_avx2`].
#[archmage::rite(v2, import_intrinsics)]
pub(crate) fn nibble_ssse3<T: NibbleFactors>(tables: &T) -> NibbleSsse3 {
    let mut lo = [_mm_setzero_si128(); 4];
    let mut hi = [_mm_setzero_si128(); 4];
    for slot in 0..4 {
        lo[slot] = _mm_loadu_si128(tables.lo(slot));
        hi[slot] = _mm_loadu_si128(tables.hi(slot));
    }
    NibbleSsse3 {
        lo,
        hi,
        nibble: _mm_set1_epi8(0x0f),
        even: _mm_set1_epi16(0x00ff),
        swap: swap_mask_ssse3(),
    }
}

/// Split `value` into its low and high nibbles, both as byte indices.
#[archmage::rite(v2)]
pub(crate) fn split(value: __m128i, nibble: __m128i) -> (__m128i, __m128i) {
    (
        _mm_and_si128(value, nibble),
        _mm_and_si128(_mm_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[archmage::rite(v2)]
pub(crate) fn lookup(lo: __m128i, hi: __m128i, split: (__m128i, __m128i)) -> __m128i {
    _mm_xor_si128(_mm_shuffle_epi8(lo, split.0), _mm_shuffle_epi8(hi, split.1))
}

#[archmage::rite(v2)]
pub(crate) fn scale_ssse3(src: __m128i, tables: &NibbleSsse3) -> __m128i {
    let swapped = _mm_shuffle_epi8(src, tables.swap);
    let direct = split(src, tables.nibble);
    let crossed = split(swapped, tables.nibble);
    let even = _mm_xor_si128(
        lookup(tables.lo[0], tables.hi[0], direct),
        lookup(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm_xor_si128(
        lookup(tables.lo[1], tables.hi[1], direct),
        lookup(tables.lo[3], tables.hi[3], crossed),
    );
    _mm_xor_si128(
        _mm_and_si128(even, tables.even),
        _mm_andnot_si128(tables.even, odd),
    )
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 16-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_ssse3(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_elements("gf16::mul_add_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, scale_ssse3(x, &vectors)));
    }
    mul_add_scalar(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 16-byte lanes.
///
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], tables: &TowerTables) {
    check_elements("gf16::mul_assign_ssse3", dst.len());
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    mul_assign_scalar(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
///
/// Unlike the GFNI and AVX2 forms this keeps ordinary stores at every size.
/// Non-temporal stores only pay when the loop is store-bound, and this one is
/// not: eight `PSHUFB` per 16 bytes hold it well under the host's write
/// bandwidth, and 16-byte non-temporal stores from a slow loop flush
/// write-combining buffers before a line fills. The GF(2^8) SSSE3 kernel does
/// reach the ceiling and does use them. See BENCHMARKS.md.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_ssse3(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_elements("gf16::mul_into_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    mul_into_scalar(dst_tail, tables.coeff, src_tail);
}

/// One source into many rows using four SSSE3 table sets at a time.
///
/// No destination alignment peel, unlike [`mul_add_scatter_avx2`](super::mul_add_scatter_avx2). A 16-byte
/// access only straddles a cache line when it is not 16-byte aligned, and
/// every allocator this runs behind already returns 16-byte-aligned memory,
/// so the peel has nothing to fix and measured as a small loss
/// (BENCHMARKS.md).
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// As [`mul_add_scatter_gfni`](super::mul_add_scatter_gfni).
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
    check_elements("gf16::mul_add_scatter_ssse3", row_len);
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
    unsafe { scatter_rows(rows, row_len, coeffs, src) };
}

/// The SSSE3 scatter body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide SSSE3, `rows.len()` must be at least
/// `coeffs.len() * row_len`, and `src.len()` must equal `row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v2)]
unsafe fn scatter_rows<C: TableCoefficient>(
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

/// Many sources into one destination, eight coefficients prepared per pass.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_ssse3<C: TableCoefficient>(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    coeffs: &[C],
    srcs: &[&[u8]],
) {
    check_elements("gf16::mul_add_gather_ssse3", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "gf16::mul_add_gather_ssse3: coefficients is {} but sources is {}",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "gf16::mul_add_gather_ssse3: dst is {} bytes but source {index} is {} bytes",
            dst.len(),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    let tail_start = dst.len() & !15;
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for block in (0..coeffs.len()).step_by(TERM_TILE) {
        let count = (coeffs.len() - block).min(TERM_TILE);
        let vectors: [NibbleSsse3; TERM_TILE] = core::array::from_fn(|i| {
            coeffs[block + i.min(count - 1)].with_tables(|tables| nibble_ssse3(tables))
        });
        for (w, dst_lane) in dst_lanes.iter_mut().enumerate() {
            let mut acc = _mm_loadu_si128(&*dst_lane);
            for i in 0..count {
                let (src_lanes, _) = srcs[block + i].as_chunks::<16>();
                let source = _mm_loadu_si128(&src_lanes[w]);
                acc = _mm_xor_si128(acc, scale_ssse3(source, &vectors[i]));
            }
            _mm_storeu_si128(dst_lane, acc);
        }
        for i in 0..count {
            mul_add_scalar(
                dst_tail,
                coeffs[block + i].coefficient(),
                &srcs[block + i][tail_start..],
            );
        }
    }
}

/// Many sources into many rows using SSSE3 nibble shuffles.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// As [`mul_add_matrix_gfni`](super::mul_add_matrix_gfni).
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3(
    token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    check_elements("gf16::mul_add_matrix_ssse3", row_len);
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
    check_elements("gf16::mul_add_matrix_ssse3_with", row_len);
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
    unsafe { matrix_rows(rows, row_len, span, nrows, terms) };
}

/// The SSSE3 matrix body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide SSSE3, `rows.len()` must be at least
/// `nrows * row_len`, every term must supply `nrows` coefficients, and
/// `span` must be at most `row_len` and at most every term's source length.
#[allow(unsafe_code)]
#[archmage::rite(v2)]
unsafe fn matrix_rows<C: TableCoefficient, M: Matrix<C> + ?Sized>(
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

/// The 16-byte form of [`scale_delta`].
#[archmage::rite(v2)]
fn scale_delta(v: __m128i, lo: __m128i, hi: __m128i, nibble: __m128i) -> __m128i {
    _mm_xor_si128(
        _mm_shuffle_epi8(lo, _mm_and_si128(v, nibble)),
        _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16::<4>(v), nibble)),
    )
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements, SSSE3.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_elements("gf16::mul_elementwise_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_ssse3: dst is {} bytes but a is {} bytes",
        dst.len(),
        a.len(),
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_ssse3: dst is {} bytes but b is {} bytes",
        dst.len(),
        b.len(),
    );
    let swap = swap_mask_ssse3();
    let even = _mm_set1_epi16(0x00ff);
    let nibble = _mm_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (_mm_loadu_si128(&delta.lo), _mm_loadu_si128(&delta.hi));
    let (a_lanes, a_rest) = a.as_chunks::<16>();
    let (b_lanes, b_rest) = b.as_chunks::<16>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    for ((dst_lane, x_lane), y_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm_loadu_si128(x_lane);
        let y = _mm_loadu_si128(y_lane);
        let direct = gf8::multiply_vectors_ssse3::<0x1b>(x, y);
        let crossed = gf8::multiply_vectors_ssse3::<0x1b>(x, _mm_shuffle_epi8(y, swap));
        let delta_bd = scale_delta(_mm_shuffle_epi8(direct, swap), delta_lo, delta_hi, nibble);
        let constant = _mm_xor_si128(direct, delta_bd);
        let cross_sum = _mm_xor_si128(crossed, _mm_shuffle_epi8(crossed, swap));
        let extension = _mm_xor_si128(cross_sum, direct);
        let product = _mm_xor_si128(
            _mm_and_si128(constant, even),
            _mm_andnot_si128(even, extension),
        );
        _mm_storeu_si128(dst_lane, product);
    }
    for ((d, x), y) in dst_rest
        .chunks_exact_mut(2)
        .zip(a_rest.chunks_exact(2))
        .zip(b_rest.chunks_exact(2))
    {
        d.copy_from_slice(
            &Elem::from_bytes([x[0], x[1]])
                .mul(Elem::from_bytes([y[0], y[1]]))
                .to_bytes(),
        );
    }
}

/// `dst[i] = dst[i] * src[i]` over interleaved tower elements, SSSE3.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_elements("gf16::mul_elementwise_assign_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask_ssse3();
    let even = _mm_set1_epi16(0x00ff);
    let nibble = _mm_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (_mm_loadu_si128(&delta.lo), _mm_loadu_si128(&delta.hi));
    let (src_lanes, src_rest) = src.as_chunks::<16>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    for (dst_lane, y_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(&*dst_lane);
        let y = _mm_loadu_si128(y_lane);
        let direct = gf8::multiply_vectors_ssse3::<0x1b>(x, y);
        let crossed = gf8::multiply_vectors_ssse3::<0x1b>(x, _mm_shuffle_epi8(y, swap));
        let delta_bd = scale_delta(_mm_shuffle_epi8(direct, swap), delta_lo, delta_hi, nibble);
        let constant = _mm_xor_si128(direct, delta_bd);
        let cross_sum = _mm_xor_si128(crossed, _mm_shuffle_epi8(crossed, swap));
        let extension = _mm_xor_si128(cross_sum, direct);
        let product = _mm_xor_si128(
            _mm_and_si128(constant, even),
            _mm_andnot_si128(even, extension),
        );
        _mm_storeu_si128(dst_lane, product);
    }
    for (d, y) in dst_rest.chunks_exact_mut(2).zip(src_rest.chunks_exact(2)) {
        let x = [d[0], d[1]];
        d.copy_from_slice(
            &Elem::from_bytes(x)
                .mul(Elem::from_bytes([y[0], y[1]]))
                .to_bytes(),
        );
    }
}
