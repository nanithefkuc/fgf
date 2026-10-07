//! AVX-512 GFNI GF(2^16) kernels over 64-byte lanes, under `simd512`.
//!
//! Each lane contains complete two-byte elements. The adjacent-byte exchange
//! and coefficient broadcasts share the tower identity of the narrower GFNI
//! kernels, and remainders descend through them. Matrix tiles borrow disjoint
//! row slices, so the destination geometry needs no raw pointers; the
//! destination is loaded once per block of terms.
//!
//! Every entry is a safe [`archmage`] capability-token function taking an
//! `X64V4xToken`.

use super::{broadcast_words, check_elements};
use crate::kernel::Matrix;
use crate::kernel::proven_checks::check_terms;
use crate::kernel::tables::TowerCoeff;

use super::gfni;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const SWAP_ADJACENT_AVX512: [u8; 64] = {
    let mut mask = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        mask[i] = super::SWAP_ADJACENT[i % 16];
        i += 1;
    }
    mask
};

#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn swap_mask() -> __m512i {
    _mm512_loadu_si512(&SWAP_ADJACENT_AVX512)
}

#[archmage::rite(v4x)]
pub(super) fn factors(coeff: TowerCoeff) -> (__m512i, __m512i) {
    let (same, cross) = broadcast_words(coeff);
    (_mm512_set1_epi16(same), _mm512_set1_epi16(cross))
}

#[archmage::rite(v4x)]
pub(super) fn scale(x: __m512i, swapped: __m512i, same: __m512i, cross: __m512i) -> __m512i {
    _mm512_xor_si512(
        _mm512_gf2p8mul_epi8(x, same),
        _mm512_gf2p8mul_epi8(swapped, cross),
    )
}

/// Shortest single-row destination that peels its head to a 64-byte
/// boundary. `mul_add` and `mul_into` share the floor.
///
/// The peel runs up to 62 bytes through the 32-byte GFNI kernel, which
/// repays only once the 64-byte body is long enough.
pub(crate) const MUL_ADD_PEEL_MIN: usize = 3584;

/// Head bytes that put `dst`'s 64-byte accesses on a cache-line boundary,
/// or `0` below [`MUL_ADD_PEEL_MIN`] or when the head would split an element.
#[inline]
fn peel(dst: &[u8]) -> usize {
    let head = dst.as_ptr().align_offset(64);
    if dst.len() < MUL_ADD_PEEL_MIN || head >= dst.len() || !head.is_multiple_of(2) {
        0
    } else {
        head
    }
}

/// Accumulates a scaled source into the destination over GF(2^16).
///
/// # Panics
/// Panics if the slices differ in length or contain a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512(token: archmage::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    check_elements("gf16::mul_add_avx512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_avx512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_add_gfni(token.v3_gfni_crypto(), dst, coeff, src);
    }
    // The peel fires only from `MUL_ADD_PEEL_MIN` bytes, far above one
    // lane, so the body below always has a whole 64-byte lane left.
    let head = peel(dst);
    let (dst_head, dst) = dst.split_at_mut(head);
    let (src_head, src) = src.split_at(head);
    if head > 0 {
        gfni::mul_add_gfni(token.v3_gfni_crypto(), dst_head, coeff, src_head);
    }
    let (same, cross) = factors(coeff);
    let swap = swap_mask();
    // Four independent multiply chains per 256-byte tile, as in the 32-byte
    // GFNI body: a single destination has no other work to hide the
    // multiplier latency behind.
    let (dtiles, dmid) = dst.as_chunks_mut::<256>();
    let (stiles, smid) = src.as_chunks::<256>();
    for (dtile, stile) in dtiles.iter_mut().zip(stiles) {
        let (d, _) = dtile.as_chunks_mut::<64>();
        let (s, _) = stile.as_chunks::<64>();
        for (dlane, slane) in d.iter_mut().zip(s) {
            let x = _mm512_loadu_si512(slane);
            let old = _mm512_loadu_si512(&*dlane);
            let product = scale(x, _mm512_shuffle_epi8(x, swap), same, cross);
            _mm512_storeu_si512(dlane, _mm512_xor_si512(old, product));
        }
    }
    let (dlanes, drest) = dmid.as_chunks_mut::<64>();
    let (slanes, srest) = smid.as_chunks::<64>();
    for (d, s) in dlanes.iter_mut().zip(slanes) {
        let x = _mm512_loadu_si512(s);
        let old = _mm512_loadu_si512(&*d);
        _mm512_storeu_si512(
            d,
            _mm512_xor_si512(old, scale(x, _mm512_shuffle_epi8(x, swap), same, cross)),
        );
    }
    if !drest.is_empty() {
        gfni::mul_add_gfni(token.v3_gfni_crypto(), drest, coeff, srest);
    }
}

/// Scales the destination in place over GF(2^16).
///
/// # Panics
/// Panics if the destination contains a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx512(token: archmage::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff) {
    check_elements("gf16::mul_assign_avx512", dst.len());
    if dst.len() < 64 {
        return gfni::mul_assign_gfni(token.v3_gfni_crypto(), dst, coeff);
    }
    let (same, cross) = factors(coeff);
    let swap = swap_mask();
    let (lanes, rest) = dst.as_chunks_mut::<64>();
    for lane in lanes {
        let x = _mm512_loadu_si512(&*lane);
        _mm512_storeu_si512(lane, scale(x, _mm512_shuffle_epi8(x, swap), same, cross));
    }
    gfni::mul_assign_gfni(token.v3_gfni_crypto(), rest, coeff);
}

/// Overwrites the destination with a scaled source over GF(2^16).
///
/// # Panics
/// Panics if the slices differ in length or contain a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx512(
    token: archmage::X64V4xToken,
    dst: &mut [u8],
    coeff: TowerCoeff,
    src: &[u8],
) {
    check_elements("gf16::mul_into_avx512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_avx512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_into_gfni(token.v3_gfni_crypto(), dst, coeff, src);
    }
    // The peel fires only from `MUL_ADD_PEEL_MIN` bytes, far above one
    // lane, so the body below always has a whole 64-byte lane left.
    let head = peel(dst);
    let (dst_head, dst) = dst.split_at_mut(head);
    let (src_head, src) = src.split_at(head);
    if head > 0 {
        gfni::mul_into_gfni(token.v3_gfni_crypto(), dst_head, coeff, src_head);
    }
    let (same, cross) = factors(coeff);
    let swap = swap_mask();
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (slanes, srest) = src.as_chunks::<64>();
    for (d, s) in dlanes.iter_mut().zip(slanes) {
        let x = _mm512_loadu_si512(s);
        _mm512_storeu_si512(d, scale(x, _mm512_shuffle_epi8(x, swap), same, cross));
    }
    gfni::mul_into_gfni(token.v3_gfni_crypto(), drest, coeff, srest);
}

#[archmage::rite(v4x)]
fn multiply(x: __m512i, y: __m512i, swap: __m512i, even: __m512i, delta: __m512i) -> __m512i {
    let direct = _mm512_gf2p8mul_epi8(x, y);
    let crossed = _mm512_gf2p8mul_epi8(x, _mm512_shuffle_epi8(y, swap));
    let delta_bd = _mm512_gf2p8mul_epi8(_mm512_shuffle_epi8(direct, swap), delta);
    let constant = _mm512_xor_si512(direct, delta_bd);
    let extension = _mm512_xor_si512(
        _mm512_xor_si512(crossed, _mm512_shuffle_epi8(crossed, swap)),
        direct,
    );
    _mm512_xor_si512(
        _mm512_and_si512(constant, even),
        _mm512_andnot_si512(even, extension),
    )
}

/// Overwrites the destination with elementwise tower products.
///
/// # Panics
/// Panics unless all buffers have the same complete-element length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx512(
    token: archmage::X64V4xToken,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
    b_raw: u8,
    reduction: u8,
) {
    check_elements("gf16::mul_elementwise_avx512", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_avx512: dst and a lengths differ"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_avx512: dst and b lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_elementwise_gfni(token.v3_gfni_crypto(), dst, a, b, b_raw, reduction);
    }
    let swap = swap_mask();
    let even = _mm512_set1_epi16(0x00ff);
    let delta = _mm512_set1_epi16(i16::from_ne_bytes([b_raw, 0]));
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (alanes, arest) = a.as_chunks::<64>();
    let (blanes, brest) = b.as_chunks::<64>();
    for ((d, a), b) in dlanes.iter_mut().zip(alanes).zip(blanes) {
        _mm512_storeu_si512(
            d,
            multiply(
                _mm512_loadu_si512(a),
                _mm512_loadu_si512(b),
                swap,
                even,
                delta,
            ),
        );
    }
    gfni::mul_elementwise_gfni(
        token.v3_gfni_crypto(),
        drest,
        arest,
        brest,
        b_raw,
        reduction,
    );
}

/// Multiplies the destination in place by the elementwise source.
///
/// # Panics
/// Panics if the slices differ in length or contain a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx512(
    token: archmage::X64V4xToken,
    dst: &mut [u8],
    src: &[u8],
    b_raw: u8,
    reduction: u8,
) {
    check_elements("gf16::mul_elementwise_assign_avx512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_avx512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_elementwise_assign_gfni(
            token.v3_gfni_crypto(),
            dst,
            src,
            b_raw,
            reduction,
        );
    }
    let swap = swap_mask();
    let even = _mm512_set1_epi16(0x00ff);
    let delta = _mm512_set1_epi16(i16::from_ne_bytes([b_raw, 0]));
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (slanes, srest) = src.as_chunks::<64>();
    for (d, s) in dlanes.iter_mut().zip(slanes) {
        let x = _mm512_loadu_si512(&*d);
        _mm512_storeu_si512(d, multiply(x, _mm512_loadu_si512(s), swap, even, delta));
    }
    gfni::mul_elementwise_assign_gfni(token.v3_gfni_crypto(), drest, srest, b_raw, reduction);
}

/// Accumulates every `(coefficients, source)` term into destination rows.
///
/// # Panics
/// Panics if row geometry overflows, a row span exceeds `rows`, or a term
/// does not provide the requested coefficients and source bytes.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512<E>(
    token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[E], &[u8])],
) where
    for<'a> [(&'a [E], &'a [u8])]: Matrix<TowerCoeff>,
{
    check_elements("gf16::mul_add_matrix_avx512", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_avx512: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_avx512: rows cannot hold requested rows"
    );
    check_terms("gf16::mul_add_matrix_avx512", row_len, nrows, terms);
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_avx512_with(token, rows, row_len, nrows, terms);
}

/// Accumulates a borrowed matrix provider into destination rows.
///
/// # Panics
/// Panics if the row length contains a partial element or the requested row
/// span overflows or exceeds `rows`.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx512_with<M: Matrix<TowerCoeff> + ?Sized>(
    token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    check_elements("gf16::mul_add_matrix_avx512_with", row_len);
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        return;
    }
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix_avx512_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix_avx512_with: rows cannot hold requested rows"
    );
    let mut span = row_len;
    for term in 0..terms.len() {
        span = span.min(terms.source(term).len());
    }
    if span == 0 {
        return;
    }
    if span < 64 {
        return gfni::mul_add_matrix_gfni_with(token.v3_gfni_crypto(), rows, row_len, nrows, terms);
    }
    let mut iter = rows[..used].chunks_exact_mut(row_len);
    let swap = swap_mask();
    let mut first = 0;
    while first + 4 <= nrows {
        let group: [&mut [u8]; 4] =
            core::array::from_fn(|_| iter.next().expect("checked row span"));
        matrix_group(token, group, first, span, terms, swap);
        first += 4;
    }
    if first + 2 <= nrows {
        let group: [&mut [u8]; 2] =
            core::array::from_fn(|_| iter.next().expect("checked row span"));
        matrix_group(token, group, first, span, terms, swap);
        first += 2;
    }
    if first < nrows {
        let group: [&mut [u8]; 1] = [iter.next().expect("checked row span")];
        matrix_group(token, group, first, span, terms, swap);
    }
}

#[archmage::rite(v4x, import_intrinsics)]
fn matrix_group<const N: usize, M: Matrix<TowerCoeff> + ?Sized>(
    token: archmage::X64V4xToken,
    mut rows: [&mut [u8]; N],
    first: usize,
    span: usize,
    terms: &M,
    swap: __m512i,
) {
    const TERM_BLOCK: usize = 16;
    let full = span / 64 * 64;
    for block in (0..terms.len()).step_by(TERM_BLOCK) {
        let count = (terms.len() - block).min(TERM_BLOCK);
        let mut factors: [[(__m512i, __m512i); N]; TERM_BLOCK] =
            core::array::from_fn(|_| [(_mm512_setzero_si512(), _mm512_setzero_si512()); N]);
        for (term, group) in factors.iter_mut().enumerate().take(count) {
            for (k, pair) in group.iter_mut().enumerate() {
                let (same, cross) = broadcast_words(terms.coefficient(block + term, first + k));
                // One broadcast per term and row for the whole block: the
                // offset loop reuses these instead of re-issuing them for
                // every 64-byte lane.
                *pair = (_mm512_set1_epi16(same), _mm512_set1_epi16(cross));
            }
        }
        for offset in (0..full).step_by(64) {
            let mut acc: [__m512i; N] = core::array::from_fn(|k| {
                let lane: &[u8; 64] = rows[k][offset..offset + 64]
                    .try_into()
                    .expect("complete destination lane");
                _mm512_loadu_si512(lane)
            });
            for (term, group) in factors.iter().enumerate().take(count) {
                let source: &[u8; 64] = terms.source(block + term)[offset..offset + 64]
                    .try_into()
                    .expect("complete source lane");
                let x = _mm512_loadu_si512(source);
                let swapped = _mm512_shuffle_epi8(x, swap);
                for (value, &(same, cross)) in acc.iter_mut().zip(group) {
                    *value = _mm512_xor_si512(*value, scale(x, swapped, same, cross));
                }
            }
            for (row, value) in rows.iter_mut().zip(acc) {
                let lane: &mut [u8; 64] = (&mut row[offset..offset + 64])
                    .try_into()
                    .expect("complete destination lane");
                _mm512_storeu_si512(lane, value);
            }
        }
        for (k, row) in rows.iter_mut().enumerate() {
            for term in block..block + count {
                gfni::mul_add_gfni(
                    token.v3_gfni_crypto(),
                    &mut row[full..span],
                    terms.coefficient(term, first + k),
                    &terms.source(term)[full..span],
                );
            }
        }
    }
}
