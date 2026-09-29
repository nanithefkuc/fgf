//! AVX-512 GFNI multiplication over interleaved tower elements.
//!
//! Each lane contains complete two-byte elements. The adjacent-byte exchange
//! and coefficient broadcasts share the same tower identity as the narrower
//! GFNI kernels; remainders descend through their proven token.

use crate::kernel::tables::TowerCoeff;

use super::{broadcast_words, elementwise, gfni};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const SWAP_512: [u8; 64] = {
    let mut mask = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        mask[i] = super::SWAP_ADJACENT[i % 16];
        i += 1;
    }
    mask
};

#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn swap512() -> __m512i {
    _mm512_loadu_si512(&SWAP_512)
}

#[archmage::rite(v4x)]
pub(super) fn factors512(coeff: TowerCoeff) -> (__m512i, __m512i) {
    let (same, cross) = broadcast_words(coeff);
    (_mm512_set1_epi16(same), _mm512_set1_epi16(cross))
}

#[archmage::rite(v4x)]
pub(super) fn scale512(x: __m512i, swapped: __m512i, same: __m512i, cross: __m512i) -> __m512i {
    _mm512_xor_si512(
        _mm512_gf2p8mul_epi8(x, same),
        _mm512_gf2p8mul_epi8(swapped, cross),
    )
}

/// Shortest single-row destination that peels its head to a 64-byte
/// boundary. `mul_add` and `mul_into` share the floor.
///
/// The peel runs up to 62 bytes through the 32-byte GFNI kernel, which
/// repays only once the 64-byte body is long enough. Set by the threshold
/// variants in `BENCHMARKS.md` ("AVX-512 byte, popcount, and prime-field
/// kernels").
pub(crate) const MUL_ADD_PEEL_MIN: usize = 3584;

/// Head bytes that put `dst`'s 64-byte accesses on a cache-line boundary,
/// or `0` below [`MUL_ADD_PEEL_MIN`] or when the head would split an element.
#[inline]
fn peel64(dst: &[u8]) -> usize {
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
pub fn mul_add512(token: archmage::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    super::check_elements("gf16::mul_add512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_add_gfni(token.v3_gfni_crypto(), dst, coeff, src);
    }
    // The peel fires only from `MUL_ADD_PEEL_MIN` bytes, far above one
    // lane, so the body below always has a whole 64-byte lane left.
    let head = peel64(dst);
    let (dst_head, dst) = dst.split_at_mut(head);
    let (src_head, src) = src.split_at(head);
    if head > 0 {
        gfni::mul_add_gfni(token.v3_gfni_crypto(), dst_head, coeff, src_head);
    }
    let (same, cross) = factors512(coeff);
    let swap = swap512();
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
            let product = scale512(x, _mm512_shuffle_epi8(x, swap), same, cross);
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
            _mm512_xor_si512(old, scale512(x, _mm512_shuffle_epi8(x, swap), same, cross)),
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
pub fn mul_assign512(token: archmage::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff) {
    super::check_elements("gf16::mul_assign512", dst.len());
    if dst.len() < 64 {
        return gfni::mul_assign_gfni(token.v3_gfni_crypto(), dst, coeff);
    }
    let (same, cross) = factors512(coeff);
    let swap = swap512();
    let (lanes, rest) = dst.as_chunks_mut::<64>();
    for lane in lanes {
        let x = _mm512_loadu_si512(&*lane);
        _mm512_storeu_si512(lane, scale512(x, _mm512_shuffle_epi8(x, swap), same, cross));
    }
    gfni::mul_assign_gfni(token.v3_gfni_crypto(), rest, coeff);
}

/// Overwrites the destination with a scaled source over GF(2^16).
///
/// # Panics
/// Panics if the slices differ in length or contain a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into512(token: archmage::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    super::check_elements("gf16::mul_into512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return gfni::mul_into_gfni(token.v3_gfni_crypto(), dst, coeff, src);
    }
    // The peel fires only from `MUL_ADD_PEEL_MIN` bytes, far above one
    // lane, so the body below always has a whole 64-byte lane left.
    let head = peel64(dst);
    let (dst_head, dst) = dst.split_at_mut(head);
    let (src_head, src) = src.split_at(head);
    if head > 0 {
        gfni::mul_into_gfni(token.v3_gfni_crypto(), dst_head, coeff, src_head);
    }
    let (same, cross) = factors512(coeff);
    let swap = swap512();
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (slanes, srest) = src.as_chunks::<64>();
    for (d, s) in dlanes.iter_mut().zip(slanes) {
        let x = _mm512_loadu_si512(s);
        _mm512_storeu_si512(d, scale512(x, _mm512_shuffle_epi8(x, swap), same, cross));
    }
    gfni::mul_into_gfni(token.v3_gfni_crypto(), drest, coeff, srest);
}

#[archmage::rite(v4x)]
fn multiply512(x: __m512i, y: __m512i, swap: __m512i, even: __m512i, delta: __m512i) -> __m512i {
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
pub fn mul_elementwise512(token: archmage::X64V4xToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    super::check_elements("gf16::mul_elementwise512", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise512: dst and a lengths differ"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise512: dst and b lengths differ"
    );
    if dst.len() < 64 {
        return elementwise::mul_elementwise_gfni(token.v3_gfni_crypto(), dst, a, b);
    }
    let swap = swap512();
    let even = _mm512_set1_epi16(0x00ff);
    let delta = _mm512_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (alanes, arest) = a.as_chunks::<64>();
    let (blanes, brest) = b.as_chunks::<64>();
    for ((d, a), b) in dlanes.iter_mut().zip(alanes).zip(blanes) {
        _mm512_storeu_si512(
            d,
            multiply512(
                _mm512_loadu_si512(a),
                _mm512_loadu_si512(b),
                swap,
                even,
                delta,
            ),
        );
    }
    elementwise::mul_elementwise_gfni(token.v3_gfni_crypto(), drest, arest, brest);
}

/// Multiplies the destination in place by the elementwise source.
///
/// # Panics
/// Panics if the slices differ in length or contain a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign512(token: archmage::X64V4xToken, dst: &mut [u8], src: &[u8]) {
    super::check_elements("gf16::mul_elementwise_assign512", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign512: dst and src lengths differ"
    );
    if dst.len() < 64 {
        return elementwise::mul_elementwise_assign_gfni(token.v3_gfni_crypto(), dst, src);
    }
    let swap = swap512();
    let even = _mm512_set1_epi16(0x00ff);
    let delta = _mm512_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let (dlanes, drest) = dst.as_chunks_mut::<64>();
    let (slanes, srest) = src.as_chunks::<64>();
    for (d, s) in dlanes.iter_mut().zip(slanes) {
        let x = _mm512_loadu_si512(&*d);
        _mm512_storeu_si512(d, multiply512(x, _mm512_loadu_si512(s), swap, even, delta));
    }
    elementwise::mul_elementwise_assign_gfni(token.v3_gfni_crypto(), drest, srest);
}
