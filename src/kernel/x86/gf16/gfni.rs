//! Single-buffer GF(2^16) kernels over `GF2P8MULB`.
//!
//! The multiply strategy with no tables at all: two `vpbroadcastw` give the
//! alternating coefficients and two byte multiplies give a 32-byte lane, so
//! what is left here is the `mul_add`/`mul_assign`/`mul_into` wiring around
//! [`scale_gfni`] — the chain depth that hides the multiply latency, the
//! non-temporal store split, and the 16-byte step each kernel takes before
//! its scalar tail.

use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::TowerCoeff;

use super::{broadcast_words, scale_gfni, swap_mask256};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `coeff * src` for one 16-byte lane, the SSE-width [`scale_gfni`].
///
/// Each 256-bit kernel takes one 128-bit step before its scalar tail, so a
/// 16..31-byte remainder is not multiplied a byte at a time. The feature list
/// is the callers' — every one of them is already `avx2,gfni`, and that also
/// guarantees the SSE2 baseline on 32-bit x86.
#[archmage::rite(v3_gfni_crypto)]
fn scale_gfni128(src: __m128i, swapped: __m128i, same: __m128i, cross: __m128i) -> __m128i {
    _mm_xor_si128(
        _mm_gf2p8mul_epi8(src, same),
        _mm_gf2p8mul_epi8(swapped, cross),
    )
}

/// `dst ^= coeff * src` with `GF2P8MULB` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: TowerCoeff,
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();
    // Four independent multiply chains per 128-byte tile: `GF2P8MULB` has
    // far more throughput than latency, and a single-destination update has
    // no other work to hide it behind.
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_rest) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let x = _mm256_loadu_si256(src_lane);
            let p = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
            let d = _mm256_loadu_si256(&*dst_lane);
            _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, p));
        }
    }
    let (dst_lanes, dst_rest) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let p = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
        let d = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, p));
    }
    // One 128-bit step down before the scalar tail. The casts are register
    // aliases, not instructions: the low half of a broadcast is the same
    // broadcast.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    let (same128, cross128, swap128) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let p = scale_gfni128(x, _mm_shuffle_epi8(x, swap128), same128, cross128);
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, p));
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary.
    mul_add_scalar(dst_tail, coeff.coeff, src_tail);
}

/// `dst = coeff * dst` with `GF2P8MULB` over 32-byte lanes.
///
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gfni(_token: archmage::X64V3GfniCryptoToken, dst: &mut [u8], coeff: TowerCoeff) {
    super::check_elements("gf16::mul_assign_gfni", dst.len());
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(
            dst_lane,
            scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross),
        );
    }
    // One 128-bit step down before the scalar tail; the casts are register
    // aliases, not instructions.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (same128, cross128, swap128) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );
    for dst_lane in dst_lanes {
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(
            dst_lane,
            scale_gfni128(x, _mm_shuffle_epi8(x, swap128), same128, cross128),
        );
    }
    // Both steps above are a whole number of elements, so the tail starts on
    // an element boundary.
    mul_assign_scalar(dst_tail, coeff.coeff);
}

/// `dst = coeff * src` with `GF2P8MULB` over 32-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: TowerCoeff,
    src: &[u8],
) {
    super::check_elements("gf16::mul_into_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    match super::nt_split(dst, 2) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_gfni_lane::<false>(head, coeff, src_head);
            mul_into_gfni_lane::<true>(body, coeff, src_body);
            _mm_sfence();
        }
        None => mul_into_gfni_lane::<false>(dst, coeff, src),
    }
}

/// The `mul_into` body: four independent multiply chains per 128-byte tile,
/// stored non-temporally when `NT`.
///
/// Only the streaming stores remain unsafe; every load and every ordinary
/// store goes through a checked slice window. A `NT` body is entered only
/// through the [`super::nt_split`] peel in [`mul_into_gfni`], which starts it
/// on a 32-byte boundary, and the caller issues the matching `_mm_sfence`.
///
/// This operation has no panic conditions beyond those established by the
/// entrypoint.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_into_gfni_lane<const NT: bool>(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();

    // The cursor walks whole tiles over the remaining destination so the
    // prefetch can name a line ahead of it without leaving the slice it
    // came from; see [`super::prefetch_dst`].
    let prefetch = super::prefetch_dst(dst, NT);
    let tiles = dst.len() / 128 * 128;
    let (mut dst_rest_tiles, dst_rest) = dst.split_at_mut(tiles);
    let (mut src_rest_tiles, src_rest) = src.split_at(tiles);
    while !dst_rest_tiles.is_empty() {
        if prefetch && dst_rest_tiles.len() >= super::PREFETCH_AHEAD + 128 {
            super::prefetch_tile(&dst_rest_tiles[super::PREFETCH_AHEAD..]);
        }
        let (dst_tile, dst_next) = dst_rest_tiles.split_at_mut(128);
        let (src_tile, src_next) = src_rest_tiles.split_at(128);
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let x = _mm256_loadu_si256(src_lane);
            let p = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `dst_lane` is a live `&mut [u8; 32]` window of `dst`,
            //        and `store256` writes exactly 32 bytes at its pointer.
            // THUS: the store remains within `dst`.
            //
            // ALIGNMENT
            // SINCE: `NT` bodies are entered only through the `nt_split`
            //        peel in `mul_into_gfni`, whose contract starts the body
            //        destination on a 32-byte boundary, and tiles advance in
            //        whole 32-byte lanes.
            // THUS: the pointer meets `store256`'s 32-byte alignment
            //        requirement when `NT`.
            unsafe { super::store256::<NT>(dst_lane.as_mut_ptr(), p) };
        }
        dst_rest_tiles = dst_next;
        src_rest_tiles = src_next;
    }
    let (dst_lanes, dst_rest) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let p = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
        _mm256_storeu_si256(dst_lane, p);
    }
    // One 128-bit step down before the scalar tail; the casts are register
    // aliases, not instructions.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    let (same128, cross128, swap128) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let p = scale_gfni128(x, _mm_shuffle_epi8(x, swap128), same128, cross128);
        _mm_storeu_si128(dst_lane, p);
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary.
    mul_into_scalar(dst_tail, coeff.coeff, src_tail);
}
