//! Single-buffer GF(2^16) kernels.
//!
//! The multiply strategy with no tables at all: two `vpbroadcastw` give the
//! alternating coefficients and two byte multiplies give a 32-byte lane, so
//! what is left here is the `mul_add`/`mul_assign`/`mul_into` wiring around
//! [`scale_gfni`](super::scale_gfni) — the chain depth that hides the multiply latency, the
//! non-temporal store split, and the 16-byte step each kernel takes before
//! its sub-lane tail.

use super::super::{broadcast_words, check_elements, swap_mask_avx2};

use crate::kernel::tables::TowerCoeff;
use crate::kernel::x86::{
    HALF_LANE_PEEL_MIN, PREFETCH_AHEAD, nt_split, prefetch_dst, prefetch_tile, store_avx2,
};

use super::scale_gfni;
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
fn scale_half(src: __m128i, swapped: __m128i, same: __m128i, cross: __m128i) -> __m128i {
    _mm_xor_si128(
        _mm_gf2p8mul_epi8(src, same),
        _mm_gf2p8mul_epi8(swapped, cross),
    )
}

/// Scale at most one 128-bit lane of elements through one staged multiply.
///
/// A zero-padded staging array carries the bytes through the same
/// `GF2P8MULB` pair the lanes use; the padding is never copied back.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn scale_tail(
    bytes: &[u8],
    same_half: __m128i,
    cross_half: __m128i,
    swap_half: __m128i,
) -> [u8; 16] {
    let mut staged = [0u8; 16];
    staged[..bytes.len()].copy_from_slice(bytes);
    let x = _mm_loadu_si128(&staged);
    let product = scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half);
    let mut out = [0u8; 16];
    _mm_storeu_si128(&mut out, product);
    out
}

/// `dst ^= coeff * src` for the sub-lane remainder of [`mul_add_gfni`].
///
/// The remainder is element-aligned and shorter than one 16-byte lane; one
/// staged [`scale_tail`] multiply covers it, and its padding is never copied
/// back. Entries call this through a `core::hint::black_box`ed function
/// pointer, so the staging arrays stay out of the entry bodies.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_add_tail(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(_mm256_set1_epi16(same_word)),
        _mm256_castsi256_si128(_mm256_set1_epi16(cross_word)),
        _mm256_castsi256_si128(swap_mask_avx2()),
    );
    let scaled = scale_tail(src, same_half, cross_half, swap_half);
    for (d, &s) in dst.iter_mut().zip(&scaled[..src.len()]) {
        *d ^= s;
    }
}

/// `dst = coeff * dst` for the sub-lane remainder of [`mul_assign_gfni`].
///
/// The remainder is element-aligned and shorter than one 16-byte lane; one
/// staged [`scale_tail`] multiply covers it and replaces the remainder.
/// Entries call this through a `core::hint::black_box`ed function pointer,
/// so the staging arrays stay out of the entry bodies.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_assign_tail(dst: &mut [u8], coeff: TowerCoeff) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(_mm256_set1_epi16(same_word)),
        _mm256_castsi256_si128(_mm256_set1_epi16(cross_word)),
        _mm256_castsi256_si128(swap_mask_avx2()),
    );
    let scaled = scale_tail(dst, same_half, cross_half, swap_half);
    dst.copy_from_slice(&scaled[..dst.len()]);
}

/// `dst = coeff * src` for the sub-lane remainder of [`mul_into_gfni`].
///
/// The remainder is element-aligned and shorter than one 16-byte lane; one
/// staged [`scale_tail`] multiply covers it and replaces the destination
/// remainder. Entries call this through a `core::hint::black_box`ed function
/// pointer, so the staging arrays stay out of the entry bodies.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_into_tail(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(_mm256_set1_epi16(same_word)),
        _mm256_castsi256_si128(_mm256_set1_epi16(cross_word)),
        _mm256_castsi256_si128(swap_mask_avx2()),
    );
    let scaled = scale_tail(src, same_half, cross_half, swap_half);
    dst.copy_from_slice(&scaled[..dst.len()]);
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
    check_elements("gf16::mul_add_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    // The half-lane case pays one 128-bit head so the 32-byte lanes stop
    // splitting cache lines; the floor is HALF_LANE_PEEL_MIN. The aligned
    // path hands its slices to the body untouched.
    if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
        let (same_word, cross_word) = broadcast_words(coeff);
        let (same_half, cross_half, swap_half) = (
            _mm256_castsi256_si128(_mm256_set1_epi16(same_word)),
            _mm256_castsi256_si128(_mm256_set1_epi16(cross_word)),
            _mm256_castsi256_si128(swap_mask_avx2()),
        );
        let x = _mm_loadu_si128(&src[..16].as_chunks::<16>().0[0]);
        let p = scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half);
        let d16 = &mut dst[..16].as_chunks_mut::<16>().0[0];
        let d = _mm_loadu_si128(d16);
        _mm_storeu_si128(d16, _mm_xor_si128(d, p));
        mul_add_impl(&mut dst[16..], coeff, &src[16..]);
    } else {
        mul_add_impl(dst, coeff, src);
    }
}

/// The `mul_add` lanes over the slices the entry hands them.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_add_impl(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask_avx2();
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );

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
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let p = scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half);
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, p));
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary. The `black_box`ed function pointer keeps the tail
    // out of line, so this body carries no staging arrays.
    if !src_tail.is_empty() {
        let tail: fn(&mut [u8], TowerCoeff, &[u8]) = mul_add_tail;
        core::hint::black_box(tail)(dst_tail, coeff, src_tail);
    }
}

/// `dst = coeff * dst` with `GF2P8MULB` over 32-byte lanes.
///
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gfni(_token: archmage::X64V3GfniCryptoToken, dst: &mut [u8], coeff: TowerCoeff) {
    check_elements("gf16::mul_assign_gfni", dst.len());
    // The half-lane case pays one 128-bit head so the 32-byte lanes stop
    // splitting cache lines; the floor is HALF_LANE_PEEL_MIN. The aligned
    // path hands its slices to the body untouched.
    if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
        let (same_word, cross_word) = broadcast_words(coeff);
        let (same_half, cross_half, swap_half) = (
            _mm256_castsi256_si128(_mm256_set1_epi16(same_word)),
            _mm256_castsi256_si128(_mm256_set1_epi16(cross_word)),
            _mm256_castsi256_si128(swap_mask_avx2()),
        );
        let d16 = &mut dst[..16].as_chunks_mut::<16>().0[0];
        let x = _mm_loadu_si128(d16);
        let p = scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half);
        _mm_storeu_si128(d16, p);
        mul_assign_impl(&mut dst[16..], coeff);
    } else {
        mul_assign_impl(dst, coeff);
    }
}

/// The `mul_assign` lanes over the slices the entry hands them.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_assign_impl(dst: &mut [u8], coeff: TowerCoeff) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask_avx2();
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );
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
    for dst_lane in dst_lanes {
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(
            dst_lane,
            scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half),
        );
    }
    // Both steps above are a whole number of elements, so the tail starts on
    // an element boundary. The `black_box`ed function pointer keeps the tail
    // out of line, so this body carries no staging arrays.
    if !dst_tail.is_empty() {
        let tail: fn(&mut [u8], TowerCoeff) = mul_assign_tail;
        core::hint::black_box(tail)(dst_tail, coeff);
    }
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
    check_elements("gf16::mul_into_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    if let Some(peel) = nt_split(dst, 2) {
        let (head, body) = dst.split_at_mut(peel);
        let (src_head, src_body) = src.split_at(peel);
        mul_into_impl::<false>(head, coeff, src_head);
        mul_into_impl::<true>(body, coeff, src_body);
        _mm_sfence();
    } else {
        // Below the streaming threshold a half-lane destination still
        // pays split lines; one narrow head aligns the temporal body.
        let head = if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
            16
        } else {
            0
        };
        if head != 0 {
            let (dst_head, dst) = dst.split_at_mut(head);
            let (src_head, src) = src.split_at(head);
            mul_into_impl::<false>(dst_head, coeff, src_head);
            mul_into_impl::<false>(dst, coeff, src);
        } else {
            mul_into_impl::<false>(dst, coeff, src);
        }
    }
}

/// The `mul_into` body: four independent multiply chains per 128-byte tile,
/// stored non-temporally when `NT`.
///
/// Only the streaming stores remain unsafe; every load and every ordinary
/// store goes through a checked slice window. A `NT` body is entered only
/// through the [`nt_split`] peel in [`mul_into_gfni`], which starts it
/// on a 32-byte boundary, and the caller issues the matching `_mm_sfence`.
///
/// This operation has no panic conditions beyond those established by the
/// entrypoint.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn mul_into_impl<const NT: bool>(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask_avx2();

    // The cursor walks whole tiles over the remaining destination so the
    // prefetch can name a line ahead of it without leaving the slice it
    // came from; see [`prefetch_dst`].
    let prefetch = prefetch_dst(dst, NT);
    let tiles = dst.len() / 128 * 128;
    let (mut dst_rest_tiles, dst_rest) = dst.split_at_mut(tiles);
    let (mut src_rest_tiles, src_rest) = src.split_at(tiles);
    while !dst_rest_tiles.is_empty() {
        if prefetch && dst_rest_tiles.len() >= PREFETCH_AHEAD + 128 {
            prefetch_tile(&dst_rest_tiles[PREFETCH_AHEAD..]);
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
            //        and `store_avx2` writes exactly 32 bytes at its pointer.
            // THUS: the store remains within `dst`.
            //
            // ALIGNMENT
            // SINCE: `NT` bodies are entered only through the `nt_split`
            //        peel in `mul_into_gfni`, whose contract starts the body
            //        destination on a 32-byte boundary, and tiles advance in
            //        whole 32-byte lanes.
            // THUS: the pointer meets `store_avx2`'s 32-byte alignment
            //        requirement when `NT`.
            unsafe { store_avx2::<NT>(dst_lane.as_mut_ptr(), p) };
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
    let (same_half, cross_half, swap_half) = (
        _mm256_castsi256_si128(same),
        _mm256_castsi256_si128(cross),
        _mm256_castsi256_si128(swap),
    );
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let p = scale_half(x, _mm_shuffle_epi8(x, swap_half), same_half, cross_half);
        _mm_storeu_si128(dst_lane, p);
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary. The `black_box`ed function pointer keeps the tail
    // out of line, so this body carries no staging arrays.
    if !src_tail.is_empty() {
        let tail: fn(&mut [u8], TowerCoeff, &[u8]) = mul_into_tail;
        core::hint::black_box(tail)(dst_tail, coeff, src_tail);
    }
}
