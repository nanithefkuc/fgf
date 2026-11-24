//! `QuadMersenne31` (`(re, im)` Mersenne31 limb pairs) SSE4.2 additive kernels.
//!
//! Two complex elements per 128-bit vector: four 32-bit lanes holding
//! `[re0, im0, re1, im1]`, little-endian, so the even 32-bit lanes are the
//! real limbs and the odd lanes the imaginary limbs. The limb canonicalization
//! — the direct `pminud` min chain — is shared with
//! [`crate::kernel::x86::mersenne31`].
//!
//! Packed buffers accept arbitrary limb bit patterns, as specified by
//! [`crate::ops`]; every stored limb is canonical (`< p`). Each entry is a
//! safe capability-token function taking an `X64V2Token`, validating its own
//! geometry, and handing any trailing single element to the portable
//! [`crate::kernel::prime`] path.

use crate::field::Elem;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::prime;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::x86::mersenne31::{P, min_chain_sse42};

/// `dst += src` in GF((2^31 - 1)^2), SSE4.2.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "quad_mersenne31::add_assign_sse42",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::add_assign_sse42", dst.len(), 8);
    let p = _mm_set1_epi32(P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
            let s = min_chain_sse42(_mm_loadu_si128(src_lane), p);
            let sum = _mm_add_epi32(d, s); // <= 2p - 2
            _mm_storeu_si128(dst_lane, _mm_min_epu32(sum, _mm_sub_epi32(sum, p)));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
        let s = min_chain_sse42(_mm_loadu_si128(src_lane), p);
        let sum = _mm_add_epi32(d, s); // <= 2p - 2
        _mm_storeu_si128(dst_lane, _mm_min_epu32(sum, _mm_sub_epi32(sum, p)));
    }
    prime::add_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst -= src` in GF((2^31 - 1)^2), SSE4.2.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "quad_mersenne31::sub_assign_sse42",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::sub_assign_sse42", dst.len(), 8);
    let p = _mm_set1_epi32(P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
            let s = min_chain_sse42(_mm_loadu_si128(src_lane), p);
            let diff = _mm_sub_epi32(d, s); // wraps iff negative
            _mm_storeu_si128(dst_lane, _mm_min_epu32(diff, _mm_add_epi32(diff, p)));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
        let s = min_chain_sse42(_mm_loadu_si128(src_lane), p);
        let diff = _mm_sub_epi32(d, s); // wraps iff negative
        _mm_storeu_si128(dst_lane, _mm_min_epu32(diff, _mm_add_epi32(diff, p)));
    }
    prime::sub_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst[i] += value` in GF((2^31 - 1)^2), SSE4.2.
///
/// The destination limbs canonicalize on load; the canonical sum reduces
/// with one subtract-and-min, and every stored limb is canonical.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_sse42(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    value: Elem<QuadMersenne31>,
) {
    check_elem_multiple("quad_mersenne31::add_assign_scalar_sse42", dst.len(), 8);
    let [vr, vi] = value.to_raw();
    // Broadcast [vr, vi] onto the two complex-element slots [re0, im0, re1, im1].
    let pair = (u64::from(vi) << 32) | u64::from(vr);
    let svec = _mm_set1_epi64x(pair.cast_signed());
    let p = _mm_set1_epi32(P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    for dst_tile in dst_tiles {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        for dst_lane in dst_lanes {
            let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
            let sum = _mm_add_epi32(d, svec); // <= 2p - 2
            _mm_storeu_si128(dst_lane, _mm_min_epu32(sum, _mm_sub_epi32(sum, p)));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
        let sum = _mm_add_epi32(d, svec); // <= 2p - 2
        _mm_storeu_si128(dst_lane, _mm_min_epu32(sum, _mm_sub_epi32(sum, p)));
    }
    prime::add_assign_scalar::<QuadMersenne31>(dst_tail, value);
}

/// `dst[i] -= value` in GF((2^31 - 1)^2), SSE4.2.
///
/// The destination limbs canonicalize on load; the canonical difference is
/// canonical after one unsigned min with its `+p` fixup, matching
/// [`sub_assign_sse42`].
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_sse42(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    value: Elem<QuadMersenne31>,
) {
    check_elem_multiple("quad_mersenne31::sub_assign_scalar_sse42", dst.len(), 8);
    let [vr, vi] = value.to_raw();
    let pair = (u64::from(vi) << 32) | u64::from(vr);
    let svec = _mm_set1_epi64x(pair.cast_signed());
    let p = _mm_set1_epi32(P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    for dst_tile in dst_tiles {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        for dst_lane in dst_lanes {
            let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
            let diff = _mm_sub_epi32(d, svec); // wraps iff negative
            _mm_storeu_si128(dst_lane, _mm_min_epu32(diff, _mm_add_epi32(diff, p)));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = min_chain_sse42(_mm_loadu_si128(&*dst_lane), p);
        let diff = _mm_sub_epi32(d, svec); // wraps iff negative
        _mm_storeu_si128(dst_lane, _mm_min_epu32(diff, _mm_add_epi32(diff, p)));
    }
    prime::sub_assign_scalar::<QuadMersenne31>(dst_tail, value);
}
