//! Mersenne31 (`u32` lanes) SSE4.2 kernels.
//!
//! Four lanes per vector, same Mersenne fold as the AVX2 path. Serves the
//! `V2` backend (x86-64-v2 guarantees SSE4.1 `pminud` and SSE4.2
//! `pcmpgtq`): each entry is a safe capability-token function taking an
//! `X64V2Token` and validating its own geometry.

use super::{M31_P, check_elem_multiple, check_equal, m31_mulmod128};
use crate::field::mersenne31::{self, Mersenne31};
use crate::kernel::prime;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Fold four raw `u32` lanes to `0..=p` (`2^31 ≡ 1`, no conditional
/// subtract), tolerating `= p` outputs like the AVX2 twin.
#[archmage::rite(v2)]
fn m31_fold128(x: __m128i, p: __m128i) -> __m128i {
    let lo = _mm_and_si128(x, p);
    let hi = _mm_srli_epi32(x, 31);
    _mm_add_epi32(lo, hi)
}

/// Canonicalize lanes in `0..=2p`: two unconditional subtract-and-min steps.
#[archmage::rite(v2)]
fn m31_min_chain128(x: __m128i, p: __m128i) -> __m128i {
    let u = _mm_min_epu32(x, _mm_sub_epi32(x, p));
    _mm_min_epu32(u, _mm_sub_epi32(u, p))
}

/// `dst += src (mod p)`, Mersenne31, SSE4.2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal("m31::add_assign_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::add_assign_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
            let s = m31_fold128(_mm_loadu_si128(src_lane), p);
            let sum = _mm_add_epi32(d, s); // <= 2p
            _mm_storeu_si128(dst_lane, m31_min_chain128(sum, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
        let s = m31_fold128(_mm_loadu_si128(src_lane), p);
        let sum = _mm_add_epi32(d, s);
        _mm_storeu_si128(dst_lane, m31_min_chain128(sum, p));
    }
    prime::add_assign::<Mersenne31>(dst_tail, src_tail);
}

/// `dst -= src (mod p)`, Mersenne31, SSE4.2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal("m31::sub_assign_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::sub_assign_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
            let s = m31_fold128(_mm_loadu_si128(src_lane), p);
            let diff = _mm_sub_epi32(d, s); // wraps iff negative
            let fixed = _mm_min_epu32(diff, _mm_add_epi32(diff, p));
            _mm_storeu_si128(dst_lane, m31_min_chain128(fixed, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
        let s = m31_fold128(_mm_loadu_si128(src_lane), p);
        let diff = _mm_sub_epi32(d, s);
        let fixed = _mm_min_epu32(diff, _mm_add_epi32(diff, p));
        _mm_storeu_si128(dst_lane, m31_min_chain128(fixed, p));
    }
    prime::sub_assign::<Mersenne31>(dst_tail, src_tail);
}

/// `dst += coeff * src (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal("m31::mul_add_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::mul_add_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    // The coefficient is loop-invariant and contract-canonical: fold it once.
    let cred = {
        let cvec = _mm_set1_epi32(coeff as i32);
        let f = m31_fold128(cvec, p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let prod = m31_mulmod128(cred, m31_fold128(_mm_loadu_si128(src_lane), p), p);
            let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
            let sum = _mm_add_epi32(d, prod); // <= 2p - 1
            let u = _mm_min_epu32(sum, _mm_sub_epi32(sum, p));
            _mm_storeu_si128(dst_lane, u);
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let prod = m31_mulmod128(cred, m31_fold128(_mm_loadu_si128(src_lane), p), p);
        let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
        let sum = _mm_add_epi32(d, prod);
        let u = _mm_min_epu32(sum, _mm_sub_epi32(sum, p));
        _mm_storeu_si128(dst_lane, u);
    }
    prime::mul_add::<Mersenne31>(dst_tail, mersenne31::Elem(coeff), src_tail);
}

/// `dst *= coeff (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u32) {
    check_elem_multiple("m31::mul_assign_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    // The coefficient is loop-invariant and contract-canonical: fold it once.
    let cred = {
        let f = m31_fold128(_mm_set1_epi32(coeff as i32), p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    for dst_tile in dst_tiles {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        for dst_lane in dst_lanes {
            let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
            _mm_storeu_si128(dst_lane, m31_mulmod128(cred, d, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = m31_fold128(_mm_loadu_si128(&*dst_lane), p);
        _mm_storeu_si128(dst_lane, m31_mulmod128(cred, d, p));
    }
    prime::mul_assign::<Mersenne31>(dst_tail, mersenne31::Elem(coeff));
}

/// `dst = coeff * src (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal("m31::mul_into_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::mul_into_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    // The coefficient is loop-invariant and contract-canonical: fold it once.
    let cred = {
        let f = m31_fold128(_mm_set1_epi32(coeff as i32), p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_tiles, src_rest) = src.as_chunks::<64>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (src_lanes, _) = src_tile.as_chunks::<16>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let s = m31_fold128(_mm_loadu_si128(src_lane), p);
            _mm_storeu_si128(dst_lane, m31_mulmod128(cred, s, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = m31_fold128(_mm_loadu_si128(src_lane), p);
        _mm_storeu_si128(dst_lane, m31_mulmod128(cred, s, p));
    }
    // Scalar tail.
    for (d, s) in dst_tail.chunks_exact_mut(4).zip(src_tail.chunks_exact(4)) {
        let v =
            mersenne31::Elem(coeff) * mersenne31::Elem(u32::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, SSE4.2.
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_m31_sse42(_token: archmage::X64V2Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("m31::mul_elementwise_sse42", "dst", dst.len(), "a", a.len());
    check_equal("m31::mul_elementwise_sse42", "dst", dst.len(), "b", b.len());
    check_elem_multiple("m31::mul_elementwise_sse42", dst.len(), 4);
    let p = _mm_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<64>();
    let (a_tiles, a_rest) = a.as_chunks::<64>();
    let (b_tiles, b_rest) = b.as_chunks::<64>();
    for (dst_tile, (a_tile, b_tile)) in dst_tiles.iter_mut().zip(a_tiles.iter().zip(b_tiles)) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<16>();
        let (a_lanes, _) = a_tile.as_chunks::<16>();
        let (b_lanes, _) = b_tile.as_chunks::<16>();
        for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
            let va = m31_fold128(_mm_loadu_si128(a_lane), p);
            let vb = m31_fold128(_mm_loadu_si128(b_lane), p);
            _mm_storeu_si128(dst_lane, m31_mulmod128(va, vb, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (a_lanes, a_tail) = a_rest.as_chunks::<16>();
    let (b_lanes, b_tail) = b_rest.as_chunks::<16>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = m31_fold128(_mm_loadu_si128(a_lane), p);
        let vb = m31_fold128(_mm_loadu_si128(b_lane), p);
        _mm_storeu_si128(dst_lane, m31_mulmod128(va, vb, p));
    }
    prime::mul_elementwise::<Mersenne31>(dst_tail, a_tail, b_tail);
}
