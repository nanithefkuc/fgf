//! Mersenne31 (`u32` lanes) AVX2 kernels.
//!
//! Eight lanes per vector. `2^31 ≡ 1 (mod p)`, so a multiply folds the
//! 62-bit product with a mask/shift/add and one conditional subtract; the
//! products come from `vpmuludq` on the even and odd 32-bit lanes. Serves
//! the `V3`/`V3GfniCrypto` backends: each entry is a safe capability-token
//! function taking an `X64V3Token` and validating its own geometry.
//!
//! The lane helpers are `pub(super)`: the `QuadMersenne31` extension is
//! pairs of these lanes, and `qm31_avx2` builds on the same arithmetic.

use super::{M31_P, check_elem_multiple, check_equal};
use crate::field::mersenne31::{self, Mersenne31};
use crate::kernel::prime;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Fold eight raw `u32` lanes to `0..=p` (`2^31 ≡ 1`, no conditional
/// subtract). The result is `p` only for the raw lane `p` itself; every
/// consumer below tolerates `= p` inputs and canonicalizes with min chains.
#[archmage::rite(v3)]
pub(super) fn m31_fold256(x: __m256i, p: __m256i) -> __m256i {
    let lo = _mm256_and_si256(x, p);
    let hi = _mm256_srli_epi32(x, 31);
    _mm256_add_epi32(lo, hi)
}

/// Canonicalize lanes in `0..=2p`: two unconditional subtract-and-min steps.
/// `min(x, x-p)` then `min(u, u-p)` handles both the `p` and the `2p` edge
/// cases a single conditional subtract misses (raw lane `p` ≡ 0).
#[archmage::rite(v3)]
pub(super) fn m31_min_chain256(x: __m256i, p: __m256i) -> __m256i {
    let u = _mm256_min_epu32(x, _mm256_sub_epi32(x, p));
    _mm256_min_epu32(u, _mm256_sub_epi32(u, p))
}

/// `a * b (mod p)` over eight 31-bit lanes (`0..=p`; fold output allowed).
///
/// Doubled-odd-limb form: `vpmuludq` on the even lanes and on the odd lanes
/// shifted down 31 (doubling them), blend-based recombination, one Mersenne
/// fold and one conditional subtract. Twelve arithmetic ops per vector; the
/// odd-lane gather is a port-5 `movehdup`, not a shift.
#[archmage::rite(v3)]
#[must_use]
pub(super) fn m31_mulmod256(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let a_odd = _mm256_srli_epi64::<31>(a); // odd lanes, doubled
    let b_odd = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(b)));
    let prod_odd_dbl = _mm256_mul_epu32(a_odd, b_odd);
    let prod_evn = _mm256_mul_epu32(a, b);
    let prod_odd_lo = _mm256_slli_epi64::<31>(prod_odd_dbl);
    let prod_evn_hi = _mm256_srli_epi64::<31>(prod_evn);
    let lo_dirty = _mm256_blend_epi32::<0b1010_1010>(prod_evn, prod_odd_lo);
    let hi = _mm256_blend_epi32::<0b1010_1010>(prod_evn_hi, prod_odd_dbl);
    let lo = _mm256_and_si256(lo_dirty, p);
    let t = _mm256_add_epi32(lo, hi); // per dword < 2^32; t = p ⇒ prod ≡ 0
    _mm256_min_epu32(t, _mm256_sub_epi32(t, p))
}

/// `dst += src (mod p)`, Mersenne31, AVX2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal("m31::add_assign_avx2", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::add_assign_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_rest) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
            let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
            let sum = _mm256_add_epi32(d, s); // <= 2p
            _mm256_storeu_si256(dst_lane, m31_min_chain256(sum, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
        let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
        let sum = _mm256_add_epi32(d, s);
        _mm256_storeu_si256(dst_lane, m31_min_chain256(sum, p));
    }
    prime::add_assign::<Mersenne31>(dst_tail, src_tail);
}

/// `dst -= src (mod p)`, Mersenne31, AVX2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal("m31::sub_assign_avx2", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::sub_assign_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_rest) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
            let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
            let diff = _mm256_sub_epi32(d, s); // wraps iff negative
            let fixed = _mm256_min_epu32(diff, _mm256_add_epi32(diff, p));
            _mm256_storeu_si256(dst_lane, m31_min_chain256(fixed, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
        let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
        let diff = _mm256_sub_epi32(d, s);
        let fixed = _mm256_min_epu32(diff, _mm256_add_epi32(diff, p));
        _mm256_storeu_si256(dst_lane, m31_min_chain256(fixed, p));
    }
    prime::sub_assign::<Mersenne31>(dst_tail, src_tail);
}

/// `dst += coeff * src (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal("m31::mul_add_avx2", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::mul_add_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cvec = _mm256_set1_epi32(coeff as i32);
    let cred = m31_min_chain256(m31_fold256(cvec, p), p);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_rest) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let prod = m31_mulmod256(cred, m31_fold256(_mm256_loadu_si256(src_lane), p), p);
            let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
            let sum = _mm256_add_epi32(d, prod); // <= 2p - 1
            let u = _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p));
            _mm256_storeu_si256(dst_lane, u);
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let prod = m31_mulmod256(cred, m31_fold256(_mm256_loadu_si256(src_lane), p), p);
        let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
        let sum = _mm256_add_epi32(d, prod);
        let u = _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p));
        _mm256_storeu_si256(dst_lane, u);
    }
    prime::mul_add::<Mersenne31>(dst_tail, mersenne31::Elem(coeff), src_tail);
}

/// `dst *= coeff (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u32) {
    check_elem_multiple("m31::mul_assign_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cred = m31_min_chain256(m31_fold256(_mm256_set1_epi32(coeff as i32), p), p);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    for dst_tile in dst_tiles {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        for dst_lane in dst_lanes {
            let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
            _mm256_storeu_si256(dst_lane, m31_mulmod256(cred, d, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = m31_fold256(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, m31_mulmod256(cred, d, p));
    }
    prime::mul_assign::<Mersenne31>(dst_tail, mersenne31::Elem(coeff));
}

/// `dst = coeff * src (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal("m31::mul_into_avx2", "dst", dst.len(), "src", src.len());
    check_elem_multiple("m31::mul_into_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cred = m31_min_chain256(m31_fold256(_mm256_set1_epi32(coeff as i32), p), p);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_rest) = src.as_chunks::<128>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (src_lanes, _) = src_tile.as_chunks::<32>();
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
            _mm256_storeu_si256(dst_lane, m31_mulmod256(cred, s, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = m31_fold256(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, m31_mulmod256(cred, s, p));
    }
    // Scalar tail.
    for (d, s) in dst_tail.chunks_exact_mut(4).zip(src_tail.chunks_exact(4)) {
        let v =
            mersenne31::Elem(coeff) * mersenne31::Elem(u32::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX2.
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_m31_avx2(_token: archmage::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("m31::mul_elementwise_avx2", "dst", dst.len(), "a", a.len());
    check_equal("m31::mul_elementwise_avx2", "dst", dst.len(), "b", b.len());
    check_elem_multiple("m31::mul_elementwise_avx2", dst.len(), 4);
    let p = _mm256_set1_epi32(M31_P);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<128>();
    let (a_tiles, a_rest) = a.as_chunks::<128>();
    let (b_tiles, b_rest) = b.as_chunks::<128>();
    for (dst_tile, (a_tile, b_tile)) in dst_tiles.iter_mut().zip(a_tiles.iter().zip(b_tiles)) {
        let (dst_lanes, _) = dst_tile.as_chunks_mut::<32>();
        let (a_lanes, _) = a_tile.as_chunks::<32>();
        let (b_lanes, _) = b_tile.as_chunks::<32>();
        for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
            let va = m31_fold256(_mm256_loadu_si256(a_lane), p);
            let vb = m31_fold256(_mm256_loadu_si256(b_lane), p);
            _mm256_storeu_si256(dst_lane, m31_mulmod256(va, vb, p));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<32>();
    let (a_lanes, a_tail) = a_rest.as_chunks::<32>();
    let (b_lanes, b_tail) = b_rest.as_chunks::<32>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = m31_fold256(_mm256_loadu_si256(a_lane), p);
        let vb = m31_fold256(_mm256_loadu_si256(b_lane), p);
        _mm256_storeu_si256(dst_lane, m31_mulmod256(va, vb, p));
    }
    prime::mul_elementwise::<Mersenne31>(dst_tail, a_tail, b_tail);
}
