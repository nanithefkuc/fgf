//! Fan–Paar GF(2^64) AVX2 kernels.

use super::super::gf32::avx2::{FanPaar32LanesAvx2, fan_paar32_lanes_avx2, scale_fan_paar32_avx2};
use crate::field::Elem;
use crate::field::binary::tower::FanPaar64Field;
use crate::kernel::scalar;
use crate::kernel::x86::gf32::load_mask;
use crate::kernel::x86::gf64::SWAP4;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// The three fp32 sub-scales of one fp64 scale.
struct Fp64Lanes {
    /// `[c0, c0 ^ mul_alpha(c1), c1]` as fp32 sub-scales — `a` even fp32
    /// lanes, `b` odd, `c` (uniform) the half-swapped source.
    a: FanPaar32LanesAvx2,
    b: FanPaar32LanesAvx2,
    c: FanPaar32LanesAvx2,
    /// `0x00000000ffffffff`: keeps each 8-byte group's low (even) fp32 lane.
    even: __m256i,
    /// `SWAP4`: exchanges the two fp32 halves of every fp64 element.
    swap4: __m256i,
}

#[archmage::rite(v3)]
fn fp64_lanes(coeff: Elem<FanPaar64Field>) -> Fp64Lanes {
    let [c0, c1] = coeff.to_components();
    let a_coeff = c0;
    let b_coeff = c0.add(c1.mul_alpha());
    let c_coeff = c1;
    Fp64Lanes {
        a: fan_paar32_lanes_avx2(a_coeff),
        b: fan_paar32_lanes_avx2(b_coeff),
        c: fan_paar32_lanes_avx2(c_coeff),
        even: _mm256_set1_epi64x(i64::from_ne_bytes(0x0000_0000_ffff_ffffu64.to_ne_bytes())),
        swap4: load_mask(&SWAP4),
    }
}

/// `coeff * src` for one 32-byte lane, given the precomputed fp32 sub-scales.
#[archmage::rite(v3)]
fn scale_fp64(x: __m256i, lanes: &Fp64Lanes) -> __m256i {
    // even fp32 lanes ← a·x, odd fp32 lanes ← b·x.
    let xa = scale_fan_paar32_avx2(x, &lanes.a);
    let xb = scale_fan_paar32_avx2(x, &lanes.b);
    let same = _mm256_xor_si256(
        _mm256_and_si256(xa, lanes.even),
        _mm256_andnot_si256(lanes.even, xb),
    );
    // uniform c on the half-swapped source.
    let cross = scale_fan_paar32_avx2(_mm256_shuffle_epi8(x, lanes.swap4), &lanes.c);
    _mm256_xor_si256(same, cross)
}

/// `dst ^= coeff * src` (Fan–Paar GF(2^64), AVX2).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar64Field>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf64::mul_add_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::gf64::mul_add_avx2: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = fp64_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let d = _mm256_loadu_si256(&*dst_lane);
        let r = scale_fp64(x, &l);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, r));
    }
    // Every 32-byte lane is a whole number of 8-byte elements, so the tail
    // starts on an element boundary.
    scalar::mul_add::<FanPaar64Field>(dst_tail, coeff, src_tail);
}

/// `dst = coeff * dst` (Fan–Paar GF(2^64), AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem<FanPaar64Field>) {
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::gf64::mul_assign_avx2: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = fp64_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_fp64(x, &l));
    }
    scalar::mul_assign::<FanPaar64Field>(dst_tail, coeff);
}

/// `dst = coeff * src` (Fan–Paar GF(2^64), AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar64Field>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf64::mul_into_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::gf64::mul_into_avx2: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = fp64_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, scale_fp64(x, &l));
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<FanPaar64Field>(dst_tail, coeff);
}
