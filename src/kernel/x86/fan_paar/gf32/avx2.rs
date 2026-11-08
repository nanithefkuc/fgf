//! Fan–Paar GF(2^32) AVX2 kernels.

use crate::field::Elem;
use crate::field::binary::tower::FanPaar32Field;
use crate::kernel::scalar;
use crate::kernel::tables::FpTowerTables;
use crate::kernel::x86::align_spill_slots;
use crate::kernel::x86::gf16::{nibble_avx2, scale_avx2};
use crate::kernel::x86::gf32::{SWAP2, load_mask};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// The three fp16 sub-coefficients of one fp32 scale, in broadcast-ready form.
pub(in crate::kernel::x86::fan_paar) struct FanPaar32LanesAvx2 {
    /// `[c0, c0 ^ mul_alpha(c1), c1]` — `a` scales the even fp16 lanes of the
    /// source, `b` the odd, `c` (uniform) the half-swapped source.
    a: crate::kernel::x86::gf16::NibbleAvx2,
    b: crate::kernel::x86::gf16::NibbleAvx2,
    c: crate::kernel::x86::gf16::NibbleAvx2,
    /// `0x0000ffff`: keeps each 4-byte group's low (even) fp16 lane.
    even: __m256i,
    /// `SWAP2`: exchanges the two fp16 halves of every fp32 element.
    swap2: __m256i,
}

#[archmage::rite(v3)]
pub(in crate::kernel::x86::fan_paar) fn fan_paar32_lanes_avx2(
    coeff: Elem<FanPaar32Field>,
) -> FanPaar32LanesAvx2 {
    let [c0, c1] = coeff.to_components();
    let a_coeff = c0;
    let b_coeff = c0.add(c1.mul_alpha());
    let c_coeff = c1;
    FanPaar32LanesAvx2 {
        a: nibble_avx2(&FpTowerTables::new(a_coeff)),
        b: nibble_avx2(&FpTowerTables::new(b_coeff)),
        c: nibble_avx2(&FpTowerTables::new(c_coeff)),
        even: _mm256_set1_epi32(i32::from_ne_bytes(0x0000_ffffu32.to_ne_bytes())),
        swap2: load_mask(&SWAP2),
    }
}

/// `coeff * src` for one 32-byte lane, given the precomputed fp16 lane tables.
#[archmage::rite(v3)]
pub(in crate::kernel::x86::fan_paar) fn scale_fan_paar32_avx2(
    x: __m256i,
    lanes: &FanPaar32LanesAvx2,
) -> __m256i {
    // even fp16 lanes ← a·x, odd fp16 lanes ← b·x.
    let xa = scale_avx2(x, &lanes.a);
    let xb = scale_avx2(x, &lanes.b);
    let same = _mm256_xor_si256(
        _mm256_and_si256(xa, lanes.even),
        _mm256_andnot_si256(lanes.even, xb),
    );
    // uniform c on the half-swapped source.
    let cross = scale_avx2(_mm256_shuffle_epi8(x, lanes.swap2), &lanes.c);
    _mm256_xor_si256(same, cross)
}

/// `dst ^= coeff * src` (Fan–Paar GF(2^32), AVX2).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar32Field>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf32::mul_add_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::gf32::mul_add_avx2: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    align_spill_slots();
    let l = fan_paar32_lanes_avx2(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let d = _mm256_loadu_si256(&*dst_lane);
        let r = scale_fan_paar32_avx2(x, &l);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, r));
    }
    // Every 32-byte lane is a whole number of 4-byte elements, so the tail
    // starts on an element boundary.
    scalar::mul_add::<FanPaar32Field>(dst_tail, coeff, src_tail);
}

/// `dst = coeff * dst` (Fan–Paar GF(2^32), AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem<FanPaar32Field>) {
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::gf32::mul_assign_avx2: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    align_spill_slots();
    let l = fan_paar32_lanes_avx2(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_fan_paar32_avx2(x, &l));
    }
    scalar::mul_assign::<FanPaar32Field>(dst_tail, coeff);
}

/// `dst = coeff * src` (Fan–Paar GF(2^32), AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar32Field>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf32::mul_into_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::gf32::mul_into_avx2: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    align_spill_slots();
    let l = fan_paar32_lanes_avx2(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, scale_fan_paar32_avx2(x, &l));
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<FanPaar32Field>(dst_tail, coeff);
}
