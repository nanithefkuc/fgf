//! Fan–Paar AVX2 kernels over 32-byte lanes.

use crate::field::Elem;
use crate::field::binary::tower::{FanPaar16, FanPaar32, FanPaar64};
use crate::kernel::scalar;
use crate::kernel::tables::FpTowerTables;
use crate::kernel::x86::gf16::{nibble_avx2, nibble_ssse3, scale_avx2, scale_ssse3};
use crate::kernel::x86::gf32::{SWAP2, load_mask};
use crate::kernel::x86::gf64::SWAP4;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `dst ^= coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2_fp16(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_add_avx2_fp16: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_add_avx2_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_avx2(tables);
    let (lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let d = _mm256_loadu_si256(&*dst_lane);
        let scaled = scale_avx2(x, &vectors);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, scaled));
    }
    // One 128-bit step down before the scalar tail; SSSE3 is implied by AVX2.
    let (narrow, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_narrow, src_tail) = src_rest.as_chunks::<16>();
    if let Some(dst_lane) = narrow.first_mut() {
        let table = nibble_ssse3(tables);
        let x = _mm_loadu_si128(&src_narrow[0]);
        let d = _mm_loadu_si128(&*dst_lane);
        let scaled = scale_ssse3(x, &table);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, scaled));
    }
    // Both steps above are a whole number of 2-byte elements, so the tail
    // starts on an element boundary. Recover the coefficient from the tables
    // it was built from for the portable fallback.
    scalar::mul_add::<FanPaar16>(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 32-byte lanes (AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2_fp16(_token: archmage::X64V3Token, dst: &mut [u8], tables: &FpTowerTables) {
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_assign_avx2_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_avx2(tables);
    let (lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_avx2(x, &vectors));
    }
    // One 128-bit step down before the scalar tail; SSSE3 is implied by AVX2.
    let (narrow, dst_tail) = dst_rest.as_chunks_mut::<16>();
    if let Some(dst_lane) = narrow.first_mut() {
        let table = nibble_ssse3(tables);
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &table));
    }
    scalar::mul_assign::<FanPaar16>(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2_fp16(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_into_avx2_fp16: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_into_avx2_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_avx2(tables);
    let (lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, scale_avx2(x, &vectors));
    }
    // One 128-bit step down before the scalar tail; SSSE3 is implied by AVX2.
    let (narrow, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_narrow, src_tail) = src_rest.as_chunks::<16>();
    if let Some(dst_lane) = narrow.first_mut() {
        let table = nibble_ssse3(tables);
        let x = _mm_loadu_si128(&src_narrow[0]);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &table));
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<FanPaar16>(dst_tail, tables.coeff);
}

// ---------------------------------------------------------------------------
// Fan–Paar GF(2^32): two fp16 lane multiplies under period-2 fp16
// coefficients, plus a 2-byte half-swap. The `mul_alpha` fold lands in
// coefficient preparation (`B = c0 ^ mul_alpha(c1)`), so the kernel is three
// uniform fp16 scales (`scale_avx2`) merged by fp16 parity. AVX2 only; SSSE3
// falls to the portable scalar path in dispatch.
// ---------------------------------------------------------------------------

/// The three fp16 sub-coefficients of one fp32 scale, in broadcast-ready form.
struct Fp32Lanes {
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
fn fp32_lanes(coeff: Elem<FanPaar32>) -> Fp32Lanes {
    let (c0, c1) = coeff.to_components();
    let a_coeff = c0;
    let b_coeff = c0.add(c1.mul_alpha());
    let c_coeff = c1;
    Fp32Lanes {
        a: nibble_avx2(&FpTowerTables::new(a_coeff)),
        b: nibble_avx2(&FpTowerTables::new(b_coeff)),
        c: nibble_avx2(&FpTowerTables::new(c_coeff)),
        even: _mm256_set1_epi32(i32::from_ne_bytes(0x0000_ffffu32.to_ne_bytes())),
        swap2: load_mask(&SWAP2),
    }
}

/// `coeff * src` for one 32-byte lane, given the precomputed fp16 lane tables.
#[archmage::rite(v3)]
fn scale_fp32(x: __m256i, lanes: &Fp32Lanes) -> __m256i {
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
pub fn mul_add_avx2_fp32(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar32>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_add_avx2_fp32: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::mul_add_avx2_fp32: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = fp32_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let d = _mm256_loadu_si256(&*dst_lane);
        let r = scale_fp32(x, &l);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, r));
    }
    // Every 32-byte lane is a whole number of 4-byte elements, so the tail
    // starts on an element boundary.
    scalar::mul_add::<FanPaar32>(dst_tail, coeff, src_tail);
}

/// `dst = coeff * dst` (Fan–Paar GF(2^32), AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2_fp32(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem<FanPaar32>) {
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::mul_assign_avx2_fp32: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = fp32_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_fp32(x, &l));
    }
    scalar::mul_assign::<FanPaar32>(dst_tail, coeff);
}

/// `dst = coeff * src` (Fan–Paar GF(2^32), AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2_fp32(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar32>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_into_avx2_fp32: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "fan_paar::mul_into_avx2_fp32: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = fp32_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        _mm256_storeu_si256(dst_lane, scale_fp32(x, &l));
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<FanPaar32>(dst_tail, coeff);
}

// ---------------------------------------------------------------------------
// Fan–Paar GF(2^64): the same identity one level up — three fp32 scales
// (each itself three fp16 scales) merged by fp32 parity, plus a 4-byte
// half-swap. Nine fp16 scales per 32-byte lane.
// ---------------------------------------------------------------------------

/// The three fp32 sub-scales of one fp64 scale.
struct Fp64Lanes {
    /// `[c0, c0 ^ mul_alpha(c1), c1]` as fp32 sub-scales — `a` even fp32
    /// lanes, `b` odd, `c` (uniform) the half-swapped source.
    a: Fp32Lanes,
    b: Fp32Lanes,
    c: Fp32Lanes,
    /// `0x00000000ffffffff`: keeps each 8-byte group's low (even) fp32 lane.
    even: __m256i,
    /// `SWAP4`: exchanges the two fp32 halves of every fp64 element.
    swap4: __m256i,
}

#[archmage::rite(v3)]
fn fp64_lanes(coeff: Elem<FanPaar64>) -> Fp64Lanes {
    let (c0, c1) = coeff.to_components();
    let a_coeff = c0;
    let b_coeff = c0.add(c1.mul_alpha());
    let c_coeff = c1;
    Fp64Lanes {
        a: fp32_lanes(a_coeff),
        b: fp32_lanes(b_coeff),
        c: fp32_lanes(c_coeff),
        even: _mm256_set1_epi64x(i64::from_ne_bytes(0x0000_0000_ffff_ffffu64.to_ne_bytes())),
        swap4: load_mask(&SWAP4),
    }
}

/// `coeff * src` for one 32-byte lane, given the precomputed fp32 sub-scales.
#[archmage::rite(v3)]
fn scale_fp64(x: __m256i, lanes: &Fp64Lanes) -> __m256i {
    // even fp32 lanes ← a·x, odd fp32 lanes ← b·x.
    let xa = scale_fp32(x, &lanes.a);
    let xb = scale_fp32(x, &lanes.b);
    let same = _mm256_xor_si256(
        _mm256_and_si256(xa, lanes.even),
        _mm256_andnot_si256(lanes.even, xb),
    );
    // uniform c on the half-swapped source.
    let cross = scale_fp32(_mm256_shuffle_epi8(x, lanes.swap4), &lanes.c);
    _mm256_xor_si256(same, cross)
}

/// `dst ^= coeff * src` (Fan–Paar GF(2^64), AVX2).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2_fp64(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar64>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_add_avx2_fp64: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::mul_add_avx2_fp64: buffer of {} bytes is not a whole number of 8-byte elements",
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
    scalar::mul_add::<FanPaar64>(dst_tail, coeff, src_tail);
}

/// `dst = coeff * dst` (Fan–Paar GF(2^64), AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2_fp64(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem<FanPaar64>) {
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::mul_assign_avx2_fp64: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = fp64_lanes(coeff);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_fp64(x, &l));
    }
    scalar::mul_assign::<FanPaar64>(dst_tail, coeff);
}

/// `dst = coeff * src` (Fan–Paar GF(2^64), AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2_fp64(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<FanPaar64>,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_into_avx2_fp64: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "fan_paar::mul_into_avx2_fp64: buffer of {} bytes is not a whole number of 8-byte elements",
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
    scalar::mul_assign::<FanPaar64>(dst_tail, coeff);
}
