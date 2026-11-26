//! Fan–Paar GF(2^16) AVX2 kernels.

use crate::field::binary::tower::FanPaar16Field;
use crate::kernel::scalar;
use crate::kernel::tables::FpTowerTables;
use crate::kernel::x86::align_spill_slots;
use crate::kernel::x86::gf16::{nibble_avx2, scale_avx2};
use crate::kernel::x86::gf16::{nibble_ssse3, scale_ssse3};

/// `dst ^= coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf16::mul_add_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::gf16::mul_add_avx2: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    align_spill_slots();
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
    scalar::mul_add::<FanPaar16Field>(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 32-byte lanes (AVX2).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], tables: &FpTowerTables) {
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::gf16::mul_assign_avx2: buffer of {} bytes is not a whole number of 2-byte elements",
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
    scalar::mul_assign::<FanPaar16Field>(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::gf16::mul_into_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::gf16::mul_into_avx2: buffer of {} bytes is not a whole number of 2-byte elements",
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
    scalar::mul_assign::<FanPaar16Field>(dst_tail, tables.coeff);
}
