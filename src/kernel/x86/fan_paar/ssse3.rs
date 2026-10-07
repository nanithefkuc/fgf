//! Fan–Paar SSSE3 kernels over 16-byte lanes.
use crate::field::binary::Binary;
use crate::field::binary::tower::{FanPaar16, Tower};
use crate::kernel::scalar;
use crate::kernel::tables::FpTowerTables;
use crate::kernel::x86::gf16::{nibble_ssse3, scale_ssse3};

/// `dst ^= coeff * src` with `PSHUFB` lookups over 16-byte lanes (SSSE3).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_ssse3_fp16(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_add_ssse3_fp16: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_add_ssse3_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let d = _mm_loadu_si128(&*dst_lane);
        let scaled = scale_ssse3(x, &vectors);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, scaled));
    }
    scalar::mul_add::<Binary<16, Tower<FanPaar16>>>(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 16-byte lanes (SSSE3).
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_ssse3_fp16(_token: archmage::X64V2Token, dst: &mut [u8], tables: &FpTowerTables) {
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_assign_ssse3_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    scalar::mul_assign::<Binary<16, Tower<FanPaar16>>>(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes (SSSE3), fused.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_ssse3_fp16(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &FpTowerTables,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "fan_paar::mul_into_ssse3_fp16: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(2),
        "fan_paar::mul_into_ssse3_fp16: buffer of {} bytes is not a whole number of 2-byte elements",
        dst.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<Binary<16, Tower<FanPaar16>>>(dst_tail, tables.coeff);
}
