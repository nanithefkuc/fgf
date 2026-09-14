//! GF(2^32) tower kernels for x86 / `x86_64`.
//!
//! A GF(2^32) multiply is two GF(2^16) lane multiplies under a period-2
//! coefficient — one of the source and one of the source with its two
//! GF(2^16) halves exchanged — just as a GF(2^16) multiply is two byte
//! multiplies under a period-2 GF(2^8) coefficient. Each GF(2^16) lane
//! multiply is the existing [`TowerCoeff`] identity folded into a 4-byte
//! periodic broadcast, so the whole GF(2^32) scale is four `GF2P8MULB` plus
//! three byte shuffles per 32-byte lane, against ~16 base multiplies per
//! element for the scalar Karatsuba.
//!
//! Only the GFNI path is written here. Shuffle-only x86 (AVX2/SSSE3) and
//! every non-x86 target fall back to the portable scalar kernel, which is
//! also the differential oracle. The GFNI kernel runs unchanged on an
//! AVX-512 host, so the dispatch routes both `Gfni` and `Avx512` here.
//! The multiply entries are safe [`archmage`] capability-token functions:
//! each takes the GFNI proof token, asserts its own buffer geometry before
//! the lane loop, and walks the buffer with reference-based loads and
//! stores over 32-byte chunks, handing the sub-lane tail to the scalar
//! reference.
//!
//! [`archmage`]: https://docs.rs/archmage

use crate::field::gf32;
use crate::kernel::scalar;
use crate::kernel::tables::{Tower2Coeff, TowerCoeff};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Pack two GF(2^16) [`TowerCoeff`] broadcast words into one 4-byte tile.
///
/// The low word applies to the even GF(2^16) lane, the high word to the odd
/// lane, so a `_mm256_set1_epi32` of the result repeats the pair at the
/// 4-byte period the GF(2^32) lane multiply needs.
#[inline]
const fn pack32(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

/// The four 4-byte GFNI broadcast tiles for one GF(2^32) coefficient.
///
/// `same_a`/`cross_a` scale the source's two GF(2^16) lanes; `same_b`/
/// `cross_b` scale the half-swapped source's. Each tile packs the two-byte
/// [`TowerCoeff`] broadcast of its GF(2^16) lane coefficient. Built once in
/// `prepare` and reused across the whole buffer; GF(2^64) reuses this to
/// build its own 8-byte tiles.
#[inline]
#[must_use]
pub fn gf32_tiles(coeff: gf32::Elem) -> [u32; 4] {
    let (c0, c1) = coeff.to_components();
    // `same = [c0, c0 + c1]` multiplies the source; `cross = [DELTA*c1, c1]`
    // the half-swapped source.
    let t = Tower2Coeff::derive(c0, c1, gf32::DELTA);
    let [s0, s1] = t.same;
    let [x0, x1] = t.cross;
    let same_a = pack32(TowerCoeff::new(s0).same, TowerCoeff::new(s1).same);
    let cross_a = pack32(TowerCoeff::new(s0).cross, TowerCoeff::new(s1).cross);
    let same_b = pack32(TowerCoeff::new(x0).same, TowerCoeff::new(x1).same);
    let cross_b = pack32(TowerCoeff::new(x0).cross, TowerCoeff::new(x1).cross);
    [same_a, cross_a, same_b, cross_b]
}

/// `PSHUFB` control that exchanges adjacent bytes within every 2-byte
/// element — the GF(2^16) tower swap. The 16-byte pattern repeats because
/// `PSHUFB` indexes within each 128-bit lane independently.
pub(crate) const SWAP1: [u8; 32] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13,
    12, 15, 14,
];

/// `PSHUFB` control that exchanges the two GF(2^16) halves of every GF(2^32)
/// element — a 2-byte-granular swap, period 4.
pub(crate) const SWAP2: [u8; 32] = [
    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14,
    15, 12, 13,
];

/// `PSHUFB` control that reverses every 4-byte group — `SWAP1` composed
/// with `SWAP2`, the swap the second GF(2^16) lane multiply applies to its
/// already-half-swapped input.
pub(crate) const REV4: [u8; 32] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15,
    14, 13, 12,
];

/// Load a 32-byte shuffle control. The mask arrays are exactly one vector
/// wide, so the load is of a complete, in-bounds lane.
///
/// A V3 [`archmage::rite`] helper: the Fan–Paar lane builders hold exactly
/// V3, and the GFNI tower lane builders cover V3 in turn.
#[archmage::rite(v3, import_intrinsics)]
pub(crate) fn load_mask(m: &[u8; 32]) -> __m256i {
    _mm256_loadu_si256(m)
}

/// Broadcast one 4-byte tile across a 32-byte lane.
#[archmage::rite(v3)]
fn set1_tile(tile: u32) -> __m256i {
    _mm256_set1_epi32(i32::from_ne_bytes(tile.to_ne_bytes()))
}

/// `coeff * src` for one 32-byte lane, given the source and the three
/// precomputed shuffles of it.
///
/// Four byte-wide multiplies: the source under `same_a`, its adjacent-swap
/// under `cross_a`, its half-swap under `same_b`, and the reversed-half-swap
/// under `cross_b`. The shuffles are the caller's job so the body stays a
/// straight line of independent multiplies.
#[archmage::rite(v3_gfni_crypto)]
fn scale32(x: __m256i, lane: &Lane32) -> __m256i {
    _mm256_xor_si256(
        _mm256_xor_si256(
            _mm256_gf2p8mul_epi8(x, lane.same_a),
            _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s1), lane.cross_a),
        ),
        _mm256_xor_si256(
            _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s2), lane.same_b),
            _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.rev4), lane.cross_b),
        ),
    )
}

/// Broadcast tiles and shuffle masks, materialized once per kernel call.
struct Lane32 {
    same_a: __m256i,
    cross_a: __m256i,
    same_b: __m256i,
    cross_b: __m256i,
    s1: __m256i,
    s2: __m256i,
    rev4: __m256i,
}

#[archmage::rite(v3_gfni_crypto)]
fn lane32(tiles: [u32; 4]) -> Lane32 {
    Lane32 {
        same_a: set1_tile(tiles[0]),
        cross_a: set1_tile(tiles[1]),
        same_b: set1_tile(tiles[2]),
        cross_b: set1_tile(tiles[3]),
        s1: load_mask(&SWAP1),
        s2: load_mask(&SWAP2),
        rev4: load_mask(&REV4),
    }
}

/// `dst ^= coeff * src` with `GF2P8MULB` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: gf32::Elem,
    tiles: [u32; 4],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "gf32::mul_add_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "gf32::mul_add_gfni: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = lane32(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let r = scale32(x, &l);
        let d = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, r));
    }
    // Every 32-byte lane is a whole number of 4-byte elements, so the tail
    // starts on an element boundary.
    scalar::mul_add::<gf32::Gf32>(dst_tail, coeff, src_tail);
}

/// `dst = coeff * dst` with `GF2P8MULB` over 32-byte lanes.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: gf32::Elem,
    tiles: [u32; 4],
) {
    assert!(
        dst.len().is_multiple_of(4),
        "gf32::mul_assign_gfni: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = lane32(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        let r = scale32(x, &l);
        _mm256_storeu_si256(dst_lane, r);
    }
    scalar::mul_assign::<gf32::Gf32>(dst_tail, coeff);
}

/// `dst = coeff * src` with `GF2P8MULB` over 32-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: gf32::Elem,
    tiles: [u32; 4],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "gf32::mul_into_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(4),
        "gf32::mul_into_gfni: buffer of {} bytes is not a whole number of 4-byte elements",
        dst.len(),
    );
    let l = lane32(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let r = scale32(x, &l);
        _mm256_storeu_si256(dst_lane, r);
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<gf32::Gf32>(dst_tail, coeff);
}
