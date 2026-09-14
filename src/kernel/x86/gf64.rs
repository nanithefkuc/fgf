//! GF(2^64) tower kernels for x86 / `x86_64`.
//!
//! One more level of the same identity as [`super::gf32`]: a GF(2^64)
//! multiply is two GF(2^32) lane multiplies under a period-2 coefficient,
//! and each GF(2^32) lane multiply is the four-`GF2P8MULB` [`super::gf32`]
//! scale folded into an 8-byte periodic broadcast. The whole GF(2^64) scale
//! is therefore eight `GF2P8MULB` plus seven byte shuffles per 32-byte lane
//! — four GF(2^16)-scale units — against ~64 base multiplies per element
//! for the scalar Karatsuba.
//!
//! As with GF(2^32), only the GFNI path is written here; everything else
//! falls back to the portable scalar kernel. The multiply entries are safe
//! [`archmage`] capability-token functions: each takes the GFNI proof
//! token, asserts its own buffer geometry before the lane loop, and walks
//! the buffer with reference-based loads and stores over 32-byte chunks,
//! handing the sub-lane tail to the scalar reference.
//!
//! [`archmage`]: https://docs.rs/archmage

use crate::field::gf64;
use crate::kernel::scalar;
use crate::kernel::tables::Tower2Coeff;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::gf32::{REV4, SWAP1, SWAP2, gf32_tiles, load_mask};

/// Pack two 4-byte GF(2^32) tiles into one 8-byte tile.
///
/// The low tile applies to the even GF(2^32) lane, the high tile to the odd
/// lane, so a `_mm256_set1_epi64x` of the result repeats the pair at the
/// 8-byte period the GF(2^64) lane multiply needs.
#[inline]
const fn pack64(lo: u32, hi: u32) -> u64 {
    (lo as u64) | ((hi as u64) << 32)
}

/// The eight 8-byte GFNI broadcast tiles for one GF(2^64) coefficient.
///
/// Tiles 0–3 are the four GF(2^32) tiles of the `same` pair `[c0, c0 + c1]`
/// (applied to the source and its inner shuffles); tiles 4–7 are those of
/// the `cross` pair `[DELTA·c1, c1]` (applied to the half-swapped source).
/// Each 8-byte tile packs two 4-byte [`super::gf32::gf32_tiles`] entries, so
/// GF(2^64) reuses the GF(2^32) tile derivation unchanged.
#[inline]
#[must_use]
pub fn gf64_tiles(coeff: gf64::Elem) -> [u64; 8] {
    let (c0, c1) = coeff.to_components();
    let tower = Tower2Coeff::derive(c0, c1, gf64::DELTA);
    let [s0, s1] = tower.same;
    let [x0, x1] = tower.cross;
    let same0 = gf32_tiles(s0);
    let same1 = gf32_tiles(s1);
    let cross0 = gf32_tiles(x0);
    let cross1 = gf32_tiles(x1);
    [
        pack64(same0[0], same1[0]),
        pack64(same0[1], same1[1]),
        pack64(same0[2], same1[2]),
        pack64(same0[3], same1[3]),
        pack64(cross0[0], cross1[0]),
        pack64(cross0[1], cross1[1]),
        pack64(cross0[2], cross1[2]),
        pack64(cross0[3], cross1[3]),
    ]
}

/// `PSHUFB` control that exchanges the two GF(2^32) halves of every
/// GF(2^64) element — a 4-byte-granular swap, period 8.
pub(crate) const SWAP4: [u8; 32] = [
    4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15,
    8, 9, 10, 11,
];

/// `SWAP1` composed with `SWAP4`.
const SWAP1_4: [u8; 32] = [
    5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14,
    9, 8, 11, 10,
];

/// `SWAP2` composed with `SWAP4`.
const SWAP2_4: [u8; 32] = [
    6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13,
    10, 11, 8, 9,
];

/// `REV4` composed with `SWAP4` — a full reversal of every 8-byte element.
const REV8: [u8; 32] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
    11, 10, 9, 8,
];

/// Broadcast one 8-byte tile across a 32-byte lane.
#[archmage::rite(v3)]
fn set1_tile(tile: u64) -> __m256i {
    _mm256_set1_epi64x(i64::from_ne_bytes(tile.to_ne_bytes()))
}

/// `coeff * src` for one 32-byte lane, given the source and the seven
/// precomputed shuffles of it.
///
/// Eight byte-wide multiplies: the source and its six distinct permutations
/// under the swap group `{SWAP1, SWAP2, REV4, SWAP4}`. The first four terms
/// are one GF(2^32) lane multiply; the last four, over the half-swapped
/// source, are the other.
#[archmage::rite(v3_gfni_crypto)]
fn scale64(x: __m256i, lane: &Lane64) -> __m256i {
    _mm256_xor_si256(
        _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_gf2p8mul_epi8(x, lane.same_a),
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s1), lane.cross_a),
            ),
            _mm256_xor_si256(
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s2), lane.same_b),
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.rev4), lane.cross_b),
            ),
        ),
        _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s4), lane.same_c),
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s1_4), lane.cross_c),
            ),
            _mm256_xor_si256(
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.s2_4), lane.same_d),
                _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(x, lane.rev8), lane.cross_d),
            ),
        ),
    )
}

/// Broadcast tiles and shuffle masks, materialized once per kernel call.
struct Lane64 {
    same_a: __m256i,
    cross_a: __m256i,
    same_b: __m256i,
    cross_b: __m256i,
    same_c: __m256i,
    cross_c: __m256i,
    same_d: __m256i,
    cross_d: __m256i,
    s1: __m256i,
    s2: __m256i,
    rev4: __m256i,
    s4: __m256i,
    s1_4: __m256i,
    s2_4: __m256i,
    rev8: __m256i,
}

#[archmage::rite(v3_gfni_crypto)]
fn lane64(tiles: [u64; 8]) -> Lane64 {
    Lane64 {
        same_a: set1_tile(tiles[0]),
        cross_a: set1_tile(tiles[1]),
        same_b: set1_tile(tiles[2]),
        cross_b: set1_tile(tiles[3]),
        same_c: set1_tile(tiles[4]),
        cross_c: set1_tile(tiles[5]),
        same_d: set1_tile(tiles[6]),
        cross_d: set1_tile(tiles[7]),
        s1: load_mask(&SWAP1),
        s2: load_mask(&SWAP2),
        rev4: load_mask(&REV4),
        s4: load_mask(&SWAP4),
        s1_4: load_mask(&SWAP1_4),
        s2_4: load_mask(&SWAP2_4),
        rev8: load_mask(&REV8),
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
    coeff: gf64::Elem,
    tiles: [u64; 8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "gf64::mul_add_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "gf64::mul_add_gfni: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = lane64(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let r = scale64(x, &l);
        let d = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, r));
    }
    // Every 32-byte lane is a whole number of 8-byte elements, so the tail
    // starts on an element boundary.
    scalar::mul_add::<gf64::Gf64>(dst_tail, coeff, src_tail);
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
    coeff: gf64::Elem,
    tiles: [u64; 8],
) {
    assert!(
        dst.len().is_multiple_of(8),
        "gf64::mul_assign_gfni: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = lane64(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in lanes.iter_mut() {
        let x = _mm256_loadu_si256(&*dst_lane);
        let r = scale64(x, &l);
        _mm256_storeu_si256(dst_lane, r);
    }
    scalar::mul_assign::<gf64::Gf64>(dst_tail, coeff);
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
    coeff: gf64::Elem,
    tiles: [u64; 8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "gf64::mul_into_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert!(
        dst.len().is_multiple_of(8),
        "gf64::mul_into_gfni: buffer of {} bytes is not a whole number of 8-byte elements",
        dst.len(),
    );
    let l = lane64(tiles);
    let (lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let r = scale64(x, &l);
        _mm256_storeu_si256(dst_lane, r);
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst_tail.copy_from_slice(src_tail);
    scalar::mul_assign::<gf64::Gf64>(dst_tail, coeff);
}
