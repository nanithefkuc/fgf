//! GFNI tower kernels over GF(2⁶⁴) lanes.

use super::super::gf32::{REV4, SWAP1, SWAP2, load_mask};
use super::{REV8, SWAP1_4, SWAP2_4, SWAP4};
use crate::field::Elem;
use crate::field::binary::tower::Rijndael64Field;
use crate::kernel::scalar;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
    coeff: Elem<Rijndael64Field>,
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
    scalar::mul_add::<Rijndael64Field>(dst_tail, coeff, src_tail);
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
    coeff: Elem<Rijndael64Field>,
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
    scalar::mul_assign::<Rijndael64Field>(dst_tail, coeff);
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
    coeff: Elem<Rijndael64Field>,
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
    scalar::mul_assign::<Rijndael64Field>(dst_tail, coeff);
}
