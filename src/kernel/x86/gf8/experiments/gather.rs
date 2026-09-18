//! Gather tile-width and remainder controls.
//!
//! Measurement scaffolding for the production gather body, which is held
//! fixed while the accumulator count, the source-chain split, and the
//! remainder composition vary. This module also carries the prepared
//! `Gf8B` affine gather — benchmark-only production-policy evidence sharing
//! the production body.

use super::super::gather::{
    check_gather, gather_remainder, mul_add_gather_gfni, mul_add_gather_impl,
};
use super::super::{Affine8B, Affine8BFactor, Gfni};
use crate::field::gf8b::Elem;

/// Experimental `Gf8B` gather using prepared `VGF2P8AFFINEQB` maps.
///
/// This is benchmark-only production-policy evidence. It deliberately shares
/// the native gather body and the short-row fusion rule with
/// [`mul_add_gather_gfni`], so the multiply instruction and prepared factor
/// representation are the only differences.
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_affine_8b(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    factors: &[Affine8BFactor],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_affine_8b", dst, factors.len(), srcs);
    let remainder = dst.len() & 127;
    let fused =
        dst.len() < 128 && factors.len() > 2 && remainder != 0 && remainder.trailing_zeros() >= 5;
    if fused {
        mul_add_gather_impl::<Affine8B, true, 4>(token, dst, factors, srcs);
    } else {
        mul_add_gather_impl::<Affine8B, false, 4>(token, dst, factors, srcs);
    }
}

/// Pre-fusion GFNI gather retained only as an interleaved benchmark control.
///
/// The 128-byte body is identical to [`mul_add_gather_gfni`], but every
/// remainder is composed from single-source AXPY calls. This is not a
/// dispatch candidate.
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni_axpy_tail(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_gfni_axpy_tail", dst, coeffs.len(), srcs);
    mul_add_gather_impl::<Gfni, false, 4>(token, dst, coeffs, srcs);
}

/// Benchmark a native GFNI gather with `TILE_LANES` 32-byte accumulators.
///
/// Width four delegates to [`mul_add_gather_gfni`] and is the exact production
/// control, including its measured short-row fusion. Widths one through three
/// use the same gather body with the existing single-source AXPY remainder.
///
/// # Panics
/// Panics unless `TILE_LANES` is in `1..=4`, and as [`mul_add_gather_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni_tile<const TILE_LANES: usize>(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    assert!(
        (1..=4).contains(&TILE_LANES),
        "GFNI gather tile must contain 1–4 lanes"
    );
    if TILE_LANES == 4 {
        mul_add_gather_gfni(token, dst, coeffs, srcs);
        return;
    }
    check_gather("mul_add_gather_gfni_tile", dst, coeffs.len(), srcs);
    mul_add_gather_impl::<Gfni, false, TILE_LANES>(token, dst, coeffs, srcs);
}

/// Benchmark the production 128-byte tile with even/odd source chains split.
///
/// The second accumulator bank breaks each lane's source dependency chain.
/// This is counter-driven evidence only; production continues through
/// [`mul_add_gather_gfni`].
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni_split(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_gfni_split", dst, coeffs.len(), srcs);
    mul_add_gather_gfni_split_impl(token, dst, coeffs, srcs);
}

/// The split-chain gather walk: an `#[inline(never)]` driver over the
/// token-bearing tile body, mirroring [`mul_add_gather_impl`]'s isolation.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn mul_add_gather_gfni_split_impl(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    const TILE: usize = 128;
    let (dst_tiles, rest) = dst.as_chunks_mut::<TILE>();
    split_gather_tiles(token, dst_tiles, coeffs, srcs);
    gather_remainder::<Gfni>(token, rest, coeffs, srcs, dst_tiles.len() * TILE);
}

/// The split-chain tile body.
///
/// Two accumulator banks break each lane's source dependency chain; the
/// feature context is the tier's, so the seam helpers it calls stay safely
/// reachable. Reachable only from the entry above.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
fn split_gather_tiles(
    _token: archmage::X64V3GfniCryptoToken,
    dst_tiles: &mut [[u8; 128]],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    const TILE: usize = 128;
    let mut offset = 0;
    for dtile in dst_tiles {
        let (dl, _) = dtile.as_chunks_mut::<32>();
        // `offset + TILE <= dst.len() <= every src.len()` (asserted by the
        // entry), so each accumulator load stays inside `dst`.
        let mut even: [__m256i; 4] = core::array::from_fn(|lane| _mm256_loadu_si256(&dl[lane]));
        let mut odd = [_mm256_setzero_si256(); 4];
        for pair in 0..coeffs.len() / 2 {
            let index = pair * 2;
            let even_factor = _mm256_set1_epi8(coeffs[index].0.cast_signed());
            let odd_factor = _mm256_set1_epi8(coeffs[index + 1].0.cast_signed());
            let even_src = &srcs[index][offset..offset + TILE];
            let odd_src = &srcs[index + 1][offset..offset + TILE];
            let (es, _) = even_src.as_chunks::<32>();
            let (os, _) = odd_src.as_chunks::<32>();
            for lane in 0..4 {
                let even_x = _mm256_loadu_si256(&es[lane]);
                let odd_x = _mm256_loadu_si256(&os[lane]);
                even[lane] =
                    _mm256_xor_si256(even[lane], _mm256_gf2p8mul_epi8(even_x, even_factor));
                odd[lane] = _mm256_xor_si256(odd[lane], _mm256_gf2p8mul_epi8(odd_x, odd_factor));
            }
        }
        if coeffs.len() & 1 != 0 {
            let index = coeffs.len() - 1;
            let factor = _mm256_set1_epi8(coeffs[index].0.cast_signed());
            let (ss, _) = srcs[index][offset..offset + TILE].as_chunks::<32>();
            for (lane, slot) in even.iter_mut().enumerate() {
                let x = _mm256_loadu_si256(&ss[lane]);
                *slot = _mm256_xor_si256(*slot, _mm256_gf2p8mul_epi8(x, factor));
            }
        }
        for (lane, &value) in even.iter().enumerate() {
            _mm256_storeu_si256(&mut dl[lane], _mm256_xor_si256(value, odd[lane]));
        }
        offset += TILE;
    }
}
