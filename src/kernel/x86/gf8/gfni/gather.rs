//! Many sources into one destination.
//!
//! Register-blocked over the destination tile, so the destination is read and
//! written once per tile no matter how many sources participate, and the
//! remainder fuses across sources on short rows.
//!
//! Every entry is a safe [`archmage`] capability-token function carrying the
//! full geometry contract: one coefficient per source, every source matching
//! the destination length.

use super::super::check_gather;
use super::{bmul_gfni, brem_gfni};
use crate::kernel::gf8::Coeffs;

#[cfg(target_arch = "x86")]
use core::arch::x86::__m256i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256i;

/// Many sources into one destination, register-blocked over 128-byte tiles
/// with source-fused 64/32-byte tails where at least three sources participate.
///
/// The `NATIVE` const selects native AES or affine multiplication at compile
/// time; every representation uses the same tile geometry.
///
/// # Panics
/// Panics unless `srcs.len() == coeffs.len()` and every source matches `dst`
/// in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni<const NATIVE: bool>(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_gfni", dst, coeffs.count(), srcs);
    let remainder = dst.len() & 127;
    let fused =
        dst.len() < 128 && coeffs.count() > 2 && remainder != 0 && remainder.trailing_zeros() >= 5;
    // The const alternatives keep the measured AXPY path for one source,
    // 16-byte/sub-lane tails, compound scalar remainders, and rows that
    // already execute the 128-byte main body.
    if fused {
        mul_add_gather_impl::<NATIVE, true, 4>(token, dst, coeffs, srcs);
    } else {
        mul_add_gather_impl::<NATIVE, false, 4>(token, dst, coeffs, srcs);
    }
}

/// The register-blocked gather walk the entries share: an `#[inline(never)]`
/// driver over token-bearing tile bodies.
///
/// The driver owns the geometry split and the remainder selection. The tile
/// bodies are [`archmage::arcane`] entries, so their feature context is a
/// boundary the driver cannot inline across — the measured contract that
/// keeps the dispatch decision and the remainder walk out of the timed tile
/// body, while the driver itself stays a separate symbol. Reachable only
/// from the feature-matching entries above.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub(super) fn mul_add_gather_impl<
    const NATIVE: bool,
    const FUSED_NATIVE_TAIL: bool,
    const TILE_LANES: usize,
>(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
) {
    assert!((1..=4).contains(&TILE_LANES));
    let tile = 32 * TILE_LANES;
    let len = dst.len() / tile * tile;
    let (tiles, rest) = dst.split_at_mut(len);
    gather_tiles::<NATIVE, TILE_LANES>(token, tiles, coeffs, srcs);
    // The driver selects the fused specialization only for measured
    // multi-source remainders that consist entirely of 32-byte lanes; the
    // tile is at most 128 bytes, so at most one 64-byte block and one 32-byte
    // lane survive.
    let consumed = if FUSED_NATIVE_TAIL {
        gather_native_tail::<NATIVE>(token, rest, coeffs, srcs, len)
    } else {
        0
    };
    gather_remainder::<NATIVE>(token, &mut rest[consumed..], coeffs, srcs, len + consumed);
}

/// The main register-blocked tile loop: `TILE_LANES` 32-byte accumulators
/// over whole tiles, every source folded in before the destination stores.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
fn gather_tiles<const NATIVE: bool, const TILE_LANES: usize>(
    _token: archmage::X64V3GfniCryptoToken,
    tiles: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
) {
    let tile = 32 * TILE_LANES;
    let mut offset = 0;
    for dtile in tiles.chunks_exact_mut(tile) {
        let (dl, _) = dtile.as_chunks_mut::<32>();
        // `offset + tile <= tiles.len() <= dst.len() <= every src.len()`
        // (asserted by the entries), so each accumulator load stays inside
        // `dst`.
        let mut acc: [__m256i; TILE_LANES] =
            core::array::from_fn(|lane| _mm256_loadu_si256(&dl[lane]));
        for (k, &src) in (0..coeffs.count()).zip(srcs) {
            let factor = factor_at::<NATIVE>(coeffs, k);
            let (sl, _) = src[offset..offset + tile].as_chunks::<32>();
            for (lane, slot) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_si256(&sl[lane]);
                *slot = _mm256_xor_si256(*slot, bmul_gfni::<NATIVE>(x, factor));
            }
        }
        for (lane, &value) in acc.iter().enumerate() {
            _mm256_storeu_si256(&mut dl[lane], value);
        }
        offset += tile;
    }
}

/// The source-fused remainder blocks: at most one 64-byte pair and one
/// 32-byte lane, every source folded in before each store.
///
/// Returns the bytes consumed, so the driver advances the remainder and the
/// source offsets past the fused blocks.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
fn gather_native_tail<const NATIVE: bool>(
    _token: archmage::X64V3GfniCryptoToken,
    rest: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
    tail: usize,
) -> usize {
    let mut consumed = 0;
    let rest = if rest.len() >= 64 {
        let (pair, after) = rest.split_at_mut(64);
        let (dl, _) = pair.as_chunks_mut::<32>();
        let (mut a0, mut a1) = (_mm256_loadu_si256(&dl[0]), _mm256_loadu_si256(&dl[1]));
        for (k, &src) in (0..coeffs.count()).zip(srcs) {
            let factor = factor_at::<NATIVE>(coeffs, k);
            let (sl, _) = src[tail..tail + 64].as_chunks::<32>();
            let x0 = _mm256_loadu_si256(&sl[0]);
            let x1 = _mm256_loadu_si256(&sl[1]);
            a0 = _mm256_xor_si256(a0, bmul_gfni::<NATIVE>(x0, factor));
            a1 = _mm256_xor_si256(a1, bmul_gfni::<NATIVE>(x1, factor));
        }
        _mm256_storeu_si256(&mut dl[0], a0);
        _mm256_storeu_si256(&mut dl[1], a1);
        consumed += 64;
        after
    } else {
        rest
    };
    if rest.len() >= 32 {
        let (lane, _) = rest.split_at_mut(32);
        let (dl, _) = lane.as_chunks_mut::<32>();
        let mut acc = _mm256_loadu_si256(&dl[0]);
        for (k, &src) in (0..coeffs.count()).zip(srcs) {
            let factor = factor_at::<NATIVE>(coeffs, k);
            let (sl, _) = src[tail + consumed..tail + consumed + 32].as_chunks::<32>();
            let x = _mm256_loadu_si256(&sl[0]);
            acc = _mm256_xor_si256(acc, bmul_gfni::<NATIVE>(x, factor));
        }
        _mm256_storeu_si256(&mut dl[0], acc);
        consumed += 32;
    }
    consumed
}

/// The single-source AXPY remainder over what no tile or fused block
/// covered.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
fn gather_remainder<const NATIVE: bool>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
    tail: usize,
) {
    if dst.is_empty() {
        return;
    }
    for (k, &src) in (0..coeffs.count()).zip(srcs) {
        brem_gfni::<NATIVE>(dst, coeffs.resolved(k), &src[tail..]);
    }
}

/// Broadcast coefficient `index` into a 256-bit multiply factor, reading only
/// the facet the multiply consumes: the raw byte for the native multiply, the
/// affine map qword otherwise.
#[inline]
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn factor_at<const NATIVE: bool>(coeffs: &(impl Coeffs + ?Sized), index: usize) -> __m256i {
    if NATIVE {
        _mm256_set1_epi8(coeffs.byte(index).cast_signed())
    } else {
        _mm256_set1_epi64x(coeffs.affine(index).cast_signed())
    }
}
