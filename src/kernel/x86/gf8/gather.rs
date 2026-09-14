//! Many sources into one destination: the production gather family.
//!
//! Register-blocked over the destination tile, so the destination is read and
//! written once per tile no matter how many sources participate. The GFNI
//! forms fuse the remainder across sources on short rows; the nibble forms
//! group sources and reuse each source load across the group.
//!
//! Every entry is a safe [`archmage`] capability-token function carrying the
//! full geometry contract: one coefficient per source, every source matching
//! the destination length.

use super::nibble::{mul_add_avx2, mul_add_ssse3};
use super::{Affine8D, Blocked, Gfni, bfactor, bmul, brem};
use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::tables::scale_table;

/// Geometry contract shared by the gather entries: one coefficient per
/// source, and every source exactly as long as the destination.
pub(super) fn check_gather(name: &str, dst: &[u8], coeffs: usize, srcs: &[&[u8]]) {
    assert_eq!(coeffs, srcs.len(), "{name}: one coefficient per source");
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            dst.len(),
            "{name}: source {index} does not match the destination length"
        );
    }
}

/// Many sources into one destination, register-blocked over 128-byte tiles
/// with source-fused 64/32-byte tails where at least three sources participate.
///
/// The main tile stays statically 128 bytes: narrower and split-chain
/// candidates reversed across neighboring page layouts. See the GFNI gather
/// tile record in `BENCHMARKS.md`.
///
/// # Panics
/// Panics unless `srcs.len() == coeffs.len()` and every source matches `dst`
/// in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_gfni", dst, coeffs.len(), srcs);
    let remainder = dst.len() & 127;
    let fused =
        dst.len() < 128 && coeffs.len() > 2 && remainder != 0 && remainder.trailing_zeros() >= 5;
    // The const alternatives keep the measured AXPY path for one source,
    // 16-byte/sub-lane tails, compound scalar remainders, and rows that
    // already execute the 128-byte main body.
    if fused {
        mul_add_gather_impl::<Gfni, true, 4>(token, dst, coeffs, srcs);
    } else {
        mul_add_gather_impl::<Gfni, false, 4>(token, dst, coeffs, srcs);
    }
}

/// [`mul_add_gather_gfni`] under `0x11D`: many sources into one destination, each
/// folded in with its `VGF2P8AFFINEQB` map.
///
/// Short rows take the same source-fused body `mul_add_gather_gfni` selects: below
/// the 128-byte main tile the per-row loop would otherwise degenerate into
/// one single-source AXPY per source, reloading the destination every time.
/// The rule is `mul_add_gather_gfni`'s verbatim — the `Blocked` seam monomorphizes
/// one body, so the affine form crosses at the same shapes the `GF2P8MULB`
/// form was measured at.
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_affine(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[gf8d::Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_affine", dst, coeffs.len(), srcs);
    let remainder = dst.len() & 127;
    let fused =
        dst.len() < 128 && coeffs.len() > 2 && remainder != 0 && remainder.trailing_zeros() >= 5;
    if fused {
        mul_add_gather_impl::<Affine8D, true, 4>(token, dst, coeffs, srcs);
    } else {
        mul_add_gather_impl::<Affine8D, false, 4>(token, dst, coeffs, srcs);
    }
}

/// The register-blocked gather walk the entries and the measurement variants
/// share: an `#[inline(never)]` driver over token-bearing tile bodies.
///
/// The driver owns the geometry split and the remainder selection. The tile
/// bodies are [`archmage::arcane`] entries, so their feature context is a
/// boundary the driver cannot inline across — the measured contract that
/// keeps the dispatch decision and the remainder walk out of the timed tile
/// body, while the driver itself stays a separate symbol. Reachable only
/// from the feature-matching entries above and the `experiments` controls.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub(super) fn mul_add_gather_impl<
    S: Blocked,
    const FUSED_NATIVE_TAIL: bool,
    const TILE_LANES: usize,
>(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[S::Coeff],
    srcs: &[&[u8]],
) {
    assert!((1..=4).contains(&TILE_LANES));
    let tile = 32 * TILE_LANES;
    let len = dst.len() / tile * tile;
    let (tiles, rest) = dst.split_at_mut(len);
    gather_tiles::<S, TILE_LANES>(token, tiles, coeffs, srcs);
    // The driver selects the fused specialization only for measured
    // multi-source remainders that consist entirely of 32-byte lanes; the
    // tile is at most 128 bytes, so at most one 64-byte block and one 32-byte
    // lane survive.
    let consumed = if FUSED_NATIVE_TAIL {
        gather_native_tail::<S>(token, rest, coeffs, srcs, len)
    } else {
        0
    };
    gather_remainder::<S>(token, &mut rest[consumed..], coeffs, srcs, len + consumed);
}

/// The main register-blocked tile loop: `TILE_LANES` 32-byte accumulators
/// over whole tiles, every source folded in before the destination stores.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
fn gather_tiles<S: Blocked, const TILE_LANES: usize>(
    _token: archmage::X64V3GfniCryptoToken,
    tiles: &mut [u8],
    coeffs: &[S::Coeff],
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
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = bfactor::<S>(coeff);
            let (sl, _) = src[offset..offset + tile].as_chunks::<32>();
            for (lane, slot) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_si256(&sl[lane]);
                *slot = _mm256_xor_si256(*slot, bmul::<S>(x, factor));
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
fn gather_native_tail<S: Blocked>(
    _token: archmage::X64V3GfniCryptoToken,
    rest: &mut [u8],
    coeffs: &[S::Coeff],
    srcs: &[&[u8]],
    tail: usize,
) -> usize {
    let mut consumed = 0;
    let rest = if rest.len() >= 64 {
        let (pair, after) = rest.split_at_mut(64);
        let (dl, _) = pair.as_chunks_mut::<32>();
        let (mut a0, mut a1) = (_mm256_loadu_si256(&dl[0]), _mm256_loadu_si256(&dl[1]));
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = bfactor::<S>(coeff);
            let (sl, _) = src[tail..tail + 64].as_chunks::<32>();
            let x0 = _mm256_loadu_si256(&sl[0]);
            let x1 = _mm256_loadu_si256(&sl[1]);
            a0 = _mm256_xor_si256(a0, bmul::<S>(x0, factor));
            a1 = _mm256_xor_si256(a1, bmul::<S>(x1, factor));
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
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = bfactor::<S>(coeff);
            let (sl, _) = src[tail + consumed..tail + consumed + 32].as_chunks::<32>();
            let x = _mm256_loadu_si256(&sl[0]);
            acc = _mm256_xor_si256(acc, bmul::<S>(x, factor));
        }
        _mm256_storeu_si256(&mut dl[0], acc);
        consumed += 32;
    }
    consumed
}

/// The single-source AXPY remainder over what no tile or fused block
/// covered.
///
/// Shared with the `experiments` controls, which pair it with their own tile
/// bodies.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gather_remainder<S: Blocked>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &[S::Coeff],
    srcs: &[&[u8]],
    tail: usize,
) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        brem::<S>(dst, coeff, &src[tail..]);
    }
}

/// Many sources into one destination using AVX2 nibble shuffles.
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_avx2(
    token: archmage::X64V3Token,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_avx2", dst, coeffs.len(), srcs);
    let len = dst.len() & !63;
    let mask = _mm256_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm256_setzero_si256(); 4];
        let mut hi = [_mm256_setzero_si256(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
            hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
        }
        let (tiles, rest) = dst.as_chunks_mut::<64>();
        for (t, dtile) in tiles.iter_mut().enumerate() {
            let offset = t * 64;
            let (d2, _) = dtile.as_chunks_mut::<32>();
            let mut acc0 = _mm256_loadu_si256(&d2[0]);
            let mut acc1 = _mm256_loadu_si256(&d2[1]);
            for slot in 0..count {
                let (s2, _) = srcs[group + slot][offset..offset + 64].as_chunks::<32>();
                let x0 = _mm256_loadu_si256(&s2[0]);
                let x1 = _mm256_loadu_si256(&s2[1]);
                let p0 = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x0, mask)),
                    _mm256_shuffle_epi8(
                        hi[slot],
                        _mm256_and_si256(_mm256_srli_epi16::<4>(x0), mask),
                    ),
                );
                let p1 = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x1, mask)),
                    _mm256_shuffle_epi8(
                        hi[slot],
                        _mm256_and_si256(_mm256_srli_epi16::<4>(x1), mask),
                    ),
                );
                acc0 = _mm256_xor_si256(acc0, p0);
                acc1 = _mm256_xor_si256(acc1, p1);
            }
            _mm256_storeu_si256(&mut d2[0], acc0);
            _mm256_storeu_si256(&mut d2[1], acc1);
        }
        for slot in 0..count {
            // The destination and source remainders keep equal lengths.
            mul_add_avx2(
                token,
                rest,
                scale_table(coeffs[group + slot]),
                &srcs[group + slot][len..],
            );
        }
    }
}

/// Many sources into one destination using SSSE3 nibble shuffles.
///
/// # Panics
/// As [`mul_add_gather_gfni`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_ssse3(
    token: archmage::X64V2Token,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_ssse3", dst, coeffs.len(), srcs);
    let len = dst.len() & !63;
    let mask = _mm_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm_loadu_si128(&table.lo);
            hi[slot] = _mm_loadu_si128(&table.hi);
        }
        let (tiles, rest) = dst.as_chunks_mut::<64>();
        for (t, dtile) in tiles.iter_mut().enumerate() {
            let offset = t * 64;
            let (d4, _) = dtile.as_chunks_mut::<16>();
            let mut acc = [
                _mm_loadu_si128(&d4[0]),
                _mm_loadu_si128(&d4[1]),
                _mm_loadu_si128(&d4[2]),
                _mm_loadu_si128(&d4[3]),
            ];
            for slot in 0..count {
                let (s4, _) = srcs[group + slot][offset..offset + 64].as_chunks::<16>();
                for (lane, value) in acc.iter_mut().enumerate() {
                    let x = _mm_loadu_si128(&s4[lane]);
                    let product = _mm_xor_si128(
                        _mm_shuffle_epi8(lo[slot], _mm_and_si128(x, mask)),
                        _mm_shuffle_epi8(hi[slot], _mm_and_si128(_mm_srli_epi16::<4>(x), mask)),
                    );
                    *value = _mm_xor_si128(*value, product);
                }
            }
            for (lane, &value) in acc.iter().enumerate() {
                _mm_storeu_si128(&mut d4[lane], value);
            }
        }
        for slot in 0..count {
            // The destination and source remainders keep equal lengths.
            mul_add_ssse3(
                token,
                rest,
                scale_table(coeffs[group + slot]),
                &srcs[group + slot][len..],
            );
        }
    }
}
