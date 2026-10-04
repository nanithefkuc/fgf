//! Many sources into one destination.
//!
//! The destination is read and written once per group of
//! [`SOURCE_GROUP`] sources, whose broadcast pairs stay resident.

use super::super::{broadcast_words, check_elements, swap_mask_avx2};
use crate::kernel::gf16::Coeffs;
use crate::kernel::tables::TowerCoeff;

use super::scale_gfni;
use super::single::mul_add_gfni;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// How many sources [`mul_add_gather_gfni`] folds into its accumulator at once.
///
/// A GFNI coefficient is two broadcast vectors, so four sources occupy eight
/// of the sixteen AVX2 registers and leave room for the accumulator, the
/// exchange control and the multiply temporaries. Folding *every* source in
/// one pass cannot: a variable coefficient count has no register home at all,
/// so the broadcasts end up re-derived inside the byte loop.
const SOURCE_GROUP: usize = 4;

/// GFNI gather: a coefficient is two broadcast vectors, so sources are folded
/// four at a time.
///
/// Every broadcast of a group is derived once and stays in a register for the
/// whole pass, and the destination is read and written once per group instead
/// of once per source. Folding all sources in a single pass, as this kernel
/// first did, cannot keep a variable number of broadcasts anywhere, so it
/// paid two base multiplies and two `vpbroadcastw` per source per tile.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_gfni(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: &(impl Coeffs + ?Sized),
    srcs: &[&[u8]],
) {
    check_elements("gf16::mul_add_gather_gfni", dst.len());
    assert_eq!(
        coeffs.count(),
        srcs.len(),
        "gf16::mul_add_gather_gfni: coefficients is {} but sources is {}",
        coeffs.count(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "gf16::mul_add_gather_gfni: dst is {} bytes but source {index} is {} bytes",
            dst.len(),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    let swap = swap_mask_avx2();
    let mut i = 0;
    while i + SOURCE_GROUP <= coeffs.count() {
        let group: [TowerCoeff; SOURCE_GROUP] = core::array::from_fn(|k| coeffs.compact(i + k));
        let mut sources: [&[u8]; SOURCE_GROUP] = [&[]; SOURCE_GROUP];
        sources.copy_from_slice(&srcs[i..i + SOURCE_GROUP]);
        gather_group(token, dst, group, sources, swap);
        i += SOURCE_GROUP;
    }
    if i + 2 <= coeffs.count() {
        let group: [TowerCoeff; 2] = core::array::from_fn(|k| coeffs.compact(i + k));
        let mut sources: [&[u8]; 2] = [&[]; 2];
        sources.copy_from_slice(&srcs[i..i + 2]);
        gather_group(token, dst, group, sources, swap);
        i += 2;
    }
    if i < coeffs.count() {
        // The unrolled single-coefficient kernel is the better shape for the
        // last source.
        mul_add_gfni(token, dst, coeffs.compact(i), srcs[i]);
    }
}

/// Fold `N` sources into `dst` in one pass over the destination.
///
/// `N` is a constant, so the `2 * N` broadcast vectors have a register home
/// for the whole pass; they are derived once per group, never inside the byte
/// loop.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn gather_group<const N: usize>(
    token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeffs: [TowerCoeff; N],
    srcs: [&[u8]; N],
    swap: __m256i,
) {
    let tail_start = dst.len() & !31;
    let mut same = [_mm256_setzero_si256(); N];
    let mut cross = [_mm256_setzero_si256(); N];
    for (k, coeff) in coeffs.iter().enumerate() {
        let (same_word, cross_word) = broadcast_words(*coeff);
        same[k] = _mm256_set1_epi16(same_word);
        cross[k] = _mm256_set1_epi16(cross_word);
    }

    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for (w, dst_lane) in dst_lanes.iter_mut().enumerate() {
        let mut acc = _mm256_loadu_si256(&*dst_lane);
        for (k, &src) in srcs.iter().enumerate() {
            let (src_lanes, _) = src.as_chunks::<32>();
            let x = _mm256_loadu_si256(&src_lanes[w]);
            let swapped = _mm256_shuffle_epi8(x, swap);
            acc = _mm256_xor_si256(acc, scale_gfni(x, swapped, same[k], cross[k]));
        }
        _mm256_storeu_si256(dst_lane, acc);
    }

    if !dst_tail.is_empty() {
        for (k, coeff) in coeffs.iter().enumerate() {
            mul_add_gfni(token, dst_tail, *coeff, &srcs[k][tail_start..]);
        }
    }
}
