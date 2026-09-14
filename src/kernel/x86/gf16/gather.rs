//! Many sources into one destination.
//!
//! The fan shape whose destination is read and written once per *group* of
//! sources rather than once per source, so the question in every backend is
//! how much coefficient state can stay resident: eight prepared table sets
//! for the nibble strategies, four broadcast pairs for GFNI.

use crate::field::gf16::Elem;
use crate::kernel::gf16::{factor_tables, mul_add_scalar};
use crate::kernel::tables::{ScaleTable, TowerCoeff};

use super::gfni::mul_add_gfni;
use super::{
    NibbleSsse3, TERM_TILE, TableCoefficient, broadcast_words, lane_avx2, nibble_ssse3, scale_gfni,
    scale_split_avx2, scale_ssse3, split_source_avx2, swap_mask256,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Many sources into one destination, eight coefficients prepared per pass.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length.
#[allow(clippy::used_underscore_binding)]
#[cfg_attr(not(test), allow(dead_code))]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_avx2<C: TableCoefficient>(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeffs: &[C],
    srcs: &[&[u8]],
) {
    super::check_elements("gf16::mul_add_gather_avx2", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "gf16::mul_add_gather_avx2: coefficients is {} but sources is {}",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "gf16::mul_add_gather_avx2: dst is {} bytes but source {index} is {} bytes",
            dst.len(),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    let lanes = lane_avx2();
    let tail_start = dst.len() & !31;
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for block in (0..coeffs.len()).step_by(TERM_TILE) {
        let count = (coeffs.len() - block).min(TERM_TILE);
        // Four bank borrows per coefficient, resolved once per block. The
        // earlier shape widened each into a full `NibbleAvx2` here — eighty-
        // eight vectors against a sixteen-register file, so every tile read
        // the whole block back off the stack.
        let factors: [[&'static ScaleTable; 4]; TERM_TILE] =
            core::array::from_fn(|i| factor_tables(coeffs[block + i.min(count - 1)].coefficient()));
        for (w, dst_lane) in dst_lanes.iter_mut().enumerate() {
            let mut acc = _mm256_loadu_si256(&*dst_lane);
            for (i, factor) in factors.iter().take(count).enumerate() {
                let (src_lanes, _) = srcs[block + i].as_chunks::<32>();
                let source = _mm256_loadu_si256(&src_lanes[w]);
                let split = split_source_avx2(source, &lanes);
                acc = _mm256_xor_si256(acc, scale_split_avx2(&split, factor, lanes.even));
            }
            _mm256_storeu_si256(dst_lane, acc);
        }
        for i in 0..count {
            mul_add_scalar(
                dst_tail,
                coeffs[block + i].coefficient(),
                &srcs[block + i][tail_start..],
            );
        }
    }
}

/// Many sources into one destination, eight coefficients prepared per pass.
///
/// # Panics
/// As [`mul_add_gather_avx2`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_ssse3<C: TableCoefficient>(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    coeffs: &[C],
    srcs: &[&[u8]],
) {
    super::check_elements("gf16::mul_add_gather_ssse3", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "gf16::mul_add_gather_ssse3: coefficients is {} but sources is {}",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "gf16::mul_add_gather_ssse3: dst is {} bytes but source {index} is {} bytes",
            dst.len(),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    let tail_start = dst.len() & !15;
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for block in (0..coeffs.len()).step_by(TERM_TILE) {
        let count = (coeffs.len() - block).min(TERM_TILE);
        let vectors: [NibbleSsse3; TERM_TILE] = core::array::from_fn(|i| {
            coeffs[block + i.min(count - 1)].with_tables(|tables| nibble_ssse3(tables))
        });
        for (w, dst_lane) in dst_lanes.iter_mut().enumerate() {
            let mut acc = _mm_loadu_si128(&*dst_lane);
            for i in 0..count {
                let (src_lanes, _) = srcs[block + i].as_chunks::<16>();
                let source = _mm_loadu_si128(&src_lanes[w]);
                acc = _mm_xor_si128(acc, scale_ssse3(source, &vectors[i]));
            }
            _mm_storeu_si128(dst_lane, acc);
        }
        for i in 0..count {
            mul_add_scalar(
                dst_tail,
                coeffs[block + i].coefficient(),
                &srcs[block + i][tail_start..],
            );
        }
    }
}

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
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    super::check_elements("gf16::mul_add_gather_gfni", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "gf16::mul_add_gather_gfni: coefficients is {} but sources is {}",
        coeffs.len(),
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
    let swap = swap_mask256();
    let mut i = 0;
    while i + SOURCE_GROUP <= coeffs.len() {
        let mut group = [Elem(0); SOURCE_GROUP];
        group.copy_from_slice(&coeffs[i..i + SOURCE_GROUP]);
        let mut sources: [&[u8]; SOURCE_GROUP] = [&[]; SOURCE_GROUP];
        sources.copy_from_slice(&srcs[i..i + SOURCE_GROUP]);
        gather_group_gfni(dst, group, sources, swap);
        i += SOURCE_GROUP;
    }
    if i + 2 <= coeffs.len() {
        let mut group = [Elem(0); 2];
        group.copy_from_slice(&coeffs[i..i + 2]);
        let mut sources: [&[u8]; 2] = [&[]; 2];
        sources.copy_from_slice(&srcs[i..i + 2]);
        gather_group_gfni(dst, group, sources, swap);
        i += 2;
    }
    if i < coeffs.len() {
        // The unrolled single-coefficient kernel is the better shape for the
        // last source.
        mul_add_gfni(token, dst, TowerCoeff::new(coeffs[i]), srcs[i]);
    }
}

/// Fold `N` sources into `dst` in one pass over the destination.
///
/// `N` is a constant, so the `2 * N` broadcast vectors have a register home
/// for the whole pass; they are derived once per group, never inside the byte
/// loop.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn gather_group_gfni<const N: usize>(
    dst: &mut [u8],
    coeffs: [Elem; N],
    srcs: [&[u8]; N],
    swap: __m256i,
) {
    let tail_start = dst.len() & !31;
    let mut same = [_mm256_setzero_si256(); N];
    let mut cross = [_mm256_setzero_si256(); N];
    for (k, &coeff) in coeffs.iter().enumerate() {
        let (same_word, cross_word) = broadcast_words(TowerCoeff::new(coeff));
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

    for (k, &coeff) in coeffs.iter().enumerate() {
        mul_add_scalar(dst_tail, coeff, &srcs[k][tail_start..]);
    }
}
