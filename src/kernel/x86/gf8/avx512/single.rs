//! `Gf8D` AVX-512 affine single-buffer kernels (`VGF2P8AFFINEQB`, 64-byte lanes).
//!
//! The 64-byte tier for the Reed–Solomon field GF(2^8)/`0x11D`, dispatched on
//! `V4x` under the `simd512` feature.
//! `GF2P8MULB` bakes in the AES polynomial and must not multiply here; each
//! kernel folds its coefficient in with the const-derived per-8-byte affine
//! map instead. The immediate is the XOR constant, so `<0>` selects the pure
//! linear map; the factor replicates one 64-bit map qword to all eight
//! 64-bit lanes, one map per 8-byte group.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics, mirroring the AVX2 affine bodies in
//! [`super::super::gfni`] lane-for-lane at 64 bytes with the same four-chain unroll.
//! Only complete 64-byte lanes run here; the remainder descends through the
//! AVX2 ladder to the scalar nibble kernel.
//!
//! The kernels dispatch on the `V4x` tier under the `simd512` feature and stay
//! reachable through the `internals` facade for differential tests and
//! measurement on AVX-512 hardware. `mul_into` keeps ordinary stores: the streaming-store policy needs
//! its own 64-byte alignment proof and measurement before it lands here.

use super::{bfactor_avx512, bmul_avx512, peel_avx512};
use crate::kernel::tables::ScaleTable;

/// `dst ^= coeff * src` using `VGF2P8AFFINEQB` over 64-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    mul_add_avx512_impl(dst, map, table, src);
}

/// The affine `mul_add` body over equal-length slices.
///
/// Four independent multiply chains per tile cover the multiplier latency,
/// as in the AVX2 form.
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn mul_add_avx512_impl(dst: &mut [u8], map: u64, table: &ScaleTable, src: &[u8]) {
    let factor = bfactor_avx512(map);

    // Head bytes run scalar until the destination is 64-byte aligned: a
    // 64-byte access at 16 mod 64 splits two cache lines, and the aligned
    // body repays the peel wherever the vector loop runs. The scalar head
    // also covers the source, which shares the destination's alignment in
    // every dispatched shape.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        super::super::gfni::mul_add_gfni_8d_impl(&mut dst[..head], map, table, &src[..head]);
    }
    let dst = &mut dst[head..];
    let src = &src[head..];

    // Four independent multiply chains per iteration. A single destination
    // AXPY is latency-bound on `VGF2P8AFFINEQB`, and the unroll is what
    // covers it.
    let (dst_tiles, dst_mid) = dst.as_chunks_mut::<256>();
    let (src_tiles, src_mid) = src.as_chunks::<256>();
    for (dtile, stile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d, _) = dtile.as_chunks_mut::<64>();
        let (s, _) = stile.as_chunks::<64>();
        let x0 = _mm512_loadu_si512(&s[0]);
        let x1 = _mm512_loadu_si512(&s[1]);
        let x2 = _mm512_loadu_si512(&s[2]);
        let x3 = _mm512_loadu_si512(&s[3]);
        let d0 = _mm512_loadu_si512(&d[0]);
        let d1 = _mm512_loadu_si512(&d[1]);
        let d2 = _mm512_loadu_si512(&d[2]);
        let d3 = _mm512_loadu_si512(&d[3]);
        let r0 = _mm512_xor_si512(d0, bmul_avx512(x0, factor));
        let r1 = _mm512_xor_si512(d1, bmul_avx512(x1, factor));
        let r2 = _mm512_xor_si512(d2, bmul_avx512(x2, factor));
        let r3 = _mm512_xor_si512(d3, bmul_avx512(x3, factor));
        _mm512_storeu_si512(&mut d[0], r0);
        _mm512_storeu_si512(&mut d[1], r1);
        _mm512_storeu_si512(&mut d[2], r2);
        _mm512_storeu_si512(&mut d[3], r3);
    }

    let (dst_lanes, dst_rest) = dst_mid.as_chunks_mut::<64>();
    let (src_lanes, src_rest) = src_mid.as_chunks::<64>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm512_loadu_si512(slane);
        let d = _mm512_loadu_si512(dlane);
        let r = _mm512_xor_si512(d, bmul_avx512(x, factor));
        _mm512_storeu_si512(dlane, r);
    }

    super::super::gfni::mul_add_gfni_8d_impl(dst_rest, map, table, src_rest);
}

/// `dst = coeff * dst` using `VGF2P8AFFINEQB` over 64-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx512(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
) {
    let factor = bfactor_avx512(map);
    // As in `mul_add_avx512_impl`: peel to 64-byte alignment first.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        super::super::gfni::mul_assign_gfni_8d_impl(&mut dst[..head], map, table);
    }
    let dst = &mut dst[head..];
    // Four lanes per tile cut loop overhead against the single-accumulator
    // form; whether the gap was scheduling or fixed costs is measurement,
    // not shown by the old 256-byte cell alone.
    let (tiles, rest) = dst.as_chunks_mut::<256>();
    for tile in tiles {
        let (d, _) = tile.as_chunks_mut::<64>();
        let d0 = _mm512_loadu_si512(&d[0]);
        let d1 = _mm512_loadu_si512(&d[1]);
        let d2 = _mm512_loadu_si512(&d[2]);
        let d3 = _mm512_loadu_si512(&d[3]);
        _mm512_storeu_si512(&mut d[0], bmul_avx512(d0, factor));
        _mm512_storeu_si512(&mut d[1], bmul_avx512(d1, factor));
        _mm512_storeu_si512(&mut d[2], bmul_avx512(d2, factor));
        _mm512_storeu_si512(&mut d[3], bmul_avx512(d3, factor));
    }
    let (lanes, tail) = rest.as_chunks_mut::<64>();
    for lane in lanes {
        let d = _mm512_loadu_si512(&*lane);
        _mm512_storeu_si512(lane, bmul_avx512(d, factor));
    }

    assign_rest(tail, map, table);
}

/// `dst = coeff * src` using `VGF2P8AFFINEQB` over 64-byte lanes.
///
/// Fused out-of-place multiply: one pass, one read of `src` and one write of
/// `dst`, versus the copy-then-scale pair the trait default runs.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx512(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    // As in `mul_add_avx512_impl`: peel to 64-byte alignment first.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        super::super::gfni::mul_into_gfni_8d_impl::<false>(
            &mut dst[..head],
            map,
            table,
            &src[..head],
        );
    }
    let (dst, src) = (&mut dst[head..], &src[head..]);
    let factor = bfactor_avx512(map);

    // Four independent multiply chains, as in the AXPY: with no destination
    // read there is even less other work to hide the multiplier latency
    // behind. The cursor walks whole tiles so the prefetch can name the lines
    // ahead of it: one `prefetch_tile` per 128-byte half of the 256-byte tile
    // 512 bytes ahead. The AVX2 24 KiB threshold carries over as a candidate;
    // the lead distance wants its own sweep (see the measurement record).
    let prefetch = crate::kernel::x86::prefetch_dst(dst, false);
    let tiles = dst.len() / 256 * 256;
    let (mut drest, dst_mid) = dst.split_at_mut(tiles);
    let (mut srest, src_mid) = src.split_at(tiles);
    while !drest.is_empty() {
        if prefetch && drest.len() >= crate::kernel::x86::PREFETCH_AHEAD + 256 {
            crate::kernel::x86::prefetch_tile(&drest[crate::kernel::x86::PREFETCH_AHEAD..]);
            crate::kernel::x86::prefetch_tile(&drest[crate::kernel::x86::PREFETCH_AHEAD + 128..]);
        }
        let (dtile, dnext) = drest.split_at_mut(256);
        let (stile, snext) = srest.split_at(256);
        let (d, _) = dtile.as_chunks_mut::<64>();
        let (s, _) = stile.as_chunks::<64>();
        let x0 = _mm512_loadu_si512(&s[0]);
        let x1 = _mm512_loadu_si512(&s[1]);
        let x2 = _mm512_loadu_si512(&s[2]);
        let x3 = _mm512_loadu_si512(&s[3]);
        _mm512_storeu_si512(&mut d[0], bmul_avx512(x0, factor));
        _mm512_storeu_si512(&mut d[1], bmul_avx512(x1, factor));
        _mm512_storeu_si512(&mut d[2], bmul_avx512(x2, factor));
        _mm512_storeu_si512(&mut d[3], bmul_avx512(x3, factor));
        drest = dnext;
        srest = snext;
    }
    let (dst_lanes, dst_rest) = dst_mid.as_chunks_mut::<64>();
    let (src_lanes, src_rest) = src_mid.as_chunks::<64>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm512_loadu_si512(slane);
        _mm512_storeu_si512(dlane, bmul_avx512(x, factor));
    }

    into_rest(dst_rest, map, table, src_rest);
}

/// The 32/16-byte and sub-lane tail of [`mul_assign_avx512`]: the AVX2
/// ladder below 64 bytes, mirroring `super::super::gfni::mul_assign_gfni_8d` without
/// disturbing the production body.
#[archmage::rite(v4x, import_intrinsics)]
fn assign_rest(dst: &mut [u8], map: u64, table: &ScaleTable) {
    use crate::kernel::gf8::mul_assign_nibble;
    let factor = _mm256_set1_epi64x(map.cast_signed());
    let factor_half = _mm256_castsi256_si128(factor);
    let (lanes, rest) = dst.as_chunks_mut::<32>();
    for lane in lanes {
        let d = _mm256_loadu_si256(&*lane);
        _mm256_storeu_si256(lane, _mm256_gf2p8affine_epi64_epi8::<0>(d, factor));
    }
    let (sixteens, tail) = rest.as_chunks_mut::<16>();
    for s16 in sixteens {
        let d = _mm_loadu_si128(&*s16);
        _mm_storeu_si128(s16, _mm_gf2p8affine_epi64_epi8::<0>(d, factor_half));
    }
    mul_assign_nibble(tail, table);
}

/// The 32/16-byte and sub-lane tail of [`mul_into_avx512`]: the temporal
/// AVX2 descent below 64 bytes. No prefetch, no streaming stores; the 512-bit
/// store policy routes past 2 MiB to the AVX2 streaming body instead.
#[archmage::rite(v4x, import_intrinsics)]
fn into_rest(dst: &mut [u8], map: u64, table: &ScaleTable, src: &[u8]) {
    use crate::kernel::gf8::mul_into_nibble;
    let factor = _mm256_set1_epi64x(map.cast_signed());
    let factor_half = _mm256_castsi256_si128(factor);
    let (dst_tiles, dst_mid) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_mid) = src.as_chunks::<128>();
    for (dtile, stile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d, _) = dtile.as_chunks_mut::<32>();
        let (s, _) = stile.as_chunks::<32>();
        let x0 = _mm256_loadu_si256(&s[0]);
        let x1 = _mm256_loadu_si256(&s[1]);
        let x2 = _mm256_loadu_si256(&s[2]);
        let x3 = _mm256_loadu_si256(&s[3]);
        _mm256_storeu_si256(&mut d[0], _mm256_gf2p8affine_epi64_epi8::<0>(x0, factor));
        _mm256_storeu_si256(&mut d[1], _mm256_gf2p8affine_epi64_epi8::<0>(x1, factor));
        _mm256_storeu_si256(&mut d[2], _mm256_gf2p8affine_epi64_epi8::<0>(x2, factor));
        _mm256_storeu_si256(&mut d[3], _mm256_gf2p8affine_epi64_epi8::<0>(x3, factor));
    }
    let (dst_lanes, dst_mid2) = dst_mid.as_chunks_mut::<32>();
    let (src_lanes, src_mid2) = src_mid.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, _mm256_gf2p8affine_epi64_epi8::<0>(x, factor));
    }
    let (dst16, dst_tail) = dst_mid2.as_chunks_mut::<16>();
    let (src16, src_tail) = src_mid2.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        _mm_storeu_si128(d16, _mm_gf2p8affine_epi64_epi8::<0>(x, factor_half));
    }
    mul_into_nibble(dst_tail, table, src_tail);
}
