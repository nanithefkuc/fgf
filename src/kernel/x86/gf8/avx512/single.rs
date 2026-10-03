//! AVX-512 affine single-buffer GF(2^8) kernels (`VGF2P8AFFINEQB`, 64-byte
//! lanes).
//!
//! Each kernel folds its coefficient in with the const-derived per-8-byte
//! affine map of its polynomial. The immediate is the XOR constant, so `<0>`
//! selects the pure linear map; the factor replicates one 64-bit map qword to
//! all eight 64-bit lanes, one map per 8-byte group.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics, mirroring the AVX2 affine bodies in
//! [`super::super::gfni`] lane-for-lane at 64 bytes with the same four-chain
//! unroll. Only complete 64-byte lanes run here; the remainder descends
//! through the 256-bit affine ladder to the scalar nibble kernel.
//!
//! The kernels dispatch on the `V4x` tier under the `simd512` feature and
//! stay reachable through the `internals` facade for differential tests and
//! measurement on AVX-512 hardware. `mul_into` keeps ordinary stores: the
//! streaming-store policy needs its own 64-byte alignment proof and
//! measurement before it lands here.

use super::super::gfni::{Affine, mul_add_gfni_impl, mul_assign_gfni_impl, mul_into_gfni_impl};
use super::{Blocked, bfactor_avx512, bmul_avx512, peel_avx512};

/// `dst ^= coeff * src` using `VGF2P8AFFINEQB` over 64-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512<C: Blocked>(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    coeff: C,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_add_avx512: dst and src differ in length"
    );
    mul_add_avx512_impl(dst, coeff, src);
}

/// The affine `mul_add` body over equal-length slices.
///
/// Four independent multiply chains per tile cover the multiplier latency,
/// as in the AVX2 form.
#[archmage::rite(v4x, import_intrinsics)]
pub(super) fn mul_add_avx512_impl<C: Blocked>(dst: &mut [u8], coeff: C, src: &[u8]) {
    let factor = bfactor_avx512(coeff.map());

    // Head bytes run on the 256-bit body until the destination is 64-byte
    // aligned: a 64-byte access at 16 mod 64 splits two cache lines, and the
    // aligned body repays the peel wherever the vector loop runs. The head
    // also covers the source, which shares the destination's alignment in
    // every dispatched shape.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        mul_add_gfni_impl(&mut dst[..head], Affine(coeff), &src[..head]);
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

    mul_add_gfni_impl(dst_rest, Affine(coeff), src_rest);
}

/// `dst = coeff * dst` using `VGF2P8AFFINEQB` over 64-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx512<C: Blocked>(_token: archmage::X64V4xToken, dst: &mut [u8], coeff: C) {
    let factor = bfactor_avx512(coeff.map());
    // As in `mul_add_avx512_impl`: peel to 64-byte alignment first.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        mul_assign_gfni_impl(&mut dst[..head], Affine(coeff));
    }
    let dst = &mut dst[head..];
    // Four lanes per tile cut loop overhead against the single-accumulator
    // form.
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

    // The 32/16-byte and sub-lane tail: the AVX2 ladder below 64 bytes. The
    // tail is shorter than one half-lane peel, so that body takes no head.
    mul_assign_gfni_impl(tail, Affine(coeff));
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
pub fn mul_into_avx512<C: Blocked>(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    coeff: C,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_into_avx512: dst and src differ in length"
    );
    // As in `mul_add_avx512_impl`: peel to 64-byte alignment first.
    let head = peel_avx512(dst.as_ptr(), dst.len());
    if head > 0 {
        mul_into_gfni_impl::<_, false>(&mut dst[..head], Affine(coeff), &src[..head]);
    }
    let (dst, src) = (&mut dst[head..], &src[head..]);
    let factor = bfactor_avx512(coeff.map());

    // Four independent multiply chains, as in the AXPY: with no destination
    // read there is even less other work to hide the multiplier latency
    // behind. The cursor walks whole tiles so the prefetch can name the lines
    // ahead of it: one `prefetch_tile` per 128-byte half of the 256-byte
    // tile.
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

    // The temporal AVX2 descent below 64 bytes: no prefetch and no streaming
    // stores, which route past the streaming threshold to the AVX2 body.
    mul_into_gfni_impl::<_, false>(dst_rest, Affine(coeff), src_rest);
}
