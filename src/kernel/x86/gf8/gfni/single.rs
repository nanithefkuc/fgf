//! Single-buffer GF(2^8) AXPY kernels.
//!
//! `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply across 32 byte lanes,
//! so a `Gf8<Poly<AES>>` coefficient is nothing but a broadcast byte;
//! `VGF2P8AFFINEQB` applies an arbitrary 8x8 GF(2) map per lane and so serves
//! every other polynomial. Both are pipelined but not single-cycle, which
//! shapes every loop here: four independent multiply chains stay in flight to
//! cover the latency. The [`Blocked`] coefficient selects the instruction at
//! compile time, so each polynomial monomorphizes to one multiply with the
//! same unroll and the same lane descent.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics. The fused `mul_into` bodies keep the
//! non-temporal store residue in [`archmage::rite`] helpers, the one unsafe
//! dependency this family retains.

use super::{Blocked, bfactor_gfni, bmul_gfni, bmul_half_gfni};
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::x86::HALF_LANE_PEEL_MIN;

/// `dst ^= coeff * src` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gfni<S: Blocked>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: S,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_add_gfni: dst and src differ in length"
    );
    // The network-payload half-lane case pays for one narrow head: its
    // remaining AVX2 stores then avoid split cache lines. The aligned path
    // hands its slices to the body untouched.
    if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
        let factor_half = _mm256_castsi256_si128(bfactor_gfni(coeff));
        let (d, _) = dst[..16].as_chunks_mut::<16>();
        let (s, _) = src[..16].as_chunks::<16>();
        let x = _mm_loadu_si128(&s[0]);
        let d0 = _mm_loadu_si128(&d[0]);
        _mm_storeu_si128(
            &mut d[0],
            _mm_xor_si128(d0, bmul_half_gfni::<S>(x, factor_half)),
        );
        mul_add_gfni_impl(&mut dst[16..], coeff, &src[16..]);
    } else {
        mul_add_gfni_impl(dst, coeff, src);
    }
}

/// The `mul_add` body over equal-length slices, shared with the blocked
/// kernels' remainder seam `brem_gfni` and the 512-bit entries' peels.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
pub(in crate::kernel::x86::gf8) fn mul_add_gfni_impl<S: Blocked>(
    dst: &mut [u8],
    coeff: S,
    src: &[u8],
) {
    let factor = bfactor_gfni(coeff);
    let factor_half = _mm256_castsi256_si128(factor);

    // Four independent multiply chains per iteration. A single destination
    // AXPY is latency-bound on the multiply, and the unroll is what covers
    // it.
    let (dst_tiles, dst_mid) = dst.as_chunks_mut::<128>();
    let (src_tiles, src_mid) = src.as_chunks::<128>();
    for (dtile, stile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d, _) = dtile.as_chunks_mut::<32>();
        let (s, _) = stile.as_chunks::<32>();
        let x0 = _mm256_loadu_si256(&s[0]);
        let x1 = _mm256_loadu_si256(&s[1]);
        let x2 = _mm256_loadu_si256(&s[2]);
        let x3 = _mm256_loadu_si256(&s[3]);
        let d0 = _mm256_loadu_si256(&d[0]);
        let d1 = _mm256_loadu_si256(&d[1]);
        let d2 = _mm256_loadu_si256(&d[2]);
        let d3 = _mm256_loadu_si256(&d[3]);
        let r0 = _mm256_xor_si256(d0, bmul_gfni::<S>(x0, factor));
        let r1 = _mm256_xor_si256(d1, bmul_gfni::<S>(x1, factor));
        let r2 = _mm256_xor_si256(d2, bmul_gfni::<S>(x2, factor));
        let r3 = _mm256_xor_si256(d3, bmul_gfni::<S>(x3, factor));
        _mm256_storeu_si256(&mut d[0], r0);
        _mm256_storeu_si256(&mut d[1], r1);
        _mm256_storeu_si256(&mut d[2], r2);
        _mm256_storeu_si256(&mut d[3], r3);
    }

    let (dst_lanes, dst_rest) = dst_mid.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src_mid.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        let d = _mm256_loadu_si256(dlane);
        let r = _mm256_xor_si256(d, bmul_gfni::<S>(x, factor));
        _mm256_storeu_si256(dlane, r);
    }
    let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src16, src_tail) = src_rest.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        let d = _mm_loadu_si128(d16);
        let r = _mm_xor_si128(d, bmul_half_gfni::<S>(x, factor_half));
        _mm_storeu_si128(d16, r);
    }

    mul_add_nibble(dst_tail, coeff.table(), src_tail);
}

/// `dst = coeff * dst` over 32-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gfni<S: Blocked>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: S,
) {
    mul_assign_gfni_impl(dst, coeff);
}

/// The `mul_assign` body over one slice, shared with the 512-bit entry's
/// alignment peel.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
pub(in crate::kernel::x86::gf8) fn mul_assign_gfni_impl<S: Blocked>(dst: &mut [u8], coeff: S) {
    let factor = bfactor_gfni(coeff);
    let factor_half = _mm256_castsi256_si128(factor);
    // The network-payload half-lane case pays for one narrow head: its
    // remaining AVX2 stores then avoid split cache lines.
    let head = if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
        16
    } else {
        0
    };
    if head != 0 {
        let (d, _) = dst[..head].as_chunks_mut::<16>();
        let value = bmul_half_gfni::<S>(_mm_loadu_si128(&d[0]), factor_half);
        _mm_storeu_si128(&mut d[0], value);
    }
    let dst = &mut dst[head..];
    // In-place scaling is store-bound and rare next to the AXPY shapes, so
    // one accumulator is enough; the loads have no dependency to cover.
    let (lanes, rest) = dst.as_chunks_mut::<32>();
    for lane in lanes {
        let d = _mm256_loadu_si256(&*lane);
        _mm256_storeu_si256(lane, bmul_gfni::<S>(d, factor));
    }
    let (sixteens, tail) = rest.as_chunks_mut::<16>();
    for s16 in sixteens {
        let d = _mm_loadu_si128(&*s16);
        _mm_storeu_si128(s16, bmul_half_gfni::<S>(d, factor_half));
    }

    mul_assign_nibble(tail, coeff.table());
}

/// `dst = coeff * src` over 32-byte lanes.
///
/// Fused out-of-place multiply: one pass, one read of `src` and one write of
/// `dst`, versus the copy-then-scale pair the trait default runs.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gfni<S: Blocked>(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: S,
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_into_gfni: dst and src differ in length"
    );
    if let Some(peel) = crate::kernel::x86::nt_split(dst, 1) {
        let (head, body) = dst.split_at_mut(peel);
        let (src_head, src_body) = src.split_at(peel);
        mul_into_gfni_impl::<S, false>(head, coeff, src_head);
        mul_into_gfni_impl::<S, true>(body, coeff, src_body);
        _mm_sfence();
    } else {
        // Below the streaming threshold a half-lane destination still
        // pays split lines; one narrow head aligns the temporal body.
        let head = if dst.len() >= HALF_LANE_PEEL_MIN && dst.as_ptr().align_offset(32) == 16 {
            16
        } else {
            0
        };
        if head != 0 {
            let (dst_head, dst) = dst.split_at_mut(head);
            let (src_head, src) = src.split_at(head);
            mul_into_gfni_impl::<S, false>(dst_head, coeff, src_head);
            mul_into_gfni_impl::<S, false>(dst, coeff, src);
        } else {
            mul_into_gfni_impl::<S, false>(dst, coeff, src);
        }
    }
}

/// Fused body over equal-length slices, `NT`-selected tile stores.
///
/// Loads and the sub-tile stores are reference-based; only the tile stores
/// run through the crate's streaming-store primitive, which is the residue
/// this body keeps.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
#[allow(unsafe_code)]
pub(in crate::kernel::x86::gf8) fn mul_into_gfni_impl<S: Blocked, const NT: bool>(
    dst: &mut [u8],
    coeff: S,
    src: &[u8],
) {
    let factor = bfactor_gfni(coeff);
    let factor_half = _mm256_castsi256_si128(factor);

    // Four independent multiply chains, as in the AXPY: the multiply is
    // pipelined, and with no destination read there is even less other work
    // to hide its latency behind. The cursor walks whole tiles over the
    // remaining destination, which is what lets the prefetch name a line
    // ahead of it without leaving the slice it came from.
    let prefetch = crate::kernel::x86::prefetch_dst(dst, NT);
    let tiles = dst.len() / 128 * 128;
    let (mut drest, dst_mid) = dst.split_at_mut(tiles);
    let (mut srest, src_mid) = src.split_at(tiles);
    while !drest.is_empty() {
        if prefetch && drest.len() >= crate::kernel::x86::PREFETCH_AHEAD + 128 {
            crate::kernel::x86::prefetch_tile(&drest[crate::kernel::x86::PREFETCH_AHEAD..]);
        }
        let (dtile, dnext) = drest.split_at_mut(128);
        let (stile, snext) = srest.split_at(128);
        let (s, _) = stile.as_chunks::<32>();
        let r0 = bmul_gfni::<S>(_mm256_loadu_si256(&s[0]), factor);
        let r1 = bmul_gfni::<S>(_mm256_loadu_si256(&s[1]), factor);
        let r2 = bmul_gfni::<S>(_mm256_loadu_si256(&s[2]), factor);
        let r3 = bmul_gfni::<S>(_mm256_loadu_si256(&s[3]), factor);
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `dtile` is a 128-byte split of the destination, and the
        //        four stores advance 0, 32, 64 and 96 bytes into it.
        // THUS: every 32-byte store writes inside the destination slice.
        //
        // NON-TEMPORAL STORE ALIGNMENT
        // SINCE: the entry selected `NT` only after `nt_split` peeled the
        //        destination to a 32-byte boundary, and tile stores advance
        //        in 32-byte steps from that boundary.
        // THUS: each streaming store address is 32-byte aligned.
        unsafe {
            let dp = dtile.as_mut_ptr();
            crate::kernel::x86::store_avx2::<NT>(dp, r0);
            crate::kernel::x86::store_avx2::<NT>(dp.add(32), r1);
            crate::kernel::x86::store_avx2::<NT>(dp.add(64), r2);
            crate::kernel::x86::store_avx2::<NT>(dp.add(96), r3);
        }
        drest = dnext;
        srest = snext;
    }
    let (dst_lanes, dst_mid2) = dst_mid.as_chunks_mut::<32>();
    let (src_lanes, src_mid2) = src_mid.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, bmul_gfni::<S>(x, factor));
    }
    let (dst16, dst_tail) = dst_mid2.as_chunks_mut::<16>();
    let (src16, src_tail) = src_mid2.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        _mm_storeu_si128(d16, bmul_half_gfni::<S>(x, factor_half));
    }

    mul_into_nibble(dst_tail, coeff.table(), src_tail);
}
