//! Single-buffer GF(2^8) AXPY kernels on the GFNI multiply.
//!
//! `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply across 32 byte lanes,
//! so a `Gf8B` coefficient is nothing but a broadcast byte. It is pipelined
//! but not single-cycle, which shapes every loop here: four independent
//! multiply chains stay in flight to cover the latency. `VGF2P8AFFINEQB`
//! applies an arbitrary 8x8 GF(2) map per lane and so serves `Gf8D`, with the
//! same unroll and the same lane descent.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics. The fused `mul_into` bodies keep the
//! non-temporal store residue in [`archmage::rite`] helpers, the one unsafe
//! dependency this family retains.

use crate::field::gf8b::Elem;
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::tables::{ScaleTable, scale_table};

/// `dst ^= coeff * src` using `GF2P8MULB` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: Elem,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    mul_add_gfni_impl(dst, coeff, src);
}

/// The `mul_add` body over equal-length slices, shared with the blocked
/// kernels' remainder seam `brem`.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
pub(super) fn mul_add_gfni_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);

    // Four independent multiply chains per iteration. A single destination
    // AXPY is latency-bound on `GF2P8MULB`, and the unroll is what covers it.
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
        let r0 = _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x0, factor));
        let r1 = _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x1, factor));
        let r2 = _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x2, factor));
        let r3 = _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x3, factor));
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
        let r = _mm256_xor_si256(d, _mm256_gf2p8mul_epi8(x, factor));
        _mm256_storeu_si256(dlane, r);
    }
    let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src16, src_tail) = src_rest.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        let d = _mm_loadu_si128(d16);
        let r = _mm_xor_si128(d, _mm_gf2p8mul_epi8(x, factor128));
        _mm_storeu_si128(d16, r);
    }

    mul_add_nibble(dst_tail, scale_table(coeff), src_tail);
}

/// `dst = coeff * dst` using `GF2P8MULB` over 32-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gfni(_token: archmage::X64V3GfniCryptoToken, dst: &mut [u8], coeff: Elem) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);
    // In-place scaling is store-bound and rare next to the AXPY shapes, so
    // one accumulator is enough; the loads have no dependency to cover.
    let (lanes, rest) = dst.as_chunks_mut::<32>();
    for lane in lanes {
        let d = _mm256_loadu_si256(&*lane);
        _mm256_storeu_si256(lane, _mm256_gf2p8mul_epi8(d, factor));
    }
    let (sixteens, tail) = rest.as_chunks_mut::<16>();
    for s16 in sixteens {
        let d = _mm_loadu_si128(&*s16);
        _mm_storeu_si128(s16, _mm_gf2p8mul_epi8(d, factor128));
    }

    mul_assign_nibble(tail, scale_table(coeff));
}

/// `dst = coeff * src` using `GF2P8MULB` over 32-byte lanes.
///
/// Fused out-of-place multiply: one pass, one read of `src` and one write of
/// `dst`, versus the copy-then-scale pair the trait default runs. On large
/// buffers that halves destination traffic and is worth roughly 2x.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    coeff: Elem,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    match super::super::nt_split(dst, 1) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_gfni_impl::<false>(head, coeff, src_head);
            mul_into_gfni_impl::<true>(body, coeff, src_body);
            _mm_sfence();
        }
        None => mul_into_gfni_impl::<false>(dst, coeff, src),
    }
}

/// Fused `GF2P8MULB` body over equal-length slices, `NT`-selected tile stores.
///
/// Loads and the sub-tile stores are reference-based; only the tile stores
/// run through the crate's streaming-store primitive, which is the residue
/// this body keeps.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
#[allow(unsafe_code)]
fn mul_into_gfni_impl<const NT: bool>(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);

    // Four independent multiply chains, as in the AXPY: `GF2P8MULB` is
    // pipelined, and with no destination read there is even less other work
    // to hide its latency behind. The cursor walks whole tiles over the
    // remaining destination, which is what lets the prefetch name a line
    // ahead of it without leaving the slice it came from.
    let prefetch = super::super::prefetch_dst(dst, NT);
    let tiles = dst.len() / 128 * 128;
    let (mut drest, dst_mid) = dst.split_at_mut(tiles);
    let (mut srest, src_mid) = src.split_at(tiles);
    while !drest.is_empty() {
        if prefetch && drest.len() >= super::super::PREFETCH_AHEAD + 128 {
            super::super::prefetch_tile(&drest[super::super::PREFETCH_AHEAD..]);
        }
        let (dtile, dnext) = drest.split_at_mut(128);
        let (stile, snext) = srest.split_at(128);
        let (s, _) = stile.as_chunks::<32>();
        let r0 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(&s[0]), factor);
        let r1 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(&s[1]), factor);
        let r2 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(&s[2]), factor);
        let r3 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(&s[3]), factor);
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
            super::super::store256::<NT>(dp, r0);
            super::super::store256::<NT>(dp.add(32), r1);
            super::super::store256::<NT>(dp.add(64), r2);
            super::super::store256::<NT>(dp.add(96), r3);
        }
        drest = dnext;
        srest = snext;
    }
    let (dst_lanes, dst_mid2) = dst_mid.as_chunks_mut::<32>();
    let (src_lanes, src_mid2) = src_mid.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, _mm256_gf2p8mul_epi8(x, factor));
    }
    let (dst16, dst_tail) = dst_mid2.as_chunks_mut::<16>();
    let (src16, src_tail) = src_mid2.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        _mm_storeu_si128(d16, _mm_gf2p8mul_epi8(x, factor128));
    }

    mul_into_nibble(dst_tail, scale_table(coeff), src_tail);
}

// ---------------------------------------------------------------------------
// Single buffer: GFNI affine.
//
// `VGF2P8AFFINEQB` applies an arbitrary 8×8 GF(2) linear map per byte lane,
// so it multiplies in *any* GF(2^8): the polynomial lives in the prepared
// matrix qword, not in the instruction. `Gf8D` (`0x11D`) uses these kernels;
// `Gf8B` has the native `GF2P8MULB` and never needs them. Each loop mirrors
// its `GF2P8MULB` counterpart exactly — same unroll, same lane descent — and
// only the multiply instruction differs. The coefficient enters as its
// affine matrix qword (see `tables::affine_8d`); its nibble table covers the
// sub-XMM tail, so both prepared forms arrive as arguments and no kernel
// below reaches for a bank.
// ---------------------------------------------------------------------------

/// `dst ^= coeff * src` using `VGF2P8AFFINEQB` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_affine(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    mul_add_affine_impl(dst, map, table, src);
}

/// The affine `mul_add` body over equal-length slices, shared with the
/// blocked kernels' remainder seam `brem`.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
pub(super) fn mul_add_affine_impl(dst: &mut [u8], map: u64, table: &ScaleTable, src: &[u8]) {
    let factor = _mm256_set1_epi64x(map.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);

    // Four independent multiply chains per iteration. A single destination
    // AXPY is latency-bound on `VGF2P8AFFINEQB`, and the unroll is what
    // covers it.
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
        let r0 = _mm256_xor_si256(d0, _mm256_gf2p8affine_epi64_epi8::<0>(x0, factor));
        let r1 = _mm256_xor_si256(d1, _mm256_gf2p8affine_epi64_epi8::<0>(x1, factor));
        let r2 = _mm256_xor_si256(d2, _mm256_gf2p8affine_epi64_epi8::<0>(x2, factor));
        let r3 = _mm256_xor_si256(d3, _mm256_gf2p8affine_epi64_epi8::<0>(x3, factor));
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
        let r = _mm256_xor_si256(d, _mm256_gf2p8affine_epi64_epi8::<0>(x, factor));
        _mm256_storeu_si256(dlane, r);
    }
    let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src16, src_tail) = src_rest.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(s16);
        let d = _mm_loadu_si128(d16);
        let r = _mm_xor_si128(d, _mm_gf2p8affine_epi64_epi8::<0>(x, factor128));
        _mm_storeu_si128(d16, r);
    }

    mul_add_nibble(dst_tail, table, src_tail);
}

/// `dst = coeff * dst` using `VGF2P8AFFINEQB` over 32-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_affine(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
) {
    let factor = _mm256_set1_epi64x(map.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);
    // In-place scaling is store-bound and rare next to the AXPY shapes, so
    // one accumulator is enough; the loads have no dependency to cover.
    let (lanes, rest) = dst.as_chunks_mut::<32>();
    for lane in lanes {
        let d = _mm256_loadu_si256(&*lane);
        _mm256_storeu_si256(lane, _mm256_gf2p8affine_epi64_epi8::<0>(d, factor));
    }
    let (sixteens, tail) = rest.as_chunks_mut::<16>();
    for s16 in sixteens {
        let d = _mm_loadu_si128(&*s16);
        _mm_storeu_si128(s16, _mm_gf2p8affine_epi64_epi8::<0>(d, factor128));
    }

    mul_assign_nibble(tail, table);
}

/// `dst = coeff * src` using `VGF2P8AFFINEQB` over 32-byte lanes.
///
/// Fused out-of-place multiply: one pass, one read of `src` and one write of
/// `dst`, versus the copy-then-scale pair the trait default runs.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_affine(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    map: u64,
    table: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    match super::super::nt_split(dst, 1) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_affine_impl::<false>(head, map, table, src_head);
            mul_into_affine_impl::<true>(body, map, table, src_body);
            _mm_sfence();
        }
        None => mul_into_affine_impl::<false>(dst, map, table, src),
    }
}

/// Fused `VGF2P8AFFINEQB` body over equal-length slices, `NT`-selected tile
/// stores.
///
/// As [`mul_into_gfni_impl`]: reference-based loads and sub-tile stores, with
/// the streaming-store primitive carrying the tile stores.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
#[allow(unsafe_code)]
fn mul_into_affine_impl<const NT: bool>(dst: &mut [u8], map: u64, table: &ScaleTable, src: &[u8]) {
    let factor = _mm256_set1_epi64x(map.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);

    // Four independent multiply chains, as in the AXPY: `VGF2P8AFFINEQB` is
    // pipelined, and with no destination read there is even less other work
    // to hide its latency behind. As in [`mul_into_gfni_impl`], the cursor
    // walks whole tiles so the prefetch can name a line ahead of it.
    let prefetch = super::super::prefetch_dst(dst, NT);
    let tiles = dst.len() / 128 * 128;
    let (mut drest, dst_mid) = dst.split_at_mut(tiles);
    let (mut srest, src_mid) = src.split_at(tiles);
    while !drest.is_empty() {
        if prefetch && drest.len() >= super::super::PREFETCH_AHEAD + 128 {
            super::super::prefetch_tile(&drest[super::super::PREFETCH_AHEAD..]);
        }
        let (dtile, dnext) = drest.split_at_mut(128);
        let (stile, snext) = srest.split_at(128);
        let (s, _) = stile.as_chunks::<32>();
        let r0 = _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_loadu_si256(&s[0]), factor);
        let r1 = _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_loadu_si256(&s[1]), factor);
        let r2 = _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_loadu_si256(&s[2]), factor);
        let r3 = _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_loadu_si256(&s[3]), factor);
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
            super::super::store256::<NT>(dp, r0);
            super::super::store256::<NT>(dp.add(32), r1);
            super::super::store256::<NT>(dp.add(64), r2);
            super::super::store256::<NT>(dp.add(96), r3);
        }
        drest = dnext;
        srest = snext;
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
        _mm_storeu_si128(d16, _mm_gf2p8affine_epi64_epi8::<0>(x, factor128));
    }

    mul_into_nibble(dst_tail, table, src_tail);
}
