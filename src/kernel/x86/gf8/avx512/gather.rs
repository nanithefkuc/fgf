//! Many sources into one destination for `Gf8D` over 64-byte lanes.
//!
//! Mirrors the GFNI gather at 64-byte lanes with `VGF2P8AFFINEQB`: the
//! destination tile stays in registers across every source.

use super::super::check_gather;
use super::{MapCoeff, bfactor_avx512, bmul_avx512, brem_avx512};

/// Shortest gather destination that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per source, so it repays only once the
/// tile body is long enough. Set by the misaligned-gather floor sweep in
/// `BENCHMARKS.md` ("AVX-512 alignment peel floors").
pub(crate) const GATHER_PEEL_MIN: usize = 3072;

// ---------------------------------------------------------------------------
// Gather: many sources into one destination.
// ---------------------------------------------------------------------------

/// Many sources into one destination, register-blocked over 256-byte tiles.
///
/// The destination tile stays in registers across every source, so it is
/// read and written once per tile, not once per source.
///
/// # Panics
/// Panics unless `srcs.len() == coeffs.len()` and every source matches `dst`
/// in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_avx512<C: MapCoeff>(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    coeffs: &[C],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_avx512", dst, coeffs.len(), srcs);
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    // Head bytes run through the single-source seam until the destination
    // is 64-byte aligned; the sources share the destination's alignment in
    // every dispatched shape.
    let head = dst.as_ptr().align_offset(64);
    let head = if head == usize::MAX || head >= dst.len() || dst.len() < GATHER_PEEL_MIN {
        0
    } else {
        head
    };
    if head > 0 {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            brem_avx512(
                &mut dst[..head],
                C::map(coeff),
                C::table(coeff),
                &src[..head],
            );
        }
    }
    let (dhead, dtile) = dst.split_at_mut(head);
    let _ = dhead;
    gather_tiles(dtile, coeffs, srcs, head);
    let tile = dtile.len() / 256 * 256;
    gather_remainder(&mut dtile[tile..], coeffs, srcs, head + tile);
}

/// The main register-blocked tile loop: four 64-byte accumulators over whole
/// tiles, every source folded in before the destination stores.
#[archmage::rite(v4x, import_intrinsics)]
fn gather_tiles<C: MapCoeff>(dst: &mut [u8], coeffs: &[C], srcs: &[&[u8]], base: usize) {
    let mut offset = base;
    for dtile in dst.chunks_exact_mut(256) {
        let (dl, _) = dtile.as_chunks_mut::<64>();
        // `offset + 256 <= dst.len() <= every src.len()` (asserted by the
        // entry), so each accumulator load stays inside `dst`.
        let mut acc = [
            _mm512_loadu_si512(&dl[0]),
            _mm512_loadu_si512(&dl[1]),
            _mm512_loadu_si512(&dl[2]),
            _mm512_loadu_si512(&dl[3]),
        ];
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = bfactor_avx512(C::map(coeff));
            let (sl, _) = src[offset..offset + 256].as_chunks::<64>();
            let x0 = _mm512_loadu_si512(&sl[0]);
            let x1 = _mm512_loadu_si512(&sl[1]);
            let x2 = _mm512_loadu_si512(&sl[2]);
            let x3 = _mm512_loadu_si512(&sl[3]);
            acc[0] = _mm512_xor_si512(acc[0], bmul_avx512(x0, factor));
            acc[1] = _mm512_xor_si512(acc[1], bmul_avx512(x1, factor));
            acc[2] = _mm512_xor_si512(acc[2], bmul_avx512(x2, factor));
            acc[3] = _mm512_xor_si512(acc[3], bmul_avx512(x3, factor));
        }
        _mm512_storeu_si512(&mut dl[0], acc[0]);
        _mm512_storeu_si512(&mut dl[1], acc[1]);
        _mm512_storeu_si512(&mut dl[2], acc[2]);
        _mm512_storeu_si512(&mut dl[3], acc[3]);
        offset += 256;
    }
}

/// The single-source AXPY remainder over what no tile covered.
#[archmage::rite(v4x, import_intrinsics)]
fn gather_remainder<C: MapCoeff>(dst: &mut [u8], coeffs: &[C], srcs: &[&[u8]], tail: usize) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        brem_avx512(dst, C::map(coeff), C::table(coeff), &src[tail..]);
    }
}
