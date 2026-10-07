//! GF(2^8) lane-wise products over 64-byte lanes: `dst[i] = a[i] * b[i]`.
//!
//! Both operands vary per lane. In the AES encoding the product is one
//! vector-by-vector `GF2P8MULB`; every other representation conjugates that
//! multiply by the isomorphism onto `0x11B`: two affine maps in, one native
//! multiply, one affine map out. The isomorphism qwords and the reduction
//! byte arrive as explicit arguments, and `NATIVE` selects the native versus
//! conjugated bodies.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics. Complete 64-byte lanes run here; the remainder
//! descends through the matching 32-byte entry in [`super::super::gfni`],
//! which carries its own 16-byte and scalar tail. The kernels dispatch on the
//! `V4x` tier under the `simd512` feature.

use super::super::gfni::{mul_elementwise_assign_gfni, mul_elementwise_gfni};
use crate::kernel::tables::Isomorphism;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `dst[i] = a[i] * b[i]` over 64-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx512<const NATIVE: bool>(
    token: archmage::X64V4xToken,
    iso: Isomorphism,
    reduction: u8,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_avx512: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_avx512: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_rest) = a.as_chunks::<64>();
    let (b_lanes, b_rest) = b.as_chunks::<64>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm512_loadu_si512(alane);
        let y = _mm512_loadu_si512(blane);
        _mm512_storeu_si512(dlane, multiply_vectors::<NATIVE>(x, y, iso));
    }
    mul_elementwise_gfni::<NATIVE>(
        token.v3_gfni_crypto(),
        iso,
        reduction,
        dst_rest,
        a_rest,
        b_rest,
    );
}

/// `dst[i] = dst[i] * src[i]` over 64-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx512<const NATIVE: bool>(
    token: archmage::X64V4xToken,
    iso: Isomorphism,
    reduction: u8,
    dst: &mut [u8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_avx512: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_rest) = src.as_chunks::<64>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm512_loadu_si512(&*dlane);
        let y = _mm512_loadu_si512(slane);
        _mm512_storeu_si512(dlane, multiply_vectors::<NATIVE>(x, y, iso));
    }
    mul_elementwise_assign_gfni::<NATIVE>(
        token.v3_gfni_crypto(),
        iso,
        reduction,
        dst_rest,
        src_rest,
    );
}

/// `x * y` over a 512-bit lane: the native multiply when `NATIVE`, else
/// `φ⁻¹(GF2P8MULB(φx, φy))` with the maps elided in the native case.
#[inline]
#[archmage::rite(v4x, import_intrinsics)]
fn multiply_vectors<const NATIVE: bool>(x: __m512i, y: __m512i, iso: Isomorphism) -> __m512i {
    if NATIVE {
        return _mm512_gf2p8mul_epi8(x, y);
    }
    let forward = _mm512_set1_epi64(iso.forward.cast_signed());
    let inverse = _mm512_set1_epi64(iso.inverse.cast_signed());
    let px = _mm512_gf2p8affine_epi64_epi8::<0>(x, forward);
    let py = _mm512_gf2p8affine_epi64_epi8::<0>(y, forward);
    _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_gf2p8mul_epi8(px, py), inverse)
}
