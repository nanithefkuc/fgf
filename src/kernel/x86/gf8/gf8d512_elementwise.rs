//! GF(2^8) lane-wise products over 64-byte lanes: `dst[i] = a[i] * b[i]`.
//!
//! Both byte fields multiply two varying vectors here. The AES field
//! (`Gf8B`, `0x11B`) is one vector-by-vector `GF2P8MULB`. The Reed–Solomon
//! field (`Gf8D`, `0x11D`) conjugates that multiply by the field isomorphism
//! [`ISO_11D_TO_11B`](super::elementwise::ISO_11D_TO_11B): two affine maps in,
//! one native multiply, one affine map out.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics. Complete 64-byte lanes run here; the remainder
//! descends through the matching 32-byte entry in [`super::elementwise`], which
//! carries its own 16-byte and scalar tail. The kernels dispatch on the `V4x`
//! tier under the `simd512` feature.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::elementwise::{
    ISO_11D_TO_11B, mul_elementwise_assign_gfni, mul_elementwise_assign_iso_8d,
    mul_elementwise_gfni, mul_elementwise_iso_8d,
};

/// `dst[i] = a[i] * b[i]` in the AES field using 64-byte `GF2P8MULB`.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gfni512(token: archmage::X64V4xToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_gfni512: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_gfni512: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_rest) = a.as_chunks::<64>();
    let (b_lanes, b_rest) = b.as_chunks::<64>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm512_loadu_si512(alane);
        let y = _mm512_loadu_si512(blane);
        _mm512_storeu_si512(dlane, _mm512_gf2p8mul_epi8(x, y));
    }
    mul_elementwise_gfni(token.v3_gfni_crypto(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = dst[i] * src[i]` in the AES field using 64-byte `GF2P8MULB`.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gfni512(token: archmage::X64V4xToken, dst: &mut [u8], src: &[u8]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_gfni512: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_rest) = src.as_chunks::<64>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm512_loadu_si512(&*dlane);
        let y = _mm512_loadu_si512(slane);
        _mm512_storeu_si512(dlane, _mm512_gf2p8mul_epi8(x, y));
    }
    mul_elementwise_assign_gfni(token.v3_gfni_crypto(), dst_rest, src_rest);
}

/// `dst[i] = a[i] * b[i]` under `0x11D` through the AES-field isomorphism
/// over 64-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_iso512_8d(token: archmage::X64V4xToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_iso512_8d: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_iso512_8d: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_rest) = a.as_chunks::<64>();
    let (b_lanes, b_rest) = b.as_chunks::<64>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm512_loadu_si512(alane);
        let y = _mm512_loadu_si512(blane);
        _mm512_storeu_si512(dlane, multiply_vectors_iso512(x, y));
    }
    mul_elementwise_iso_8d(token.v3_gfni_crypto(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = dst[i] * src[i]` under `0x11D` through the AES-field
/// isomorphism over 64-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_iso512_8d(token: archmage::X64V4xToken, dst: &mut [u8], src: &[u8]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_iso512_8d: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_rest) = src.as_chunks::<64>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm512_loadu_si512(&*dlane);
        let y = _mm512_loadu_si512(slane);
        _mm512_storeu_si512(dlane, multiply_vectors_iso512(x, y));
    }
    mul_elementwise_assign_iso_8d(token.v3_gfni_crypto(), dst_rest, src_rest);
}

/// `x * y` under `0x11D` over a 512-bit lane: `φ⁻¹(GF2P8MULB(φx, φy))`.
#[inline]
#[archmage::rite(v4x, import_intrinsics)]
fn multiply_vectors_iso512(x: __m512i, y: __m512i) -> __m512i {
    let phi = _mm512_set1_epi64(ISO_11D_TO_11B.cast_signed());
    let px = _mm512_gf2p8affine_epi64_epi8::<0>(x, phi);
    let py = _mm512_gf2p8affine_epi64_epi8::<0>(y, phi);
    _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_gf2p8mul_epi8(px, py), phi)
}
