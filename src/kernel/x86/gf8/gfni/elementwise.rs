//! Lane-wise products of two buffers: `dst[i] = a[i] * b[i]`.
//!
//! No coefficient and no table: both operands vary per lane. In the AES
//! encoding the product is one vector-by-vector `GF2P8MULB`; every other
//! representation conjugates that multiply by the isomorphism onto `0x11B`,
//! `a · b = φ⁻¹(GF2P8MULB(φa, φb))` — two affine maps in, one native
//! multiply, one affine map out. The isomorphism qwords and the reduction
//! byte arrive as explicit arguments; nothing here derives a field fact from
//! a polynomial.
//!
//! The buffer entries are safe [`archmage`] capability-token functions over
//! reference-based intrinsics.

use super::super::ssse3::{mul_elementwise_assign_ssse3, mul_elementwise_ssse3};
use crate::kernel::tables::Isomorphism;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `x * y` over a 256-bit lane: the native multiply when `NATIVE`, else the
/// isomorphism-conjugated one.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(in crate::kernel::x86::gf8) fn multiply_vectors_gfni<const NATIVE: bool>(
    x: __m256i,
    y: __m256i,
    iso: Isomorphism,
) -> __m256i {
    if NATIVE {
        return _mm256_gf2p8mul_epi8(x, y);
    }
    let forward = _mm256_set1_epi64x(iso.forward.cast_signed());
    let inverse = _mm256_set1_epi64x(iso.inverse.cast_signed());
    let px = _mm256_gf2p8affine_epi64_epi8::<0>(x, forward);
    let py = _mm256_gf2p8affine_epi64_epi8::<0>(y, forward);
    _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_gf2p8mul_epi8(px, py), inverse)
}

/// `x * y` over a 128-bit lane.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
fn multiply_vectors_half_gfni<const NATIVE: bool>(
    x: __m128i,
    y: __m128i,
    iso: Isomorphism,
) -> __m128i {
    if NATIVE {
        return _mm_gf2p8mul_epi8(x, y);
    }
    let forward = _mm_set1_epi64x(iso.forward.cast_signed());
    let inverse = _mm_set1_epi64x(iso.inverse.cast_signed());
    let px = _mm_gf2p8affine_epi64_epi8::<0>(x, forward);
    let py = _mm_gf2p8affine_epi64_epi8::<0>(y, forward);
    _mm_gf2p8affine_epi64_epi8::<0>(_mm_gf2p8mul_epi8(px, py), inverse)
}

/// `dst[i] = a[i] * b[i]`.
///
/// `NATIVE` selects the AES-encoding bodies; otherwise the lanes conjugate
/// by `iso` and the remainder threads `reduction` through the shift/reduce
/// seam.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gfni<const NATIVE: bool>(
    _token: archmage::X64V3GfniCryptoToken,
    iso: Isomorphism,
    reduction: u8,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_gfni: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_gfni: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(alane);
        let y = _mm256_loadu_si256(blane);
        _mm256_storeu_si256(dlane, multiply_vectors_gfni::<NATIVE>(x, y, iso));
    }
    if NATIVE {
        let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
        let (a16, a_tail) = a_rest.as_chunks::<16>();
        let (b16, b_tail) = b_rest.as_chunks::<16>();
        for ((d16, a16s), b16s) in dst16.iter_mut().zip(a16).zip(b16) {
            let x = _mm_loadu_si128(a16s);
            let y = _mm_loadu_si128(b16s);
            _mm_storeu_si128(d16, multiply_vectors_half_gfni::<NATIVE>(x, y, iso));
        }
        for ((d, &x), &y) in dst_tail.iter_mut().zip(a_tail).zip(b_tail) {
            // `NATIVE` means the encoding is the AES polynomial basis, so the
            // caller's reduction byte is 0x1b by construction.
            *d = super::super::ssse3::gf_mul_ref(x, y, reduction);
        }
    } else {
        // GFNI implies AVX2 and SSSE3; the remainder keeps the shift/reduce
        // seam at 128-bit lanes and scalar.
        mul_elementwise_ssse3(_token.v3().v2(), reduction, dst_rest, a_rest, b_rest);
    }
}

/// `dst[i] = dst[i] * src[i]`.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gfni<const NATIVE: bool>(
    _token: archmage::X64V3GfniCryptoToken,
    iso: Isomorphism,
    reduction: u8,
    dst: &mut [u8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_gfni: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dlane);
        let y = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, multiply_vectors_gfni::<NATIVE>(x, y, iso));
    }
    if NATIVE {
        let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
        let (src16, src_tail) = src_rest.as_chunks::<16>();
        for (d16, s16) in dst16.iter_mut().zip(src16) {
            let x = _mm_loadu_si128(&*d16);
            let y = _mm_loadu_si128(s16);
            _mm_storeu_si128(d16, multiply_vectors_half_gfni::<NATIVE>(x, y, iso));
        }
        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = super::super::ssse3::gf_mul_ref(*d, s, reduction);
        }
    } else {
        // As in `mul_elementwise_gfni`: the shift/reduce seam finishes the
        // 128-bit lanes and the scalar tail.
        mul_elementwise_assign_ssse3(_token.v3().v2(), reduction, dst_rest, src_rest);
    }
}
