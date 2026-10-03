//! Lane-wise products of two buffers: `dst[i] = a[i] * b[i]`.
//!
//! No coefficient and no table: both operands vary per lane. Under the AES
//! polynomial the product is one vector-by-vector `GF2P8MULB`; under every
//! other polynomial it conjugates that multiply by the field isomorphism onto
//! `0x11B`, `a · b = φ⁻¹(GF2P8MULB(φa, φb))` — two affine maps in, one
//! native multiply, one affine map out.
//!
//! The buffer entries are safe [`archmage`] capability-token functions over
//! reference-based intrinsics.

use super::super::ssse3::{mul_elementwise_assign_ssse3, mul_elementwise_ssse3};
use crate::field::poly::AES;
use crate::field::{Elem, Poly};
use crate::kernel::tables::isomorphism_to_aes;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `x * y` under `POLY` over a 256-bit lane.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(in crate::kernel::x86::gf8) fn multiply_vectors_gfni<const POLY: u32>(
    x: __m256i,
    y: __m256i,
) -> __m256i {
    if POLY == AES {
        return _mm256_gf2p8mul_epi8(x, y);
    }
    let iso = isomorphism_to_aes::<POLY>();
    let forward = _mm256_set1_epi64x(iso.forward.cast_signed());
    let inverse = _mm256_set1_epi64x(iso.inverse.cast_signed());
    let px = _mm256_gf2p8affine_epi64_epi8::<0>(x, forward);
    let py = _mm256_gf2p8affine_epi64_epi8::<0>(y, forward);
    _mm256_gf2p8affine_epi64_epi8::<0>(_mm256_gf2p8mul_epi8(px, py), inverse)
}

/// `x * y` under `POLY` over a 128-bit lane.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
fn multiply_vectors_half_gfni<const POLY: u32>(x: __m128i, y: __m128i) -> __m128i {
    if POLY == AES {
        return _mm_gf2p8mul_epi8(x, y);
    }
    let iso = isomorphism_to_aes::<POLY>();
    let forward = _mm_set1_epi64x(iso.forward.cast_signed());
    let inverse = _mm_set1_epi64x(iso.inverse.cast_signed());
    let px = _mm_gf2p8affine_epi64_epi8::<0>(x, forward);
    let py = _mm_gf2p8affine_epi64_epi8::<0>(y, forward);
    _mm_gf2p8affine_epi64_epi8::<0>(_mm_gf2p8mul_epi8(px, py), inverse)
}

/// `dst[i] = a[i] * b[i]` under `POLY`.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gfni<const POLY: u32>(
    _token: archmage::X64V3GfniCryptoToken,
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
        _mm256_storeu_si256(dlane, multiply_vectors_gfni::<POLY>(x, y));
    }
    if POLY == AES {
        let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
        let (a16, a_tail) = a_rest.as_chunks::<16>();
        let (b16, b_tail) = b_rest.as_chunks::<16>();
        for ((d16, a16s), b16s) in dst16.iter_mut().zip(a16).zip(b16) {
            let x = _mm_loadu_si128(a16s);
            let y = _mm_loadu_si128(b16s);
            _mm_storeu_si128(d16, multiply_vectors_half_gfni::<POLY>(x, y));
        }
        for ((d, &x), &y) in dst_tail.iter_mut().zip(a_tail).zip(b_tail) {
            *d = Elem::<8, Poly<POLY>>::from_raw(x)
                .mul(Elem::<8, Poly<POLY>>::from_raw(y))
                .0;
        }
    } else {
        // GFNI implies AVX2 and SSSE3; the remainder keeps the shift/reduce
        // seam at 128-bit lanes and scalar.
        mul_elementwise_ssse3::<POLY>(_token.v3().v2(), dst_rest, a_rest, b_rest);
    }
}

/// `dst[i] = dst[i] * src[i]` under `POLY`.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gfni<const POLY: u32>(
    _token: archmage::X64V3GfniCryptoToken,
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
        _mm256_storeu_si256(dlane, multiply_vectors_gfni::<POLY>(x, y));
    }
    if POLY == AES {
        let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
        let (src16, src_tail) = src_rest.as_chunks::<16>();
        for (d16, s16) in dst16.iter_mut().zip(src16) {
            let x = _mm_loadu_si128(&*d16);
            let y = _mm_loadu_si128(s16);
            _mm_storeu_si128(d16, multiply_vectors_half_gfni::<POLY>(x, y));
        }
        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = Elem::<8, Poly<POLY>>::from_raw(*d)
                .mul(Elem::<8, Poly<POLY>>::from_raw(s))
                .0;
        }
    } else {
        // As in `mul_elementwise_gfni`: the shift/reduce seam finishes the
        // 128-bit lanes and the scalar tail.
        mul_elementwise_assign_ssse3::<POLY>(_token.v3().v2(), dst_rest, src_rest);
    }
}
