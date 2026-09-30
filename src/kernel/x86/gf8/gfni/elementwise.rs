//! Lane-wise products of two buffers: `dst[i] = a[i] * b[i]`.
//!
//! No coefficient and no table: both operands vary per lane, so the `Gf8B`
//! form is one vector-by-vector `GF2P8MULB` and the `Gf8D` form conjugates
//! it by the field isomorphism onto `0x11B`.
//!
//! The buffer entries are safe [`archmage`] capability-token functions over
//! reference-based intrinsics.

use super::super::ssse3::{mul_elementwise_assign_ssse3, mul_elementwise_ssse3};
use crate::field::gf8b::Elem;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `dst[i] = a[i] * b[i]` using vector-by-vector `GF2P8MULB`.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gfni(
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
        _mm256_storeu_si256(dlane, _mm256_gf2p8mul_epi8(x, y));
    }
    let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (a16, a_tail) = a_rest.as_chunks::<16>();
    let (b16, b_tail) = b_rest.as_chunks::<16>();
    for ((d16, a16s), b16s) in dst16.iter_mut().zip(a16).zip(b16) {
        let x = _mm_loadu_si128(a16s);
        let y = _mm_loadu_si128(b16s);
        _mm_storeu_si128(d16, _mm_gf2p8mul_epi8(x, y));
    }
    for ((d, &x), &y) in dst_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}

/// `dst[i] = dst[i] * src[i]` using vector-by-vector `GF2P8MULB`.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gfni(
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
        _mm256_storeu_si256(dlane, _mm256_gf2p8mul_epi8(x, y));
    }
    let (dst16, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src16, src_tail) = src_rest.as_chunks::<16>();
    for (d16, s16) in dst16.iter_mut().zip(src16) {
        let x = _mm_loadu_si128(&*d16);
        let y = _mm_loadu_si128(s16);
        _mm_storeu_si128(d16, _mm_gf2p8mul_epi8(x, y));
    }
    for (d, s) in dst_tail.iter_mut().zip(src_tail) {
        *d = Elem(*d).mul(Elem(*s)).0;
    }
}

/// Lane-parallel GF(2^8)/`0x11D` multiplication via the AES field.
///
/// `φ` (`ISO_11D_TO_11B`) is a GF(2)-linear isomorphism onto GF(2^8)/`0x11B`,
/// so `a · b = φ⁻¹(GF2P8MULB(φa, φb))`: two affine maps around one native
/// multiply. Bit-identical to [`gf_mul_ref`] under `0x1d`; the map is an
/// involution, so one constant serves both directions.
/// `VGF2P8AFFINEQB` computes `dst.bit[b] = parity(map.byte[7 - b] & x)`,
/// so the isomorphism rows land row-reversed in the qword (the same
/// transpose the [`affine_8d`](crate::kernel::tables::affine_8d) bank uses).
pub(in crate::kernel::x86::gf8) const ISO_11D_TO_11B: u64 = 0xffaa_cc88_f0a0_c080;

/// `dst[i] = a[i] * b[i]` under `0x11D` through the AES-field isomorphism.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
fn multiply_vectors_8d(x: __m256i, y: __m256i) -> __m256i {
    use core::arch::x86_64::*;
    let phi = _mm256_set1_epi64x(ISO_11D_TO_11B.cast_signed());
    let px = _mm256_gf2p8affine_epi64_epi8::<0>(x, phi);
    let py = _mm256_gf2p8affine_epi64_epi8::<0>(y, phi);
    let p = _mm256_gf2p8mul_epi8(px, py);
    _mm256_gf2p8affine_epi64_epi8::<0>(p, phi)
}

/// `dst[i] = a[i] * b[i]` under `0x11D` through the AES-field isomorphism.
///
/// Four GFNI instructions per lane instead of eight shift/reduce rounds.
/// Requires the GFNI token: the affine maps and the native multiply are all
/// GFNI instructions.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gfni_8d(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_gfni_8d: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_gfni_8d: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(alane);
        let y = _mm256_loadu_si256(blane);
        _mm256_storeu_si256(dlane, multiply_vectors_8d(x, y));
    }

    // GFNI implies AVX2 and SSSE3; the remainder keeps the shift/reduce seam
    // (bit-identical under `0x1d`) at 128-bit lanes and scalar.
    mul_elementwise_ssse3::<0x1d>(_token.v3().v2(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = dst[i] * src[i]` under `0x11D` through the AES-field
/// isomorphism.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gfni_8d(
    _token: archmage::X64V3GfniCryptoToken,
    dst: &mut [u8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_gfni_8d: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dlane);
        let y = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, multiply_vectors_8d(x, y));
    }

    // As in `mul_elementwise_gfni_8d`: the shift/reduce seam finishes the
    // 128-bit lanes and the scalar tail.
    mul_elementwise_assign_ssse3::<0x1d>(_token.v3().v2(), dst_rest, src_rest);
}
