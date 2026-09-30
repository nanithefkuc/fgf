//! `dst[i] = a[i] * b[i]` over interleaved GF(2^16) elements.
//!
//! Both operands vary, so no coefficient becomes a table first: the base
//! field multiplies directly with `GF2P8MULB`.

use super::super::{check_elements, swap_mask_avx2};
use crate::field::gf16::Elem;

/// `dst[i] = a[i] * b[i]` over interleaved tower elements using GFNI.
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
    check_elements("gf16::mul_elementwise_gfni", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_gfni: dst is {} bytes but a is {} bytes",
        dst.len(),
        a.len(),
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_gfni: dst is {} bytes but b is {} bytes",
        dst.len(),
        b.len(),
    );
    let swap = swap_mask_avx2();
    let even = _mm256_set1_epi16(0x00ff);
    let delta_even = _mm256_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for ((dst_lane, x_lane), y_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        // For x=[a,b], y=[c,d]:
        // constant = ac ^ DELTA*bd
        // extension = ad ^ bc ^ bd.
        let x = _mm256_loadu_si256(x_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = _mm256_gf2p8mul_epi8(x, y); // [ac, bd]
        let crossed = _mm256_gf2p8mul_epi8(x, _mm256_shuffle_epi8(y, swap)); // [ad, bc]
        let delta_bd = _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(direct, swap), delta_even);
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    for ((d, x), y) in dst_rest
        .chunks_exact_mut(2)
        .zip(a_rest.chunks_exact(2))
        .zip(b_rest.chunks_exact(2))
    {
        d.copy_from_slice(
            &Elem::from_bytes([x[0], x[1]])
                .mul(Elem::from_bytes([y[0], y[1]]))
                .to_bytes(),
        );
    }
}

/// `dst[i] = dst[i] * src[i]` over interleaved tower elements using GFNI.
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
    check_elements("gf16::mul_elementwise_assign_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask_avx2();
    let even = _mm256_set1_epi16(0x00ff);
    let delta_even = _mm256_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for (dst_lane, y_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dst_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = _mm256_gf2p8mul_epi8(x, y); // [ac, bd]
        let crossed = _mm256_gf2p8mul_epi8(x, _mm256_shuffle_epi8(y, swap)); // [ad, bc]
        let delta_bd = _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(direct, swap), delta_even);
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    for (d, y) in dst_rest.chunks_exact_mut(2).zip(src_rest.chunks_exact(2)) {
        let x = [d[0], d[1]];
        d.copy_from_slice(
            &Elem::from_bytes(x)
                .mul(Elem::from_bytes([y[0], y[1]]))
                .to_bytes(),
        );
    }
}
