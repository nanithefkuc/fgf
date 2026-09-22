//! `dst[i] = a[i] * b[i]` over interleaved GF(2^16) elements.
//!
//! The one shape whose operands both vary, so no coefficient can be turned
//! into a table first. GFNI multiplies the base field directly; AVX2 and
//! SSSE3 borrow the shift/reduce vector multiply from the GF(2^8) kernels
//! and keep a nibble shuffle only for the constant `DELTA` factor.

use crate::field::gf16::Elem;
use crate::kernel::tables::scale_table;

use super::{swap_mask128, swap_mask256};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
    super::check_elements("gf16::mul_elementwise_gfni", dst.len());
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
    let swap = swap_mask256();
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

/// `DELTA * v` for every byte lane, by split-nibble `PSHUFB`.
///
/// The tower product needs one multiply by the *constant* `DELTA`, which —
/// unlike the two varying-operand products around it — does have a nibble
/// table. Two shuffles beat eight shift/reduce rounds.
#[archmage::rite(v3)]
fn scale_delta_avx2(v: __m256i, lo: __m256i, hi: __m256i, nibble: __m256i) -> __m256i {
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo, _mm256_and_si256(v, nibble)),
        _mm256_shuffle_epi8(hi, _mm256_and_si256(_mm256_srli_epi16::<4>(v), nibble)),
    )
}

/// The 16-byte form of [`scale_delta_avx2`].
#[archmage::rite(v2)]
fn scale_delta_sse(v: __m128i, lo: __m128i, hi: __m128i, nibble: __m128i) -> __m128i {
    _mm_xor_si128(
        _mm_shuffle_epi8(lo, _mm_and_si128(v, nibble)),
        _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16::<4>(v), nibble)),
    )
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements, AVX2.
///
/// Three base products, as everywhere else: `direct = [ac, bd]`,
/// `crossed = [ad, bc]`, and `DELTA*bd`. The first two have two varying
/// operands and use the shift/reduce vector multiply from
/// [`super::gf8`]; the third is a constant multiply and stays a nibble
/// shuffle.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2(token: archmage::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    super::check_elements("gf16::mul_elementwise_avx2", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_avx2: dst is {} bytes but a is {} bytes",
        dst.len(),
        a.len(),
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_avx2: dst is {} bytes but b is {} bytes",
        dst.len(),
        b.len(),
    );
    let swap = swap_mask256();
    let even = _mm256_set1_epi16(0x00ff);
    let nibble = _mm256_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.lo)),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.hi)),
    );
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for ((dst_lane, x_lane), y_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(x_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = super::gf8::multiply_vectors_avx2::<0x1b>(x, y);
        let crossed = super::gf8::multiply_vectors_avx2::<0x1b>(x, _mm256_shuffle_epi8(y, swap));
        let delta_bd = scale_delta_avx2(
            _mm256_shuffle_epi8(direct, swap),
            delta_lo,
            delta_hi,
            nibble,
        );
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    // AVX2 implies SSSE3, and the remainders keep equal lengths.
    mul_elementwise_ssse3(token.v2(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements, SSSE3.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    super::check_elements("gf16::mul_elementwise_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_ssse3: dst is {} bytes but a is {} bytes",
        dst.len(),
        a.len(),
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_ssse3: dst is {} bytes but b is {} bytes",
        dst.len(),
        b.len(),
    );
    let swap = swap_mask128();
    let even = _mm_set1_epi16(0x00ff);
    let nibble = _mm_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (_mm_loadu_si128(&delta.lo), _mm_loadu_si128(&delta.hi));
    let (a_lanes, a_rest) = a.as_chunks::<16>();
    let (b_lanes, b_rest) = b.as_chunks::<16>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    for ((dst_lane, x_lane), y_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm_loadu_si128(x_lane);
        let y = _mm_loadu_si128(y_lane);
        let direct = super::gf8::multiply_vectors_sse::<0x1b>(x, y);
        let crossed = super::gf8::multiply_vectors_sse::<0x1b>(x, _mm_shuffle_epi8(y, swap));
        let delta_bd = scale_delta_sse(_mm_shuffle_epi8(direct, swap), delta_lo, delta_hi, nibble);
        let constant = _mm_xor_si128(direct, delta_bd);
        let cross_sum = _mm_xor_si128(crossed, _mm_shuffle_epi8(crossed, swap));
        let extension = _mm_xor_si128(cross_sum, direct);
        let product = _mm_xor_si128(
            _mm_and_si128(constant, even),
            _mm_andnot_si128(even, extension),
        );
        _mm_storeu_si128(dst_lane, product);
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
    super::check_elements("gf16::mul_elementwise_assign_gfni", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_gfni: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask256();
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

/// `dst[i] = dst[i] * src[i]` over interleaved tower elements, AVX2.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2(token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    super::check_elements("gf16::mul_elementwise_assign_avx2", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask256();
    let even = _mm256_set1_epi16(0x00ff);
    let nibble = _mm256_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.lo)),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.hi)),
    );
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for (dst_lane, y_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dst_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = super::gf8::multiply_vectors_avx2::<0x1b>(x, y);
        let crossed = super::gf8::multiply_vectors_avx2::<0x1b>(x, _mm256_shuffle_epi8(y, swap));
        let delta_bd = scale_delta_avx2(
            _mm256_shuffle_epi8(direct, swap),
            delta_lo,
            delta_hi,
            nibble,
        );
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    // AVX2 implies SSSE3, and the remainders keep equal lengths.
    mul_elementwise_assign_ssse3(token.v2(), dst_rest, src_rest);
}

/// `dst[i] = dst[i] * src[i]` over interleaved tower elements, SSSE3.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    super::check_elements("gf16::mul_elementwise_assign_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask128();
    let even = _mm_set1_epi16(0x00ff);
    let nibble = _mm_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (_mm_loadu_si128(&delta.lo), _mm_loadu_si128(&delta.hi));
    let (src_lanes, src_rest) = src.as_chunks::<16>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    for (dst_lane, y_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(&*dst_lane);
        let y = _mm_loadu_si128(y_lane);
        let direct = super::gf8::multiply_vectors_sse::<0x1b>(x, y);
        let crossed = super::gf8::multiply_vectors_sse::<0x1b>(x, _mm_shuffle_epi8(y, swap));
        let delta_bd = scale_delta_sse(_mm_shuffle_epi8(direct, swap), delta_lo, delta_hi, nibble);
        let constant = _mm_xor_si128(direct, delta_bd);
        let cross_sum = _mm_xor_si128(crossed, _mm_shuffle_epi8(crossed, swap));
        let extension = _mm_xor_si128(cross_sum, direct);
        let product = _mm_xor_si128(
            _mm_and_si128(constant, even),
            _mm_andnot_si128(even, extension),
        );
        _mm_storeu_si128(dst_lane, product);
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
