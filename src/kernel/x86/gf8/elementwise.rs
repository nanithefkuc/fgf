//! Lane-wise products of two buffers: `dst[i] = a[i] * b[i]`.
//!
//! No coefficient and no table: both operands vary per lane, so the GFNI form
//! is a vector-by-vector `GF2P8MULB` and the shuffle forms carry the
//! Russian-peasant multiply in registers. The vector helpers are shared with
//! the GF(2^16) tower kernels, which multiply their halves in this field.
//!
//! The buffer entries are safe [`archmage`] capability-token functions over
//! reference-based intrinsics.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::field::gf8b::Elem;

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

/// Reference GF(2^8) product under reduction-polynomial low byte `RED`, for
/// the sub-lane elementwise tail. Field-agnostic: `RED` is `0x1b` for the AES
/// field and `0x1d` for the Reed–Solomon field.
#[inline]
fn gf_mul_ref<const RED: u8>(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= RED;
        }
        b >>= 1;
    }
    product
}

/// Lane-parallel GF(2^8) multiplication of two varying byte vectors.
///
/// Without `GF2P8MULB` there is no fixed coefficient to build a nibble table
/// from, so the product comes from eight branchless shift/reduce rounds — the
/// sequence already proven on NEON and wasm. x86 has no byte shift: `PADDB`
/// doubles each lane (a left shift that drops the carry) and the signed
/// `PCMPGTB` against zero recovers the bit it dropped.
#[archmage::rite(v3)]
pub fn multiply_vectors_avx2<const RED: u8>(mut a: __m256i, mut b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi8(1);
    let reduction = _mm256_set1_epi8(RED.cast_signed());
    let low7 = _mm256_set1_epi8(0x7f);
    let mut product = zero;
    for round in 0..8 {
        let active = _mm256_cmpeq_epi8(_mm256_and_si256(b, one), one);
        product = _mm256_xor_si256(product, _mm256_and_si256(a, active));
        if round == 7 {
            break;
        }
        let carry = _mm256_cmpgt_epi8(zero, a);
        a = _mm256_xor_si256(_mm256_add_epi8(a, a), _mm256_and_si256(reduction, carry));
        b = _mm256_and_si256(_mm256_srli_epi16::<1>(b), low7);
    }
    product
}

/// The 16-byte form of [`multiply_vectors_avx2`].
#[archmage::rite(v2)]
pub fn multiply_vectors_sse<const RED: u8>(mut a: __m128i, mut b: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let reduction = _mm_set1_epi8(RED.cast_signed());
    let low7 = _mm_set1_epi8(0x7f);
    let mut product = zero;
    for round in 0..8 {
        let active = _mm_cmpeq_epi8(_mm_and_si128(b, one), one);
        product = _mm_xor_si128(product, _mm_and_si128(a, active));
        if round == 7 {
            break;
        }
        let carry = _mm_cmpgt_epi8(zero, a);
        a = _mm_xor_si128(_mm_add_epi8(a, a), _mm_and_si128(reduction, carry));
        b = _mm_and_si128(_mm_srli_epi16::<1>(b), low7);
    }
    product
}

/// `dst[i] = a[i] * b[i]` by branchless shift/reduce over 32-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2<const RED: u8>(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_avx2: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_avx2: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(alane);
        let y = _mm256_loadu_si256(blane);
        _mm256_storeu_si256(dlane, multiply_vectors_avx2::<RED>(x, y));
    }

    // AVX2 implies SSSE3; the remainders keep equal lengths.
    mul_elementwise_ssse3::<RED>(_token.v2(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = a[i] * b[i]` by branchless shift/reduce over 16-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_ssse3<const RED: u8>(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_ssse3: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_ssse3: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    let (a_lanes, a_rest) = a.as_chunks::<16>();
    let (b_lanes, b_rest) = b.as_chunks::<16>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm_loadu_si128(alane);
        let y = _mm_loadu_si128(blane);
        _mm_storeu_si128(dlane, multiply_vectors_sse::<RED>(x, y));
    }
    for ((d, &x), &y) in dst_rest.iter_mut().zip(a_rest).zip(b_rest) {
        *d = gf_mul_ref::<RED>(x, y);
    }
}
