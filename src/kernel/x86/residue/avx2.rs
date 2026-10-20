//! AVX2 kernels for the generic residue rows.
//!
//! One submodule per fan shape: `single` holds the elementwise and
//! single-row fixed-coefficient entries, `matrix` the delayed-reduction
//! matrix entries. The vector helpers both shapes share live here with an
//! `_avx2` suffix.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

mod matrix;
mod single;

pub use matrix::{mul_add_matrix_u32_avx2_with, mul_into_matrix_u32_avx2_with};
pub use single::{
    mul_add_u8_avx2, mul_add_u32_avx2, mul_add_u64_avx2, mul_assign_u8_avx2, mul_assign_u32_avx2,
    mul_assign_u64_avx2, mul_elementwise_assign_u32_avx2, mul_elementwise_u32_avx2,
    mul_into_u8_avx2, mul_into_u32_avx2, mul_into_u64_avx2,
};

/// One conditional subtract of `p` over four packed 64-bit lanes.
#[archmage::rite(v3)]
fn subtract_p(r: __m256i, p: __m256i) -> __m256i {
    let ge = _mm256_cmpgt_epi64(r, _mm256_sub_epi64(p, _mm256_set1_epi64x(1)));
    _mm256_sub_epi64(r, _mm256_and_si256(ge, p))
}

/// Reduce a 64-bit product below `2^32 * P` modulo `p` through the narrow
/// Barrett form.
///
/// With one factor canonical the product is below `2^32 * P`; the floor
/// quotient estimate undershoots by at most two, so two conditional
/// subtractions canonicalize. `floor(t * mu / 2^32)` is assembled from two
/// exact 32x32 products without losing a carry: the low half of the first
/// product contributes only its high 32 bits, the second product is already
/// a high half, and their sum holds the full intermediate without
/// overflow.
#[archmage::rite(v3)]
pub(super) fn narrow_avx2(t: __m256i, mu: __m256i, p: __m256i, shift: i32) -> __m256i {
    let mid = _mm256_add_epi64(
        _mm256_srli_epi64::<32>(_mm256_mul_epu32(t, mu)),
        _mm256_mul_epu32(_mm256_srli_epi64::<32>(t), mu),
    );
    let q = _mm256_srl_epi64(mid, _mm_cvtsi32_si128(shift));
    let r = _mm256_sub_epi64(t, _mm256_mul_epu32(q, p));
    subtract_p(subtract_p(r, p), p)
}

/// Canonicalize raw 32-bit lanes: the Barrett quotient estimate leaves a
/// value below `2P` and one unsigned-min correction canonicalizes it,
/// including for `P > 2^31`, where the wrapping subtraction still selects
/// the canonical representative through the unsigned min.
#[archmage::rite(v3)]
pub(super) fn normalize_avx2(x: __m256i, norm_mu: __m256i, p32: __m256i) -> __m256i {
    let e = _mm256_srli_epi64::<32>(_mm256_mul_epu32(x, norm_mu));
    let o = _mm256_srli_epi64::<32>(_mm256_mul_epu32(_mm256_srli_epi64::<32>(x), norm_mu));
    let q = _mm256_or_si256(e, _mm256_slli_epi64::<32>(o));
    let r = _mm256_sub_epi32(x, _mm256_mullo_epi32(q, p32));
    _mm256_min_epu32(r, _mm256_sub_epi32(r, p32))
}

/// Interleave four `u32` results per 128-bit half from the even and odd
/// widened lane sets into element order.
#[archmage::rite(v3)]
#[must_use]
pub(super) fn combine_halves_avx2(ev: __m256i, od: __m256i) -> __m256i {
    _mm256_unpacklo_epi64(_mm256_unpacklo_epi32(ev, od), _mm256_unpackhi_epi32(ev, od))
}

/// Widen two canonical 32-bit lane sets to 64-bit lanes and add them
/// modulo `p`: each pairwise sum sits below `2P < 2^33`, so one unsigned
/// 64-bit comparison canonicalizes it. The result is eight canonical
/// 32-bit lanes in element order.
#[archmage::rite(v3)]
pub(super) fn widen_add_avx2(a: __m256i, b: __m256i, p64: __m256i, p_m1: __m256i) -> __m256i {
    let ae = _mm256_mul_epu32(a, _mm256_set1_epi64x(1));
    let ao = _mm256_mul_epu32(_mm256_shuffle_epi32(a, 0xB1), _mm256_set1_epi64x(1));
    let be = _mm256_mul_epu32(b, _mm256_set1_epi64x(1));
    let bo = _mm256_mul_epu32(_mm256_shuffle_epi32(b, 0xB1), _mm256_set1_epi64x(1));
    let se = _mm256_add_epi64(ae, be);
    let so = _mm256_add_epi64(ao, bo);
    let se = _mm256_sub_epi64(se, _mm256_and_si256(_mm256_cmpgt_epi64(se, p_m1), p64));
    let so = _mm256_sub_epi64(so, _mm256_and_si256(_mm256_cmpgt_epi64(so, p_m1), p64));
    combine_halves_avx2(se, so)
}
