//! AVX-512 kernels for the generic residue rows.
//!
//! One submodule per fan shape: `single` holds the elementwise and
//! single-row fixed-coefficient entries, `matrix` the delayed-reduction
//! matrix entries. The vector helpers both shapes share live here with an
//! `_avx512` suffix.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub(crate) mod matrix;
pub(crate) mod single;

/// One conditional subtract of `p` over eight packed 64-bit lanes.
#[archmage::rite(v4)]
fn subtract_p(r: __m512i, p: __m512i) -> __m512i {
    let ge = _mm512_cmp_epu64_mask::<{ _MM_CMPINT_NLT }>(r, p);
    _mm512_mask_sub_epi64(r, ge, r, p)
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
#[archmage::rite(v4)]
pub(super) fn narrow_avx512(t: __m512i, mu: __m512i, p: __m512i, shift: i32) -> __m512i {
    let mid = _mm512_add_epi64(
        _mm512_srli_epi64::<32>(_mm512_mul_epu32(t, mu)),
        _mm512_mul_epu32(_mm512_srli_epi64::<32>(t), mu),
    );
    let q = _mm512_srl_epi64(mid, _mm_cvtsi32_si128(shift));
    let r = _mm512_sub_epi64(t, _mm512_mul_epu32(q, p));
    subtract_p(subtract_p(r, p), p)
}

/// Canonicalize raw 32-bit lanes: the Barrett quotient estimate leaves a
/// value below `2P` and one unsigned-min correction canonicalizes it,
/// including for `P > 2^31`, where the wrapping subtraction still selects
/// the canonical representative through the unsigned min.
#[archmage::rite(v4)]
pub(super) fn normalize_avx512(x: __m512i, norm_mu: __m512i, p32: __m512i) -> __m512i {
    let e = _mm512_srli_epi64::<32>(_mm512_mul_epu32(x, norm_mu));
    let o = _mm512_srli_epi64::<32>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(x), norm_mu));
    let q = _mm512_or_si512(e, _mm512_slli_epi64::<32>(o));
    let r = _mm512_sub_epi32(x, _mm512_mullo_epi32(q, p32));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, p32))
}

/// Interleave eight `u32` results per lane set from the even and odd
/// widened lane sets into element order.
#[archmage::rite(v4)]
#[must_use]
pub(super) fn combine_halves_avx512(ev: __m512i, od: __m512i) -> __m512i {
    _mm512_or_si512(ev, _mm512_slli_epi64::<32>(od))
}

/// Widen two canonical 32-bit lane sets to 64-bit lanes and add them
/// modulo `p`: each pairwise sum sits below `2P < 2^33`, so one unsigned
/// 64-bit comparison canonicalizes it. The result is sixteen canonical
/// 32-bit lanes in element order.
#[archmage::rite(v4)]
pub(super) fn widen_add_avx512(a: __m512i, b: __m512i, p64: __m512i, p_m1: __m512i) -> __m512i {
    let ae = _mm512_mul_epu32(a, _mm512_set1_epi64(1));
    let ao = _mm512_mul_epu32(_mm512_shuffle_epi32::<0xB1>(a), _mm512_set1_epi64(1));
    let be = _mm512_mul_epu32(b, _mm512_set1_epi64(1));
    let bo = _mm512_mul_epu32(_mm512_shuffle_epi32::<0xB1>(b), _mm512_set1_epi64(1));
    let se = _mm512_add_epi64(ae, be);
    let so = _mm512_add_epi64(ao, bo);
    let se = _mm512_mask_sub_epi64(
        se,
        _mm512_cmp_epu64_mask::<{ _MM_CMPINT_NLE }>(se, p_m1),
        se,
        p64,
    );
    let so = _mm512_mask_sub_epi64(
        so,
        _mm512_cmp_epu64_mask::<{ _MM_CMPINT_NLE }>(so, p_m1),
        so,
        p64,
    );
    combine_halves_avx512(se, so)
}
