//! `QuadMersenne31` kernels over 512-bit lanes (`V4x`).
//!
//! Element-for-element ports of the `avx2` multiply bodies at eight complex
//! elements per vector: the same deferred 64-bit complex products, the same
//! split against `2^31 ≡ 1 (mod p)`, and the same direct min-chain
//! canonicalization, so every element produces the bytes the AVX2 kernel
//! produces. Only AVX-512F instructions are used, so each entry takes
//! `X64V4Token`. Complete 64-byte lanes run here; the remainder is handed
//! to the matching AVX2 entry.
//!
//! The entries screen limbs for bit 31 instead of canonicalizing every
//! load: a limb with bit 31 clear lies in `0..=p`, which [`cmul`],
//! [`cmul_vec`], and [`addmod`] accept as loaded. The out-of-place entries
//! multiply optimistically and recompute every element through
//! [`min_chain`] when any source limb carries bit 31; the accumulating
//! entry loads [`SCREEN_LANES`] lanes of each operand into registers and
//! canonicalizes them only when one carries it, before any store.

use crate::field::Elem;
use crate::field::mersenne31;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::x86::mersenne31::P;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Odd 32-bit lanes of a 512-bit vector, as a blend mask.
const ODD_LANES: __mmask16 = 0xAAAA;

/// Lanes per register group of the accumulating multiply: every operand
/// lane of a group is loaded and screened before the group's first store.
const SCREEN_LANES: usize = 4;

/// Whether any 32-bit limb of `seen` carries bit 31, the one bit outside
/// the `0..=p` limb range of [`cmul`] and [`cmul_vec`].
#[archmage::rite(v4)]
fn any_bit31(seen: __m512i) -> bool {
    _mm512_test_epi32_mask(seen, _mm512_set1_epi32(i32::MIN)) != 0
}

/// Duplicate each element's imaginary limb into both lanes of its pair.
#[archmage::rite(v4)]
#[must_use]
fn dup_im(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)))
}

/// Canonicalize arbitrary raw `u32` lanes below `p` in two unconditional
/// subtract-and-min steps; every lane value fits `0..=2p + 1`
/// (`u32::MAX == 2p + 1`).
#[archmage::rite(v4)]
#[must_use]
fn min_chain(x: __m512i, p: __m512i) -> __m512i {
    let u = _mm512_min_epu32(x, _mm512_sub_epi32(x, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// `a + b (mod p)` over sixteen lanes (`0..=p` in, canonical out).
#[archmage::rite(v4)]
#[must_use]
fn addmod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let t = _mm512_add_epi32(a, b);
    let u = _mm512_min_epu32(t, _mm512_sub_epi32(t, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// Reduce deferred 64-bit product accumulators into packed canonical limb
/// pairs: the even dwords carry the real part and the odd dwords the
/// imaginary part, each split against `2^31 ≡ 1 (mod p)` and canonicalized
/// with the min chain.
#[archmage::rite(v4)]
#[must_use]
fn reduce_pair(re: __m512i, im: __m512i, p: __m512i) -> __m512i {
    let hi = _mm512_mask_blend_epi32(
        ODD_LANES,
        _mm512_srli_epi64::<31>(re),
        _mm512_slli_epi64::<1>(im),
    );
    let hi = _mm512_min_epu32(hi, _mm512_sub_epi32(hi, p));
    let lo = _mm512_and_si512(
        _mm512_mask_blend_epi32(ODD_LANES, re, _mm512_slli_epi64::<32>(im)),
        p,
    );
    let t = _mm512_add_epi32(lo, hi);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// Multiply eight complex elements with limbs in `0..=p` by prepared
/// coefficient broadcasts: `cr` and `ci` as-is, `nci` holding `p - ci`.
/// The real lanes accumulate `re*cr + im*(p-ci) ≡ re*cr - im*ci` and the
/// imaginary lanes `im*cr + re*ci`, both deferred at 64 bits for
/// [`reduce_pair`].
#[archmage::rite(v4)]
#[must_use]
fn cmul(x: __m512i, cr: __m512i, ci: __m512i, nci: __m512i, p: __m512i) -> __m512i {
    let xi = dup_im(x);
    let re = _mm512_add_epi64(_mm512_mul_epu32(x, cr), _mm512_mul_epu32(xi, nci));
    let im = _mm512_add_epi64(_mm512_mul_epu32(xi, cr), _mm512_mul_epu32(x, ci));
    reduce_pair(re, im, p)
}

/// Complex multiply of two vectors with limbs in `0..=p`,
/// `(ar*br - ai*bi, ar*bi + ai*br)` per element, negating the duplicated
/// imaginary limbs of `b` by XOR with `p` (exact on `0..=p`) and reducing
/// the deferred 64-bit accumulators.
#[archmage::rite(v4)]
#[must_use]
fn cmul_vec(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let ai = dup_im(a);
    let bi = dup_im(b);
    let neg_bi = _mm512_xor_si512(bi, p);
    let re = _mm512_add_epi64(_mm512_mul_epu32(a, b), _mm512_mul_epu32(ai, neg_bi));
    let im = _mm512_add_epi64(_mm512_mul_epu32(a, bi), _mm512_mul_epu32(ai, b));
    reduce_pair(re, im, p)
}

/// The coefficient broadcasts every coefficient multiply entry builds:
/// `cr` and `ci` as-is, plus `p - ci` for the negated imaginary product.
/// Both limbs enter canonical.
#[archmage::rite(v4)]
fn coeff_vectors(cr: u32, ci: u32) -> (__m512i, __m512i, __m512i) {
    let nci = mersenne31::MODULUS - ci;
    (
        _mm512_set1_epi32(cr.cast_signed()),
        _mm512_set1_epi32(ci.cast_signed()),
        _mm512_set1_epi32(nci.cast_signed()),
    )
}

/// `dst = coeff * src` in GF((2^31 - 1)^2), AVX-512. The destination is
/// write-only: its prior contents are ignored, never read.
///
/// Every input limb bit pattern is legal: the first pass multiplies the
/// limbs as loaded, a second pass recomputes every element from
/// canonicalized loads when any source limb has bit 31 set, and every
/// stored limb is canonical. A zero coefficient zeroes `dst`, mirroring
/// the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx512(
    token: archmage::X64V4Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_into_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_into_avx512", dst.len(), 8);
    let [cr, ci] = coeff.to_raw();
    let p = _mm512_set1_epi32(P);
    let (cr_v, ci_v, nci) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    let mut seen = _mm512_setzero_si512();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm512_loadu_si512(src_lane);
        seen = _mm512_or_si512(seen, s);
        _mm512_storeu_si512(dst_lane, cmul(s, cr_v, ci_v, nci, p));
    }
    // The destination is write-only and never aliases the source, so the
    // recomputing pass reads the same source lanes.
    if any_bit31(seen) {
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let s = min_chain(_mm512_loadu_si512(src_lane), p);
            _mm512_storeu_si512(dst_lane, cmul(s, cr_v, ci_v, nci, p));
        }
    }
    if !dst_tail.is_empty() {
        super::mul_into_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst += coeff * src` in GF((2^31 - 1)^2), AVX-512.
///
/// Every input limb bit pattern is legal: a register group whose limbs all
/// have bit 31 clear multiplies as loaded, any other group and the lanes
/// past the last whole group canonicalize first, and every stored limb is
/// canonical. A zero coefficient leaves `dst` untouched and the coefficient
/// `1 + 0i` adds `src` elementwise, mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512(
    token: archmage::X64V4Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_add_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_add_avx512", dst.len(), 8);
    let [cr, ci] = coeff.to_raw();
    if cr == 0 && ci == 0 {
        return;
    }
    if cr == 1 && ci == 0 {
        super::add_assign_avx2(token.v3(), dst, src);
        return;
    }
    let p = _mm512_set1_epi32(P);
    let (cr_v, ci_v, nci) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    let (dst_groups, dst_rest) = dst_lanes.as_chunks_mut::<SCREEN_LANES>();
    let (src_groups, src_rest) = src_lanes.as_chunks::<SCREEN_LANES>();
    for (dst_group, src_group) in dst_groups.iter_mut().zip(src_groups) {
        let mut d = [_mm512_setzero_si512(); SCREEN_LANES];
        let mut s = [_mm512_setzero_si512(); SCREEN_LANES];
        let mut seen = _mm512_setzero_si512();
        for (((dv, sv), dst_lane), src_lane) in
            d.iter_mut().zip(&mut s).zip(&*dst_group).zip(src_group)
        {
            *dv = _mm512_loadu_si512(dst_lane);
            *sv = _mm512_loadu_si512(src_lane);
            seen = _mm512_or_si512(seen, _mm512_or_si512(*dv, *sv));
        }
        if any_bit31(seen) {
            for v in &mut d {
                *v = min_chain(*v, p);
            }
            for v in &mut s {
                *v = min_chain(*v, p);
            }
        }
        for ((dst_lane, dv), sv) in dst_group.iter_mut().zip(&d).zip(&s) {
            _mm512_storeu_si512(dst_lane, addmod(*dv, cmul(*sv, cr_v, ci_v, nci, p), p));
        }
    }
    for (dst_lane, src_lane) in dst_rest.iter_mut().zip(src_rest) {
        let s = min_chain(_mm512_loadu_si512(src_lane), p);
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        _mm512_storeu_si512(dst_lane, addmod(d, cmul(s, cr_v, ci_v, nci, p), p));
    }
    if !dst_tail.is_empty() {
        super::mul_add_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i]` in GF((2^31 - 1)^2), AVX-512.
///
/// Every input limb bit pattern is legal: the first pass multiplies the
/// limbs as loaded, a second pass recomputes every element from
/// canonicalized loads when any source limb has bit 31 set, and every
/// stored limb is canonical.
///
/// # Panics
/// Panics unless all three buffers match in length (whole elements).
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx512(token: archmage::X64V4Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "quad_mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_elementwise_avx512", dst.len(), 8);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a.as_chunks::<64>();
    let (b_lanes, b_tail) = b.as_chunks::<64>();
    let mut seen = _mm512_setzero_si512();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = _mm512_loadu_si512(a_lane);
        let vb = _mm512_loadu_si512(b_lane);
        seen = _mm512_or_si512(seen, _mm512_or_si512(va, vb));
        _mm512_storeu_si512(dst_lane, cmul_vec(va, vb, p));
    }
    // The destination is write-only and never aliases a source, so the
    // recomputing pass reads the same source lanes.
    if any_bit31(seen) {
        for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
            let va = min_chain(_mm512_loadu_si512(a_lane), p);
            let vb = min_chain(_mm512_loadu_si512(b_lane), p);
            _mm512_storeu_si512(dst_lane, cmul_vec(va, vb, p));
        }
    }
    if !dst_tail.is_empty() {
        super::mul_elementwise_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}
