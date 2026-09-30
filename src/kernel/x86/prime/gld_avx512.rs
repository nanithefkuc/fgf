//! Goldilocks kernels over 512-bit lanes (`V4x`).
//!
//! Eight 64-bit lanes per vector. The 64×64→128 product is the same four
//! `vpmuludq` half-products as `gld_avx2`, and the split-fold reduction uses
//! the same identities (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`). AVX-512F supplies
//! unsigned 64-bit compares into mask registers, so wrap and range
//! corrections are masked adds and subtracts instead of the AVX2 kernels'
//! sign-shifted domain. Every output lane is the canonical residue, so each
//! lane stores the bytes the AVX2 kernel stores. Only AVX-512F instructions
//! are used, so each entry takes `X64V4Token`. Complete 64-byte lanes run
//! here; the remainder descends through the matching AVX2 entry.

use super::{
    EPS, GLD_P, add_assign_gld_avx2, add_assign_scalar_gld_avx2, check_elem_multiple, check_equal,
    mul_add_gld_avx2, mul_assign_gld_avx2, mul_elementwise_assign_gld_avx2,
    mul_elementwise_gld_avx2, mul_into_gld_avx2, sub_assign_gld_avx2, sub_assign_scalar_gld_avx2,
};
use crate::field::goldilocks;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Odd 32-bit lanes of a 512-bit vector, as a blend mask.
const ODD_LANES: __mmask16 = 0xAAAA;

/// The lane constants every entry needs.
#[derive(Clone, Copy)]
struct Consts {
    /// The modulus in every lane.
    p: __m512i,
    /// `2^32 − 1 ≡ 2^64` in every lane.
    eps: __m512i,
}

#[archmage::rite(v4)]
fn consts() -> Consts {
    Consts {
        p: _mm512_set1_epi64(GLD_P),
        eps: _mm512_set1_epi64(EPS),
    }
}

/// `x mod p` for any 64-bit word: `2^64 < 2p`, so one subtraction suffices.
#[archmage::rite(v4)]
fn canon512(x: __m512i, c: Consts) -> __m512i {
    let ge = _mm512_cmpge_epu64_mask(x, c.p);
    _mm512_mask_sub_epi64(x, ge, x, c.p)
}

/// `(a + b) mod p` for canonical `a` and `b`.
#[archmage::rite(v4)]
fn addmod512(a: __m512i, b: __m512i, c: Consts) -> __m512i {
    let s = _mm512_add_epi64(a, b);
    // A wrapped lane lost `2^64 ≡ eps`; the corrected lane is already `< p`.
    let carry = _mm512_cmplt_epu64_mask(s, b);
    let s = _mm512_mask_add_epi64(s, carry, s, c.eps);
    canon512(s, c)
}

/// `(a − b) mod p` for canonical `a` and `b`.
#[archmage::rite(v4)]
fn submod512(a: __m512i, b: __m512i, c: Consts) -> __m512i {
    let d = _mm512_sub_epi64(a, b);
    // A borrowed lane gained `2^64`; subtracting `eps` leaves `a − b + p < p`.
    let borrow = _mm512_cmplt_epu64_mask(a, b);
    _mm512_mask_sub_epi64(d, borrow, d, c.eps)
}

/// `(x · y) mod p` for any 64-bit words, canonical.
#[archmage::rite(v4)]
fn mulmod512(x: __m512i, y: __m512i, c: Consts) -> __m512i {
    let x_hi = _mm512_srli_epi64::<32>(x);
    let y_hi = _mm512_srli_epi64::<32>(y);
    let ll = _mm512_mul_epu32(x, y);
    let lh = _mm512_mul_epu32(x, y_hi);
    let hl = _mm512_mul_epu32(x_hi, y);
    let hh = _mm512_mul_epu32(x_hi, y_hi);
    // Big-endian bignum assembly; every partial sum fits by construction.
    let t0 = _mm512_add_epi64(hl, _mm512_srli_epi64::<32>(ll));
    let t1 = _mm512_add_epi64(lh, _mm512_and_si512(t0, c.eps));
    let hi = _mm512_add_epi64(
        _mm512_add_epi64(hh, _mm512_srli_epi64::<32>(t0)),
        _mm512_srli_epi64::<32>(t1),
    );
    let lo = _mm512_mask_blend_epi32(ODD_LANES, ll, _mm512_slli_epi64::<32>(t1));
    // lo + 2^64 (2^32 hi_hi + hi_lo) ≡ lo − hi_hi + hi_lo · eps.
    let hi_hi = _mm512_srli_epi64::<32>(hi);
    let borrow = _mm512_cmplt_epu64_mask(lo, hi_hi);
    let t = _mm512_sub_epi64(lo, hi_hi);
    // A borrow gained 2^64: subtract eps. `t >= 2^64 − 2^32 + 1` there.
    let t = _mm512_mask_sub_epi64(t, borrow, t, c.eps);
    let m = _mm512_mul_epu32(hi, c.eps);
    let r = _mm512_add_epi64(t, m);
    // A wrap lost 2^64: add eps back. The corrected lane stays below 2^64.
    let carry = _mm512_cmplt_epu64_mask(r, m);
    let r = _mm512_mask_add_epi64(r, carry, r, c.eps);
    canon512(r, c)
}

/// `dst += src (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal("gld::add_assign_avx512", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::add_assign_avx512", dst.len(), 8);
    let c = consts();
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = canon512(_mm512_loadu_si512(&*dst_lane), c);
        let s = canon512(_mm512_loadu_si512(src_lane), c);
        _mm512_storeu_si512(dst_lane, addmod512(d, s, c));
    }
    if !dst_tail.is_empty() {
        add_assign_gld_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst -= src (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal("gld::sub_assign_avx512", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::sub_assign_avx512", dst.len(), 8);
    let c = consts();
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = canon512(_mm512_loadu_si512(&*dst_lane), c);
        let s = canon512(_mm512_loadu_si512(src_lane), c);
        _mm512_storeu_si512(dst_lane, submod512(d, s, c));
    }
    if !dst_tail.is_empty() {
        sub_assign_gld_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst += coeff * src (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal("gld::mul_add_avx512", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::mul_add_avx512", dst.len(), 8);
    let c = consts();
    let cvec = _mm512_set1_epi64(coeff.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = canon512(_mm512_loadu_si512(&*dst_lane), c);
        let prod = mulmod512(cvec, _mm512_loadu_si512(src_lane), c);
        _mm512_storeu_si512(dst_lane, addmod512(d, prod, c));
    }
    if !dst_tail.is_empty() {
        mul_add_gld_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst *= coeff (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u64) {
    check_elem_multiple("gld::mul_assign_avx512", dst.len(), 8);
    let c = consts();
    let cvec = _mm512_set1_epi64(coeff.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = _mm512_loadu_si512(&*dst_lane);
        _mm512_storeu_si512(dst_lane, mulmod512(cvec, d, c));
    }
    if !dst_tail.is_empty() {
        mul_assign_gld_avx2(token.v3(), dst_tail, coeff);
    }
}

/// `dst = coeff * src (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal("gld::mul_into_avx512", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::mul_into_avx512", dst.len(), 8);
    let c = consts();
    let cvec = _mm512_set1_epi64(coeff.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        _mm512_storeu_si512(dst_lane, mulmod512(cvec, _mm512_loadu_si512(src_lane), c));
    }
    if !dst_tail.is_empty() {
        mul_into_gld_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "gld::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "gld::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("gld::mul_elementwise_avx512", dst.len(), 8);
    let c = consts();
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a.as_chunks::<64>();
    let (b_lanes, b_tail) = b.as_chunks::<64>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let product = mulmod512(_mm512_loadu_si512(a_lane), _mm512_loadu_si512(b_lane), c);
        _mm512_storeu_si512(dst_lane, product);
    }
    if !dst_tail.is_empty() {
        mul_elementwise_gld_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Goldilocks, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "gld::mul_elementwise_assign_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("gld::mul_elementwise_assign_avx512", dst.len(), 8);
    let c = consts();
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let product = mulmod512(
            _mm512_loadu_si512(&*dst_lane),
            _mm512_loadu_si512(src_lane),
            c,
        );
        _mm512_storeu_si512(dst_lane, product);
    }
    if !dst_tail.is_empty() {
        mul_elementwise_assign_gld_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst[i] += value (mod p)`, Goldilocks, AVX-512. The raw `value` word is
/// canonicalized once on entry.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("gld::add_assign_scalar_avx512", dst.len(), 8);
    let c = consts();
    let svec = _mm512_set1_epi64(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = canon512(_mm512_loadu_si512(&*dst_lane), c);
        _mm512_storeu_si512(dst_lane, addmod512(d, svec, c));
    }
    if !dst_tail.is_empty() {
        add_assign_scalar_gld_avx2(token.v3(), dst_tail, value);
    }
}

/// `dst[i] -= value (mod p)`, Goldilocks, AVX-512. The raw `value` word is
/// canonicalized once on entry.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_gld_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("gld::sub_assign_scalar_avx512", dst.len(), 8);
    let c = consts();
    let svec = _mm512_set1_epi64(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = canon512(_mm512_loadu_si512(&*dst_lane), c);
        _mm512_storeu_si512(dst_lane, submod512(d, svec, c));
    }
    if !dst_tail.is_empty() {
        sub_assign_scalar_gld_avx2(token.v3(), dst_tail, value);
    }
}
