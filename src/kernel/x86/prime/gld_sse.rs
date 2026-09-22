//! Goldilocks (`u64` lanes) SSE4.2 kernels.
//!
//! Two lanes per vector, same split-fold reduction as the AVX2 path but in
//! the unshifted domain with explicit `pcmpgtq` sign-flip compares. Serves
//! the `V2` backend: each entry is a safe capability-token function taking
//! an `X64V2Token` and validating its own geometry.

use super::{EPS, GLD_P, GLD_PM1, check_elem_multiple, check_equal};
use crate::field::goldilocks::{self, Goldilocks};
use crate::kernel::{prime, scalar};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Low-32-bit lane mask, used by the unshifted-domain fold.
const M32: i64 = 0xFFFF_FFFF;

/// Unsigned `a > b` over two 64-bit lanes, via a sign-bit flip.
#[archmage::rite(v2)]
fn gt_epu64_128(a: __m128i, b: __m128i) -> __m128i {
    let bias = _mm_set1_epi64x(i64::MIN);
    _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias))
}

/// Canonicalize two raw `u64` lanes to `0..p` (one conditional subtract).
#[archmage::rite(v2)]
fn gld_canon128(x: __m128i, pcst: __m128i, pm1: __m128i) -> __m128i {
    let ge = gt_epu64_128(x, pm1);
    _mm_sub_epi64(x, _mm_and_si128(ge, pcst))
}

/// `a + b (mod p)` over two canonicalized `u64` lanes.
#[archmage::rite(v2)]
fn gld_addmod128(a: __m128i, b: __m128i, pcst: __m128i, pm1: __m128i, eps: __m128i) -> __m128i {
    let s = _mm_add_epi64(a, b);
    let carry = gt_epu64_128(a, s);
    let s = _mm_add_epi64(s, _mm_and_si128(carry, eps));
    let ge = gt_epu64_128(s, pm1);
    _mm_sub_epi64(s, _mm_and_si128(ge, pcst))
}

/// `a - b (mod p)` over two canonicalized `u64` lanes.
#[archmage::rite(v2)]
fn gld_submod128(a: __m128i, b: __m128i, pcst: __m128i, eps: __m128i) -> __m128i {
    let d = _mm_sub_epi64(a, b);
    let borrow = gt_epu64_128(b, a);
    let d = _mm_sub_epi64(d, _mm_and_si128(borrow, eps));
    let ge = gt_epu64_128(d, _mm_sub_epi64(pcst, _mm_set1_epi64x(1)));
    _mm_sub_epi64(d, _mm_and_si128(ge, pcst))
}

/// `a * b (mod p)` over two lanes: `vpmuludq` half-products, bignum
/// assembly, then the split fold in the unshifted domain.
#[archmage::rite(v2)]
fn gld_mul128(
    a: __m128i,
    b: __m128i,
    m32: __m128i,
    eps: __m128i,
    pcst: __m128i,
    pm1: __m128i,
) -> __m128i {
    let a_lo = _mm_and_si128(a, m32);
    let a_hi = _mm_srli_epi64(a, 32);
    let b_lo = _mm_and_si128(b, m32);
    let b_hi = _mm_srli_epi64(b, 32);
    let ll = _mm_mul_epu32(a_lo, b_lo);
    let lh = _mm_mul_epu32(a_lo, b_hi);
    let hl = _mm_mul_epu32(a_hi, b_lo);
    let hh = _mm_mul_epu32(a_hi, b_hi);
    let mid = _mm_add_epi64(
        _mm_srli_epi64(ll, 32),
        _mm_add_epi64(_mm_and_si128(lh, m32), _mm_and_si128(hl, m32)),
    );
    let c1 = _mm_and_si128(mid, m32);
    let carry1 = _mm_srli_epi64(mid, 32);
    let hi = _mm_add_epi64(
        _mm_add_epi64(_mm_srli_epi64(lh, 32), _mm_srli_epi64(hl, 32)),
        _mm_add_epi64(hh, carry1),
    );
    let lo = _mm_or_si128(_mm_and_si128(ll, m32), _mm_slli_epi64(c1, 32));
    let x_hi_hi = _mm_srli_epi64(hi, 32);
    let x_hi_lo = _mm_and_si128(hi, m32);
    let t0 = _mm_sub_epi64(lo, x_hi_hi);
    let borrow = gt_epu64_128(x_hi_hi, lo);
    let t0 = _mm_sub_epi64(t0, _mm_and_si128(borrow, eps));
    let t1 = _mm_mul_epu32(x_hi_lo, eps);
    let res = _mm_add_epi64(t0, t1);
    let carry = gt_epu64_128(t0, res);
    let res = _mm_add_epi64(res, _mm_and_si128(carry, eps));
    let ge = gt_epu64_128(res, pm1);
    _mm_sub_epi64(res, _mm_and_si128(ge, pcst))
}

/// `dst += src (mod p)`, Goldilocks, SSE4.2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal("gld::add_assign_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::add_assign_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = gld_canon128(_mm_loadu_si128(&*dst_lane), pcst, pm1);
        let s = gld_canon128(_mm_loadu_si128(src_lane), pcst, pm1);
        _mm_storeu_si128(dst_lane, gld_addmod128(d, s, pcst, pm1, eps));
    }
    prime::add_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst -= src (mod p)`, Goldilocks, SSE4.2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal("gld::sub_assign_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::sub_assign_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let eps = _mm_set1_epi64x(EPS);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = gld_canon128(_mm_loadu_si128(&*dst_lane), pcst, _mm_set1_epi64x(GLD_PM1));
        let s = gld_canon128(_mm_loadu_si128(src_lane), pcst, _mm_set1_epi64x(GLD_PM1));
        _mm_storeu_si128(dst_lane, gld_submod128(d, s, pcst, eps));
    }
    prime::sub_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst += coeff * src (mod p)`, Goldilocks, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal("gld::mul_add_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::mul_add_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let cvec = _mm_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm_loadu_si128(src_lane);
        let prod = gld_mul128(cvec, s, m32, eps, pcst, pm1);
        let d = gld_canon128(_mm_loadu_si128(&*dst_lane), pcst, pm1);
        _mm_storeu_si128(dst_lane, gld_addmod128(d, prod, pcst, pm1, eps));
    }
    prime::mul_add::<Goldilocks>(dst_tail, goldilocks::Elem(coeff), src_tail);
}

/// `dst *= coeff (mod p)`, Goldilocks, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u64) {
    check_elem_multiple("gld::mul_assign_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let cvec = _mm_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, gld_mul128(cvec, d, m32, eps, pcst, pm1));
    }
    prime::mul_assign::<Goldilocks>(dst_tail, goldilocks::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, SSE4.2.
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("gld::mul_elementwise_sse42", "dst", dst.len(), "a", a.len());
    check_equal("gld::mul_elementwise_sse42", "dst", dst.len(), "b", b.len());
    check_elem_multiple("gld::mul_elementwise_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (a_lanes, a_tail) = a.as_chunks::<16>();
    let (b_lanes, b_tail) = b.as_chunks::<16>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = _mm_loadu_si128(a_lane);
        let vb = _mm_loadu_si128(b_lane);
        _mm_storeu_si128(dst_lane, gld_mul128(va, vb, m32, eps, pcst, pm1));
    }
    prime::mul_elementwise::<Goldilocks>(dst_tail, a_tail, b_tail);
}

/// `dst = coeff * src (mod p)`, Goldilocks, SSE4.2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal("gld::mul_into_sse42", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gld::mul_into_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let cvec = _mm_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, gld_mul128(cvec, s, m32, eps, pcst, pm1));
    }
    // Scalar tail.
    for (d, s) in dst_tail.chunks_exact_mut(8).zip(src_tail.chunks_exact(8)) {
        let v =
            goldilocks::Elem(coeff) * goldilocks::Elem(u64::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Goldilocks, SSE4.2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "gld::mul_elementwise_assign_sse42",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("gld::mul_elementwise_assign_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = _mm_loadu_si128(&*dst_lane);
        let s = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, gld_mul128(d, s, m32, eps, pcst, pm1));
    }
    scalar::mul_elementwise_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst[i] += value (mod p)`, Goldilocks, SSE4.2. The raw `value` word is
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("gld::add_assign_scalar_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    // The broadcast value is loop-invariant: canonicalize the raw word once.
    let svec = _mm_set1_epi64x(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = gld_canon128(_mm_loadu_si128(&*dst_lane), pcst, pm1);
        _mm_storeu_si128(dst_lane, gld_addmod128(d, svec, pcst, pm1, eps));
    }
    prime::add_assign_scalar::<Goldilocks>(dst_tail, goldilocks::Elem(value));
}

/// `dst[i] -= value (mod p)`, Goldilocks, SSE4.2. The raw `value` word is
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_gld_sse42(_token: archmage::X64V2Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("gld::sub_assign_scalar_sse42", dst.len(), 8);
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let svec = _mm_set1_epi64x(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let d = gld_canon128(_mm_loadu_si128(&*dst_lane), pcst, pm1);
        _mm_storeu_si128(dst_lane, gld_submod128(d, svec, pcst, eps));
    }
    prime::sub_assign_scalar::<Goldilocks>(dst_tail, goldilocks::Elem(value));
}
