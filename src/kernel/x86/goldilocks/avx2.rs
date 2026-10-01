//! Goldilocks (`u64` lanes) AVX2 kernels.
//!
//! Four lanes per vector. A 64×64→128 product built from `vpmuludq`
//! 32-bit half-products, then the split-fold reduction (`2^64 ≡ 2^32 − 1`,
//! `2^96 ≡ −1`). Unsigned 64-bit compares are synthesized from `vpcmpgtq`
//! with a sign-bit flip, and whole loop bodies stay in the shifted domain.
//! Serves the `V3`/`V3GfniCrypto` backends: each entry is a safe
//! capability-token function taking an `X64V3Token` and validating its own
//! geometry.

use super::{EPS, P, PM1};
use crate::field::goldilocks::{self, Goldilocks};
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::{prime, scalar};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Unsigned `a > b` over four 64-bit lanes, via a sign-bit flip.
#[archmage::rite(v3)]
fn gt_epu64(a: __m256i, b: __m256i) -> __m256i {
    let bias = _mm256_set1_epi64x(i64::MIN);
    _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias))
}

/// Canonicalize four raw `u64` lanes to `0..p` (one conditional subtract).
#[archmage::rite(v3)]
fn canon(x: __m256i, pcst: __m256i, pm1: __m256i) -> __m256i {
    let ge = gt_epu64(x, pm1);
    _mm256_sub_epi64(x, _mm256_and_si256(ge, pcst))
}

// Shifted-domain helpers. Values XORed with 2^63 let signed `vpcmpgtq`
// emulate unsigned compares; the bias cancels when two shifted values add,
// so intermediate results stay shifted through whole loop bodies. A wrap
// moves a lane by `2^64 == eps (mod p)`: a wrapped subtraction has gained
// it and a wrapped addition has lost it, so the fold corrections subtract
// and add `eps` respectively, via `srli_epi64::<32>` of the all-ones
// compare mask instead of a separate AND with the epsilon.
const SIGN: i64 = i64::MIN;
#[allow(clippy::cast_possible_wrap)]
const SHIFTED_P: i64 = 0x7FFF_FFFF_0000_0001_u64 as i64;

/// Canonicalize shifted lanes in `0..2p` (one conditional subtract),
/// returning shifted lanes in `0..p`.
#[archmage::rite(v3)]
fn canon_s(x_s: __m256i, sp: __m256i, eps: __m256i) -> __m256i {
    // All-ones where x_s < p (already canonical), zero otherwise.
    let mask = _mm256_cmpgt_epi64(sp, x_s);
    // Where not canonical: add 0xFFFFFFFF ≡ -p.
    _mm256_add_epi64(x_s, _mm256_andnot_si256(mask, eps))
}

/// Full 64x64 -> 128 product over four lanes via three `vpmuludq` pairs;
/// the high-limb gathers ride the float-domain shuffle port.
/// Returns `(hi, lo)`.
#[archmage::rite(v3)]
fn mul_wide(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
    let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));
    let ll = _mm256_mul_epu32(x, y);
    let lh = _mm256_mul_epu32(x, y_hi);
    let hl = _mm256_mul_epu32(x_hi, y);
    let hh = _mm256_mul_epu32(x_hi, y_hi);
    // Big-endian bignum assembly; every partial sum fits by construction.
    let ll_hi = _mm256_srli_epi64::<32>(ll);
    let t0 = _mm256_add_epi64(hl, ll_hi);
    let t0_lo = _mm256_and_si256(t0, _mm256_set1_epi64x(0xFFFF_FFFF));
    let t0_hi = _mm256_srli_epi64::<32>(t0);
    let t1 = _mm256_add_epi64(lh, t0_lo);
    let t2 = _mm256_add_epi64(hh, t0_hi);
    let res_hi = _mm256_add_epi64(t2, _mm256_srli_epi64::<32>(t1));
    let t1_lo = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
    let res_lo = _mm256_blend_epi32::<0b1010_1010>(ll, t1_lo);
    (res_hi, res_lo)
}

/// Split-fold a 128-bit product into a shifted-domain lane in `0..2p`.
/// `lo` enters shifted; everything below stays shifted.
#[archmage::rite(v3)]
fn reduce_wide_s(hi: __m256i, lo: __m256i, eps: __m256i) -> __m256i {
    let lo_s = _mm256_xor_si256(lo, _mm256_set1_epi64x(SIGN));
    let hi_hi = _mm256_srli_epi64::<32>(hi);
    // lo - hi_hi; a lane where this wraps (lo < hi_hi) has gained
    // 2^64 == eps (mod p), so the correction subtracts it back.
    let sub_wrapped = _mm256_sub_epi64(lo_s, hi_hi);
    let sub_mask = _mm256_cmpgt_epi32(sub_wrapped, lo_s);
    let folded_s = _mm256_sub_epi64(sub_wrapped, _mm256_srli_epi64::<32>(sub_mask));
    // + hi_lo * epsilon; `vpmuludq` reads only the low dwords.
    let t1 = _mm256_mul_epu32(hi, eps);
    // lo1 + t1 with carry correction (t1 <= 0xffffffff00000000).
    let add_wrapped = _mm256_add_epi64(folded_s, t1);
    let add_mask = _mm256_cmpgt_epi32(folded_s, add_wrapped);
    _mm256_add_epi64(add_wrapped, _mm256_srli_epi64::<32>(add_mask))
}

/// `a + b (mod p)` over four canonicalized `u64` lanes.
///
/// Every input lane must be canonical (below `p`), and every result lane is.
/// With `q = p - b`, the integer `a - q + p` equals `a + b`. The borrow
/// marks `q > a`, which is `a + b < p`: there `a - q` wraps and the final
/// add of `p` wraps back, leaving exactly `a + b` below `p`. Without the
/// borrow, `q <= a` holds and the direct difference `a - q = a + b - p` is
/// already canonical.
#[archmage::rite(v3)]
fn addmod(a: __m256i, b: __m256i, pcst: __m256i, _pm1: __m256i, _eps: __m256i) -> __m256i {
    let q = _mm256_sub_epi64(pcst, b);
    let borrow = gt_epu64(q, a);
    let d = _mm256_sub_epi64(a, q);
    _mm256_add_epi64(d, _mm256_and_si256(borrow, pcst))
}

/// `a - b (mod p)` over four canonicalized `u64` lanes.
///
/// Every input lane must be canonical (below `p`), and every result lane is.
/// The borrow marks `b > a`; the wrapped `a - b` then carries a spurious
/// `2^64`, which equals `p + eps` over this field, so subtracting `eps`
/// leaves `a - b + p`, below `p` exactly because `b > a`. Without the
/// borrow, `a - b` is a plain difference of lanes below `p` and needs no
/// trailing reduction.
#[archmage::rite(v3)]
fn submod(a: __m256i, b: __m256i, _pcst: __m256i, eps: __m256i) -> __m256i {
    let d = _mm256_sub_epi64(a, b);
    let borrow = gt_epu64(b, a);
    _mm256_sub_epi64(d, _mm256_and_si256(borrow, eps))
}

/// `dst += src (mod p)`, Goldilocks, AVX2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "goldilocks::add_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("goldilocks::add_assign_avx2", dst.len(), 8);
    let pcst = _mm256_set1_epi64x(P);
    let pm1 = _mm256_set1_epi64x(PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = canon(_mm256_loadu_si256(&*dst_lane), pcst, pm1);
        let s = canon(_mm256_loadu_si256(src_lane), pcst, pm1);
        _mm256_storeu_si256(dst_lane, addmod(d, s, pcst, pm1, eps));
    }
    prime::add_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst -= src (mod p)`, Goldilocks, AVX2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "goldilocks::sub_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("goldilocks::sub_assign_avx2", dst.len(), 8);
    let pcst = _mm256_set1_epi64x(P);
    let pm1 = _mm256_set1_epi64x(PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = canon(_mm256_loadu_si256(&*dst_lane), pcst, pm1);
        let s = canon(_mm256_loadu_si256(src_lane), pcst, pm1);
        _mm256_storeu_si256(dst_lane, submod(d, s, pcst, eps));
    }
    prime::sub_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst += coeff * src (mod p)`, Goldilocks, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal(
        "goldilocks::mul_add_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("goldilocks::mul_add_avx2", dst.len(), 8);
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(SIGN);
    let sp_c = _mm256_set1_epi64x(SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        // Canonical dst: shifted compare, then an exact unshift (XOR is
        // involutive even across the 2^64 wrap of the shifted domain).
        let d_s = canon_s(
            _mm256_xor_si256(_mm256_loadu_si256(&*dst_lane), sign),
            sp_c,
            eps,
        );
        let d = _mm256_xor_si256(d_s, sign);
        let (hi, lo) = mul_wide(cvec, _mm256_loadu_si256(src_lane));
        let mut red_s = reduce_wide_s(hi, lo, eps);
        let mask = _mm256_cmpgt_epi64(sp_c, red_s);
        red_s = _mm256_add_epi64(red_s, _mm256_andnot_si256(mask, eps));
        // d < p and red < p: shifted add with wrap == -p correction.
        let sum_wrapped = _mm256_add_epi64(d, red_s);
        let carry = _mm256_cmpgt_epi64(red_s, sum_wrapped);
        let sum_s = _mm256_add_epi64(sum_wrapped, _mm256_srli_epi64::<32>(carry));
        let mask = _mm256_cmpgt_epi64(sp_c, sum_s);
        let res = _mm256_xor_si256(
            _mm256_add_epi64(sum_s, _mm256_andnot_si256(mask, eps)),
            sign,
        );
        _mm256_storeu_si256(dst_lane, res);
    }
    prime::mul_add::<Goldilocks>(dst_tail, goldilocks::Elem(coeff), src_tail);
}

/// `dst *= coeff (mod p)`, Goldilocks, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u64) {
    check_elem_multiple("goldilocks::mul_assign_avx2", dst.len(), 8);
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(SIGN);
    let sp_c = _mm256_set1_epi64x(SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let (hi, lo) = mul_wide(cvec, _mm256_loadu_si256(&*dst_lane));
        let res_s = reduce_wide_s(hi, lo, eps);
        let mask = _mm256_cmpgt_epi64(sp_c, res_s);
        let res = _mm256_xor_si256(
            _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps)),
            sign,
        );
        _mm256_storeu_si256(dst_lane, res);
    }
    prime::mul_assign::<Goldilocks>(dst_tail, goldilocks::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, AVX2.
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2(_token: archmage::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "goldilocks::mul_elementwise_avx2",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "goldilocks::mul_elementwise_avx2",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("goldilocks::mul_elementwise_avx2", dst.len(), 8);
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(SIGN);
    let sp_c = _mm256_set1_epi64x(SHIFTED_P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_tail) = a.as_chunks::<32>();
    let (b_lanes, b_tail) = b.as_chunks::<32>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let (hi, lo) = mul_wide(_mm256_loadu_si256(a_lane), _mm256_loadu_si256(b_lane));
        let res_s = reduce_wide_s(hi, lo, eps);
        let mask = _mm256_cmpgt_epi64(sp_c, res_s);
        let res = _mm256_xor_si256(
            _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps)),
            sign,
        );
        _mm256_storeu_si256(dst_lane, res);
    }
    prime::mul_elementwise::<Goldilocks>(dst_tail, a_tail, b_tail);
}

/// `dst = coeff * src (mod p)`, Goldilocks, AVX2. `coeff` must be canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: u64, src: &[u8]) {
    check_equal(
        "goldilocks::mul_into_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("goldilocks::mul_into_avx2", dst.len(), 8);
    let eps = _mm256_set1_epi64x(EPS);
    let sp_c = _mm256_set1_epi64x(SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm256_loadu_si256(src_lane);
        let (hi, lo) = mul_wide(cvec, s);
        let res_s = reduce_wide_s(hi, lo, eps);
        // One conditional subtract, then back out of the shifted domain.
        let mask = _mm256_cmpgt_epi64(sp_c, res_s);
        let res = _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps));
        let res = _mm256_xor_si256(res, _mm256_set1_epi64x(SIGN));
        _mm256_storeu_si256(dst_lane, res);
    }
    // Scalar tail.
    for (d, s) in dst_tail.chunks_exact_mut(8).zip(src_tail.chunks_exact(8)) {
        let v =
            goldilocks::Elem(coeff) * goldilocks::Elem(u64::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Goldilocks, AVX2.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "goldilocks::mul_elementwise_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("goldilocks::mul_elementwise_assign_avx2", dst.len(), 8);
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(SIGN);
    let sp_c = _mm256_set1_epi64x(SHIFTED_P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let (hi, lo) = mul_wide(_mm256_loadu_si256(&*dst_lane), _mm256_loadu_si256(src_lane));
        let res_s = reduce_wide_s(hi, lo, eps);
        let mask = _mm256_cmpgt_epi64(sp_c, res_s);
        let res = _mm256_xor_si256(
            _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps)),
            sign,
        );
        _mm256_storeu_si256(dst_lane, res);
    }
    scalar::mul_elementwise_assign::<Goldilocks>(dst_tail, src_tail);
}

/// `dst[i] += value (mod p)`, Goldilocks, AVX2. The raw `value` word is
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_avx2(_token: archmage::X64V3Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("goldilocks::add_assign_scalar_avx2", dst.len(), 8);
    let pcst = _mm256_set1_epi64x(P);
    let pm1 = _mm256_set1_epi64x(PM1);
    let eps = _mm256_set1_epi64x(EPS);
    // The broadcast value is loop-invariant: canonicalize the raw word once.
    let svec = _mm256_set1_epi64x(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = canon(_mm256_loadu_si256(&*dst_lane), pcst, pm1);
        _mm256_storeu_si256(dst_lane, addmod(d, svec, pcst, pm1, eps));
    }
    prime::add_assign_scalar::<Goldilocks>(dst_tail, goldilocks::Elem(value));
}

/// `dst[i] -= value (mod p)`, Goldilocks, AVX2. The raw `value` word is
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_avx2(_token: archmage::X64V3Token, dst: &mut [u8], value: u64) {
    check_elem_multiple("goldilocks::sub_assign_scalar_avx2", dst.len(), 8);
    let pcst = _mm256_set1_epi64x(P);
    let pm1 = _mm256_set1_epi64x(PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let svec = _mm256_set1_epi64x(goldilocks::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = canon(_mm256_loadu_si256(&*dst_lane), pcst, pm1);
        _mm256_storeu_si256(dst_lane, submod(d, svec, pcst, eps));
    }
    prime::sub_assign_scalar::<Goldilocks>(dst_tail, goldilocks::Elem(value));
}
