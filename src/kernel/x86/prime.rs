//! Prime-field integer-SIMD kernels for x86 / `x86_64`.
//!
//! Two lane widths, one reduction discipline each:
//!
//! - **Mersenne31 (`u32` lanes).** `2^31 ≡ 1 (mod p)`, so a multiply folds the
//!   62-bit product with a mask/shift/add and one conditional subtract; the
//!   products come from `vpmuludq` on the even and odd 32-bit lanes.
//! - **Goldilocks (`u64` lanes).** A 64×64→128 product built from four
//!   `vpmuludq` 32-bit half-products, then the split-fold reduction
//!   (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`). Unsigned 64-bit compares are synthesized
//!   from `vpcmpgtq` with a sign-bit flip, since AVX2 has no native one.
//!
//! Every kernel processes whole vector lanes and hands the sub-lane remainder
//! to the portable [`crate::kernel::prime`] path, which is also the differential
//! oracle. All lane operations are validated against a `u128 % p` model. The
//! AVX2 kernels serve `V3`/`V3GfniCrypto`; the SSE4.2 kernels serve `V2`
//! (x86-64-v2 guarantees SSE4.1 `pminud` and SSE4.2 `pcmpgtq`).

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]

use crate::field::goldilocks::{self, Goldilocks};
use crate::field::mersenne31::{self, Mersenne31};
use crate::kernel::prime;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const M31_P: i32 = mersenne31::MODULUS as i32; // 0x7FFF_FFFF
const M31_P64: i64 = mersenne31::MODULUS as i64;
const GLD_P: i64 = goldilocks::MODULUS as i64; // 0xFFFF_FFFF_0000_0001
const GLD_PM1: i64 = (goldilocks::MODULUS - 1) as i64; // 0xFFFF_FFFF_0000_0000
const EPS: i64 = 0xFFFF_FFFF; // 2^32 - 1
const M32: i64 = 0xFFFF_FFFF;

// ===========================================================================
// Mersenne31 (u32 lanes) — AVX2
// ===========================================================================

/// Canonicalize eight raw `u32` lanes to `0..p` (fold + conditional subtract).
#[inline]
#[target_feature(enable = "avx2")]
fn m31_reduce256(x: __m256i, p: __m256i) -> __m256i {
    let vand = _mm256_and_si256(x, p);
    let vshr = _mm256_srli_epi32(x, 31);
    let s = _mm256_add_epi32(vand, vshr);
    let sub = _mm256_sub_epi32(s, p);
    _mm256_min_epu32(s, sub)
}

/// Reduce four 62-bit products (one per 64-bit lane) to canonical `u32`s held
/// in the low 32 bits of each 64-bit lane.
#[inline]
#[target_feature(enable = "avx2")]
fn m31_reduce_mul256(prod: __m256i, pm64: __m256i) -> __m256i {
    let lo = _mm256_and_si256(prod, pm64);
    let hi = _mm256_srli_epi64(prod, 31);
    let s = _mm256_add_epi64(lo, hi);
    let sub = _mm256_sub_epi64(s, pm64);
    _mm256_min_epu32(s, sub)
}

/// `a * b (mod p)` over eight canonicalized `u32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn m31_mul256(a: __m256i, b: __m256i, p: __m256i, pm64: __m256i) -> __m256i {
    let ca = m31_reduce256(a, p);
    let cb = m31_reduce256(b, p);
    let prod_even = _mm256_mul_epu32(ca, cb);
    let a_odd = _mm256_srli_epi64(ca, 32);
    let b_odd = _mm256_srli_epi64(cb, 32);
    let prod_odd = _mm256_mul_epu32(a_odd, b_odd);
    let even = m31_reduce_mul256(prod_even, pm64);
    let odd = m31_reduce_mul256(prod_odd, pm64);
    _mm256_or_si256(even, _mm256_slli_epi64(odd, 32))
}

/// `a + b (mod p)` over eight canonicalized `u32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn m31_addmod256(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let s = _mm256_add_epi32(a, b);
    let sub = _mm256_sub_epi32(s, p);
    _mm256_min_epu32(s, sub)
}

/// `dst += src (mod p)`, Mersenne31, AVX2.
pub fn add_assign_m31_avx2(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { add_assign_m31_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn add_assign_m31_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_reduce256(_mm256_loadu_si256(dp.add(off).cast()), p);
            let s = m31_reduce256(_mm256_loadu_si256(sp.add(off).cast()), p);
            _mm256_storeu_si256(dp.add(off).cast(), m31_addmod256(d, s, p));
        }
        off += 32;
    }
    prime::add_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst -= src (mod p)`, Mersenne31, AVX2.
pub fn sub_assign_m31_avx2(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { sub_assign_m31_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_m31_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_reduce256(_mm256_loadu_si256(dp.add(off).cast()), p);
            let s = m31_reduce256(_mm256_loadu_si256(sp.add(off).cast()), p);
            // a - b = a + (p - b); p - b needs no wrap since b < p.
            let pmb = _mm256_sub_epi32(p, s);
            _mm256_storeu_si256(dp.add(off).cast(), m31_addmod256(d, pmb, p));
        }
        off += 32;
    }
    prime::sub_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst += coeff * src (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
pub fn mul_add_m31_avx2(dst: &mut [u8], coeff: u32, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_add_m31_avx2_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_add_m31_avx2_impl(dst: &mut [u8], coeff: u32, src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let pm64 = _mm256_set1_epi64x(M31_P64);
    let cvec = _mm256_set1_epi32(coeff as i32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let s = _mm256_loadu_si256(sp.add(off).cast());
            let prod = m31_mul256(cvec, s, p, pm64);
            let d = m31_reduce256(_mm256_loadu_si256(dp.add(off).cast()), p);
            _mm256_storeu_si256(dp.add(off).cast(), m31_addmod256(d, prod, p));
        }
        off += 32;
    }
    prime::mul_add::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff), &src[off..]);
}

/// `dst *= coeff (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
pub fn mul_assign_m31_avx2(dst: &mut [u8], coeff: u32) {
    // SAFETY: caller selected an AVX2 backend.
    unsafe { mul_assign_m31_avx2_impl(dst, coeff) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_m31_avx2_impl(dst: &mut [u8], coeff: u32) {
    let len = dst.len() & !31;
    let dp = dst.as_mut_ptr();
    let p = _mm256_set1_epi32(M31_P);
    let pm64 = _mm256_set1_epi64x(M31_P64);
    let cvec = _mm256_set1_epi32(coeff as i32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len().
        unsafe {
            let d = _mm256_loadu_si256(dp.add(off).cast());
            _mm256_storeu_si256(dp.add(off).cast(), m31_mul256(cvec, d, p, pm64));
        }
        off += 32;
    }
    prime::mul_assign::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX2.
pub fn mul_elementwise_m31_avx2(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_elementwise_m31_avx2_impl(dst, a, b) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_elementwise_m31_avx2_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len() & !31;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let pm64 = _mm256_set1_epi64x(M31_P64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = _mm256_loadu_si256(ap.add(off).cast());
            let vb = _mm256_loadu_si256(bp.add(off).cast());
            _mm256_storeu_si256(dp.add(off).cast(), m31_mul256(va, vb, p, pm64));
        }
        off += 32;
    }
    prime::mul_elementwise::<Mersenne31>(&mut dst[off..], &a[off..], &b[off..]);
}

// ===========================================================================
// Mersenne31 (u32 lanes) — SSE4.2
// ===========================================================================

#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_reduce128(x: __m128i, p: __m128i) -> __m128i {
    let vand = _mm_and_si128(x, p);
    let vshr = _mm_srli_epi32(x, 31);
    let s = _mm_add_epi32(vand, vshr);
    let sub = _mm_sub_epi32(s, p);
    _mm_min_epu32(s, sub)
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_reduce_mul128(prod: __m128i, pm64: __m128i) -> __m128i {
    let lo = _mm_and_si128(prod, pm64);
    let hi = _mm_srli_epi64(prod, 31);
    let s = _mm_add_epi64(lo, hi);
    let sub = _mm_sub_epi64(s, pm64);
    _mm_min_epu32(s, sub)
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_mul128(a: __m128i, b: __m128i, p: __m128i, pm64: __m128i) -> __m128i {
    let ca = m31_reduce128(a, p);
    let cb = m31_reduce128(b, p);
    let prod_even = _mm_mul_epu32(ca, cb);
    let prod_odd = _mm_mul_epu32(_mm_srli_epi64(ca, 32), _mm_srli_epi64(cb, 32));
    let even = m31_reduce_mul128(prod_even, pm64);
    let odd = m31_reduce_mul128(prod_odd, pm64);
    _mm_or_si128(even, _mm_slli_epi64(odd, 32))
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_addmod128(a: __m128i, b: __m128i, p: __m128i) -> __m128i {
    let s = _mm_add_epi32(a, b);
    let sub = _mm_sub_epi32(s, p);
    _mm_min_epu32(s, sub)
}

/// `dst += src (mod p)`, Mersenne31, SSE4.2.
pub fn add_assign_m31_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { add_assign_m31_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn add_assign_m31_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_reduce128(_mm_loadu_si128(dp.add(off).cast()), p);
            let s = m31_reduce128(_mm_loadu_si128(sp.add(off).cast()), p);
            _mm_storeu_si128(dp.add(off).cast(), m31_addmod128(d, s, p));
        }
        off += 16;
    }
    prime::add_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst -= src (mod p)`, Mersenne31, SSE4.2.
pub fn sub_assign_m31_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { sub_assign_m31_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn sub_assign_m31_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_reduce128(_mm_loadu_si128(dp.add(off).cast()), p);
            let s = m31_reduce128(_mm_loadu_si128(sp.add(off).cast()), p);
            let pmb = _mm_sub_epi32(p, s);
            _mm_storeu_si128(dp.add(off).cast(), m31_addmod128(d, pmb, p));
        }
        off += 16;
    }
    prime::sub_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst += coeff * src (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
pub fn mul_add_m31_sse41(dst: &mut [u8], coeff: u32, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_add_m31_sse41_impl(dst, coeff, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_add_m31_sse41_impl(dst: &mut [u8], coeff: u32, src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let pm64 = _mm_set1_epi64x(M31_P64);
    let cvec = _mm_set1_epi32(coeff as i32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let s = _mm_loadu_si128(sp.add(off).cast());
            let prod = m31_mul128(cvec, s, p, pm64);
            let d = m31_reduce128(_mm_loadu_si128(dp.add(off).cast()), p);
            _mm_storeu_si128(dp.add(off).cast(), m31_addmod128(d, prod, p));
        }
        off += 16;
    }
    prime::mul_add::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff), &src[off..]);
}

/// `dst *= coeff (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
pub fn mul_assign_m31_sse41(dst: &mut [u8], coeff: u32) {
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_assign_m31_sse41_impl(dst, coeff) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_assign_m31_sse41_impl(dst: &mut [u8], coeff: u32) {
    let len = dst.len() & !15;
    let dp = dst.as_mut_ptr();
    let p = _mm_set1_epi32(M31_P);
    let pm64 = _mm_set1_epi64x(M31_P64);
    let cvec = _mm_set1_epi32(coeff as i32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len().
        unsafe {
            let d = _mm_loadu_si128(dp.add(off).cast());
            _mm_storeu_si128(dp.add(off).cast(), m31_mul128(cvec, d, p, pm64));
        }
        off += 16;
    }
    prime::mul_assign::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, SSE4.2.
pub fn mul_elementwise_m31_sse41(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_elementwise_m31_sse41_impl(dst, a, b) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_elementwise_m31_sse41_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len() & !15;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let pm64 = _mm_set1_epi64x(M31_P64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = _mm_loadu_si128(ap.add(off).cast());
            let vb = _mm_loadu_si128(bp.add(off).cast());
            _mm_storeu_si128(dp.add(off).cast(), m31_mul128(va, vb, p, pm64));
        }
        off += 16;
    }
    prime::mul_elementwise::<Mersenne31>(&mut dst[off..], &a[off..], &b[off..]);
}

// ===========================================================================
// Goldilocks (u64 lanes) — AVX2
// ===========================================================================

/// Unsigned `a > b` over four 64-bit lanes, via a sign-bit flip.
#[inline]
#[target_feature(enable = "avx2")]
fn gt_epu64_256(a: __m256i, b: __m256i) -> __m256i {
    let bias = _mm256_set1_epi64x(i64::MIN);
    _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias))
}

/// Canonicalize four raw `u64` lanes to `0..p` (one conditional subtract).
#[inline]
#[target_feature(enable = "avx2")]
fn gld_canon256(x: __m256i, pcst: __m256i, pm1: __m256i) -> __m256i {
    let ge = gt_epu64_256(x, pm1);
    _mm256_sub_epi64(x, _mm256_and_si256(ge, pcst))
}

/// `a + b (mod p)` over four canonicalized `u64` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn gld_addmod256(a: __m256i, b: __m256i, pcst: __m256i, pm1: __m256i, eps: __m256i) -> __m256i {
    let s = _mm256_add_epi64(a, b);
    let carry = gt_epu64_256(a, s); // s < a ⇒ overflow
    let s = _mm256_add_epi64(s, _mm256_and_si256(carry, eps));
    let ge = gt_epu64_256(s, pm1);
    _mm256_sub_epi64(s, _mm256_and_si256(ge, pcst))
}

/// `a - b (mod p)` over four canonicalized `u64` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn gld_submod256(a: __m256i, b: __m256i, pcst: __m256i, eps: __m256i) -> __m256i {
    let d = _mm256_sub_epi64(a, b);
    let borrow = gt_epu64_256(b, a); // b > a ⇒ borrow
    let d = _mm256_sub_epi64(d, _mm256_and_si256(borrow, eps));
    // After correction d is already < p; guard against the borrow case.
    let ge = gt_epu64_256(d, _mm256_sub_epi64(pcst, _mm256_set1_epi64x(1)));
    _mm256_sub_epi64(d, _mm256_and_si256(ge, pcst))
}

/// `a * b (mod p)` over four `u64` lanes (raw inputs allowed).
#[inline]
#[target_feature(enable = "avx2")]
fn gld_mul256(
    a: __m256i,
    b: __m256i,
    m32: __m256i,
    eps: __m256i,
    pcst: __m256i,
    pm1: __m256i,
) -> __m256i {
    // 64x64 -> 128 via four 32-bit half products.
    let a_lo = _mm256_and_si256(a, m32);
    let a_hi = _mm256_srli_epi64(a, 32);
    let b_lo = _mm256_and_si256(b, m32);
    let b_hi = _mm256_srli_epi64(b, 32);
    let ll = _mm256_mul_epu32(a_lo, b_lo);
    let lh = _mm256_mul_epu32(a_lo, b_hi);
    let hl = _mm256_mul_epu32(a_hi, b_lo);
    let hh = _mm256_mul_epu32(a_hi, b_hi);
    let mid = _mm256_add_epi64(
        _mm256_srli_epi64(ll, 32),
        _mm256_add_epi64(_mm256_and_si256(lh, m32), _mm256_and_si256(hl, m32)),
    );
    let c1 = _mm256_and_si256(mid, m32);
    let carry1 = _mm256_srli_epi64(mid, 32);
    let hi = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_srli_epi64(lh, 32), _mm256_srli_epi64(hl, 32)),
        _mm256_add_epi64(hh, carry1),
    );
    let lo = _mm256_or_si256(_mm256_and_si256(ll, m32), _mm256_slli_epi64(c1, 32));
    // Split-fold reduction of (lo, hi).
    let x_hi_hi = _mm256_srli_epi64(hi, 32);
    let x_hi_lo = _mm256_and_si256(hi, m32);
    let t0 = _mm256_sub_epi64(lo, x_hi_hi);
    let borrow = gt_epu64_256(x_hi_hi, lo);
    let t0 = _mm256_sub_epi64(t0, _mm256_and_si256(borrow, eps));
    let t1 = _mm256_mul_epu32(x_hi_lo, eps);
    let res = _mm256_add_epi64(t0, t1);
    let carry = gt_epu64_256(t0, res);
    let res = _mm256_add_epi64(res, _mm256_and_si256(carry, eps));
    let ge = gt_epu64_256(res, pm1);
    _mm256_sub_epi64(res, _mm256_and_si256(ge, pcst))
}

/// `dst += src (mod p)`, Goldilocks, AVX2.
pub fn add_assign_gld_avx2(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { add_assign_gld_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn add_assign_gld_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm256_set1_epi64x(GLD_P);
    let pm1 = _mm256_set1_epi64x(GLD_PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = gld_canon256(_mm256_loadu_si256(dp.add(off).cast()), pcst, pm1);
            let s = gld_canon256(_mm256_loadu_si256(sp.add(off).cast()), pcst, pm1);
            _mm256_storeu_si256(dp.add(off).cast(), gld_addmod256(d, s, pcst, pm1, eps));
        }
        off += 32;
    }
    prime::add_assign::<Goldilocks>(&mut dst[off..], &src[off..]);
}

/// `dst -= src (mod p)`, Goldilocks, AVX2.
pub fn sub_assign_gld_avx2(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { sub_assign_gld_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_gld_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm256_set1_epi64x(GLD_P);
    let pm1 = _mm256_set1_epi64x(GLD_PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = gld_canon256(_mm256_loadu_si256(dp.add(off).cast()), pcst, pm1);
            let s = gld_canon256(_mm256_loadu_si256(sp.add(off).cast()), pcst, pm1);
            _mm256_storeu_si256(dp.add(off).cast(), gld_submod256(d, s, pcst, eps));
        }
        off += 32;
    }
    prime::sub_assign::<Goldilocks>(&mut dst[off..], &src[off..]);
}

/// `dst += coeff * src (mod p)`, Goldilocks, AVX2. `coeff` must be canonical.
pub fn mul_add_gld_avx2(dst: &mut [u8], coeff: u64, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_add_gld_avx2_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_add_gld_avx2_impl(dst: &mut [u8], coeff: u64, src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm256_set1_epi64x(GLD_P);
    let pm1 = _mm256_set1_epi64x(GLD_PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let m32 = _mm256_set1_epi64x(M32);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let s = _mm256_loadu_si256(sp.add(off).cast());
            let prod = gld_mul256(cvec, s, m32, eps, pcst, pm1);
            let d = gld_canon256(_mm256_loadu_si256(dp.add(off).cast()), pcst, pm1);
            _mm256_storeu_si256(dp.add(off).cast(), gld_addmod256(d, prod, pcst, pm1, eps));
        }
        off += 32;
    }
    prime::mul_add::<Goldilocks>(&mut dst[off..], goldilocks::Elem(coeff), &src[off..]);
}

/// `dst *= coeff (mod p)`, Goldilocks, AVX2. `coeff` must be canonical.
pub fn mul_assign_gld_avx2(dst: &mut [u8], coeff: u64) {
    // SAFETY: caller selected an AVX2 backend.
    unsafe { mul_assign_gld_avx2_impl(dst, coeff) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_gld_avx2_impl(dst: &mut [u8], coeff: u64) {
    let len = dst.len() & !31;
    let dp = dst.as_mut_ptr();
    let pcst = _mm256_set1_epi64x(GLD_P);
    let pm1 = _mm256_set1_epi64x(GLD_PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let m32 = _mm256_set1_epi64x(M32);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len().
        unsafe {
            let d = _mm256_loadu_si256(dp.add(off).cast());
            _mm256_storeu_si256(dp.add(off).cast(), gld_mul256(cvec, d, m32, eps, pcst, pm1));
        }
        off += 32;
    }
    prime::mul_assign::<Goldilocks>(&mut dst[off..], goldilocks::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, AVX2.
pub fn mul_elementwise_gld_avx2(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_elementwise_gld_avx2_impl(dst, a, b) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_elementwise_gld_avx2_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len() & !31;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let pcst = _mm256_set1_epi64x(GLD_P);
    let pm1 = _mm256_set1_epi64x(GLD_PM1);
    let eps = _mm256_set1_epi64x(EPS);
    let m32 = _mm256_set1_epi64x(M32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = _mm256_loadu_si256(ap.add(off).cast());
            let vb = _mm256_loadu_si256(bp.add(off).cast());
            _mm256_storeu_si256(dp.add(off).cast(), gld_mul256(va, vb, m32, eps, pcst, pm1));
        }
        off += 32;
    }
    prime::mul_elementwise::<Goldilocks>(&mut dst[off..], &a[off..], &b[off..]);
}

// ===========================================================================
// Goldilocks (u64 lanes) — SSE4.2
// ===========================================================================

#[inline]
#[target_feature(enable = "sse4.2")]
fn gt_epu64_128(a: __m128i, b: __m128i) -> __m128i {
    let bias = _mm_set1_epi64x(i64::MIN);
    _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias))
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn gld_canon128(x: __m128i, pcst: __m128i, pm1: __m128i) -> __m128i {
    let ge = gt_epu64_128(x, pm1);
    _mm_sub_epi64(x, _mm_and_si128(ge, pcst))
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn gld_addmod128(a: __m128i, b: __m128i, pcst: __m128i, pm1: __m128i, eps: __m128i) -> __m128i {
    let s = _mm_add_epi64(a, b);
    let carry = gt_epu64_128(a, s);
    let s = _mm_add_epi64(s, _mm_and_si128(carry, eps));
    let ge = gt_epu64_128(s, pm1);
    _mm_sub_epi64(s, _mm_and_si128(ge, pcst))
}

#[inline]
#[target_feature(enable = "sse4.2")]
fn gld_submod128(a: __m128i, b: __m128i, pcst: __m128i, eps: __m128i) -> __m128i {
    let d = _mm_sub_epi64(a, b);
    let borrow = gt_epu64_128(b, a);
    let d = _mm_sub_epi64(d, _mm_and_si128(borrow, eps));
    let ge = gt_epu64_128(d, _mm_sub_epi64(pcst, _mm_set1_epi64x(1)));
    _mm_sub_epi64(d, _mm_and_si128(ge, pcst))
}

#[inline]
#[target_feature(enable = "sse4.2")]
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
pub fn add_assign_gld_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { add_assign_gld_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn add_assign_gld_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = gld_canon128(_mm_loadu_si128(dp.add(off).cast()), pcst, pm1);
            let s = gld_canon128(_mm_loadu_si128(sp.add(off).cast()), pcst, pm1);
            _mm_storeu_si128(dp.add(off).cast(), gld_addmod128(d, s, pcst, pm1, eps));
        }
        off += 16;
    }
    prime::add_assign::<Goldilocks>(&mut dst[off..], &src[off..]);
}

/// `dst -= src (mod p)`, Goldilocks, SSE4.2.
pub fn sub_assign_gld_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { sub_assign_gld_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn sub_assign_gld_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm_set1_epi64x(GLD_P);
    let eps = _mm_set1_epi64x(EPS);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = gld_canon128(
                _mm_loadu_si128(dp.add(off).cast()),
                pcst,
                _mm_set1_epi64x(GLD_PM1),
            );
            let s = gld_canon128(
                _mm_loadu_si128(sp.add(off).cast()),
                pcst,
                _mm_set1_epi64x(GLD_PM1),
            );
            _mm_storeu_si128(dp.add(off).cast(), gld_submod128(d, s, pcst, eps));
        }
        off += 16;
    }
    prime::sub_assign::<Goldilocks>(&mut dst[off..], &src[off..]);
}

/// `dst += coeff * src (mod p)`, Goldilocks, SSE4.2. `coeff` must be canonical.
pub fn mul_add_gld_sse41(dst: &mut [u8], coeff: u64, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_add_gld_sse41_impl(dst, coeff, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_add_gld_sse41_impl(dst: &mut [u8], coeff: u64, src: &[u8]) {
    let len = dst.len() & !15;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let cvec = _mm_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let s = _mm_loadu_si128(sp.add(off).cast());
            let prod = gld_mul128(cvec, s, m32, eps, pcst, pm1);
            let d = gld_canon128(_mm_loadu_si128(dp.add(off).cast()), pcst, pm1);
            _mm_storeu_si128(dp.add(off).cast(), gld_addmod128(d, prod, pcst, pm1, eps));
        }
        off += 16;
    }
    prime::mul_add::<Goldilocks>(&mut dst[off..], goldilocks::Elem(coeff), &src[off..]);
}

/// `dst *= coeff (mod p)`, Goldilocks, SSE4.2. `coeff` must be canonical.
pub fn mul_assign_gld_sse41(dst: &mut [u8], coeff: u64) {
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_assign_gld_sse41_impl(dst, coeff) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_assign_gld_sse41_impl(dst: &mut [u8], coeff: u64) {
    let len = dst.len() & !15;
    let dp = dst.as_mut_ptr();
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let cvec = _mm_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len().
        unsafe {
            let d = _mm_loadu_si128(dp.add(off).cast());
            _mm_storeu_si128(dp.add(off).cast(), gld_mul128(cvec, d, m32, eps, pcst, pm1));
        }
        off += 16;
    }
    prime::mul_assign::<Goldilocks>(&mut dst[off..], goldilocks::Elem(coeff));
}

/// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, SSE4.2.
pub fn mul_elementwise_gld_sse41(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2 ⇒ SSE4.2).
    unsafe { mul_elementwise_gld_sse41_impl(dst, a, b) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_elementwise_gld_sse41_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len() & !15;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let pcst = _mm_set1_epi64x(GLD_P);
    let pm1 = _mm_set1_epi64x(GLD_PM1);
    let eps = _mm_set1_epi64x(EPS);
    let m32 = _mm_set1_epi64x(M32);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = _mm_loadu_si128(ap.add(off).cast());
            let vb = _mm_loadu_si128(bp.add(off).cast());
            _mm_storeu_si128(dp.add(off).cast(), gld_mul128(va, vb, m32, eps, pcst, pm1));
        }
        off += 16;
    }
    prime::mul_elementwise::<Goldilocks>(&mut dst[off..], &a[off..], &b[off..]);
}
