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
const GLD_P: i64 = goldilocks::MODULUS as i64; // 0xFFFF_FFFF_0000_0001
const GLD_PM1: i64 = (goldilocks::MODULUS - 1) as i64; // 0xFFFF_FFFF_0000_0000
const EPS: i64 = 0xFFFF_FFFF; // 2^32 - 1
const M32: i64 = 0xFFFF_FFFF;

// ===========================================================================
// Mersenne31 (u32 lanes) — AVX2
// ===========================================================================

/// Fold eight raw `u32` lanes to `0..=p` (`2^31 ≡ 1`, no conditional
/// subtract). The result is `p` only for the raw lane `p` itself; every
/// consumer below tolerates `= p` inputs and canonicalizes with min chains.
#[inline]
#[target_feature(enable = "avx2")]
fn m31_fold256(x: __m256i, p: __m256i) -> __m256i {
    let lo = _mm256_and_si256(x, p);
    let hi = _mm256_srli_epi32(x, 31);
    _mm256_add_epi32(lo, hi)
}

/// Canonicalize lanes in `0..=2p`: two unconditional subtract-and-min steps.
/// `min(x, x-p)` then `min(u, u-p)` handles both the `p` and the `2p` edge
/// cases a single conditional subtract misses (raw lane `p` ≡ 0).
#[inline]
#[target_feature(enable = "avx2")]
fn m31_min_chain256(x: __m256i, p: __m256i) -> __m256i {
    let u = _mm256_min_epu32(x, _mm256_sub_epi32(x, p));
    _mm256_min_epu32(u, _mm256_sub_epi32(u, p))
}

/// `a * b (mod p)` over eight 31-bit lanes (`0..=p`; fold output allowed).
///
/// Doubled-odd-limb form: `vpmuludq` on the even lanes and on the odd lanes
/// shifted down 31 (doubling them), blend-based recombination, one Mersenne
/// fold and one conditional subtract. Twelve arithmetic ops per vector; the
/// odd-lane gather is a port-5 `movehdup`, not a shift.
#[inline]
#[target_feature(enable = "avx2")]
#[must_use]
fn m31_mulmod256(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let a_odd = _mm256_srli_epi64::<31>(a); // odd lanes, doubled
    let b_odd = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(b)));
    let prod_odd_dbl = _mm256_mul_epu32(a_odd, b_odd);
    let prod_evn = _mm256_mul_epu32(a, b);
    let prod_odd_lo = _mm256_slli_epi64::<31>(prod_odd_dbl);
    let prod_evn_hi = _mm256_srli_epi64::<31>(prod_evn);
    let lo_dirty = _mm256_blend_epi32::<0b1010_1010>(prod_evn, prod_odd_lo);
    let hi = _mm256_blend_epi32::<0b1010_1010>(prod_evn_hi, prod_odd_dbl);
    let lo = _mm256_and_si256(lo_dirty, p);
    let t = _mm256_add_epi32(lo, hi); // per dword < 2^32; t = p ⇒ prod ≡ 0
    _mm256_min_epu32(t, _mm256_sub_epi32(t, p))
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
    let fast = len & !127;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let d = m31_fold256(_mm256_loadu_si256(dp.add(off + i).cast()), p);
                let s = m31_fold256(_mm256_loadu_si256(sp.add(off + i).cast()), p);
                let sum = _mm256_add_epi32(d, s); // <= 2p
                _mm256_storeu_si256(dp.add(off + i).cast(), m31_min_chain256(sum, p));
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_fold256(_mm256_loadu_si256(dp.add(off).cast()), p);
            let s = m31_fold256(_mm256_loadu_si256(sp.add(off).cast()), p);
            let sum = _mm256_add_epi32(d, s);
            _mm256_storeu_si256(dp.add(off).cast(), m31_min_chain256(sum, p));
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
    let fast = len & !127;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let d = m31_fold256(_mm256_loadu_si256(dp.add(off + i).cast()), p);
                let s = m31_fold256(_mm256_loadu_si256(sp.add(off + i).cast()), p);
                let diff = _mm256_sub_epi32(d, s); // wraps iff negative
                let fixed = _mm256_min_epu32(diff, _mm256_add_epi32(diff, p));
                _mm256_storeu_si256(dp.add(off + i).cast(), m31_min_chain256(fixed, p));
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_fold256(_mm256_loadu_si256(dp.add(off).cast()), p);
            let s = m31_fold256(_mm256_loadu_si256(sp.add(off).cast()), p);
            let diff = _mm256_sub_epi32(d, s);
            let fixed = _mm256_min_epu32(diff, _mm256_add_epi32(diff, p));
            _mm256_storeu_si256(dp.add(off).cast(), m31_min_chain256(fixed, p));
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
    let fast = len & !127;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cvec = _mm256_set1_epi32(coeff as i32);
    let cred = m31_min_chain256(m31_fold256(cvec, p), p);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let prod = m31_mulmod256(
                    cred,
                    m31_fold256(_mm256_loadu_si256(sp.add(off + i).cast()), p),
                    p,
                );
                let d = m31_fold256(_mm256_loadu_si256(dp.add(off + i).cast()), p);
                let sum = _mm256_add_epi32(d, prod); // <= 2p - 1
                let u = _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p));
                _mm256_storeu_si256(dp.add(off + i).cast(), u);
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let prod = m31_mulmod256(
                cred,
                m31_fold256(_mm256_loadu_si256(sp.add(off).cast()), p),
                p,
            );
            let d = m31_fold256(_mm256_loadu_si256(dp.add(off).cast()), p);
            let sum = _mm256_add_epi32(d, prod);
            let u = _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p));
            _mm256_storeu_si256(dp.add(off).cast(), u);
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
    let fast = len & !127;
    let dp = dst.as_mut_ptr();
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cred = m31_min_chain256(m31_fold256(_mm256_set1_epi32(coeff as i32), p), p);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let d = m31_fold256(_mm256_loadu_si256(dp.add(off + i).cast()), p);
                _mm256_storeu_si256(dp.add(off + i).cast(), m31_mulmod256(cred, d, p));
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len().
        unsafe {
            let d = m31_fold256(_mm256_loadu_si256(dp.add(off).cast()), p);
            _mm256_storeu_si256(dp.add(off).cast(), m31_mulmod256(cred, d, p));
        }
        off += 32;
    }
    prime::mul_assign::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff));
}

/// `dst = coeff * src (mod p)`, Mersenne31, AVX2. `coeff` must be canonical.
pub fn mul_into_m31_avx2(dst: &mut [u8], coeff: u32, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_into_m31_avx2_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_into_m31_avx2_impl(dst: &mut [u8], coeff: u32, src: &[u8]) {
    let len = dst.len() & !31;
    let fast = len & !127;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    // The coefficient is loop-invariant: canonicalize it once. Raw
    // (non-canonical) coefficient lanes stay legal inputs.
    let cred = m31_min_chain256(m31_fold256(_mm256_set1_epi32(coeff as i32), p), p);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let s = m31_fold256(_mm256_loadu_si256(sp.add(off + i).cast()), p);
                _mm256_storeu_si256(dp.add(off + i).cast(), m31_mulmod256(cred, s, p));
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let s = m31_fold256(_mm256_loadu_si256(sp.add(off).cast()), p);
            _mm256_storeu_si256(dp.add(off).cast(), m31_mulmod256(cred, s, p));
        }
        off += 32;
    }
    // Scalar tail.
    let dt = &mut dst[off..];
    let st = &src[off..];
    for (d, s) in dt.chunks_exact_mut(4).zip(st.chunks_exact(4)) {
        let v =
            mersenne31::Elem(coeff) * mersenne31::Elem(u32::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
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
    let fast = len & !127;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let p = _mm256_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 128 <= fast <= dst.len() == a.len() == b.len().
        unsafe {
            for i in [0, 32, 64, 96] {
                let va = m31_fold256(_mm256_loadu_si256(ap.add(off + i).cast()), p);
                let vb = m31_fold256(_mm256_loadu_si256(bp.add(off + i).cast()), p);
                _mm256_storeu_si256(dp.add(off + i).cast(), m31_mulmod256(va, vb, p));
            }
        }
        off += 128;
    }
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = m31_fold256(_mm256_loadu_si256(ap.add(off).cast()), p);
            let vb = m31_fold256(_mm256_loadu_si256(bp.add(off).cast()), p);
            _mm256_storeu_si256(dp.add(off).cast(), m31_mulmod256(va, vb, p));
        }
        off += 32;
    }
    prime::mul_elementwise::<Mersenne31>(&mut dst[off..], &a[off..], &b[off..]);
}

// ===========================================================================
// Mersenne31 (u32 lanes) — SSE4.2
#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_fold128(x: __m128i, p: __m128i) -> __m128i {
    let lo = _mm_and_si128(x, p);
    let hi = _mm_srli_epi32(x, 31);
    _mm_add_epi32(lo, hi)
}

/// Canonicalize lanes in `0..=2p`: two unconditional subtract-and-min steps.
#[inline]
#[target_feature(enable = "sse4.2")]
fn m31_min_chain128(x: __m128i, p: __m128i) -> __m128i {
    let u = _mm_min_epu32(x, _mm_sub_epi32(x, p));
    _mm_min_epu32(u, _mm_sub_epi32(u, p))
}

/// `a * b (mod p)` over four 31-bit lanes (`0..=p`; fold output allowed).
/// Doubled-odd-limb form; the odd-lane gather of `b` is a port-5 shuffle.
#[inline]
#[target_feature(enable = "sse4.2")]
pub(crate) fn m31_mulmod128(a: __m128i, b: __m128i, p: __m128i) -> __m128i {
    let a_odd = _mm_srli_epi64::<31>(a);
    let b_odd = _mm_shuffle_epi32::<0b11_11_01_01>(b);
    let prod_odd_dbl = _mm_mul_epu32(a_odd, b_odd);
    let prod_evn = _mm_mul_epu32(a, b);
    let prod_odd_lo = _mm_slli_epi64::<31>(prod_odd_dbl);
    let prod_evn_hi = _mm_srli_epi64::<31>(prod_evn);
    // SAFETY: `sse4.2` is enabled on this function.
    let (lo_dirty, hi) = unsafe {
        (
            _mm_blend_epi32::<0b1010>(prod_evn, prod_odd_lo),
            _mm_blend_epi32::<0b1010>(prod_evn_hi, prod_odd_dbl),
        )
    };
    let lo = _mm_and_si128(lo_dirty, p);
    let t = _mm_add_epi32(lo, hi);
    _mm_min_epu32(t, _mm_sub_epi32(t, p))
}

/// `dst += src (mod p)`, Mersenne31, SSE4.2.
pub fn add_assign_m31_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { add_assign_m31_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn add_assign_m31_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let d = m31_fold128(_mm_loadu_si128(dp.add(off + i).cast()), p);
                let s = m31_fold128(_mm_loadu_si128(sp.add(off + i).cast()), p);
                let sum = _mm_add_epi32(d, s); // <= 2p
                _mm_storeu_si128(dp.add(off + i).cast(), m31_min_chain128(sum, p));
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_fold128(_mm_loadu_si128(dp.add(off).cast()), p);
            let s = m31_fold128(_mm_loadu_si128(sp.add(off).cast()), p);
            let sum = _mm_add_epi32(d, s);
            _mm_storeu_si128(dp.add(off).cast(), m31_min_chain128(sum, p));
        }
        off += 16;
    }
    prime::add_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst -= src (mod p)`, Mersenne31, SSE4.2.
pub fn sub_assign_m31_sse41(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { sub_assign_m31_sse41_impl(dst, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn sub_assign_m31_sse41_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let d = m31_fold128(_mm_loadu_si128(dp.add(off + i).cast()), p);
                let s = m31_fold128(_mm_loadu_si128(sp.add(off + i).cast()), p);
                let diff = _mm_sub_epi32(d, s); // wraps iff negative
                let fixed = _mm_min_epu32(diff, _mm_add_epi32(diff, p));
                _mm_storeu_si128(dp.add(off + i).cast(), m31_min_chain128(fixed, p));
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let d = m31_fold128(_mm_loadu_si128(dp.add(off).cast()), p);
            let s = m31_fold128(_mm_loadu_si128(sp.add(off).cast()), p);
            let diff = _mm_sub_epi32(d, s);
            let fixed = _mm_min_epu32(diff, _mm_add_epi32(diff, p));
            _mm_storeu_si128(dp.add(off).cast(), m31_min_chain128(fixed, p));
        }
        off += 16;
    }
    prime::sub_assign::<Mersenne31>(&mut dst[off..], &src[off..]);
}

/// `dst += coeff * src (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
pub fn mul_add_m31_sse41(dst: &mut [u8], coeff: u32, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { mul_add_m31_sse41_impl(dst, coeff, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_add_m31_sse41_impl(dst: &mut [u8], coeff: u32, src: &[u8]) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    // The coefficient is loop-invariant and contract-canonical: fold it once.
    let cred = {
        let cvec = _mm_set1_epi32(coeff as i32);
        let f = m31_fold128(cvec, p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let prod = m31_mulmod128(
                    cred,
                    m31_fold128(_mm_loadu_si128(sp.add(off + i).cast()), p),
                    p,
                );
                let d = m31_fold128(_mm_loadu_si128(dp.add(off + i).cast()), p);
                let sum = _mm_add_epi32(d, prod); // <= 2p - 1
                let u = _mm_min_epu32(sum, _mm_sub_epi32(sum, p));
                _mm_storeu_si128(dp.add(off + i).cast(), u);
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let prod = m31_mulmod128(cred, m31_fold128(_mm_loadu_si128(sp.add(off).cast()), p), p);
            let d = m31_fold128(_mm_loadu_si128(dp.add(off).cast()), p);
            let sum = _mm_add_epi32(d, prod);
            let u = _mm_min_epu32(sum, _mm_sub_epi32(sum, p));
            _mm_storeu_si128(dp.add(off).cast(), u);
        }
        off += 16;
    }
    prime::mul_add::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff), &src[off..]);
}

/// `dst *= coeff (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
pub fn mul_assign_m31_sse41(dst: &mut [u8], coeff: u32) {
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { mul_assign_m31_sse41_impl(dst, coeff) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_assign_m31_sse41_impl(dst: &mut [u8], coeff: u32) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let dp = dst.as_mut_ptr();
    let p = _mm_set1_epi32(M31_P);
    let cred = {
        let f = m31_fold128(_mm_set1_epi32(coeff as i32), p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let d = m31_fold128(_mm_loadu_si128(dp.add(off + i).cast()), p);
                _mm_storeu_si128(dp.add(off + i).cast(), m31_mulmod128(cred, d, p));
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len().
        unsafe {
            let d = m31_fold128(_mm_loadu_si128(dp.add(off).cast()), p);
            _mm_storeu_si128(dp.add(off).cast(), m31_mulmod128(cred, d, p));
        }
        off += 16;
    }
    prime::mul_assign::<Mersenne31>(&mut dst[off..], mersenne31::Elem(coeff));
}

/// `dst = coeff * src (mod p)`, Mersenne31, SSE4.2. `coeff` must be canonical.
pub fn mul_into_m31_sse41(dst: &mut [u8], coeff: u32, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { mul_into_m31_sse41_impl(dst, coeff, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_into_m31_sse41_impl(dst: &mut [u8], coeff: u32, src: &[u8]) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let cred = {
        let f = m31_fold128(_mm_set1_epi32(coeff as i32), p);
        let u = _mm_min_epu32(f, _mm_sub_epi32(f, p));
        _mm_min_epu32(u, _mm_sub_epi32(u, p))
    };
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len() == src.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let s = m31_fold128(_mm_loadu_si128(sp.add(off + i).cast()), p);
                _mm_storeu_si128(dp.add(off + i).cast(), m31_mulmod128(cred, s, p));
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == src.len().
        unsafe {
            let s = m31_fold128(_mm_loadu_si128(sp.add(off).cast()), p);
            _mm_storeu_si128(dp.add(off).cast(), m31_mulmod128(cred, s, p));
        }
        off += 16;
    }
    // Scalar tail.
    let dt = &mut dst[off..];
    let st = &src[off..];
    for (d, s) in dt.chunks_exact_mut(4).zip(st.chunks_exact(4)) {
        let v =
            mersenne31::Elem(coeff) * mersenne31::Elem(u32::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, SSE4.2.
pub fn mul_elementwise_m31_sse41(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { mul_elementwise_m31_sse41_impl(dst, a, b) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_elementwise_m31_sse41_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len() & !15;
    let fast = len & !63;
    let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let p = _mm_set1_epi32(M31_P);
    let mut off = 0;
    while off < fast {
        // SAFETY: off + 64 <= fast <= dst.len() == a.len() == b.len().
        unsafe {
            for i in [0, 16, 32, 48] {
                let va = m31_fold128(_mm_loadu_si128(ap.add(off + i).cast()), p);
                let vb = m31_fold128(_mm_loadu_si128(bp.add(off + i).cast()), p);
                _mm_storeu_si128(dp.add(off + i).cast(), m31_mulmod128(va, vb, p));
            }
        }
        off += 64;
    }
    while off < len {
        // SAFETY: off + 16 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let va = m31_fold128(_mm_loadu_si128(ap.add(off).cast()), p);
            let vb = m31_fold128(_mm_loadu_si128(bp.add(off).cast()), p);
            _mm_storeu_si128(dp.add(off).cast(), m31_mulmod128(va, vb, p));
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

// Shifted-domain helpers. Values XORed with 2^63 let signed `vpcmpgtq`
// emulate unsigned compares; the bias cancels when two shifted values add,
// so intermediate results stay shifted through whole loop bodies. The wrap
// correction adds `0xFFFFFFFF` (= `-p mod 2^64`) via `srli_epi64::<32>` of
// the all-ones compare mask, replacing a separate AND with the epsilon.
const GLD_SIGN: i64 = i64::MIN;
const GLD_SHIFTED_P: i64 = 0x7FFF_FFFF_0000_0001_u64 as i64;

/// Canonicalize shifted lanes in `0..2p` (one conditional subtract),
/// returning shifted lanes in `0..p`.
#[inline]
#[target_feature(enable = "avx2")]
fn gld_canon_s(x_s: __m256i, sp: __m256i, eps: __m256i) -> __m256i {
    // All-ones where x_s < p (already canonical), zero otherwise.
    let mask = _mm256_cmpgt_epi64(sp, x_s);
    // Where not canonical: add 0xFFFFFFFF ≡ -p.
    _mm256_add_epi64(x_s, _mm256_andnot_si256(mask, eps))
}

/// Full 64x64 -> 128 product over four lanes via three `vpmuludq` pairs;
/// the high-limb gathers ride the float-domain shuffle port.
/// Returns `(hi, lo)`.
#[inline]
#[target_feature(enable = "avx2")]
fn gld_mul_wide(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
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
#[inline]
#[target_feature(enable = "avx2")]
fn gld_reduce128_s(hi: __m256i, lo: __m256i, eps: __m256i) -> __m256i {
    let lo_s = _mm256_xor_si256(lo, _mm256_set1_epi64x(GLD_SIGN));
    let hi_hi = _mm256_srli_epi64::<32>(hi);
    // lo - hi_hi with borrow correction (hi_hi <= 0xffffffff).
    let sub_wrapped = _mm256_sub_epi64(lo_s, hi_hi);
    let sub_mask = _mm256_cmpgt_epi32(sub_wrapped, lo_s);
    let folded_s = _mm256_add_epi64(sub_wrapped, _mm256_srli_epi64::<32>(sub_mask));
    // + hi_lo * epsilon; `vpmuludq` reads only the low dwords.
    let t1 = _mm256_mul_epu32(hi, eps);
    // lo1 + t1 with carry correction (t1 <= 0xffffffff00000000).
    let add_wrapped = _mm256_add_epi64(folded_s, t1);
    let add_mask = _mm256_cmpgt_epi32(folded_s, add_wrapped);
    _mm256_add_epi64(add_wrapped, _mm256_srli_epi64::<32>(add_mask))
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
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(GLD_SIGN);
    let sp_c = _mm256_set1_epi64x(GLD_SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            // Canonical dst: shifted compare, then an exact unshift (XOR is
            // involutive even across the 2^64 wrap of the shifted domain).
            let d_s = gld_canon_s(
                _mm256_xor_si256(_mm256_loadu_si256(dp.add(off).cast()), sign),
                sp_c,
                eps,
            );
            let d = _mm256_xor_si256(d_s, sign);
            let (hi, lo) = gld_mul_wide(cvec, _mm256_loadu_si256(sp.add(off).cast()));
            let mut red_s = gld_reduce128_s(hi, lo, eps);
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
            _mm256_storeu_si256(dp.add(off).cast(), res);
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
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(GLD_SIGN);
    let sp_c = _mm256_set1_epi64x(GLD_SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len().
        unsafe {
            let (hi, lo) = gld_mul_wide(cvec, _mm256_loadu_si256(dp.add(off).cast()));
            let res_s = gld_reduce128_s(hi, lo, eps);
            let mask = _mm256_cmpgt_epi64(sp_c, res_s);
            let res = _mm256_xor_si256(
                _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps)),
                sign,
            );
            _mm256_storeu_si256(dp.add(off).cast(), res);
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
    let eps = _mm256_set1_epi64x(EPS);
    let sign = _mm256_set1_epi64x(GLD_SIGN);
    let sp_c = _mm256_set1_epi64x(GLD_SHIFTED_P);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == a.len() == b.len().
        unsafe {
            let (hi, lo) = gld_mul_wide(
                _mm256_loadu_si256(ap.add(off).cast()),
                _mm256_loadu_si256(bp.add(off).cast()),
            );
            let res_s = gld_reduce128_s(hi, lo, eps);
            let mask = _mm256_cmpgt_epi64(sp_c, res_s);
            let res = _mm256_xor_si256(
                _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps)),
                sign,
            );
            _mm256_storeu_si256(dp.add(off).cast(), res);
        }
        off += 32;
    }
    prime::mul_elementwise::<Goldilocks>(&mut dst[off..], &a[off..], &b[off..]);
}

/// `dst = coeff * src (mod p)`, Goldilocks, AVX2.
pub fn mul_into_gld_avx2(dst: &mut [u8], coeff: u64, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected an AVX2 backend; slices are equal-length.
    unsafe { mul_into_gld_avx2_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_into_gld_avx2_impl(dst: &mut [u8], coeff: u64, src: &[u8]) {
    let len = dst.len() & !31;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    let eps = _mm256_set1_epi64x(EPS);
    let sp_c = _mm256_set1_epi64x(GLD_SHIFTED_P);
    let cvec = _mm256_set1_epi64x(coeff as i64);
    let mut off = 0;
    while off < len {
        // SAFETY: off + 32 <= len <= dst.len() == src.len().
        unsafe {
            let s = _mm256_loadu_si256(sp.add(off).cast());
            let (hi, lo) = gld_mul_wide(cvec, s);
            let res_s = gld_reduce128_s(hi, lo, eps);
            // One conditional subtract, then back out of the shifted domain.
            let mask = _mm256_cmpgt_epi64(sp_c, res_s);
            let res = _mm256_add_epi64(res_s, _mm256_andnot_si256(mask, eps));
            let res = _mm256_xor_si256(res, _mm256_set1_epi64x(GLD_SIGN));
            _mm256_storeu_si256(dp.add(off).cast(), res);
        }
        off += 32;
    }
    // Scalar tail.
    let dt = &mut dst[off..];
    let st = &src[off..];
    for (d, s) in dt.chunks_exact_mut(8).zip(st.chunks_exact(8)) {
        let v =
            goldilocks::Elem(coeff) * goldilocks::Elem(u64::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
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

/// `dst = coeff * src (mod p)`, Goldilocks, SSE4.2.
pub fn mul_into_gld_sse41(dst: &mut [u8], coeff: u64, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: caller selected the V2 backend (x86-64-v2, hence SSE4.2).
    unsafe { mul_into_gld_sse41_impl(dst, coeff, src) }
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_into_gld_sse41_impl(dst: &mut [u8], coeff: u64, src: &[u8]) {
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
            _mm_storeu_si128(dp.add(off).cast(), gld_mul128(cvec, s, m32, eps, pcst, pm1));
        }
        off += 16;
    }
    // Scalar tail.
    let dt = &mut dst[off..];
    let st = &src[off..];
    for (d, s) in dt.chunks_exact_mut(8).zip(st.chunks_exact(8)) {
        let v =
            goldilocks::Elem(coeff) * goldilocks::Elem(u64::from_le_bytes(s.try_into().unwrap()));
        d.copy_from_slice(&v.to_raw().to_le_bytes());
    }
}
