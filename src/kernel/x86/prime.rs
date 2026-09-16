//! Prime-field integer-SIMD kernels for x86 / `x86_64`.
//!
//! Two lane widths, one reduction discipline each — plus the quadratic
//! extension field built over pairs of the Mersenne31 lanes:
//!
//! - **Mersenne31 (`u32` lanes).** `2^31 ≡ 1 (mod p)`, so a multiply folds the
//!   62-bit product with a mask/shift/add and one conditional subtract; the
//!   products come from `vpmuludq` on the even and odd 32-bit lanes.
//! - **Goldilocks (`u64` lanes).** A 64×64→128 product built from four
//!   `vpmuludq` 32-bit half-products, then the split-fold reduction
//!   (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`). Unsigned 64-bit compares are synthesized
//!   from `vpcmpgtq` with a sign-bit flip, since AVX2 has no native one.
//! - **`QuadMersenne31` (`(re, im)` Mersenne31 limb pairs).** Four complex
//!   elements per vector over the Mersenne31 lane discipline; a complex
//!   product is two limb multiplies against a sign-mixed coefficient vector.
//!
//! The `QuadMersenne31` AVX2 file sits outside process dispatch, which
//! serves the field from the portable path; its kernels are reached only
//! through the direct entries.
//!
//! Every kernel processes whole vector lanes and hands the sub-lane remainder
//! to the portable [`crate::kernel::prime`] path, which is also the differential
//! oracle. All lane operations are validated against a `u128 % p` model. The
//! AVX2 kernels serve `V3`/`V3GfniCrypto`; the SSE4.2 kernels serve `V2`
//! (x86-64-v2 guarantees SSE4.1 `pminud` and SSE4.2 `pcmpgtq`).
//!
//! Every entry is a safe capability-token function —
//! `#[arcane(import_intrinsics)]` over reference-based loads and stores —
//! taking the exact token its instructions require: `X64V3Token` for the AVX2
//! files, `X64V2Token` for the SSE4.2 files. Each entry validates its own
//! geometry (paired lengths, whole elements) before the first lane; lane
//! helpers are `#[rite]` functions. No kernel detects or summons; the one
//! unsafe residue is the SSE blend whose intrinsic feature table overstates
//! what a `V2` host needs.
//!
//! # Layout
//!
//! The submodules split on **field, then vector width** — the two axes that
//! decide the reduction strategy and the intrinsic set respectively. Each one
//! owns its width-specific lane helpers (fold, canonicalize, multiply); the
//! modulus constants below are shared, as is the 128-bit Mersenne31 multiply
//! the SSE file builds on and the geometry checks every entry opens with.
//!
//! | Module | Field | Width | Backend |
//! | --- | --- | --- | --- |
//! | `m31_avx2` | Mersenne31 | 256-bit | `V3`/`V3GfniCrypto` |
//! | `m31_sse` | Mersenne31 | 128-bit | `V2` |
//! | `gld_avx2` | Goldilocks | 256-bit | `V3`/`V3GfniCrypto` |
//! | `gld_sse` | Goldilocks | 128-bit | `V2` |
//! | `qm31_avx2` | QuadMersenne31 | 256-bit | `V3`/`V3GfniCrypto` |

mod gld_avx2;
mod gld_sse;
mod m31_avx2;
mod m31_sse;
mod qm31_avx2;

pub use gld_avx2::{
    add_assign_gld_avx2, mul_add_gld_avx2, mul_assign_gld_avx2, mul_elementwise_gld_avx2,
    mul_into_gld_avx2, sub_assign_gld_avx2,
};
pub use gld_sse::{
    add_assign_gld_sse42, mul_add_gld_sse42, mul_assign_gld_sse42, mul_elementwise_gld_sse42,
    mul_into_gld_sse42, sub_assign_gld_sse42,
};
pub use m31_avx2::{
    add_assign_m31_avx2, mul_add_m31_avx2, mul_assign_m31_avx2, mul_elementwise_m31_avx2,
    mul_into_m31_avx2, sub_assign_m31_avx2,
};
pub use m31_sse::{
    add_assign_m31_sse42, mul_add_m31_sse42, mul_assign_m31_sse42, mul_elementwise_m31_sse42,
    mul_into_m31_sse42, sub_assign_m31_sse42,
};
pub use qm31_avx2::{
    add_assign_qm31_avx2, mul_add_qm31_avx2, mul_assign_qm31_avx2, mul_elementwise_qm31_avx2,
    mul_into_qm31_avx2, sub_assign_qm31_avx2,
};

use crate::field::{goldilocks, mersenne31};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[allow(clippy::cast_possible_wrap)]
const M31_P: i32 = mersenne31::MODULUS as i32; // 0x7FFF_FFFF
#[allow(clippy::cast_possible_wrap)]
const GLD_P: i64 = goldilocks::MODULUS as i64; // 0xFFFF_FFFF_0000_0001
#[allow(clippy::cast_possible_wrap)]
const GLD_PM1: i64 = (goldilocks::MODULUS - 1) as i64; // 0xFFFF_FFFF_0000_0000
const EPS: i64 = 0xFFFF_FFFF; // 2^32 - 1

/// Paired length equality, with the facade's operand-naming message.
///
/// Every entry asserts its buffer pairs here, so a direct kernel call
/// validates exactly like the [`crate::ops`] facade and the `internals`
/// surface do.
#[inline]
fn check_equal(name: &str, left_name: &str, left: usize, right_name: &str, right: usize) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
}

/// Whole-element lengths, with the facade's element-naming message.
#[inline]
fn check_elem_multiple(name: &str, len: usize, elem_bytes: usize) {
    assert!(
        len.is_multiple_of(elem_bytes),
        "{name}: buffer of {len} bytes is not a whole number of {elem_bytes}-byte elements"
    );
}

/// `a * b (mod p)` over four 31-bit lanes (`0..=p`; fold output allowed).
/// Doubled-odd-limb form; the odd-lane gather of `b` is a port-5 shuffle.
///
/// `_mm_blend_epi16::<0b1100_1100>` selects both 16-bit halves of each odd
/// 32-bit lane. That is the SSE4.1 spelling of the same lane selection as a
/// 32-bit blend, without the standard library's stricter AVX2 gate.
#[archmage::rite(v2)]
fn m31_mulmod128(a: __m128i, b: __m128i, p: __m128i) -> __m128i {
    let a_odd = _mm_srli_epi64::<31>(a);
    let b_odd = _mm_shuffle_epi32::<0b11_11_01_01>(b);
    let prod_odd_dbl = _mm_mul_epu32(a_odd, b_odd);
    let prod_evn = _mm_mul_epu32(a, b);
    let prod_odd_lo = _mm_slli_epi64::<31>(prod_odd_dbl);
    let prod_evn_hi = _mm_srli_epi64::<31>(prod_evn);
    let lo_dirty = _mm_blend_epi16::<0b1100_1100>(prod_evn, prod_odd_lo);
    let hi = _mm_blend_epi16::<0b1100_1100>(prod_evn_hi, prod_odd_dbl);
    let lo = _mm_and_si128(lo_dirty, p);
    let t = _mm_add_epi32(lo, hi);
    _mm_min_epu32(t, _mm_sub_epi32(t, p))
}
