//! Mersenne31 kernels over 512-bit lanes (`V4x`).
//!
//! Lane-for-lane ports of the `avx2` entries at sixteen 32-bit lanes per
//! vector: the same Mersenne fold and the same `vpminud` canonicalization, so
//! every lane produces the bytes the AVX2 kernel produces. The multiply
//! recombination differs from the AVX2 blend form: both `a` limb sets are
//! doubled, the odd `b` limbs are gathered with `vmovehdup`, and masked
//! `vmovsldup`/`vmovshdup` recombination pairs each doubled partial product
//! for one right shift and one conditional subtract. Only AVX-512F
//! instructions are used, so each entry takes `X64V4Token`. Complete 64-byte
//! lanes run here; the remainder descends through the matching AVX2 entry,
//! which carries its own scalar tail.

use super::{
    P, add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
use crate::field::mersenne31;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Odd 32-bit lanes of a 512-bit vector, as a blend mask.
const ODD_LANES: __mmask16 = 0xAAAA;

#[archmage::rite(v4)]
fn fold(x: __m512i, p: __m512i) -> __m512i {
    _mm512_add_epi32(_mm512_and_si512(x, p), _mm512_srli_epi32::<31>(x))
}

#[archmage::rite(v4)]
fn min_chain(x: __m512i, p: __m512i) -> __m512i {
    let u = _mm512_min_epu32(x, _mm512_sub_epi32(x, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// `a * b (mod p)` over sixteen 31-bit lanes (`0..=p` in, canonical out).
#[archmage::rite(v4)]
fn mulmod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let a_even_dbl = _mm512_add_epi32(a, a);
    let a_odd_dbl = _mm512_srli_epi64::<31>(a);
    let b_odd = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(b)));
    let even = _mm512_mul_epu32(a_even_dbl, b);
    let odd = _mm512_mul_epu32(a_odd_dbl, b_odd);
    let lo_dbl = _mm512_castps_si512(_mm512_mask_moveldup_ps(
        _mm512_castsi512_ps(even),
        ODD_LANES,
        _mm512_castsi512_ps(odd),
    ));
    let hi = _mm512_castps_si512(_mm512_mask_movehdup_ps(
        _mm512_castsi512_ps(odd),
        !ODD_LANES,
        _mm512_castsi512_ps(even),
    ));
    let lo = _mm512_srli_epi32::<1>(lo_dbl);
    let t = _mm512_add_epi32(lo, hi);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// The canonical broadcast of a raw coefficient or value word.
#[allow(clippy::cast_possible_wrap)]
#[archmage::rite(v4)]
fn broadcast(word: u32, p: __m512i) -> __m512i {
    min_chain(fold(_mm512_set1_epi32(word as i32), p), p)
}

/// `dst += src (mod p)`, Mersenne31, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "mersenne31::add_assign_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("mersenne31::add_assign_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = fold(_mm512_loadu_si512(&*dst_lane), p);
        let s = fold(_mm512_loadu_si512(src_lane), p);
        _mm512_storeu_si512(dst_lane, min_chain(_mm512_add_epi32(d, s), p));
    }
    if !dst_tail.is_empty() {
        add_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst -= src (mod p)`, Mersenne31, AVX-512.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "mersenne31::sub_assign_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("mersenne31::sub_assign_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = fold(_mm512_loadu_si512(&*dst_lane), p);
        let s = fold(_mm512_loadu_si512(src_lane), p);
        let diff = _mm512_sub_epi32(d, s); // wraps iff negative
        let fixed = _mm512_min_epu32(diff, _mm512_add_epi32(diff, p));
        _mm512_storeu_si512(dst_lane, min_chain(fixed, p));
    }
    if !dst_tail.is_empty() {
        sub_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst += coeff * src (mod p)`, Mersenne31, AVX-512. The `dst` and `src`
/// lanes must be canonical (`0..p`); `coeff` is canonicalized on entry.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal(
        "mersenne31::mul_add_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("mersenne31::mul_add_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let cred = broadcast(coeff, p);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let prod = mulmod(cred, _mm512_loadu_si512(src_lane), p);
        let d = _mm512_loadu_si512(&*dst_lane);
        let sum = _mm512_add_epi32(d, prod); // <= 2p - 1
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
    }
    if !dst_tail.is_empty() {
        mul_add_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst *= coeff (mod p)`, Mersenne31, AVX-512. The `dst` lanes must be
/// canonical (`0..p`); `coeff` is canonicalized on entry.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u32) {
    check_elem_multiple("mersenne31::mul_assign_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let cred = broadcast(coeff, p);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = _mm512_loadu_si512(&*dst_lane);
        _mm512_storeu_si512(dst_lane, mulmod(cred, d, p));
    }
    if !dst_tail.is_empty() {
        mul_assign_avx2(token.v3(), dst_tail, coeff);
    }
}

/// `dst = coeff * src (mod p)`, Mersenne31, AVX-512. The `src` lanes must be
/// canonical (`0..p`); `coeff` is canonicalized on entry.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u32, src: &[u8]) {
    check_equal(
        "mersenne31::mul_into_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("mersenne31::mul_into_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let cred = broadcast(coeff, p);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm512_loadu_si512(src_lane);
        _mm512_storeu_si512(dst_lane, mulmod(cred, s, p));
    }
    if !dst_tail.is_empty() {
        mul_into_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX-512. The `a` and `b`
/// lanes must be canonical (`0..p`).
///
/// # Panics
/// Panics unless all three buffers match in length (whole lanes).
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx512(token: archmage::X64V4Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "mersenne31::mul_elementwise_avx512",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("mersenne31::mul_elementwise_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a.as_chunks::<64>();
    let (b_lanes, b_tail) = b.as_chunks::<64>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = _mm512_loadu_si512(a_lane);
        let vb = _mm512_loadu_si512(b_lane);
        _mm512_storeu_si512(dst_lane, mulmod(va, vb, p));
    }
    if !dst_tail.is_empty() {
        mul_elementwise_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Mersenne31, AVX-512. The `dst` and
/// `src` lanes must be canonical (`0..p`).
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx512(token: archmage::X64V4Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "mersenne31::mul_elementwise_assign_avx512",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("mersenne31::mul_elementwise_assign_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let vd = _mm512_loadu_si512(&*dst_lane);
        let vs = _mm512_loadu_si512(src_lane);
        _mm512_storeu_si512(dst_lane, mulmod(vd, vs, p));
    }
    if !dst_tail.is_empty() {
        mul_elementwise_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst[i] += value (mod p)`, Mersenne31, AVX-512. The raw `value` word is
/// canonicalized once on entry.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u32) {
    check_elem_multiple("mersenne31::add_assign_scalar_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let svec = _mm512_set1_epi32(mersenne31::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = fold(_mm512_loadu_si512(&*dst_lane), p);
        _mm512_storeu_si512(dst_lane, min_chain(_mm512_add_epi32(d, svec), p));
    }
    if !dst_tail.is_empty() {
        add_assign_scalar_avx2(token.v3(), dst_tail, value);
    }
}

/// `dst[i] -= value (mod p)`, Mersenne31, AVX-512. The raw `value` word is
/// canonicalized once on entry.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u32) {
    check_elem_multiple("mersenne31::sub_assign_scalar_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let svec = _mm512_set1_epi32(mersenne31::Elem(value).canonical().to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = fold(_mm512_loadu_si512(&*dst_lane), p);
        let diff = _mm512_sub_epi32(d, svec);
        let fixed = _mm512_min_epu32(diff, _mm512_add_epi32(diff, p));
        _mm512_storeu_si512(dst_lane, min_chain(fixed, p));
    }
    if !dst_tail.is_empty() {
        sub_assign_scalar_avx2(token.v3(), dst_tail, value);
    }
}
