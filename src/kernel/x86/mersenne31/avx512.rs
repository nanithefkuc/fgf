//! Mersenne31 kernels over 512-bit lanes (`V4x`).
//!
//! Buffers use `fgf`'s little-endian 32-bit field-lane layout. Raw lanes
//! normalize to the multiplication input range before their products are
//! computed; stored products are canonical.
//!
//! Multiplication doubles the left limbs and interleaves the low and high
//! partial-product words with fixed two-source permutations before the
//! Mersenne fold. Multiply-accumulate groups independent vectors without
//! sharing their input-range decisions. Each entry takes `X64V4Token`;
//! complete 64-byte lanes run here, and smaller remainders descend through
//! the matching AVX2 entry and its scalar tail.

use super::{
    P, add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
use crate::field::Elem;
use crate::field::mersenne31::Mersenne31;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Canonicalize arbitrary raw `u32` lanes below `p` in two unconditional
/// subtract-and-min steps; every lane value fits `0..=2p + 1`
/// (`u32::MAX == 2p + 1`).
#[archmage::rite(v4)]
fn min_chain(x: __m512i, p: __m512i) -> __m512i {
    let u = _mm512_min_epu32(x, _mm512_sub_epi32(x, p));
    _mm512_min_epu32(u, _mm512_sub_epi32(u, p))
}

/// Reduce multiplication inputs to `0..=p`, retaining redundant zero.
///
/// Lanes without their high bit set already meet the product's range.
/// Other vectors canonicalize before the product, including raw aliases.
#[archmage::rite(v4)]
#[must_use]
fn mul_input(x: __m512i, p: __m512i) -> __m512i {
    if _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), x) == 0 {
        x
    } else {
        min_chain(x, p)
    }
}

/// `a * b (mod p)` over sixteen 31-bit lanes (`0..=p` in, canonical out).
#[archmage::rite(v4)]
fn mulmod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let a_even_dbl = _mm512_add_epi32(a, a);
    let a_odd_dbl = _mm512_srli_epi64::<31>(a);
    let b_odd = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(b)));
    let even = _mm512_mul_epu32(a_even_dbl, b);
    let odd = _mm512_mul_epu32(a_odd_dbl, b_odd);
    let lo_index = _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    let hi_index = _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    let lo_dbl = _mm512_permutex2var_epi32(even, lo_index, odd);
    let hi = _mm512_permutex2var_epi32(even, hi_index, odd);
    let lo = _mm512_srli_epi32::<1>(lo_dbl);
    let t = _mm512_add_epi32(lo, hi);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// The canonical broadcast of a raw coefficient or value word.
#[allow(clippy::cast_possible_wrap)]
#[archmage::rite(v4)]
fn broadcast(word: u32, p: __m512i) -> __m512i {
    mul_input(_mm512_set1_epi32(word as i32), p)
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
        // Canonical operands sum to at most `2p - 2`: one final
        // subtract-and-min completes the reduction.
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let s = min_chain(_mm512_loadu_si512(src_lane), p);
        let sum = _mm512_add_epi32(d, s); // <= 2p - 2
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
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
        // The unsigned min of the wrapping difference and its `+p` fixup is
        // already canonical for canonical operands.
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let s = min_chain(_mm512_loadu_si512(src_lane), p);
        let diff = _mm512_sub_epi32(d, s); // wraps iff negative
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(diff, _mm512_add_epi32(diff, p)));
    }
    if !dst_tail.is_empty() {
        sub_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst += coeff * src (mod p)`, Mersenne31, AVX-512.
///
/// A zero coefficient leaves `dst` untouched, mirroring the portable
/// reference. The source canonicalizes through `mul_input` and the
/// product stays canonical, so the destination needs only the
/// redundant range `0..=p` (retaining redundant zero): `prod + dst <=`
/// `2p - 1`, and the single trailing subtract-and-min canonicalizes
/// every raw lane bit pattern. Every stored lane is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial lane.
#[allow(clippy::cast_possible_wrap)]
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
    // Reduce the raw word once; the zero check runs on the value, matching
    // the portable reference.
    let folded = (coeff & crate::field::mersenne31::MODULUS) + (coeff >> 31);
    let c = if folded >= crate::field::mersenne31::MODULUS {
        folded - crate::field::mersenne31::MODULUS
    } else {
        folded
    };
    if c == 0 {
        return;
    }
    let p = _mm512_set1_epi32(P);
    let cred = _mm512_set1_epi32(c as i32);
    let (dst_tiles, dst_rest) = dst.as_chunks_mut::<256>();
    let (src_tiles, src_rest) = src.as_chunks::<256>();
    for (dst_tile, src_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (ds, _) = dst_tile.as_chunks_mut::<64>();
        let (ss, _) = src_tile.as_chunks::<64>();
        for i in 0..4 {
            let prod = mulmod(cred, mul_input(_mm512_loadu_si512(&ss[i]), p), p);
            let d = mul_input(_mm512_loadu_si512(&ds[i]), p);
            let sum = _mm512_add_epi32(d, prod);
            _mm512_storeu_si512(&mut ds[i], _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
        }
    }
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let prod = mulmod(cred, mul_input(_mm512_loadu_si512(src_lane), p), p);
        let d = mul_input(_mm512_loadu_si512(&*dst_lane), p);
        let sum = _mm512_add_epi32(d, prod); // <= 2p - 1
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
    }
    if !dst_tail.is_empty() {
        mul_add_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst *= coeff (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: lanes canonicalize on load,
/// and every stored lane is canonical.
///
/// # Panics
/// Panics on a partial trailing lane.
#[allow(clippy::cast_possible_wrap)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx512(token: archmage::X64V4Token, dst: &mut [u8], coeff: u32) {
    check_elem_multiple("mersenne31::mul_assign_avx512", dst.len(), 4);
    // Reduce the raw word once; the unit check runs on the value, matching
    // the portable reference.
    let folded = (coeff & crate::field::mersenne31::MODULUS) + (coeff >> 31);
    let c = if folded >= crate::field::mersenne31::MODULUS {
        folded - crate::field::mersenne31::MODULUS
    } else {
        folded
    };
    if c == 1 {
        return;
    }
    let p = _mm512_set1_epi32(P);
    let cred = _mm512_set1_epi32(c as i32);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = mul_input(_mm512_loadu_si512(&*dst_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(cred, d, p));
    }
    if !dst_tail.is_empty() {
        mul_assign_avx2(token.v3(), dst_tail, coeff);
    }
}

/// `dst = coeff * src (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: lanes canonicalize on load,
/// and every stored lane is canonical.
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
        let s = mul_input(_mm512_loadu_si512(src_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(cred, s, p));
    }
    if !dst_tail.is_empty() {
        mul_into_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: lanes canonicalize on load,
/// and every stored lane is canonical. Each 256-byte group ORs its eight
/// input lanes once and runs a single mask test: a clean group skips every
/// per-vector `mul_input` (high-bit-free lanes already meet `mulmod`'s
/// `0..=p` range), a dirty group falls back to the per-vector path. The
/// sub-group remainder keeps the original 64-byte AVX-512 vector loop; only
/// the sub-64-byte tail descends to AVX2.
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
    let zero = _mm512_setzero_si512();
    let (dst_groups, dst_rest) = dst.as_chunks_mut::<256>();
    let (a_groups, a_rest) = a.as_chunks::<256>();
    let (b_groups, b_rest) = b.as_chunks::<256>();
    for (dst_group, (a_group, b_group)) in dst_groups.iter_mut().zip(a_groups.iter().zip(b_groups))
    {
        let (dst_lanes, _) = dst_group.as_chunks_mut::<64>();
        let (a_lanes, _) = a_group.as_chunks::<64>();
        let (b_lanes, _) = b_group.as_chunks::<64>();
        let va = [
            _mm512_loadu_si512(&a_lanes[0]),
            _mm512_loadu_si512(&a_lanes[1]),
            _mm512_loadu_si512(&a_lanes[2]),
            _mm512_loadu_si512(&a_lanes[3]),
        ];
        let vb = [
            _mm512_loadu_si512(&b_lanes[0]),
            _mm512_loadu_si512(&b_lanes[1]),
            _mm512_loadu_si512(&b_lanes[2]),
            _mm512_loadu_si512(&b_lanes[3]),
        ];
        let acc = _mm512_or_si512(
            _mm512_or_si512(_mm512_or_si512(va[0], va[1]), _mm512_or_si512(va[2], va[3])),
            _mm512_or_si512(_mm512_or_si512(vb[0], vb[1]), _mm512_or_si512(vb[2], vb[3])),
        );
        if _mm512_cmpgt_epi32_mask(zero, acc) == 0 {
            for ((dst_lane, &x), &y) in dst_lanes.iter_mut().zip(va.iter()).zip(vb.iter()) {
                _mm512_storeu_si512(dst_lane, mulmod(x, y, p));
            }
        } else {
            for ((dst_lane, &x), &y) in dst_lanes.iter_mut().zip(va.iter()).zip(vb.iter()) {
                _mm512_storeu_si512(dst_lane, mulmod(mul_input(x, p), mul_input(y, p), p));
            }
        }
    }
    // Sub-group remainder keeps the original 64-byte AVX-512 vector loop;
    // only the sub-64-byte tail descends to AVX2.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a_rest.as_chunks::<64>();
    let (b_lanes, b_tail) = b_rest.as_chunks::<64>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = mul_input(_mm512_loadu_si512(a_lane), p);
        let vb = mul_input(_mm512_loadu_si512(b_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(va, vb, p));
    }
    if !dst_tail.is_empty() {
        mul_elementwise_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: lanes canonicalize on load,
/// and every stored lane is canonical.
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
        let vd = mul_input(_mm512_loadu_si512(&*dst_lane), p);
        let vs = mul_input(_mm512_loadu_si512(src_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(vd, vs, p));
    }
    if !dst_tail.is_empty() {
        mul_elementwise_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst[i] += value (mod p)`, Mersenne31, AVX-512. The raw `value` word is
/// canonicalized once on entry. The destination lanes canonicalize on
/// load; the canonical sum reduces with one subtract-and-min.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u32) {
    check_elem_multiple("mersenne31::add_assign_scalar_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let svec = _mm512_set1_epi32(Elem::<Mersenne31>::from_raw(value).to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let sum = _mm512_add_epi32(d, svec); // <= 2p - 2
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
    }
    if !dst_tail.is_empty() {
        add_assign_scalar_avx2(token.v3(), dst_tail, value);
    }
}

/// `dst[i] -= value (mod p)`, Mersenne31, AVX-512. The raw `value` word is
/// canonicalized once on entry. The destination lanes canonicalize on
/// load; the canonical difference is canonical after one unsigned min with
/// its `+p` fixup.
///
/// # Panics
/// Panics on a partial trailing lane.
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_avx512(token: archmage::X64V4Token, dst: &mut [u8], value: u32) {
    check_elem_multiple("mersenne31::sub_assign_scalar_avx512", dst.len(), 4);
    let p = _mm512_set1_epi32(P);
    let svec = _mm512_set1_epi32(Elem::<Mersenne31>::from_raw(value).to_raw().cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for dst_lane in dst_lanes {
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let diff = _mm512_sub_epi32(d, svec); // wraps iff negative
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(diff, _mm512_add_epi32(diff, p)));
    }
    if !dst_tail.is_empty() {
        sub_assign_scalar_avx2(token.v3(), dst_tail, value);
    }
}
