//! Mersenne31 kernels over 512-bit lanes (`V4x`).
//!
//! Lane-for-lane ports of the `avx2` entries at sixteen 32-bit lanes per
//! vector: the same direct `vpminud` canonicalization, so every lane
//! produces the bytes the AVX2 kernel produces. The multiply
//! recombination differs from the AVX2 blend form: both `a` limb sets are
//! doubled, the odd `b` limbs are gathered with `vmovehdup`, and two
//! two-source dword permutes (`vpermt2d`) pair each doubled partial product
//! for one right shift and one conditional subtract. Only AVX-512F
//! instructions are used, so each entry takes `X64V4Token`. Complete 64-byte
//! lanes run here; the remainder descends through the matching AVX2 entry,
//! which carries its own scalar tail.
//!
//! The multiply entries screen operand lanes for bit 31 instead of
//! canonicalizing every load: a lane with bit 31 clear lies in `0..=p`,
//! which [`mulmod`] and the accumulate reduction accept as loaded. The
//! out-of-place entries multiply optimistically and recompute every lane
//! through [`min_chain`] when any source lane carries bit 31; the in-place
//! entries load [`SCREEN_LANES`] lanes of each operand into registers and
//! canonicalize them only when one carries it, before any store.

use super::{
    P, add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
use crate::field::Elem;
use crate::field::mersenne31::Mersenne31;
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
#[cfg(test)]
use crate::kernel::trace::{Substage, Unit, emit};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Lanes per register group of the in-place multiplies: every operand
/// lane of a group is loaded and screened before the group's first store.
const SCREEN_LANES: usize = 4;

/// Whether any 32-bit lane of `seen` carries bit 31, the one bit outside
/// the `0..=p` operand range of [`mulmod`].
#[archmage::rite(v4)]
fn any_bit31(seen: __m512i) -> bool {
    _mm512_test_epi32_mask(seen, _mm512_set1_epi32(i32::MIN)) != 0
}

/// Canonicalize arbitrary raw `u32` lanes below `p` in two unconditional
/// subtract-and-min steps; every lane value fits `0..=2p + 1`
/// (`u32::MAX == 2p + 1`).
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
    // Lane `2i` takes a dword of the even product `i`, lane `2i + 1` the
    // same dword of the odd product `i`: the low dwords hold `2 * lo`, the
    // high dwords `hi`.
    let lo_dbl = _mm512_permutex2var_epi32(
        even,
        _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30),
        odd,
    );
    let hi = _mm512_permutex2var_epi32(
        even,
        _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31),
        odd,
    );
    let lo = _mm512_srli_epi32::<1>(lo_dbl);
    let t = _mm512_add_epi32(lo, hi);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// The canonical broadcast of a raw coefficient or value word.
#[allow(clippy::cast_possible_wrap)]
#[archmage::rite(v4)]
fn broadcast(word: u32, p: __m512i) -> __m512i {
    min_chain(_mm512_set1_epi32(word as i32), p)
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
    #[cfg(test)]
    emit(
        "mersenne31.avx512.add_assign",
        Substage::Add,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
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
    #[cfg(test)]
    emit(
        "mersenne31.avx512.sub_assign",
        Substage::Sub,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        sub_assign_avx2(token.v3(), dst_tail, src_tail);
    }
}

/// `dst += coeff * src (mod p)`, Mersenne31, AVX-512.
///
/// A zero coefficient leaves `dst` untouched, mirroring the portable
/// reference. Every input lane bit pattern is legal: a register group whose
/// lanes all have bit 31 clear multiplies as loaded, any other group and
/// the lanes past the last whole group canonicalize first, and every stored
/// lane is canonical.
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
            let sum = _mm512_add_epi32(*dv, mulmod(cred, *sv, p)); // <= 2p - 1
            _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
        }
    }
    for (dst_lane, src_lane) in dst_rest.iter_mut().zip(src_rest) {
        let prod = mulmod(cred, min_chain(_mm512_loadu_si512(src_lane), p), p);
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let sum = _mm512_add_epi32(d, prod); // <= 2p - 1
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.mul_add",
        Substage::MulAdd,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        mul_add_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst *= coeff (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: a register group whose lanes all
/// have bit 31 clear multiplies as loaded, any other group and the lanes
/// past the last whole group canonicalize first, and every stored lane is
/// canonical.
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
    let (dst_groups, dst_rest) = dst_lanes.as_chunks_mut::<SCREEN_LANES>();
    for dst_group in dst_groups.iter_mut() {
        let mut d = [_mm512_setzero_si512(); SCREEN_LANES];
        let mut seen = _mm512_setzero_si512();
        for (dv, dst_lane) in d.iter_mut().zip(&*dst_group) {
            *dv = _mm512_loadu_si512(dst_lane);
            seen = _mm512_or_si512(seen, *dv);
        }
        if any_bit31(seen) {
            for v in &mut d {
                *v = min_chain(*v, p);
            }
        }
        for (dst_lane, dv) in dst_group.iter_mut().zip(&d) {
            _mm512_storeu_si512(dst_lane, mulmod(cred, *dv, p));
        }
    }
    for dst_lane in dst_rest.iter_mut() {
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(cred, d, p));
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.mul_assign",
        Substage::Mul,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        mul_assign_avx2(token.v3(), dst_tail, coeff);
    }
}

/// `dst = coeff * src (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: the first pass multiplies the
/// lanes as loaded, a second pass recomputes every lane from
/// canonicalized loads when any source lane has bit 31 set, and every
/// stored lane is canonical.
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
    let mut seen = _mm512_setzero_si512();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = _mm512_loadu_si512(src_lane);
        seen = _mm512_or_si512(seen, s);
        _mm512_storeu_si512(dst_lane, mulmod(cred, s, p));
    }
    // The destination is write-only and never aliases the source, so the
    // recomputing pass reads the same source lanes.
    if any_bit31(seen) {
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let s = min_chain(_mm512_loadu_si512(src_lane), p);
            _mm512_storeu_si512(dst_lane, mulmod(cred, s, p));
        }
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.mul_into",
        Substage::Mul,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        mul_into_avx2(token.v3(), dst_tail, coeff, src_tail);
    }
}

/// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: the first pass multiplies the
/// lanes as loaded, a second pass recomputes every lane from
/// canonicalized loads when any source lane has bit 31 set, and every
/// stored lane is canonical.
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
    let mut seen = _mm512_setzero_si512();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = _mm512_loadu_si512(a_lane);
        let vb = _mm512_loadu_si512(b_lane);
        seen = _mm512_or_si512(seen, _mm512_or_si512(va, vb));
        _mm512_storeu_si512(dst_lane, mulmod(va, vb, p));
    }
    // The destination is write-only and never aliases a source, so the
    // recomputing pass reads the same source lanes.
    if any_bit31(seen) {
        for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
            let va = min_chain(_mm512_loadu_si512(a_lane), p);
            let vb = min_chain(_mm512_loadu_si512(b_lane), p);
            _mm512_storeu_si512(dst_lane, mulmod(va, vb, p));
        }
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.mul_elementwise",
        Substage::MulElementwise,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        mul_elementwise_avx2(token.v3(), dst_tail, a_tail, b_tail);
    }
}

/// `dst[i] = dst[i] * src[i] (mod p)`, Mersenne31, AVX-512.
///
/// Every input lane bit pattern is legal: a register group whose lanes all
/// have bit 31 clear multiplies as loaded, any other group and the lanes
/// past the last whole group canonicalize first, and every stored lane is
/// canonical.
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
            _mm512_storeu_si512(dst_lane, mulmod(*dv, *sv, p));
        }
    }
    for (dst_lane, src_lane) in dst_rest.iter_mut().zip(src_rest) {
        let vd = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let vs = min_chain(_mm512_loadu_si512(src_lane), p);
        _mm512_storeu_si512(dst_lane, mulmod(vd, vs, p));
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.mul_elementwise_assign",
        Substage::MulElementwise,
        Unit::Elements,
        dst_lanes.len() * 16,
        crate::kernel::Backend::V4x,
    );
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
    #[cfg(test)]
    let num_lanes = dst_lanes.len();
    for dst_lane in dst_lanes {
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let sum = _mm512_add_epi32(d, svec); // <= 2p - 2
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(sum, _mm512_sub_epi32(sum, p)));
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.add_assign_scalar",
        Substage::Add,
        Unit::Elements,
        num_lanes * 16,
        crate::kernel::Backend::V4x,
    );
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
    #[cfg(test)]
    let num_lanes = dst_lanes.len();
    for dst_lane in dst_lanes {
        let d = min_chain(_mm512_loadu_si512(&*dst_lane), p);
        let diff = _mm512_sub_epi32(d, svec); // wraps iff negative
        _mm512_storeu_si512(dst_lane, _mm512_min_epu32(diff, _mm512_add_epi32(diff, p)));
    }
    #[cfg(test)]
    emit(
        "mersenne31.avx512.sub_assign_scalar",
        Substage::Sub,
        Unit::Elements,
        num_lanes * 16,
        crate::kernel::Backend::V4x,
    );
    if !dst_tail.is_empty() {
        sub_assign_scalar_avx2(token.v3(), dst_tail, value);
    }
}
