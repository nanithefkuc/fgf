//! Four-byte-lane single-row AVX-512 kernels for the generic residue rows.
//!
//! Narrow-Barrett elementwise products and Shoup fixed-coefficient
//! products over four-byte lanes: sixteen lanes per 64-byte vector body,
//! scalar tails past the body. The shared vector helpers
//! (`normalize_avx512`, `narrow_avx512`, `combine_halves_avx512`,
//! `widen_add_avx512`) live in the parent module. Entries are safe
//! capability-token functions taking an `X64V4Token` and validating their
//! own geometry.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{combine_halves_avx512, narrow_avx512, normalize_avx512, widen_add_avx512};
use crate::kernel::proven_checks::check_equal;
use crate::kernel::x86::residue::U32Params;

// ---------------------------------------------------------------------------
// u32 elementwise (narrow Barrett)
// ---------------------------------------------------------------------------

/// `dst[i] = a[i] * b[i] (mod p)` over four-byte residue lanes, AVX-512.
///
/// Sixteen lanes per vector through the narrow Barrett form. Each product's
/// low and high 32-bit halves come from `vpmuludq` on the even and odd
/// lanes, the `b` factor normalizes once per vector so every product sits
/// below `2^32 * P`, and each product reduces before recombination.
/// Every input lane bit pattern is legal: the `b` lanes normalize on load
/// and every stored lane is canonical.
///
/// # Panics
///
/// Panics unless all three buffers match in length (whole four-byte lanes).
#[allow(
    clippy::used_underscore_binding,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_possible_truncation
)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_u32_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
    params: U32Params,
) {
    check_geometry_u32(
        "residue::mul_elementwise_u32_avx512",
        dst.len(),
        "a",
        a.len(),
    );
    check_geometry_u32(
        "residue::mul_elementwise_u32_avx512",
        dst.len(),
        "b",
        b.len(),
    );
    let p = _mm512_set1_epi64(params.p as u64 as i64);
    let mu = _mm512_set1_epi64(params.barrett_mu as u64 as i64);
    let norm_mu = _mm512_set1_epi64(u64::from(params.norm_mu) as i64);
    let p32 = _mm512_set1_epi32(params.p as i32);
    let shift = params.shift as i32;
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (a_lanes, a_tail) = a.as_chunks::<64>();
    let (b_lanes, b_tail) = b.as_chunks::<64>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = _mm512_loadu_si512(a_lane);
        let vb = normalize_avx512(_mm512_loadu_si512(b_lane), norm_mu, p32);
        _mm512_storeu_si512(dst_lane, multiply(va, vb, mu, p, shift));
    }
    tail_mul(dst_tail, a_tail, b_tail, params.p);
}

/// `dst[i] = dst[i] * src[i] (mod p)` over four-byte residue lanes, AVX-512.
///
/// The in-place form of [`mul_elementwise_u32_avx512`]: the `src` lanes
/// normalize on load and every stored lane is canonical.
///
/// # Panics
///
/// Panics if the slices differ in length or hold a partial four-byte lane
/// width.
#[allow(
    clippy::used_underscore_binding,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_possible_truncation
)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_u32_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    src: &[u8],
    params: U32Params,
) {
    check_geometry_u32(
        "residue::mul_elementwise_assign_u32_avx512",
        dst.len(),
        "src",
        src.len(),
    );
    let p = _mm512_set1_epi64(params.p as u64 as i64);
    let mu = _mm512_set1_epi64(params.barrett_mu as u64 as i64);
    let norm_mu = _mm512_set1_epi64(u64::from(params.norm_mu) as i64);
    let p32 = _mm512_set1_epi32(params.p as i32);
    let shift = params.shift as i32;
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, src_tail) = src.as_chunks::<64>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let vd = _mm512_loadu_si512(&*dst_lane);
        let vs = normalize_avx512(_mm512_loadu_si512(src_lane), norm_mu, p32);
        _mm512_storeu_si512(dst_lane, multiply(vd, vs, mu, p, shift));
    }
    tail_mul_assign(dst_tail, src_tail, params.p);
}

// ---------------------------------------------------------------------------
// u32 fixed-coefficient (Shoup)
// ---------------------------------------------------------------------------

/// `dst[i] = src[i] * c mod P` from the Shoup quotient
/// `q = floor(c * 2^32 / P)`, over four-byte lanes.
///
/// The widened 64-bit path keeps the remainder exact in `[0, 2P)` for
/// arbitrary raw lanes; eight widened lanes per set combine back to sixteen
/// `u32` outputs. Serves every four-byte modulus.
///
/// # Panics
///
/// Panics unless `dst` and `src` have equal lengths holding whole
/// four-byte lanes.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_u32_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    src: &[u8],
    c: u32,
    q: u32,
    p: u32,
) {
    check_geometry_u32("residue::mul_into_u32_avx512", dst.len(), "src", src.len());
    let cv = _mm512_set1_epi32(c as i32);
    let qv = _mm512_set1_epi32(q as i32);
    let p64 = _mm512_set1_epi64(i64::from(p));
    let p_m1 = _mm512_set1_epi64(i64::from(p.wrapping_sub(1)));
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, _) = src.as_chunks::<64>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let v = _mm512_loadu_si512(s);
        let vo = _mm512_shuffle_epi32::<0xB1>(v);
        let re = shoup64(_mm512_mul_epu32(v, cv), _mm512_mul_epu32(v, qv), p64, p_m1);
        let ro = shoup64(
            _mm512_mul_epu32(vo, cv),
            _mm512_mul_epu32(vo, qv),
            p64,
            p_m1,
        );
        _mm512_storeu_si512(d, combine_halves_avx512(re, ro));
    }
    for (d, s) in dst_tail.chunks_exact_mut(4).zip(src_tail_u32(src)) {
        let x = u32::from_le_bytes(s.as_ref().try_into().unwrap());
        d.copy_from_slice(&scalar_shoup32(x, c, q, p).to_le_bytes());
    }
}

/// `dst[i] += src[i] * c mod P` through the Shoup pair.
///
/// The raw destination lane normalizes through one Barrett quotient
/// estimate and joins the canonical product in 64-bit lanes, where the
/// sum cannot wrap; the output is canonical. Serves every four-byte
/// modulus.
///
/// # Panics
///
/// Panics unless `dst` and `src` have equal lengths holding whole
/// four-byte lanes.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_u32_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    src: &[u8],
    c: u32,
    q: u32,
    p: u32,
) {
    check_geometry_u32("residue::mul_add_u32_avx512", dst.len(), "src", src.len());
    let cv = _mm512_set1_epi32(c as i32);
    let qv = _mm512_set1_epi32(q as i32);
    let p64 = _mm512_set1_epi64(i64::from(p));
    let p_m1 = _mm512_set1_epi64(i64::from(p.wrapping_sub(1)));
    let norm_mu = _mm512_set1_epi64(i64::from(norm_mu32(p)));
    let p32 = _mm512_set1_epi32(p as i32);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, _) = src.as_chunks::<64>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let normed = normalize_avx512(_mm512_loadu_si512(&*d), norm_mu, p32);
        let v = _mm512_loadu_si512(s);
        let vo = _mm512_shuffle_epi32::<0xB1>(v);
        let re = shoup64(_mm512_mul_epu32(v, cv), _mm512_mul_epu32(v, qv), p64, p_m1);
        let ro = shoup64(
            _mm512_mul_epu32(vo, cv),
            _mm512_mul_epu32(vo, qv),
            p64,
            p_m1,
        );
        let prod = combine_halves_avx512(re, ro);
        // Both halves are canonical, so each 64-bit pairwise sum sits
        // below `2P < 2^33` and one unsigned comparison canonicalizes it.
        _mm512_storeu_si512(d, widen_add_avx512(normed, prod, p64, p_m1));
    }
    let wide = u64::from(p);
    for (d, s) in dst_tail.chunks_exact_mut(4).zip(src_tail_u32(src)) {
        let acc = u64::from(u32::from_le_bytes(d.as_ref().try_into().unwrap())) % wide;
        let x = u32::from_le_bytes(s.as_ref().try_into().unwrap());
        let sum = acc + u64::from(scalar_shoup32(x, c, q, p));
        d.copy_from_slice(
            &u32::try_from(if sum >= wide { sum - wide } else { sum })
                .expect("shoup tail below P")
                .to_le_bytes(),
        );
    }
}

/// `dst[i] = dst[i] * c mod P` through the Shoup pair.
///
/// Arbitrary raw lanes reduce in place; the output is canonical. Serves
/// every four-byte modulus.
///
/// # Panics
///
/// Panics if `dst` holds a partial four-byte lane.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_u32_avx512(_token: archmage::X64V4Token, dst: &mut [u8], c: u32, q: u32, p: u32) {
    assert!(
        dst.len().is_multiple_of(4),
        "residue::mul_assign_u32_avx512: dst lane width is 4 bytes but the buffer is {} bytes",
        dst.len()
    );
    let cv = _mm512_set1_epi32(c as i32);
    let qv = _mm512_set1_epi32(q as i32);
    let p64 = _mm512_set1_epi64(i64::from(p));
    let p_m1 = _mm512_set1_epi64(i64::from(p.wrapping_sub(1)));
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for d in dst_lanes.iter_mut() {
        let v = _mm512_loadu_si512(&*d);
        let vo = _mm512_shuffle_epi32::<0xB1>(v);
        let re = shoup64(_mm512_mul_epu32(v, cv), _mm512_mul_epu32(v, qv), p64, p_m1);
        let ro = shoup64(
            _mm512_mul_epu32(vo, cv),
            _mm512_mul_epu32(vo, qv),
            p64,
            p_m1,
        );
        _mm512_storeu_si512(d, combine_halves_avx512(re, ro));
    }
    for d in dst_tail.chunks_exact_mut(4) {
        let x = u32::from_le_bytes(d.as_ref().try_into().unwrap());
        d.copy_from_slice(&scalar_shoup32(x, c, q, p).to_le_bytes());
    }
}

/// Whole-lane geometry: paired equality plus the four-byte lane width.
fn check_geometry_u32(name: &str, dst_len: usize, other_name: &str, other_len: usize) {
    check_equal(name, "dst", dst_len, other_name, other_len);
    assert!(
        dst_len.is_multiple_of(4),
        "{name}: dst lane width is 4 bytes but the buffer is {dst_len} bytes"
    );
}

/// The sub-vector tail of whole four-byte lanes past the 64-byte vector body.
fn src_tail_u32(src: &[u8]) -> impl Iterator<Item = &[u8; 4]> {
    let (_, tail) = src.as_chunks::<64>();
    tail.chunks_exact(4).map(|lane| {
        lane.try_into()
            .expect("residue u32: tail holds whole four-byte lanes")
    })
}

/// `a * b (mod p)` over sixteen lanes, `b` canonical, products exact.
#[archmage::rite(v4)]
fn multiply(a: __m512i, b: __m512i, mu: __m512i, p: __m512i, shift: i32) -> __m512i {
    let e = _mm512_mul_epu32(a, b);
    let o = _mm512_mul_epu32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    let re = narrow_avx512(e, mu, p, shift);
    let ro = narrow_avx512(o, mu, p, shift);
    _mm512_or_si512(re, _mm512_slli_epi64::<32>(ro))
}

#[allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]
#[must_use]
fn scalar_shoup32(x: u32, c: u32, q: u32, p: u32) -> u32 {
    let v = u64::from(x) * u64::from(c) - ((u64::from(x) * u64::from(q)) >> 32) * u64::from(p);
    (if v >= u64::from(p) {
        v - u64::from(p)
    } else {
        v
    }) as u32
}

/// Shoup reduction of one 64-bit widened lane set: `U = xq >> 32`,
/// `v = xc - U*P` in `[0, 2P)`, one conditional subtract.
#[archmage::rite(v4)]
#[must_use]
fn shoup64(xc: __m512i, xq: __m512i, p64: __m512i, p_m1: __m512i) -> __m512i {
    let u = _mm512_srli_epi64::<32>(xq);
    let v = _mm512_sub_epi64(xc, _mm512_mul_epu32(u, p64));
    _mm512_mask_sub_epi64(
        v,
        _mm512_cmp_epu64_mask::<{ _MM_CMPINT_NLE }>(v, p_m1),
        v,
        p64,
    )
}

/// Truncated quotient helper: `floor(2^32 / p)`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
fn norm_mu32(p: u32) -> u32 {
    ((1u128 << 32) / u128::from(p)) as u32
}

/// The portable elementwise tail: reduce both raw u32 lanes, multiply, reduce.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
fn tail_mul(dest: &mut [u8], left: &[u8], right: &[u8], modulus: u32) {
    let wide = u64::from(modulus);
    for ((lane, x), y) in dest
        .chunks_exact_mut(4)
        .zip(left.chunks_exact(4))
        .zip(right.chunks_exact(4))
    {
        let x = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % wide;
        let y = u64::from(u32::from_le_bytes(y.try_into().unwrap())) % wide;
        lane.copy_from_slice(&((x * y % wide) as u32).to_le_bytes());
    }
}

/// The portable in-place elementwise tail: `dst[i] = dst[i] * src[i]`.
#[allow(clippy::cast_possible_truncation)]
fn tail_mul_assign(dest: &mut [u8], src: &[u8], modulus: u32) {
    let wide = u64::from(modulus);
    for (lane, word) in dest.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let x = u64::from(u32::from_le_bytes(lane.try_into().unwrap())) % wide;
        let y = u64::from(u32::from_le_bytes(word.try_into().unwrap())) % wide;
        lane.copy_from_slice(&((x * y % wide) as u32).to_le_bytes());
    }
}
