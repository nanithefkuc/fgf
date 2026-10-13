//! Eight-byte-lane single-row AVX-512 kernels for the generic residue rows.
//!
//! Montgomery REDC over u64 lanes at odd `P`: eight
//! lanes per 64-byte vector body and scalar `redc64_scalar` tails past
//! the body. Entries are safe capability-token functions taking an
//! `X64V4Token` and validating their own geometry.
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::src_tail_avx512;

/// `dst[i] = REDC64(src[i] * cR)` over eight-byte lanes.
///
/// Eight lanes per vector through Montgomery REDC. The
/// coefficient arrives prepared as the u64 Montgomery residue
/// `cR = c * 2^64 mod P`; each product assembles the full 64x64->128
/// product from 32-bit partial products with all carries retained, then
/// reduces with the 129th numerator bit tracked, so one conditional
/// subtract canonicalizes. The raw source keeps `src[i]*cR < P*2^64`, so
/// the numerator stays below `2P*2^64` and the one-subtract REDC64
/// applies; the output is canonical.
///
/// # Panics
///
/// Panics unless both buffers match in length (whole eight-byte lanes),
/// `p` is odd, and `c_mont` is below `p`.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_u64_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    src: &[u8],
    c_mont: u64,
    p: u64,
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "residue::mul_into_u64_avx512: dst and src lengths differ"
    );
    assert_eq!(
        dst.len() % 8,
        0,
        "residue::mul_into_u64_avx512: dst length {} is not whole 8-byte lanes",
        dst.len()
    );
    assert!(p % 2 == 1, "residue u64: modulus {p} is even");
    assert!(
        c_mont < p,
        "residue u64: prepared coefficient {c_mont} reaches modulus {p}"
    );
    let ninv = neg_inv_mod2_64(p);
    let ninv_v = _mm512_set1_epi64(ninv.cast_signed());
    let p_v = _mm512_set1_epi64(p.cast_signed());
    let pm1_v = _mm512_set1_epi64(p.wrapping_sub(1).cast_signed());
    let cm_v = _mm512_set1_epi64(c_mont.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, _) = src.as_chunks::<64>();
    for (d, s_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let (t_hi, t_lo) = mul_wide(_mm512_loadu_si512(s_lane), cm_v);
        _mm512_storeu_si512(d, redc64_lanes(t_hi, t_lo, ninv_v, p_v, pm1_v));
    }
    let (dt, _) = dst_tail.as_chunks_mut::<8>();
    let (st, _) = src_tail_avx512(src).as_chunks::<8>();
    for (d, x) in dt.iter_mut().zip(st) {
        let t = u128::from(u64::from_le_bytes(*x)) * u128::from(c_mont);
        *d = redc64_scalar(t, p, ninv).to_le_bytes();
    }
}

/// `dst[i] += REDC64(src[i] * cR)` over eight-byte lanes.
///
/// The raw destination lane reduces through `x mod P` in 64-bit lanes
/// (one Barrett quotient estimate plus one conditional subtraction;
/// `P < 2^64` keeps every intermediate exact) and joins the canonical
/// product through the wrapping modular add; the output is canonical.
///
/// # Panics
///
/// Panics unless both buffers match in length (whole eight-byte lanes),
/// `p` is odd, and `c_mont` is below `p`.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_u64_avx512(
    _token: archmage::X64V4Token,
    dst: &mut [u8],
    src: &[u8],
    c_mont: u64,
    p: u64,
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "residue::mul_add_u64_avx512: dst and src lengths differ"
    );
    assert_eq!(
        dst.len() % 8,
        0,
        "residue::mul_add_u64_avx512: dst length {} is not whole 8-byte lanes",
        dst.len()
    );
    assert!(p % 2 == 1, "residue u64: modulus {p} is even");
    assert!(
        c_mont < p,
        "residue u64: prepared coefficient {c_mont} reaches modulus {p}"
    );
    let ninv = neg_inv_mod2_64(p);
    let ninv_v = _mm512_set1_epi64(ninv.cast_signed());
    let p_v = _mm512_set1_epi64(p.cast_signed());
    let pm1_v = _mm512_set1_epi64(p.wrapping_sub(1).cast_signed());
    let cm_v = _mm512_set1_epi64(c_mont.cast_signed());
    // `floor(2^64 / P)` sits below `2^64 / 3 < u64::MAX` for odd `P >= 3`.
    #[allow(clippy::cast_possible_truncation)]
    let mu = ((1u128 << 64) / u128::from(p)) as u64;
    let mu_v = _mm512_set1_epi64(mu.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    let (src_lanes, _) = src.as_chunks::<64>();
    for (d, s_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let normed = normalize64(_mm512_loadu_si512(&*d), mu_v, p_v, pm1_v);
        let (t_hi, t_lo) = mul_wide(_mm512_loadu_si512(s_lane), cm_v);
        let prod = redc64_lanes(t_hi, t_lo, ninv_v, p_v, pm1_v);
        _mm512_storeu_si512(d, addmod64(normed, prod, p_v, pm1_v));
    }
    let (dt, _) = dst_tail.as_chunks_mut::<8>();
    let (st, _) = src_tail_avx512(src).as_chunks::<8>();
    for (d, x) in dt.iter_mut().zip(st) {
        let acc = u64::from_le_bytes(*d) % p;
        let t = u128::from(u64::from_le_bytes(*x)) * u128::from(c_mont);
        let prod = redc64_scalar(t, p, ninv);
        let sum = u128::from(acc) + u128::from(prod);
        // Exact: the modular sum sits below `P <= u64::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let out = (if sum >= u128::from(p) {
            sum - u128::from(p)
        } else {
            sum
        }) as u64;
        *d = out.to_le_bytes();
    }
}

/// `dst[i] = REDC64(dst[i] * cR)` over eight-byte lanes.
///
/// Arbitrary raw lanes reduce in place; the output is canonical.
///
/// # Panics
///
/// Panics unless `dst` holds whole eight-byte lanes, `p` is odd, and
/// `c_mont` is below `p`.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_u64_avx512(_token: archmage::X64V4Token, dst: &mut [u8], c_mont: u64, p: u64) {
    assert_eq!(
        dst.len() % 8,
        0,
        "residue::mul_assign_u64_avx512: dst length {} is not whole 8-byte lanes",
        dst.len()
    );
    assert!(p % 2 == 1, "residue u64: modulus {p} is even");
    assert!(
        c_mont < p,
        "residue u64: prepared coefficient {c_mont} reaches modulus {p}"
    );
    let ninv = neg_inv_mod2_64(p);
    let ninv_v = _mm512_set1_epi64(ninv.cast_signed());
    let p_v = _mm512_set1_epi64(p.cast_signed());
    let pm1_v = _mm512_set1_epi64(p.wrapping_sub(1).cast_signed());
    let cm_v = _mm512_set1_epi64(c_mont.cast_signed());
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<64>();
    for d in dst_lanes.iter_mut() {
        let (t_hi, t_lo) = mul_wide(_mm512_loadu_si512(&*d), cm_v);
        _mm512_storeu_si512(d, redc64_lanes(t_hi, t_lo, ninv_v, p_v, pm1_v));
    }
    let (dt, _) = dst_tail.as_chunks_mut::<8>();
    for d in dt.iter_mut() {
        let t = u128::from(u64::from_le_bytes(*d)) * u128::from(c_mont);
        *d = redc64_scalar(t, p, ninv).to_le_bytes();
    }
}

/// Negative inverse of an odd modulus modulo `2^64`: `p*ninv = -1
/// (mod 2^64)`, by Newton doubling from one correct bit.
const fn neg_inv_mod2_64(p: u64) -> u64 {
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

/// Scalar REDC64. The `t + m*P` addition is tracked through its 129th bit
/// with `u128::overflowing_add`; a set carry is the high bit of the shifted
/// result and forces the conditional subtract on its own, not only the
/// unsigned word comparison.
#[allow(clippy::cast_possible_truncation)]
fn redc64_scalar(t: u128, p: u64, ninv: u64) -> u64 {
    let m = (t as u64).wrapping_mul(ninv);
    let (num, carry) = t.overflowing_add(u128::from(m) * u128::from(p));
    let u = (num >> 64) as u64;
    if carry || u >= p {
        u.wrapping_sub(p)
    } else {
        u
    }
}

/// Unsigned `a > b` over eight u64 lanes.
#[archmage::rite(v4)]
fn gt_epu64(a: __m512i, b: __m512i) -> __m512i {
    let mask = _mm512_cmp_epu64_mask::<{ _MM_CMPINT_NLE }>(a, b);
    _mm512_maskz_set1_epi64(mask, -1)
}

/// Full 64x64->128 product over eight lanes via four `vpmuludq` partials.
/// Each partial sum is bounded below `2^64` by construction, so no wrap is
/// hidden. Returns `(hi, lo)`.
#[archmage::rite(v4)]
fn mul_wide(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
    let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));
    let ll = _mm512_mul_epu32(x, y);
    let lh = _mm512_mul_epu32(x, y_hi);
    let hl = _mm512_mul_epu32(x_hi, y);
    let hh = _mm512_mul_epu32(x_hi, y_hi);
    // Big-endian bignum assembly; every partial sum fits by construction.
    let ll_hi = _mm512_srli_epi64::<32>(ll);
    let t0 = _mm512_add_epi64(hl, ll_hi);
    let t0_lo = _mm512_and_si512(t0, _mm512_set1_epi64(0xFFFF_FFFF));
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(lh, t0_lo);
    let t2 = _mm512_add_epi64(hh, t0_hi);
    let hi = _mm512_add_epi64(t2, _mm512_srli_epi64::<32>(t1));
    let lo = _mm512_or_si512(
        _mm512_and_si512(ll, _mm512_set1_epi64(0xFFFF_FFFF)),
        _mm512_slli_epi64::<32>(t1),
    );
    (hi, lo)
}

/// REDC64 over eight lanes: `(t + m*P) >> 64` with `m = t_lo*ninv mod
/// 2^64`, for `(t_hi, t_lo)` the full product. The numerator addition is
/// tracked through bit 128: `top` carries both carry masks, becomes the
/// 65th bit of the shifted result, and joins the unsigned comparison in
/// the subtract mask. The shifted result is below `2P`, so one
/// conditional subtract canonicalizes.
#[archmage::rite(v4)]
fn redc64_lanes(t_hi: __m512i, t_lo: __m512i, ninv: __m512i, p: __m512i, pm1: __m512i) -> __m512i {
    let m = _mm512_mullo_epi64(t_lo, ninv);
    let (mp_hi, mp_lo) = mul_wide(m, p);
    let n_lo = _mm512_add_epi64(t_lo, mp_lo);
    let c_lo = gt_epu64(t_lo, n_lo);
    let hi_sum = _mm512_add_epi64(t_hi, mp_hi);
    let c_a = gt_epu64(t_hi, hi_sum);
    // All-ones carry masks act as -1, so subtraction adds the carry.
    let n_hi = _mm512_sub_epi64(hi_sum, c_lo);
    let c_b = gt_epu64(hi_sum, n_hi);
    let top = _mm512_or_si512(c_a, c_b);
    let mask = _mm512_or_si512(top, gt_epu64(n_hi, pm1));
    _mm512_sub_epi64(n_hi, _mm512_and_si512(mask, p))
}

/// Canonical modular add of two values below `P` over 64-bit lanes: the
/// sum wraps exactly where it borrows, so the wrapped comparison selects
/// the representative and one subtract canonicalizes.
#[archmage::rite(v4)]
fn addmod64(a: __m512i, b: __m512i, p: __m512i, pm1: __m512i) -> __m512i {
    let sum = _mm512_add_epi64(a, b);
    let borrow = gt_epu64(sum, pm1);
    // Where the sum wrapped, `sum < a` and the unsigned comparison with
    // `P - 1` still reads the wrapped value as below `P`; OR-ing the
    // wrap mask restores the subtract.
    let wrapped = gt_epu64(a, sum);
    let mask = _mm512_or_si512(borrow, wrapped);
    _mm512_sub_epi64(sum, _mm512_and_si512(mask, p))
}

/// Reduce arbitrary raw 64-bit lanes into `0..P` through one Barrett
/// quotient estimate: `q = hi64(x * mu)`, `r = x - q*P` in `[0, 2P)`,
/// then one conditional subtract. Exact for every odd `P`: `mu` holds
/// `floor(2^64 / P)` and the raw-word reciprocal estimate undershoots by
/// at most one, so `q*P <= x` and the subtraction cannot borrow.
#[archmage::rite(v4)]
fn normalize64(x: __m512i, mu: __m512i, p: __m512i, pm1: __m512i) -> __m512i {
    let q = mul_wide(x, mu).0;
    let qp_lo = _mm512_mullo_epi64(q, p);
    let r = _mm512_sub_epi64(x, qp_lo);
    let under = gt_epu64(r, pm1);
    _mm512_sub_epi64(r, _mm512_and_si512(under, p))
}
