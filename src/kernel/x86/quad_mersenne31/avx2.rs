//! `QuadMersenne31` (`(re, im)` Mersenne31 limb pairs) AVX2 kernels.
//!
//! Four complex elements per 256-bit vector: eight 32-bit lanes holding
//! `[re0, im0, re1, im1, re2, im2, re3, im3]`, little-endian, so the even
//! 32-bit lanes are the real limbs and the odd lanes the imaginary limbs.
//! The limb arithmetic — the fold and the canonicalizing min chain — is
//! the Mersenne31 discipline shared with
//! [`crate::kernel::x86::mersenne31`]; this module adds the complex layer
//! on top of it.
//!
//! A complex product defers its reduction to 64-bit accumulators. The
//! coefficient multiply prepares `cr` broadcast to all lanes plus a
//! sign-mixed `ci` vector (`p - ci` on the real lanes, `ci` on the
//! imaginary lanes), and accumulates `re*cr + im*(p-ci)` and
//! `im*cr + re*ci` per element over the element's duplicated imaginary
//! limb. The elementwise product, where both factors vary, duplicates the
//! imaginary limbs of both factors, negates one copy of `b`'s imaginary
//! limbs with `p`, and accumulates `ar*br + ai*(p-bi)` and
//! `ar*bi + ai*br` the same way. Both forms reduce through [`reduce_pair`],
//! which splits each 64-bit accumulator against `2^31 ≡ 1 (mod p)` and
//! canonicalizes with the min chain, so every stored limb is below the
//! modulus. The tests validate every lane construction against an
//! independent `u128 % p` limb model.
//!
//! Buffers meet the packed canonical-lane contract: every input limb is
//! below the modulus and every stored limb is canonical, the same contract
//! as [`crate::ops`]. Coefficients are total — any raw `Elem` limbs
//! canonicalize once per call.
//!
//! Reaching these entries directly is an `internals` surface, like the
//! other direct x86 kernels: each is a safe capability-token function
//! taking an `X64V3Token`, validating its own geometry, and handing
//! whole-element tails to the portable [`crate::kernel::prime`] path. The
//! zero and one coefficient shortcuts mirror the scalar control exactly,
//! byte for byte.

use crate::field::quad_mersenne31::{Elem, QuadMersenne31};
use crate::field::{Field, mersenne31};
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::x86::mersenne31::{P, fold_avx2, min_chain_avx2};
use crate::kernel::{prime, scalar};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Duplicate each element's imaginary limb into both lanes of its pair.
#[archmage::rite(v3)]
#[must_use]
fn dup_im(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)))
}

/// `a + b (mod p)` over eight folded lanes (`0..=p` in, canonical out).
#[archmage::rite(v3)]
#[must_use]
fn addmod(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    min_chain_avx2(_mm256_add_epi32(a, b), p) // sum <= 2p
}

/// `a - b (mod p)` over eight folded lanes (`0..=p` in, canonical out).
#[archmage::rite(v3)]
#[must_use]
fn submod(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let diff = _mm256_sub_epi32(a, b); // wraps iff negative
    let fixed = _mm256_min_epu32(diff, _mm256_add_epi32(diff, p));
    min_chain_avx2(fixed, p)
}

/// Multiply four folded complex elements by a prepared coefficient:
/// `cr` broadcast to all lanes, `ci_mixed` holding `p - ci` on the real
/// lanes and `ci` on the imaginary lanes. The real lanes accumulate
/// `cr*re + (p-ci)*im = cr*re - ci*im` and the imaginary lanes
/// `cr*im + ci*re`, both deferred at 64 bits for [`reduce_pair`].
#[archmage::rite(v3)]
#[must_use]
fn cmul(x: __m256i, cr: __m256i, ci_mixed: __m256i, p: __m256i) -> __m256i {
    let xi = dup_im(x);
    let ci = dup_im(ci_mixed);
    let re = _mm256_add_epi64(_mm256_mul_epu32(x, cr), _mm256_mul_epu32(xi, ci_mixed));
    let im = _mm256_add_epi64(_mm256_mul_epu32(xi, cr), _mm256_mul_epu32(x, ci));
    reduce_pair(re, im, p)
}

/// Complex multiply of two folded vectors, `(ar*br - ai*bi, ar*bi + ai*br)`
/// per element, negating the duplicated imaginary limbs of `b` with `p`
/// and reducing the deferred 64-bit accumulators.
#[archmage::rite(v3)]
#[must_use]
fn cmul_vec(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let ai = dup_im(a);
    let bi = dup_im(b);
    let neg_bi = _mm256_xor_si256(bi, p);
    let re = _mm256_add_epi64(_mm256_mul_epu32(a, b), _mm256_mul_epu32(ai, neg_bi));
    let im = _mm256_add_epi64(_mm256_mul_epu32(a, bi), _mm256_mul_epu32(ai, b));
    reduce_pair(re, im, p)
}

/// Reduce deferred 64-bit product accumulators into packed canonical limb
/// pairs: the even dwords carry the real part and the odd dwords the
/// imaginary part, each split against `2^31 ≡ 1 (mod p)` and canonicalized
/// with the min chain.
#[archmage::rite(v3)]
#[must_use]
fn reduce_pair(re: __m256i, im: __m256i, p: __m256i) -> __m256i {
    let hi =
        _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<31>(re), _mm256_slli_epi64::<1>(im));
    let hi = _mm256_min_epu32(hi, _mm256_sub_epi32(hi, p));
    let lo = _mm256_and_si256(
        _mm256_blend_epi32::<0b1010_1010>(re, _mm256_slli_epi64::<32>(im)),
        p,
    );
    let t = _mm256_add_epi32(lo, hi);
    _mm256_min_epu32(t, _mm256_sub_epi32(t, p))
}

/// The coefficient vectors every coefficient multiply entry builds: `cr`
/// broadcast, and `ci` mixed-sign. Both limbs enter canonical.
#[archmage::rite(v3)]
#[allow(clippy::cast_possible_wrap)]
fn coeff_vectors(cr: u32, ci: u32) -> (__m256i, __m256i) {
    let neg_ci = if ci == 0 { 0 } else { mersenne31::MODULUS - ci };
    let cr_v = _mm256_set1_epi32(cr as i32);
    // Even (real) lanes negate ci by p - ci; odd (imaginary) lanes keep ci.
    let ci_mixed = _mm256_blend_epi32::<0b1010_1010>(
        _mm256_set1_epi32(neg_ci as i32),
        _mm256_set1_epi32(ci as i32),
    );
    (cr_v, ci_mixed)
}

/// `dst += src` in GF((2^31 - 1)^2), AVX2.
///
/// Both buffers hold canonical lanes — every 32-bit limb below the
/// modulus — and every stored lane is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "quad_mersenne31::add_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::add_assign_avx2", dst.len(), 8);
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let s = fold_avx2(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, addmod(d, s, p));
    }
    prime::add_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst -= src` in GF((2^31 - 1)^2), AVX2.
///
/// Both buffers hold canonical lanes — every 32-bit limb below the
/// modulus — and every stored lane is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "quad_mersenne31::sub_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::sub_assign_avx2", dst.len(), 8);
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let s = fold_avx2(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, submod(d, s, p));
    }
    prime::sub_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst += coeff * src` in GF((2^31 - 1)^2), AVX2.
///
/// Both buffers hold canonical lanes; the coefficient may hold any raw
/// limbs, canonicalized once before the loop. A zero coefficient leaves
/// `dst` untouched and the coefficient `1 + 0i` adds `src` elementwise,
/// mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem, src: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_add_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_add_avx2", dst.len(), 8);
    let (cr, ci) = coeff.canonical().to_raw();
    if cr == 0 && ci == 0 {
        return;
    }
    if cr == 1 && ci == 0 {
        add_assign_avx2(_token, dst, src);
        return;
    }
    let p = _mm256_set1_epi32(P);
    let (cr_v, ci_mixed) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let prod = cmul(
            fold_avx2(_mm256_loadu_si256(src_lane), p),
            cr_v,
            ci_mixed,
            p,
        );
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, addmod(d, prod, p));
    }
    prime::mul_add::<QuadMersenne31>(dst_tail, coeff, src_tail);
}

/// `dst *= coeff` in GF((2^31 - 1)^2), AVX2.
///
/// The buffer holds canonical lanes; the coefficient may hold any raw
/// limbs, canonicalized once before the loop. A zero coefficient zeroes
/// `dst` and the coefficient `1 + 0i` leaves it untouched, mirroring the
/// scalar control.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem) {
    check_elem_multiple("quad_mersenne31::mul_assign_avx2", dst.len(), 8);
    let (cr, ci) = coeff.canonical().to_raw();
    if cr == 0 && ci == 0 {
        dst.fill(0);
        return;
    }
    if cr == 1 && ci == 0 {
        return;
    }
    let p = _mm256_set1_epi32(P);
    let (cr_v, ci_mixed) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, cmul(d, cr_v, ci_mixed, p));
    }
    prime::mul_assign::<QuadMersenne31>(dst_tail, coeff);
}

/// `dst = coeff * src` in GF((2^31 - 1)^2), AVX2. The destination is
/// write-only: its prior contents are ignored, never read.
///
/// Both buffers hold canonical lanes; the coefficient may hold any raw
/// limbs, canonicalized once before the loop. A zero coefficient zeroes
/// `dst`, mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem, src: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_into_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_into_avx2", dst.len(), 8);
    let (cr, ci) = coeff.canonical().to_raw();
    if cr == 0 && ci == 0 {
        dst.fill(0);
        return;
    }
    let p = _mm256_set1_epi32(P);
    let (cr_v, ci_mixed) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let s = fold_avx2(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, cmul(s, cr_v, ci_mixed, p));
    }
    // Whole-element scalar tail.
    for (d, s) in dst_tail.chunks_exact_mut(8).zip(src_tail.chunks_exact(8)) {
        let v = coeff * QuadMersenne31::decode(s);
        d.copy_from_slice(&v.to_bytes());
    }
}

/// `dst[i] = a[i] * b[i]` in GF((2^31 - 1)^2), AVX2.
///
/// All three buffers hold canonical lanes — every 32-bit limb below the
/// modulus — and every stored lane is canonical.
///
/// # Panics
/// Panics unless all three buffers match in length (whole elements).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2(_token: archmage::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_elementwise_avx2",
        "dst",
        dst.len(),
        "a",
        a.len(),
    );
    check_equal(
        "quad_mersenne31::mul_elementwise_avx2",
        "dst",
        dst.len(),
        "b",
        b.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_elementwise_avx2", dst.len(), 8);
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_tail) = a.as_chunks::<32>();
    let (b_lanes, b_tail) = b.as_chunks::<32>();
    for ((dst_lane, a_lane), b_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let va = fold_avx2(_mm256_loadu_si256(a_lane), p);
        let vb = fold_avx2(_mm256_loadu_si256(b_lane), p);
        _mm256_storeu_si256(dst_lane, cmul_vec(va, vb, p));
    }
    prime::mul_elementwise::<QuadMersenne31>(dst_tail, a_tail, b_tail);
}

/// `dst[i] = dst[i] * src[i]` in GF((2^31 - 1)^2), AVX2.
///
/// All buffers hold canonical lanes — every 32-bit limb below the modulus —
/// and every stored lane is canonical.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_equal(
        "quad_mersenne31::mul_elementwise_assign_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_elementwise_assign_avx2", dst.len(), 8);
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let vd = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let vs = fold_avx2(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, cmul_vec(vd, vs, p));
    }
    scalar::mul_elementwise_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst[i] += value` in GF((2^31 - 1)^2), AVX2. The raw `value` limbs are
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_avx2(_token: archmage::X64V3Token, dst: &mut [u8], value: Elem) {
    check_elem_multiple("quad_mersenne31::add_assign_scalar_avx2", dst.len(), 8);
    // The broadcast value is loop-invariant: canonicalize the raw limbs once.
    let (vr, vi) = value.canonical().to_raw();
    // Broadcast the (re, im) pair onto the alternating limb lanes.
    let svec = _mm256_blend_epi32::<0b1010_1010>(
        _mm256_set1_epi32(vr as i32),
        _mm256_set1_epi32(vi as i32),
    );
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, addmod(d, svec, p));
    }
    prime::add_assign_scalar::<QuadMersenne31>(dst_tail, value);
}

/// `dst[i] -= value` in GF((2^31 - 1)^2), AVX2. The raw `value` limbs are
/// canonicalized once on entry.
///
/// The destination lanes are canonicalized on load; the broadcast value is
/// used as given.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_avx2(_token: archmage::X64V3Token, dst: &mut [u8], value: Elem) {
    check_elem_multiple("quad_mersenne31::sub_assign_scalar_avx2", dst.len(), 8);
    let (vr, vi) = value.canonical().to_raw();
    let svec = _mm256_blend_epi32::<0b1010_1010>(
        _mm256_set1_epi32(vr as i32),
        _mm256_set1_epi32(vi as i32),
    );
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = fold_avx2(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, submod(d, svec, p));
    }
    prime::sub_assign_scalar::<QuadMersenne31>(dst_tail, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{KernelDispatch, RawDispatch};
    use archmage::SimdToken as _;

    /// Lengths in bytes straddling the 32-byte vector lane: empty, one
    /// element, sub-vector tails of one and three elements, exact vectors,
    /// vector-plus-tail, and a large odd size.
    const LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];

    /// Coefficients: zero, one, the imaginary unit, canonical limb
    /// extremes, mixed values, the field generator, and raw non-canonical
    /// limbs — the last two exercise the coefficient canonicalization the
    /// contract allows.
    const COEFFS: [Elem; 9] = [
        Elem::ZERO,
        Elem::ONE,
        Elem::I,
        Elem::from_raw(mersenne31::MODULUS - 1, mersenne31::MODULUS - 1),
        Elem::from_raw(1, mersenne31::MODULUS - 1),
        Elem::from_raw(0x5555_5555, 0x1234_5678),
        Elem::from_raw(mersenne31::MODULUS, 0), // non-canonical zero
        Elem::from_raw(u32::MAX, 2),            // raw limb, value 1 + 2i
        Elem::from_raw(1, 12),                  // the field generator
    ];

    /// A full-work coefficient: canonical, neither zero nor one, both
    /// limbs nonzero.
    const FULL_WORK: Elem = Elem::from_raw(0x1234_5678, 0x3FFF_FFF1);

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (state >> 33).to_le_bytes()[0]
            })
            .collect()
    }

    /// Random whole elements with canonical limbs, meeting the packed
    /// canonical-lane contract.
    fn canonical(len: usize, seed: u64) -> Vec<u8> {
        let mut out = noise(len, seed);
        for chunk in out.chunks_exact_mut(8) {
            let e = Elem::from_bytes(chunk.try_into().unwrap());
            chunk.copy_from_slice(&e.canonical().to_bytes());
        }
        out
    }

    /// Canonical edge elements tiled to `len` bytes: zero, both-limb and
    /// single-limb `p - 1` (the largest canonical limb), one, and a mixed
    /// pair — the range edges in the source operands themselves.
    fn edges(len: usize) -> Vec<u8> {
        const EDGES: [Elem; 5] = [
            Elem::ZERO,
            Elem::from_raw(mersenne31::MODULUS - 1, mersenne31::MODULUS - 1),
            Elem::from_raw(mersenne31::MODULUS - 1, 0),
            Elem::ONE,
            Elem::from_raw(0x5555_5555, 0x1234_5678),
        ];
        let mut out = Vec::with_capacity(len);
        for i in 0..len / 8 {
            out.extend_from_slice(&EDGES[i % EDGES.len()].to_bytes());
        }
        out
    }

    // Independent `u128 % p` limb model: deliberately sharing no reduction
    // shape with the kernels or the field type.
    fn o_canon(x: u32) -> u32 {
        u32::try_from(u128::from(x) % u128::from(mersenne31::MODULUS)).unwrap()
    }
    fn o_add(a: u32, b: u32) -> u32 {
        u32::try_from(
            (u128::from(o_canon(a)) + u128::from(o_canon(b))) % u128::from(mersenne31::MODULUS),
        )
        .unwrap()
    }
    fn o_sub(a: u32, b: u32) -> u32 {
        let d = i128::from(o_canon(a)) - i128::from(o_canon(b));
        u32::try_from(d.rem_euclid(i128::from(mersenne31::MODULUS))).unwrap()
    }
    fn o_mul(a: u32, b: u32) -> u32 {
        u32::try_from(
            (u128::from(o_canon(a)) * u128::from(o_canon(b))) % u128::from(mersenne31::MODULUS),
        )
        .unwrap()
    }

    fn model_elem_add(a: Elem, b: Elem) -> Elem {
        let (ar, ai) = a.to_raw();
        let (br, bi) = b.to_raw();
        Elem::from_raw(o_add(ar, br), o_add(ai, bi))
    }
    fn model_elem_sub(a: Elem, b: Elem) -> Elem {
        let (ar, ai) = a.to_raw();
        let (br, bi) = b.to_raw();
        Elem::from_raw(o_sub(ar, br), o_sub(ai, bi))
    }
    fn model_elem_mul(a: Elem, b: Elem) -> Elem {
        let (ar, ai) = a.to_raw();
        let (br, bi) = b.to_raw();
        Elem::from_raw(
            o_sub(o_mul(ar, br), o_mul(ai, bi)),
            o_add(o_mul(ar, bi), o_mul(ai, br)),
        )
    }

    /// Decode a whole-element buffer, apply `f` elementwise, re-encode.
    fn model_map(buf: &[u8], f: impl Fn(Elem) -> Elem) -> Vec<u8> {
        let mut out = buf.to_vec();
        for chunk in out.chunks_exact_mut(8) {
            let e = f(Elem::from_bytes(chunk.try_into().unwrap()));
            chunk.copy_from_slice(&e.to_bytes());
        }
        out
    }

    /// Decode two whole-element buffers positionally, apply `f` pairwise,
    /// re-encode.
    fn model_zip(a: &[u8], b: &[u8], f: impl Fn(Elem, Elem) -> Elem) -> Vec<u8> {
        assert_eq!(a.len(), b.len());
        let mut out = a.to_vec();
        for (ca, cb) in out.chunks_exact_mut(8).zip(b.chunks_exact(8)) {
            let e = f(
                Elem::from_bytes(ca.try_into().unwrap()),
                Elem::from_bytes(cb.try_into().unwrap()),
            );
            ca.copy_from_slice(&e.to_bytes());
        }
        out
    }

    /// Whether the host resolves to V3 — the same single-source selection
    /// the shared differential tests use, honoring `SIMD_BACKEND`.
    fn host_v3() -> bool {
        simdispatch::Selection::new("SIMD_BACKEND")
            .supports(&[crate::kernel::Backend::V3])
            .resolve()
            != crate::kernel::Backend::Scalar
    }

    /// Every operation matches the independent limb model across boundary
    /// lengths, canonical random fixtures, and canonical edge elements in
    /// the source operands.
    #[test]
    fn avx2_matches_independent_model() {
        if !host_v3() {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = archmage::X64V3Token::summon().expect("guard passed: AVX2 summons here");
        for &len in LENS {
            for (src, base, b2) in [
                (
                    canonical(len, 0xa1 + len as u64),
                    canonical(len, 0xb2 + len as u64),
                    canonical(len, 0xc3 + len as u64),
                ),
                (edges(len), edges(len), edges(len)),
            ] {
                let mut got = base.clone();
                add_assign_avx2(token, &mut got, &src);
                assert_eq!(
                    got,
                    model_zip(&base, &src, model_elem_add),
                    "add_assign len {len}"
                );

                let mut got = base.clone();
                sub_assign_avx2(token, &mut got, &src);
                assert_eq!(
                    got,
                    model_zip(&base, &src, model_elem_sub),
                    "sub_assign len {len}"
                );

                let mut got = vec![0u8; len];
                mul_elementwise_avx2(token, &mut got, &src, &b2);
                assert_eq!(
                    got,
                    model_zip(&src, &b2, model_elem_mul),
                    "mul_elementwise len {len}"
                );

                for &coeff in &COEFFS {
                    let mut got = base.clone();
                    mul_add_avx2(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_zip(&base, &src, |d, s| {
                            model_elem_add(d, model_elem_mul(coeff, s))
                        }),
                        "mul_add len {len} coeff {coeff:?}"
                    );

                    let mut got = vec![0u8; len];
                    mul_into_avx2(token, &mut got, coeff, &src);
                    assert_eq!(
                        got,
                        model_map(&src, |e| model_elem_mul(coeff, e)),
                        "mul_into len {len} coeff {coeff:?}"
                    );

                    let mut got = base.clone();
                    mul_assign_avx2(token, &mut got, coeff);
                    assert_eq!(
                        got,
                        model_map(&base, |e| model_elem_mul(coeff, e)),
                        "mul_assign len {len} coeff {coeff:?}"
                    );
                }
            }
        }
    }

    /// Every operation agrees with the portable [`prime`] differential
    /// oracle over lengths straddling the vector lane.
    #[test]
    fn avx2_matches_portable_reference() {
        if !host_v3() {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = archmage::X64V3Token::summon().expect("guard passed: AVX2 summons here");
        // Vector-plus-tail, exact vectors, and a different tail.
        for &len in &[40usize, 64, 72] {
            let src = edges(len);
            let base = canonical(len, 0xb2 + len as u64);
            let b2 = canonical(len, 0xc3 + len as u64);

            let mut got = base.clone();
            let mut want = base.clone();
            add_assign_avx2(token, &mut got, &src);
            prime::add_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "add_assign vs reference len {len}");

            let mut got = base.clone();
            let mut want = base.clone();
            sub_assign_avx2(token, &mut got, &src);
            prime::sub_assign::<QuadMersenne31>(&mut want, &src);
            assert_eq!(got, want, "sub_assign vs reference len {len}");

            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            mul_elementwise_avx2(token, &mut got, &src, &b2);
            prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &b2);
            assert_eq!(got, want, "mul_elementwise vs reference len {len}");

            let mut got = base.clone();
            let mut want = base.clone();
            mul_add_avx2(token, &mut got, FULL_WORK, &src);
            prime::mul_add::<QuadMersenne31>(&mut want, FULL_WORK, &src);
            assert_eq!(got, want, "mul_add vs reference len {len}");

            let mut got = vec![0u8; len];
            let mut want = src.clone();
            mul_into_avx2(token, &mut got, FULL_WORK, &src);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_into vs reference len {len}");

            let mut got = base.clone();
            let mut want = base.clone();
            mul_assign_avx2(token, &mut got, FULL_WORK);
            prime::mul_assign::<QuadMersenne31>(&mut want, FULL_WORK);
            assert_eq!(got, want, "mul_assign vs reference len {len}");
        }
    }

    /// The zero and one coefficient shortcuts preserve the scalar
    /// control's observable behavior. The zero arms never read `dst`, so a
    /// raw destination pattern defends the byte-preserving contract; the
    /// one shortcut adds, so its destination holds canonical elements.
    #[test]
    fn avx2_zero_one_shortcuts_match_control() {
        if !host_v3() {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = archmage::X64V3Token::summon().expect("guard passed: AVX2 summons here");

        let len = 40;
        let src = canonical(len, 0x71);
        // Zero, including a non-canonical limb spelling of it: the no-op
        // for mul_add and the fill for mul_assign and mul_into.
        for coeff in [Elem::ZERO, Elem::from_raw(mersenne31::MODULUS, 0)] {
            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            mul_add_avx2(token, &mut got, coeff, &src);
            <QuadMersenne31 as KernelDispatch>::mul_add(RawDispatch, &mut want, &coeff, &src);
            assert_eq!(got, want, "shortcut mul_add {coeff:?}");

            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            mul_assign_avx2(token, &mut got, coeff);
            <QuadMersenne31 as KernelDispatch>::mul_assign(RawDispatch, &mut want, &coeff);
            assert_eq!(got, want, "shortcut mul_assign {coeff:?}");

            let mut got = vec![0xA5u8; len];
            let mut want = vec![0xA5u8; len];
            mul_into_avx2(token, &mut got, coeff, &src);
            <QuadMersenne31 as KernelDispatch>::mul_into(RawDispatch, &mut want, &coeff, &src);
            assert_eq!(got, want, "shortcut mul_into {coeff:?}");
        }
        // The one shortcut: mul_add adds, mul_assign is the identity.
        let mut got = canonical(len, 0x72);
        let mut want = got.clone();
        mul_add_avx2(token, &mut got, Elem::ONE, &src);
        <QuadMersenne31 as KernelDispatch>::mul_add(RawDispatch, &mut want, &Elem::ONE, &src);
        assert_eq!(got, want, "shortcut mul_add one");

        let mut got = canonical(len, 0x72);
        let mut want = got.clone();
        mul_assign_avx2(token, &mut got, Elem::ONE);
        <QuadMersenne31 as KernelDispatch>::mul_assign(RawDispatch, &mut want, &Elem::ONE);
        assert_eq!(got, want, "shortcut mul_assign one");
    }

    /// Unaligned but whole-element slices are legal: every access is an
    /// unaligned load or store. Each offset window holds canonical element
    /// bytes copied in; only its address is odd.
    #[test]
    fn avx2_unaligned_slices() {
        if !host_v3() {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = archmage::X64V3Token::summon().expect("guard passed: AVX2 summons here");
        let len = 72;
        let src_elems = edges(len);
        let dst_elems = canonical(len, 0x82);
        let coeff = Elem::from_raw(3, 5);
        for offset in [0usize, 1, 3, 5] {
            let mut src_storage = vec![0u8; len + 8];
            src_storage[offset..offset + len].copy_from_slice(&src_elems);
            let src = &src_storage[offset..offset + len];
            let mut got = vec![0u8; len];
            mul_into_avx2(token, &mut got, coeff, src);
            assert_eq!(
                got,
                model_map(src, |e| model_elem_mul(coeff, e)),
                "unaligned mul_into offset {offset}"
            );

            // Unaligned destination as well as source.
            let mut dst_storage = vec![0u8; len + 8];
            dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
            let dst = &mut dst_storage[offset..offset + len];
            add_assign_avx2(token, dst, src);
            assert_eq!(
                &dst_storage[offset..offset + len],
                &model_zip(&dst_elems, &src_elems, model_elem_add)[..],
                "unaligned add_assign offset {offset}"
            );
        }
    }

    /// The invocation must panic on bad geometry and leave `dst`
    /// unchanged: validation precedes every write.
    fn expect_geometry_panic(label: &str, dst: &mut [u8], invoke: impl FnOnce(&mut [u8])) {
        let before = dst.to_vec();
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invoke(dst))).is_err();
        assert!(panicked, "{label} must panic");
        assert_eq!(
            &*dst,
            &before[..],
            "{label} must not write before validating"
        );
    }

    /// Geometry validation fires before any write: every entry rejects a
    /// partial trailing element, and every multi-operand entry rejects a
    /// length mismatch.
    #[test]
    fn avx2_geometry_validation_precedes_writes() {
        if !host_v3() {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = archmage::X64V3Token::summon().expect("guard passed: AVX2 summons here");
        let coeff = Elem::from_raw(3, 5);

        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("add partial", &mut partial, |d| {
            add_assign_avx2(token, d, &partial_src);
        });
        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("sub partial", &mut partial, |d| {
            sub_assign_avx2(token, d, &partial_src);
        });
        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("mul_add partial", &mut partial, |d| {
            mul_add_avx2(token, d, coeff, &partial_src);
        });
        let mut partial = [1u8; 12];
        expect_geometry_panic("mul_assign partial", &mut partial, |d| {
            mul_assign_avx2(token, d, coeff);
        });
        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("mul_into partial", &mut partial, |d| {
            mul_into_avx2(token, d, coeff, &partial_src);
        });
        let mut partial = [1u8; 12];
        let partial_src = [1u8; 12];
        expect_geometry_panic("mul_elementwise partial", &mut partial, |d| {
            mul_elementwise_avx2(token, d, &partial_src, &partial_src);
        });

        let mut mismatched = [1u8; 16];
        let short_src = [1u8; 8];
        expect_geometry_panic("add length mismatch", &mut mismatched, |d| {
            add_assign_avx2(token, d, &short_src);
        });
        let mut mismatched = [1u8; 16];
        let full_src = [1u8; 16];
        let short_src = [1u8; 8];
        expect_geometry_panic("mul_elementwise b mismatch", &mut mismatched, |d| {
            mul_elementwise_avx2(token, d, &full_src, &short_src);
        });
    }
}
