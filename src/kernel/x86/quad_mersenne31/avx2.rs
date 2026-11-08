//! `QuadMersenne31` (`(re, im)` Mersenne31 limb pairs) AVX2 kernels.
//!
//! Four complex elements per 256-bit vector: eight 32-bit lanes holding
//! `[re0, im0, re1, im1, re2, im2, re3, im3]`, little-endian, so the even
//! 32-bit lanes are the real limbs and the odd lanes the imaginary limbs.
//! The limb arithmetic — the direct canonicalizing min chain — is the
//! Mersenne31 discipline shared with
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
//! Packed buffers accept arbitrary limb bit patterns, as specified by
//! [`crate::ops`]. Multiplication keeps limbs in the redundant range
//! `0..=p` internally; all arithmetic stores canonical limbs, except the
//! documented byte-preserving coefficient shortcuts. Coefficients enter
//! through canonical [`Elem`] storage.
//!
//! Reaching these entries directly is an `internals` surface, like the
//! other direct x86 kernels: each is a safe capability-token function
//! taking an `X64V3Token`, validating its own geometry, and handing
//! whole-element tails to the portable [`crate::kernel::prime`] path. The
//! zero and one coefficient shortcuts mirror the scalar control exactly,
//! byte for byte.

use crate::field::Elem;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::field::{FieldBuffer, mersenne31};
use crate::kernel::proven_checks::{check_elem_multiple, check_equal};
use crate::kernel::x86::mersenne31::{P, min_chain_avx2};
use crate::kernel::{prime, scalar};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Duplicate each element's imaginary limb into both lanes of its pair.
#[archmage::rite(v3)]
#[must_use]
fn dup_im(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)))
}

/// Reduce complex-multiply inputs to `0..=p`, retaining redundant zero.
/// Lanes without their high bit set are already below `2^31`, i.e. `0..=p`.
/// That range keeps `cmul`'s and `cmul_vec`'s deferred 64-bit accumulators at
/// most `2p^2`, which `reduce_pair` still folds to canonical limbs, and it
/// keeps `cmul_vec`'s `XOR p` negation exact (`p` is all low 31 bits set, so
/// the XOR flips only those bits). Vectors with any high bit set
/// canonicalize before the product, including raw aliases.
#[archmage::rite(v3)]
#[must_use]
fn mul_input(x: __m256i, p: __m256i) -> __m256i {
    if _mm256_movemask_ps(_mm256_castsi256_ps(x)) == 0 {
        x
    } else {
        min_chain_avx2(x, p)
    }
}

/// `a + b (mod p)` over eight lanes (`0..=p` in, canonical out).
#[archmage::rite(v3)]
#[must_use]
fn addmod(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    min_chain_avx2(_mm256_add_epi32(a, b), p) // sum <= 2p
}

/// Multiply four complex elements by a prepared coefficient, each input lane
/// `0..=p`: `cr` broadcast to all lanes, `ci_mixed` holding `p - ci` on the
/// real lanes and `ci` on the imaginary lanes. The real lanes accumulate
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

/// Complex multiply of two vectors with each input lane `0..=p`,
/// `(ar*br - ai*bi, ar*bi + ai*br)` per element, negating the duplicated
/// imaginary limbs of `b` with `p` and reducing the deferred 64-bit
/// accumulators.
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
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
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
        // Canonical limbs sum to at most `2p - 2` per lane, so one final
        // subtract-and-min completes the reduction.
        let d = min_chain_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let s = min_chain_avx2(_mm256_loadu_si256(src_lane), p);
        let sum = _mm256_add_epi32(d, s); // <= 2p - 2
        _mm256_storeu_si256(dst_lane, _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p)));
    }
    prime::add_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst -= src` in GF((2^31 - 1)^2), AVX2.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
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
        // The unsigned min of the wrapping difference and its `+p` fixup is
        // already canonical for canonical limbs.
        let d = min_chain_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let s = min_chain_avx2(_mm256_loadu_si256(src_lane), p);
        let diff = _mm256_sub_epi32(d, s); // wraps iff negative
        _mm256_storeu_si256(dst_lane, _mm256_min_epu32(diff, _mm256_add_epi32(diff, p)));
    }
    prime::sub_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst += coeff * src` in GF((2^31 - 1)^2), AVX2.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical. A zero coefficient leaves
/// `dst` untouched and the coefficient `1 + 0i` adds `src` elementwise,
/// mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_add_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_add_avx2", dst.len(), 8);
    let [cr, ci] = coeff.to_raw();
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
        // `mul_input` keeps `cmul`'s deferred reduction and the final
        // `addmod` inside their proven ranges.
        let prod = cmul(
            mul_input(_mm256_loadu_si256(src_lane), p),
            cr_v,
            ci_mixed,
            p,
        );
        let d = min_chain_avx2(_mm256_loadu_si256(&*dst_lane), p);
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
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], coeff: Elem<QuadMersenne31>) {
    check_elem_multiple("quad_mersenne31::mul_assign_avx2", dst.len(), 8);
    let [cr, ci] = coeff.to_raw();
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
        // `mul_input` suffices for `cmul`, as above.
        let d = mul_input(_mm256_loadu_si256(&*dst_lane), p);
        _mm256_storeu_si256(dst_lane, cmul(d, cr_v, ci_mixed, p));
    }
    prime::mul_assign::<QuadMersenne31>(dst_tail, coeff);
}

/// `dst = coeff * src` in GF((2^31 - 1)^2), AVX2. The destination is
/// write-only: its prior contents are ignored, never read.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical. A zero coefficient zeroes
/// `dst`, mirroring the scalar control.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    coeff: Elem<QuadMersenne31>,
    src: &[u8],
) {
    check_equal(
        "quad_mersenne31::mul_into_avx2",
        "dst",
        dst.len(),
        "src",
        src.len(),
    );
    check_elem_multiple("quad_mersenne31::mul_into_avx2", dst.len(), 8);
    let [cr, ci] = coeff.to_raw();
    if cr == 0 && ci == 0 {
        dst.fill(0);
        return;
    }
    let p = _mm256_set1_epi32(P);
    let (cr_v, ci_mixed) = coeff_vectors(cr, ci);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_tail) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        // `mul_input` suffices for `cmul`, as above.
        let s = mul_input(_mm256_loadu_si256(src_lane), p);
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
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
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
        // `mul_input` keeps every lane `0..=p`: the duplicated-imaginary
        // `XOR p` negation below stays exact and the deferred accumulators
        // stay inside `reduce_pair`'s proven range.
        let va = mul_input(_mm256_loadu_si256(a_lane), p);
        let vb = mul_input(_mm256_loadu_si256(b_lane), p);
        _mm256_storeu_si256(dst_lane, cmul_vec(va, vb, p));
    }
    prime::mul_elementwise::<QuadMersenne31>(dst_tail, a_tail, b_tail);
}

/// `dst[i] = dst[i] * src[i]` in GF((2^31 - 1)^2), AVX2.
///
/// Every input limb bit pattern is legal: limbs canonicalize on load,
/// and every stored limb is canonical.
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
        // As above: `mul_input` keeps the negation exact.
        let vd = mul_input(_mm256_loadu_si256(&*dst_lane), p);
        let vs = mul_input(_mm256_loadu_si256(src_lane), p);
        _mm256_storeu_si256(dst_lane, cmul_vec(vd, vs, p));
    }
    scalar::mul_elementwise_assign::<QuadMersenne31>(dst_tail, src_tail);
}

/// `dst[i] += value` in GF((2^31 - 1)^2), AVX2. The raw `value` limbs are
/// canonicalized once on entry.
///
/// The destination limbs canonicalize on load; the canonical sum reduces
/// with one subtract-and-min, and every stored limb is canonical.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn add_assign_scalar_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    value: Elem<QuadMersenne31>,
) {
    check_elem_multiple("quad_mersenne31::add_assign_scalar_avx2", dst.len(), 8);
    // The broadcast value is loop-invariant: canonicalize the raw limbs once.
    let [vr, vi] = value.to_raw();
    // Broadcast the (re, im) pair onto the alternating limb lanes.
    let svec = _mm256_blend_epi32::<0b1010_1010>(
        _mm256_set1_epi32(vr as i32),
        _mm256_set1_epi32(vi as i32),
    );
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = min_chain_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let sum = _mm256_add_epi32(d, svec); // <= 2p - 2
        _mm256_storeu_si256(dst_lane, _mm256_min_epu32(sum, _mm256_sub_epi32(sum, p)));
    }
    prime::add_assign_scalar::<QuadMersenne31>(dst_tail, value);
}

/// `dst[i] -= value` in GF((2^31 - 1)^2), AVX2. The raw `value` limbs are
/// canonicalized once on entry.
///
/// The destination limbs canonicalize on load; the canonical difference is
/// canonical after one unsigned min with its `+p` fixup, matching
/// [`sub_assign_avx2`].
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::cast_possible_wrap, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn sub_assign_scalar_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    value: Elem<QuadMersenne31>,
) {
    check_elem_multiple("quad_mersenne31::sub_assign_scalar_avx2", dst.len(), 8);
    let [vr, vi] = value.to_raw();
    let svec = _mm256_blend_epi32::<0b1010_1010>(
        _mm256_set1_epi32(vr as i32),
        _mm256_set1_epi32(vi as i32),
    );
    let p = _mm256_set1_epi32(P);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let d = min_chain_avx2(_mm256_loadu_si256(&*dst_lane), p);
        let diff = _mm256_sub_epi32(d, svec); // wraps iff negative
        _mm256_storeu_si256(dst_lane, _mm256_min_epu32(diff, _mm256_add_epi32(diff, p)));
    }
    prime::sub_assign_scalar::<QuadMersenne31>(dst_tail, value);
}
