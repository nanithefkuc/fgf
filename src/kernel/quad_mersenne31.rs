//! GF((2³¹ − 1)²) kernel entries.
//!
//! Extension arithmetic over Mersenne31 pairs with `i² = −1`, reached by
//! the component rows structurally equal to the shipped prime extension;
//! [`super::extension`] routes them here. On x86 hosts resolving the SSE4.2
//! class, the additive and scalar-broadcast operations run the SSE4.2 kernels
//! (two complex elements per vector) and the multiplies stay here. On x86
//! hosts resolving the AVX2 class, rows at or above the vector floors in
//! [`super::routing`] dispatch to the AVX2 extension kernels (four complex
//! elements per vector); shorter rows stay here. On hosts resolving the
//! AVX-512 class, the coefficient multiply, multiply-accumulate, and
//! elementwise product run the AVX-512 kernels (eight complex elements per
//! vector) and the remaining operations run the AVX2 kernels.
//!
//! The scalar loops canonicalize each loaded limb once and then run raw
//! modular add/sub/mul on limbs known to be `< p`: one conditional subtract
//! per op, no re-reduction of already-canonical operands. Every input limb
//! bit pattern is legal and every stored limb is canonical, matching the
//! vector kernels lane for lane.
//!
//! Each entry matches the cached `QuadMersenne31` strategy resolved once per
//! process; no entry reads the process backend.

use crate::field::quad_mersenne31::QuadMersenne31;
use crate::field::{Elem, FieldBuffer};
use crate::kernel::quad_mersenne_strategy;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::{QuadMersenneAddToken, QuadMersenneStrategy};
use crate::kernel::{prime, scalar};

/// Reduce an arbitrary 32-bit lane to the canonical range `0..p`
/// (the Mersenne fold, `2^31 ≡ 1`).
#[inline]
#[must_use]
const fn canon(x: u32) -> u32 {
    let s = (x & crate::field::mersenne31::MODULUS) + (x >> 31);
    if s >= crate::field::mersenne31::MODULUS {
        s - crate::field::mersenne31::MODULUS
    } else {
        s
    }
}

/// Modular multiply of two canonical (`< p`) limbs.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
const fn raw_mul(a: u32, b: u32) -> u32 {
    let prod = (a as u64) * (b as u64);
    let s = ((prod as u32) & crate::field::mersenne31::MODULUS) + (prod >> 31) as u32;
    if s >= crate::field::mersenne31::MODULUS {
        s - crate::field::mersenne31::MODULUS
    } else {
        s
    }
}

/// Modular add of two canonical limbs.
#[inline]
#[must_use]
const fn raw_add(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= crate::field::mersenne31::MODULUS {
        s - crate::field::mersenne31::MODULUS
    } else {
        s
    }
}

/// `ac - bd` for canonical limbs; the wrapping add of `p` absorbs the borrow.
#[inline]
#[must_use]
const fn raw_sub(ac: u32, bd: u32) -> u32 {
    let s = ac
        .wrapping_sub(bd)
        .wrapping_add(crate::field::mersenne31::MODULUS);
    if s >= crate::field::mersenne31::MODULUS {
        s - crate::field::mersenne31::MODULUS
    } else {
        s
    }
}

/// Encode two already-canonical (`< p`) limbs directly without re-reducing.
#[inline]
fn encode_canonical(dst: &mut [u8], limbs: [u32; 2]) {
    debug_assert!(limbs[0] < crate::field::mersenne31::MODULUS);
    debug_assert!(limbs[1] < crate::field::mersenne31::MODULUS);
    dst[..4].copy_from_slice(&limbs[0].to_le_bytes());
    dst[4..8].copy_from_slice(&limbs[1].to_le_bytes());
}

/// `(a+bi)(c+di)` over canonical limbs, returning canonical limbs.
#[inline]
#[must_use]
const fn qmul_limbs(ar: u32, ai: u32, br: u32, bi: u32) -> [u32; 2] {
    let ac = raw_mul(ar, br);
    let bd = raw_mul(ai, bi);
    let ad = raw_mul(ar, bi);
    let bc = raw_mul(ai, br);
    [raw_sub(ac, bd), raw_add(ad, bc)]
}

/// `(a+bi)(c+di)` over canonical limbs.
#[inline]
#[must_use]
const fn qmul(ar: u32, ai: u32, br: u32, bi: u32) -> Elem<QuadMersenne31> {
    Elem::<QuadMersenne31>::from_raw(qmul_limbs(ar, ai, br, bi))
}

/// The backend the QM31 kernels report on this host.
#[inline]
pub(crate) fn backend_report() -> crate::kernel::Backend {
    let report = quad_mersenne_strategy().report();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if report == crate::kernel::Backend::Scalar
        && matches!(
            crate::kernel::quad_mersenne_add_token(),
            QuadMersenneAddToken::V2(_)
        )
    {
        return crate::kernel::Backend::V2;
    }
    report
}

/// `dst += src`.
pub(crate) fn add_assign(dst: &mut [u8], src: &[u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if dst.len() >= crate::kernel::routing::QUAD_VECTOR_ADD_MIN_BYTES {
        match crate::kernel::quad_mersenne_add_token() {
            QuadMersenneAddToken::V3(token) => {
                crate::kernel::x86::quad_mersenne31::add_assign_avx2(token, dst, src);
                return;
            }
            QuadMersenneAddToken::V2(token) => {
                crate::kernel::x86::quad_mersenne31::add_assign_sse42(token, dst, src);
                return;
            }
            QuadMersenneAddToken::Scalar => {}
        }
    }
    add_assign_short(dst, src);
}

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), inline(never))]
fn add_assign_short(dst: &mut [u8], src: &[u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let QuadMersenneAddToken::V2(token) = crate::kernel::quad_mersenne_add_token() {
        crate::kernel::x86::quad_mersenne31::add_assign_sse42(token, dst, src);
        return;
    }
    for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
        let a = QuadMersenne31::decode(d).to_raw();
        let b = QuadMersenne31::decode(s).to_raw();
        // Both buffers are caller-owned bytes and may hold non-canonical
        // raw lanes; canonicalize each limb once, then one conditional
        // subtract folds the sum.
        QuadMersenne31::encode(
            d,
            Elem::<QuadMersenne31>::from_raw([
                raw_add(canon(a[0]), canon(b[0])),
                raw_add(canon(a[1]), canon(b[1])),
            ]),
        );
    }
}

/// `dst -= src`.
pub(crate) fn sub_assign(dst: &mut [u8], src: &[u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if dst.len() >= crate::kernel::routing::QUAD_VECTOR_ADD_MIN_BYTES {
        match crate::kernel::quad_mersenne_add_token() {
            QuadMersenneAddToken::V3(token) => {
                crate::kernel::x86::quad_mersenne31::sub_assign_avx2(token, dst, src);
                return;
            }
            QuadMersenneAddToken::V2(token) => {
                crate::kernel::x86::quad_mersenne31::sub_assign_sse42(token, dst, src);
                return;
            }
            QuadMersenneAddToken::Scalar => {}
        }
    }
    sub_assign_short(dst, src);
}

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), inline(never))]
fn sub_assign_short(dst: &mut [u8], src: &[u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let QuadMersenneAddToken::V2(token) = crate::kernel::quad_mersenne_add_token() {
        crate::kernel::x86::quad_mersenne31::sub_assign_sse42(token, dst, src);
        return;
    }
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
        let a = QuadMersenne31::decode(d).to_raw();
        let b = QuadMersenne31::decode(s).to_raw();
        // dst + (-src); negation of a canonical limb is p - limb (0 stays 0).
        let br = canon(b[0]);
        let bi = canon(b[1]);
        let ar = canon(a[0]);
        let ai = canon(a[1]);
        QuadMersenne31::encode(
            d,
            Elem::<QuadMersenne31>::from_raw([
                raw_add(
                    ar,
                    if br == 0 {
                        0
                    } else {
                        crate::field::mersenne31::MODULUS - br
                    },
                ),
                raw_add(
                    ai,
                    if bi == 0 {
                        0
                    } else {
                        crate::field::mersenne31::MODULUS - bi
                    },
                ),
            ]),
        );
    }
}

/// `dst += coeff * src`.
pub(crate) fn mul_add(dst: &mut [u8], coeff: Elem<QuadMersenne31>, src: &[u8]) {
    match quad_mersenne_strategy() {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx512
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_add_avx512(
                crate::kernel::x86_v4_token(),
                dst,
                coeff,
                src,
            );
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx2
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_add_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                coeff,
                src,
            );
            return;
        }
        _ => {}
    }
    let cr = canon(coeff.to_raw()[0]);
    let ci = canon(coeff.to_raw()[1]);
    if cr == 0 && ci == 0 {
        return;
    }
    if cr == 1 && ci == 0 {
        add_assign(dst, src);
        return;
    }
    for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
        let ar = canon(u32::from_le_bytes([d[0], d[1], d[2], d[3]]));
        let ai = canon(u32::from_le_bytes([d[4], d[5], d[6], d[7]]));
        let br = canon(u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
        let bi = canon(u32::from_le_bytes([s[4], s[5], s[6], s[7]]));
        let prod = qmul_limbs(br, bi, cr, ci);
        encode_canonical(d, [raw_add(ar, prod[0]), raw_add(ai, prod[1])]);
    }
}

/// `dst *= coeff`.
pub(crate) fn mul_assign(dst: &mut [u8], coeff: Elem<QuadMersenne31>) {
    match quad_mersenne_strategy() {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx512
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_assign_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                coeff,
            );
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx2
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_assign_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                coeff,
            );
            return;
        }
        _ => {}
    }
    let cr = canon(coeff.to_raw()[0]);
    let ci = canon(coeff.to_raw()[1]);
    if cr == 0 && ci == 0 {
        dst.fill(0);
        return;
    }
    if cr == 1 && ci == 0 {
        return;
    }
    for d in dst.chunks_exact_mut(8) {
        let a = QuadMersenne31::decode(d).to_raw();
        QuadMersenne31::encode(d, qmul(canon(a[0]), canon(a[1]), cr, ci));
    }
}

/// `dst = coeff * src`, never reading the old destination.
pub(crate) fn mul_into(dst: &mut [u8], coeff: Elem<QuadMersenne31>, src: &[u8]) {
    match quad_mersenne_strategy() {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx512
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_into_avx512(
                crate::kernel::x86_v4_token(),
                dst,
                coeff,
                src,
            );
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx2
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_into_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                coeff,
                src,
            );
            return;
        }
        _ => {}
    }
    let cr = canon(coeff.to_raw()[0]);
    let ci = canon(coeff.to_raw()[1]);
    if cr == 0 && ci == 0 {
        dst.fill(0);
        return;
    }
    for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
        let b = QuadMersenne31::decode(s).to_raw();
        QuadMersenne31::encode(d, qmul(canon(b[0]), canon(b[1]), cr, ci));
    }
}

/// `dst[i] = a[i] * b[i]`.
pub(crate) fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    match quad_mersenne_strategy() {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx512
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_elementwise_avx512(
                crate::kernel::x86_v4_token(),
                dst,
                a,
                b,
            );
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx2
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_elementwise_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                a,
                b,
            );
            return;
        }
        _ => {}
    }
    for (d, x, y) in dst
        .chunks_exact_mut(8)
        .zip(a.chunks_exact(8))
        .zip(b.chunks_exact(8))
        .map(|((d, x), y)| (d, x, y))
    {
        let xa = QuadMersenne31::decode(x).to_raw();
        let xb = QuadMersenne31::decode(y).to_raw();
        QuadMersenne31::encode(
            d,
            qmul(canon(xa[0]), canon(xa[1]), canon(xb[0]), canon(xb[1])),
        );
    }
}

/// `dst[i] *= src[i]`.
pub(crate) fn mul_elementwise_assign(dst: &mut [u8], src: &[u8]) {
    match quad_mersenne_strategy() {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx512
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_elementwise_assign_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                src,
            );
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        QuadMersenneStrategy::Avx2
            if dst.len() >= crate::kernel::routing::QUAD_VECTOR_MUL_MIN_BYTES =>
        {
            crate::kernel::x86::quad_mersenne31::mul_elementwise_assign_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                src,
            );
            return;
        }
        _ => {}
    }
    scalar::mul_elementwise_assign::<QuadMersenne31>(dst, src);
}

/// `dst[i] += value`.
pub(crate) fn add_assign_scalar(dst: &mut [u8], value: Elem<QuadMersenne31>) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if dst.len() >= crate::kernel::routing::QUAD_VECTOR_ADD_MIN_BYTES {
        match crate::kernel::quad_mersenne_add_token() {
            QuadMersenneAddToken::V3(token) => {
                crate::kernel::x86::quad_mersenne31::add_assign_scalar_avx2(token, dst, value);
                return;
            }
            QuadMersenneAddToken::V2(token) => {
                crate::kernel::x86::quad_mersenne31::add_assign_scalar_sse42(token, dst, value);
                return;
            }
            QuadMersenneAddToken::Scalar => {}
        }
    }
    add_assign_scalar_short(dst, value);
}

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), inline(never))]
fn add_assign_scalar_short(dst: &mut [u8], value: Elem<QuadMersenne31>) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let QuadMersenneAddToken::V2(token) = crate::kernel::quad_mersenne_add_token() {
        crate::kernel::x86::quad_mersenne31::add_assign_scalar_sse42(token, dst, value);
        return;
    }
    prime::add_assign_scalar::<QuadMersenne31>(dst, value);
}

/// `dst[i] -= value`.
pub(crate) fn sub_assign_scalar(dst: &mut [u8], value: Elem<QuadMersenne31>) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if dst.len() >= crate::kernel::routing::QUAD_VECTOR_ADD_MIN_BYTES {
        match crate::kernel::quad_mersenne_add_token() {
            QuadMersenneAddToken::V3(token) => {
                crate::kernel::x86::quad_mersenne31::sub_assign_scalar_avx2(token, dst, value);
                return;
            }
            QuadMersenneAddToken::V2(token) => {
                crate::kernel::x86::quad_mersenne31::sub_assign_scalar_sse42(token, dst, value);
                return;
            }
            QuadMersenneAddToken::Scalar => {}
        }
    }
    sub_assign_scalar_short(dst, value);
}

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), inline(never))]
fn sub_assign_scalar_short(dst: &mut [u8], value: Elem<QuadMersenne31>) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let QuadMersenneAddToken::V2(token) = crate::kernel::quad_mersenne_add_token() {
        crate::kernel::x86::quad_mersenne31::sub_assign_scalar_sse42(token, dst, value);
        return;
    }
    let [vr, vi] = value.to_raw();
    for d in dst.chunks_exact_mut(8) {
        let ar = canon(u32::from_le_bytes([d[0], d[1], d[2], d[3]]));
        let ai = canon(u32::from_le_bytes([d[4], d[5], d[6], d[7]]));
        encode_canonical(d, [raw_sub(ar, vr), raw_sub(ai, vi)]);
    }
}
