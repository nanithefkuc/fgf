//! GF((2³¹ − 1)²) kernel dispatch.
//!
//! Extension arithmetic over Mersenne31 pairs with `i² = −1`. Kernels are
//! composed from the base Mersenne31 lane kernels — no hand-written SIMD
//! for the extension yet — so every backend currently reports `Scalar`.
//!
//! The loops canonicalize each loaded limb once and then run raw modular
//! add/sub/mul on limbs known to be `< p`: one conditional subtract per op,
//! no re-reduction of already-canonical operands. Raw non-canonical lanes in
//! a destination are still legal input — every entry point canonicalizes what
//! it reads before computing. Measured against the canonicalize-per-helper
//! form in BENCHMARKS.md.

use crate::field::Field;
use crate::field::quad_mersenne31::{Elem, QuadMersenne31};
use crate::kernel::FieldKernels;

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

/// `(a+bi)(c+di)` over canonical limbs.
#[inline]
#[must_use]
const fn qmul(ar: u32, ai: u32, br: u32, bi: u32) -> Elem {
    let ac = raw_mul(ar, br);
    let bd = raw_mul(ai, bi);
    let ad = raw_mul(ar, bi);
    let bc = raw_mul(ai, br);
    Elem(raw_sub(ac, bd), raw_add(ad, bc))
}

impl FieldKernels for QuadMersenne31 {
    /// The canonical pair `(re, im)`, used as-is.
    type Prepared = Elem;

    #[inline]
    fn prepare(coeff: Elem) -> Elem {
        coeff.canonical()
    }

    #[inline]
    fn prepared_coeff(prepared: &Elem) -> Elem {
        *prepared
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }

    fn add_assign(dst: &mut [u8], src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let a = QuadMersenne31::read(d);
            let b = QuadMersenne31::read(s);
            // Both buffers are caller-owned bytes and may hold non-canonical
            // raw lanes; canonicalize each limb once, then one conditional
            // subtract folds the sum.
            QuadMersenne31::write(
                d,
                Elem(
                    raw_add(canon(a.0), canon(b.0)),
                    raw_add(canon(a.1), canon(b.1)),
                ),
            );
        }
    }

    fn sub_assign(dst: &mut [u8], src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let a = QuadMersenne31::read(d);
            let b = QuadMersenne31::read(s);
            // dst + (-src); negation of a canonical limb is p - limb (0 stays 0).
            let br = canon(b.0);
            let bi = canon(b.1);
            let ar = canon(a.0);
            let ai = canon(a.1);
            QuadMersenne31::write(
                d,
                Elem(
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
                ),
            );
        }
    }

    fn mul_add(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        let cr = canon(coeff.0);
        let ci = canon(coeff.1);
        if cr == 0 && ci == 0 {
            return;
        }
        if cr == 1 && ci == 0 {
            Self::add_assign(dst, src);
            return;
        }
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let a = QuadMersenne31::read(d);
            let b = QuadMersenne31::read(s);
            let prod = qmul(canon(b.0), canon(b.1), cr, ci);
            QuadMersenne31::write(
                d,
                Elem(raw_add(canon(a.0), prod.0), raw_add(canon(a.1), prod.1)),
            );
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Elem) {
        let cr = canon(coeff.0);
        let ci = canon(coeff.1);
        if cr == 0 && ci == 0 {
            dst.fill(0);
            return;
        }
        if cr == 1 && ci == 0 {
            return;
        }
        for d in dst.chunks_exact_mut(8) {
            let a = QuadMersenne31::read(d);
            QuadMersenne31::write(d, qmul(canon(a.0), canon(a.1), cr, ci));
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        let cr = canon(coeff.0);
        let ci = canon(coeff.1);
        if cr == 0 && ci == 0 {
            dst.fill(0);
            return;
        }
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let b = QuadMersenne31::read(s);
            QuadMersenne31::write(d, qmul(canon(b.0), canon(b.1), cr, ci));
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &coeff, src);
        }
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &coeff, src);
        }
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &coeff, src);
            }
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        debug_assert_eq!(dst.len(), a.len());
        debug_assert_eq!(dst.len(), b.len());
        for ((d, x), y) in dst
            .chunks_exact_mut(8)
            .zip(a.chunks_exact(8))
            .zip(b.chunks_exact(8))
        {
            let xa = QuadMersenne31::read(x);
            let xb = QuadMersenne31::read(y);
            QuadMersenne31::write(d, qmul(canon(xa.0), canon(xa.1), canon(xb.0), canon(xb.1)));
        }
    }
}
