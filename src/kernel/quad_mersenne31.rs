//! GF((2³¹ − 1)²) kernel dispatch.
//!
//! Extension arithmetic over Mersenne31 pairs with `i² = −1`. Kernels are
//! composed from the base Mersenne31 lane kernels — no hand-written SIMD
//! for the extension yet — so every backend currently reports `Scalar`.
//! Correctness is differential-tested against the portable `prime` oracle
//! adapted for the extension (the scalar `Elem` itself is the oracle).

use crate::field::Elem as ElemTrait;
use crate::field::Field;
use crate::field::quad_mersenne31::{Elem, QuadMersenne31};
use crate::kernel::FieldKernels;

impl FieldKernels for QuadMersenne31 {
    /// Canonical pair `(re, im)`, used as-is.
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
            QuadMersenne31::write(d, a.add(b));
        }
    }

    fn sub_assign(dst: &mut [u8], src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let a = QuadMersenne31::read(d);
            let b = QuadMersenne31::read(s);
            QuadMersenne31::write(d, a.sub(b));
        }
    }

    fn mul_add(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        if coeff.is_zero() {
            return;
        }
        if coeff.is_one() {
            Self::add_assign(dst, src);
            return;
        }
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let a = QuadMersenne31::read(d);
            let b = QuadMersenne31::read(s);
            QuadMersenne31::write(d, a.add(b.mul(*coeff)));
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Elem) {
        if coeff.is_one() {
            return;
        }
        if coeff.is_zero() {
            dst.fill(0);
            return;
        }
        for d in dst.chunks_exact_mut(8) {
            let a = QuadMersenne31::read(d);
            QuadMersenne31::write(d, a.mul(*coeff));
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        if coeff.is_zero() {
            dst.fill(0);
            return;
        }
        for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let v = QuadMersenne31::read(s).mul(*coeff);
            QuadMersenne31::write(d, v);
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
            QuadMersenne31::write(d, QuadMersenne31::read(x).mul(QuadMersenne31::read(y)));
        }
    }
}
