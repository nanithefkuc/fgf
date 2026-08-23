//! GF(2^64 − 2^32 + 1) kernel dispatch.
//!
//! Prime-field arithmetic over 64-bit lanes. On x86 with AVX2 (`V3`) the
//! multiply uses 32-bit-half `vpmuludq` splitting and the Goldilocks split-fold
//! (`2^64 ≡ 2^32 − 1`); SSE4.1 (`V2`) runs the same construction at half width.
//! Every other target runs the portable [`crate::kernel::prime`] path, so
//! [`FieldKernels::active_backend`] reports `Scalar` there. The prepared form
//! is the canonical lane word, broadcast by the kernels on entry.

use crate::field::goldilocks::{Elem, Goldilocks};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, prime};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

impl FieldKernels for Goldilocks {
    /// The canonical lane word, broadcast by the kernels on entry.
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
    fn active_backend() -> Backend {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            match backend() {
                Backend::V3GfniCrypto | Backend::V3 => Backend::V3,
                Backend::V2 => Backend::V2,
                _ => Backend::Scalar,
            }
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            Backend::Scalar
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            matches!(backend(), Backend::V3GfniCrypto | Backend::V3 | Backend::V2)
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            false
        }
    }

    fn add_assign(dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::add_assign_gld_avx2(dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::add_assign_gld_sse41(dst, src),
            _ => prime::add_assign::<Goldilocks>(dst, src),
        }
    }

    fn sub_assign(dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::sub_assign_gld_avx2(dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::sub_assign_gld_sse41(dst, src),
            _ => prime::sub_assign::<Goldilocks>(dst, src),
        }
    }

    fn mul_add(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::prime::mul_add_gld_avx2(dst, coeff.to_raw(), src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_add_gld_sse41(dst, coeff.to_raw(), src),
            _ => prime::mul_add::<Goldilocks>(dst, *coeff, src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Elem) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::prime::mul_assign_gld_avx2(dst, coeff.to_raw());
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_assign_gld_sse41(dst, coeff.to_raw()),
            _ => prime::mul_assign::<Goldilocks>(dst, *coeff),
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        prime::mul_add_scatter::<Goldilocks>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        prime::mul_add_gather::<Goldilocks>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        prime::mul_add_matrix::<Goldilocks>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::mul_elementwise_gld_avx2(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_elementwise_gld_sse41(dst, a, b),
            _ => prime::mul_elementwise::<Goldilocks>(dst, a, b),
        }
    }
}
