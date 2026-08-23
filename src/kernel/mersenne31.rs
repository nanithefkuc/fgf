//! GF(2^31 − 1) kernel dispatch.
//!
//! Prime-field arithmetic over 32-bit lanes. On x86 with AVX2 (`V3`) or SSE4.1
//! (`V2`) the add/sub/multiply loops run over packed integer lanes with a
//! Mersenne fold (`2^31 ≡ 1`); every other target runs the portable
//! [`crate::kernel::prime`] path, so [`FieldKernels::active_backend`] reports
//! `Scalar` there. The coefficient's prepared form is simply its canonical
//! lane word, which the kernels broadcast on entry.

use crate::field::mersenne31::{Elem, Mersenne31};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, prime};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

/// Whether the running backend has integer-SIMD prime kernels for this field.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn vectorized() -> bool {
    matches!(backend(), Backend::V3GfniCrypto | Backend::V3 | Backend::V2)
}

impl FieldKernels for Mersenne31 {
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
            vectorized()
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            false
        }
    }

    fn add_assign(dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::add_assign_m31_avx2(dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::add_assign_m31_sse41(dst, src),
            _ => prime::add_assign::<Mersenne31>(dst, src),
        }
    }

    fn sub_assign(dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::sub_assign_m31_avx2(dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::sub_assign_m31_sse41(dst, src),
            _ => prime::sub_assign::<Mersenne31>(dst, src),
        }
    }

    fn mul_add(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::prime::mul_add_m31_avx2(dst, coeff.to_raw(), src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_add_m31_sse41(dst, coeff.to_raw(), src),
            _ => prime::mul_add::<Mersenne31>(dst, *coeff, src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Elem) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::prime::mul_assign_m31_avx2(dst, coeff.to_raw());
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_assign_m31_sse41(dst, coeff.to_raw()),
            _ => prime::mul_assign::<Mersenne31>(dst, *coeff),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Elem, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::prime::mul_into_m31_avx2(dst, coeff.to_raw(), src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_into_m31_sse41(dst, coeff.to_raw(), src),
            _ => {
                for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    let v = *coeff * Elem(u32::from_le_bytes(s.try_into().unwrap()));
                    d.copy_from_slice(&v.to_raw().to_le_bytes());
                }
            }
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        prime::mul_add_scatter::<Mersenne31>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        prime::mul_add_gather::<Mersenne31>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        prime::mul_add_matrix::<Mersenne31>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::prime::mul_elementwise_m31_avx2(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::prime::mul_elementwise_m31_sse41(dst, a, b),
            _ => prime::mul_elementwise::<Mersenne31>(dst, a, b),
        }
    }
}
