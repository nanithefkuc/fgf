//! GF(2^64 − 2^32 + 1) kernel dispatch.
//!
//! Prime-field arithmetic over 64-bit lanes. On x86 with AVX-512 (`V4x`,
//! under `simd512`) or AVX2 (`V3`) the multiply uses 32-bit-half `vpmuludq`
//! splitting and the Goldilocks split-fold (`2^64 ≡ 2^32 − 1`); SSE4.1
//! (`V2`) runs the same construction at half width. Every other target runs
//! the portable [`crate::kernel::prime`] path, so [`FieldKernels::backend`]
//! reports `Scalar` there. The prepared form is the canonical lane word,
//! broadcast by the kernels on entry.

use crate::field::Elem;
use crate::field::goldilocks::Goldilocks;
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, prime, scalar};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86_v4_token as v4;

impl FieldKernels for Goldilocks {
    #[inline]
    fn backend() -> Backend {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            match backend() {
                #[cfg(feature = "simd512")]
                Backend::V4x => Backend::V4x,
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
            matches!(
                backend(),
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2
            )
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            false
        }
    }
}

impl KernelDispatch for Goldilocks {
    /// The canonical lane word, broadcast by the kernels on entry.
    type Prepared = Elem<Goldilocks>;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<Goldilocks>) -> Elem<Goldilocks> {
        coeff
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Elem<Goldilocks>) -> Elem<Goldilocks> {
        *prepared
    }

    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::add_assign_avx512(v4(), dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::add_assign_avx2(crate::kernel::x86_v3_token(), dst, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::add_assign_sse42(crate::kernel::x86_v2_token(), dst, src);
            }
            _ => prime::add_assign::<Goldilocks>(dst, src),
        }
    }

    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::sub_assign_avx512(v4(), dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::sub_assign_avx2(crate::kernel::x86_v3_token(), dst, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::sub_assign_sse42(crate::kernel::x86_v2_token(), dst, src);
            }
            _ => prime::sub_assign::<Goldilocks>(dst, src),
        }
    }

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Elem<Goldilocks>, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::mul_add_avx512(v4(), dst, coeff.to_raw(), src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::mul_add_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    coeff.to_raw(),
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::goldilocks::mul_add_sse42(
                crate::kernel::x86_v2_token(),
                dst,
                coeff.to_raw(),
                src,
            ),
            _ => prime::mul_add::<Goldilocks>(dst, *coeff, src),
        }
    }

    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Elem<Goldilocks>) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::mul_assign_avx512(v4(), dst, coeff.to_raw()),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::mul_assign_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    coeff.to_raw(),
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::mul_assign_sse42(
                    crate::kernel::x86_v2_token(),
                    dst,
                    coeff.to_raw(),
                );
            }
            _ => prime::mul_assign::<Goldilocks>(dst, *coeff),
        }
    }

    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Elem<Goldilocks>, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::mul_into_avx512(v4(), dst, coeff.to_raw(), src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::mul_into_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    coeff.to_raw(),
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::goldilocks::mul_into_sse42(
                crate::kernel::x86_v2_token(),
                dst,
                coeff.to_raw(),
                src,
            ),
            _ => {
                for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                    let v = *coeff
                        * Elem::<Goldilocks>::from_raw(u64::from_le_bytes(s.try_into().unwrap()));
                    d.copy_from_slice(&v.to_raw().to_le_bytes());
                }
            }
        }
    }

    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Goldilocks>],
        src: &[u8],
    ) {
        prime::mul_add_scatter::<Goldilocks>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<Goldilocks>],
        srcs: &[&[u8]],
    ) {
        prime::mul_add_gather::<Goldilocks>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Goldilocks>], &[u8])],
    ) {
        prime::mul_add_matrix::<Goldilocks>(rows, row_len, nrows, terms);
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::mul_elementwise_avx512(v4(), dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::mul_elementwise_avx2(crate::kernel::x86_v3_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::mul_elementwise_sse42(crate::kernel::x86_v2_token(), dst, a, b);
            }
            _ => prime::mul_elementwise::<Goldilocks>(dst, a, b),
        }
    }

    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::mul_elementwise_assign_avx512(v4(), dst, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::mul_elementwise_assign_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::mul_elementwise_assign_sse42(
                    crate::kernel::x86_v2_token(),
                    dst,
                    src,
                );
            }
            _ => scalar::mul_elementwise_assign::<Goldilocks>(dst, src),
        }
    }

    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Elem<Goldilocks>) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::add_assign_scalar_avx512(v4(), dst, value.to_raw()),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::add_assign_scalar_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    value.to_raw(),
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::add_assign_scalar_sse42(
                    crate::kernel::x86_v2_token(),
                    dst,
                    value.to_raw(),
                );
            }
            _ => prime::add_assign_scalar::<Goldilocks>(dst, *value),
        }
    }

    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Elem<Goldilocks>) {
        match backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::goldilocks::sub_assign_scalar_avx512(v4(), dst, value.to_raw()),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::goldilocks::sub_assign_scalar_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    value.to_raw(),
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::goldilocks::sub_assign_scalar_sse42(
                    crate::kernel::x86_v2_token(),
                    dst,
                    value.to_raw(),
                );
            }
            _ => prime::sub_assign_scalar::<Goldilocks>(dst, *value),
        }
    }
}
