//! Kernel dispatch for the Fan–Paar field API.
//!
//! [`FanPaar16`], [`FanPaar32`], and [`FanPaar64`] have hand-written x86
//! kernels over the canonical Wiedemann tower representation. Unsupported
//! backends and [`FanPaar8`] use the portable scalar kernel. The vector kernels
//! are differentially tested against the direct [`crate::field::wiedemann`]
//! recurrence rather than against the optimized scalar element arithmetic.
//! [`crate::kernel::FieldKernels`] defaults compose multi-row and prepared
//! operations from `mul_add`.

use crate::field::fan_paar::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, fp16, fp32, fp64};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

crate::kernel::scalar::impl_field_kernels!(FanPaar8);

impl FieldKernels for FanPaar32 {
    #[inline]
    fn backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }
}

impl KernelDispatch for FanPaar32 {
    type Prepared = fp32::Elem;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: fp32::Elem) -> Self::Prepared {
        coeff
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> fp32::Elem {
        *prepared
    }

    #[inline]
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        crate::kernel::xor_gather(region, dst, offsets);
    }

    #[inline]
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_add_fp32_avx2(crate::kernel::x86_v3_token(), dst, *coeff, src);
            }
            _ => scalar::mul_add::<FanPaar32>(dst, *coeff, src),
        }
    }

    #[inline]
    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_assign_fp32_avx2(crate::kernel::x86_v3_token(), dst, *coeff);
            }
            _ => scalar::mul_assign::<FanPaar32>(dst, *coeff),
        }
    }

    #[inline]
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_into_fp32_avx2(crate::kernel::x86_v3_token(), dst, *coeff, src);
            }
            _ => {
                dst.copy_from_slice(src);
                scalar::mul_assign::<FanPaar32>(dst, *coeff);
            }
        }
    }

    #[inline]
    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[fp32::Elem],
        src: &[u8],
    ) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
        }
    }

    #[inline]
    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[fp32::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
        }
    }

    #[inline]
    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp32::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
            }
        }
    }

    #[inline]
    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        scalar::mul_elementwise::<FanPaar32>(dst, a, b);
    }
}

impl FieldKernels for FanPaar64 {
    #[inline]
    fn backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }
}

impl KernelDispatch for FanPaar64 {
    type Prepared = fp64::Elem;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: fp64::Elem) -> Self::Prepared {
        coeff
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> fp64::Elem {
        *prepared
    }

    #[inline]
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        crate::kernel::xor_gather(region, dst, offsets);
    }

    #[inline]
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_add_fp64_avx2(crate::kernel::x86_v3_token(), dst, *coeff, src);
            }
            _ => scalar::mul_add::<FanPaar64>(dst, *coeff, src),
        }
    }

    #[inline]
    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_assign_fp64_avx2(crate::kernel::x86_v3_token(), dst, *coeff);
            }
            _ => scalar::mul_assign::<FanPaar64>(dst, *coeff),
        }
    }

    #[inline]
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => {
                x86::fan_paar::mul_into_fp64_avx2(crate::kernel::x86_v3_token(), dst, *coeff, src);
            }
            _ => {
                dst.copy_from_slice(src);
                scalar::mul_assign::<FanPaar64>(dst, *coeff);
            }
        }
    }

    #[inline]
    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[fp64::Elem],
        srcs: &[u8],
    ) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), srcs);
        }
    }

    #[inline]
    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[fp64::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
        }
    }

    #[inline]
    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp64::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
            }
        }
    }

    #[inline]
    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        scalar::mul_elementwise::<FanPaar64>(dst, a, b);
    }
}

/// A Fan–Paar GF(2^16) coefficient resolved into the form this host's backend
/// wants.
///
/// `Tables` holds the four `fp8` nibble-table factors, built once in
/// `prepare` and reused across the whole buffer; `Plain`
/// hands the element to the portable scalar kernel.
#[derive(Clone, Debug)]
pub enum Fp16Prepared {
    /// AVX2 or SSSE3: the four `fp8` nibble tables, plus the element for the
    /// scalar tail.
    #[cfg(any(
        all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")),
        feature = "internals"
    ))]
    Tables {
        /// The Fan–Paar GF(2^16) coefficient.
        coeff: fp16::Elem,
        /// [`crate::kernel::tables::FpTowerTables`] for `coeff`.
        tables: crate::kernel::tables::FpTowerTables,
    },
    /// No shuffle backend: the element itself.
    Plain(fp16::Elem),
}

impl Fp16Prepared {
    /// The coefficient this was built from.
    #[inline]
    #[must_use]
    pub const fn coeff(&self) -> fp16::Elem {
        match self {
            Self::Plain(coeff) => *coeff,
            #[cfg(any(
                all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")),
                feature = "internals"
            ))]
            Self::Tables { coeff, .. } => *coeff,
        }
    }
}

impl FieldKernels for FanPaar16 {
    #[inline]
    fn backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }
}

impl KernelDispatch for FanPaar16 {
    type Prepared = Fp16Prepared;

    fn prepare(_proof: RawDispatch, coeff: fp16::Elem) -> Fp16Prepared {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => Fp16Prepared::Tables {
                coeff,
                tables: crate::kernel::tables::FpTowerTables::new(coeff),
            },
            _ => Fp16Prepared::Plain(coeff),
        }
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Fp16Prepared) -> fp16::Elem {
        prepared.coeff()
    }

    #[inline]
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        crate::kernel::xor_gather(region, dst, offsets);
    }

    #[inline]
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Fp16Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::V2 => {
                    x86::fan_paar::mul_add_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                }
                _ => x86::fan_paar::mul_add_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
            },
            other => scalar::mul_add::<FanPaar16>(dst, other.coeff(), src),
        }
    }

    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Fp16Prepared) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::V2 => {
                    x86::fan_paar::mul_assign_ssse3(crate::kernel::x86_v2_token(), dst, tables);
                }
                _ => x86::fan_paar::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, tables),
            },
            other => scalar::mul_assign::<FanPaar16>(dst, other.coeff()),
        }
    }

    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Fp16Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::V2 => {
                    x86::fan_paar::mul_into_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                }
                _ => x86::fan_paar::mul_into_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
            },
            other => {
                dst.copy_from_slice(src);
                Self::mul_assign(RawDispatch, dst, other);
            }
        }
    }

    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[fp16::Elem],
        src: &[u8],
    ) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
        }
    }

    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[fp16::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
        }
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp16::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
            }
        }
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        // Both operands vary per lane; no fixed coefficient to broadcast.
        scalar::mul_elementwise::<FanPaar16>(dst, a, b);
    }
}
