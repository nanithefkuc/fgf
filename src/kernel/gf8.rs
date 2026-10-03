//! GF(2^8) kernel dispatch for every flat byte field.
//!
//! [`Gf8`] is one construction under any reduction polynomial, so its fields
//! share one dispatch: the split-nibble scalar operations every architecture
//! backend uses for sub-lane tails, and one routing table that diverges by
//! polynomial only where the hardware does. `GF2P8MULB` implements only the
//! AES field, so [`AES`] takes the native GFNI multiply and every other
//! polynomial its `VGF2P8AFFINEQB` map; every comparison against `AES` below
//! is a constant that folds away in each monomorphization.
//!
//! A [`Prepared`] coefficient borrows its polynomial's nibble table and
//! carries its affine matrix, so no kernel recovers a field fact from the
//! coefficient byte.
//!
//! # Backends
//!
//! `AArch64` and Wasm use polynomial-specific nibble tables for single-row
//! scaling. Their blocked and elementwise routes serve only the AES field.

use crate::field::poly::AES;
use crate::field::{Elem, Gf8, Poly};
use crate::kernel::tables::{ScaleTable, affine_map, scale_table};
use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::kernel::aarch64;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
use crate::kernel::wasm32;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

// --------------------------------------------------------------------------
// Shared preparation and portable tails.
// --------------------------------------------------------------------------

/// `dst ^= coeff * src` via split-nibble lookup.
///
/// Two table reads and two XORs per byte, versus a log/antilog pair with a
/// modular reduction. Also the tail handler for every vector kernel, which is
/// why it takes the already-built table rather than a coefficient.
pub fn mul_add_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

/// `dst *= coeff` via split-nibble lookup.
pub fn mul_assign_nibble(dst: &mut [u8], table: &ScaleTable) {
    for d in dst.iter_mut() {
        *d = table.lo[(*d & 0x0f) as usize] ^ table.hi[(*d >> 4) as usize];
    }
}

/// `dst = coeff * src` via split-nibble lookup.
///
/// Tail handler for the fused out-of-place kernels, mirroring
/// [`mul_add_nibble`].
pub fn mul_into_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

/// The backend-ready form of a GF(2^8) coefficient under `POLY`.
///
/// One preparation gives every backend what it can consume: the affine GFNI
/// and AVX-512 kernels read the `VGF2P8AFFINEQB` matrix qword, the native
/// AES kernels the coefficient byte, and the shuffle backends and sub-lane
/// scalar tails the nibble tables from the polynomial's own bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prepared<const POLY: u32> {
    pub(crate) table: &'static ScaleTable,
    pub(crate) affine: u64,
}

impl<const POLY: u32> Prepared<POLY> {
    /// Resolve `coeff` against its polynomial's banks.
    #[inline]
    #[must_use]
    pub fn new(coeff: Elem<8, Poly<POLY>>) -> Self {
        Self {
            table: scale_table(coeff),
            affine: affine_map(coeff),
        }
    }

    /// The coefficient this was built from.
    #[inline]
    #[must_use]
    pub fn coeff(&self) -> Elem<8, Poly<POLY>> {
        Elem::<8, Poly<POLY>>::from_raw(self.table.coeff)
    }
}

impl<const POLY: u32> FieldKernels for Gf8<Poly<POLY>> {
    #[inline]
    fn backend() -> Backend {
        backend()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        match backend() {
            Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => {
                cfg!(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))
            }
            Backend::NeonAes | Backend::Neon | Backend::Wasm128 => POLY == AES,
            _ => false,
        }
    }
}

impl<const POLY: u32> KernelDispatch for Gf8<Poly<POLY>> {
    type Prepared = Prepared<POLY>;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<8, Poly<POLY>>) -> Self::Prepared {
        Prepared::new(coeff)
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<8, Poly<POLY>> {
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

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_add_avx512(crate::kernel::x86_v4x_token(), dst, *coeff, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::mul_add_gfni(crate::kernel::x86_v3_gfni_token(), dst, *coeff, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_add_avx2(crate::kernel::x86_v3_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_add_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => {
                aarch64::gf8::mul_add_neon(crate::kernel::neon_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => {
                wasm32::gf8::mul_add_simd128(crate::kernel::wasm128_token(), dst, coeff.table, src);
            }
            _ => mul_add_nibble(dst, coeff.table, src),
        }
    }

    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_assign_avx512(crate::kernel::x86_v4x_token(), dst, *coeff);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                // The affine in-place form selects the nibble shuffle for
                // long buffers; the native AES multiply leads at every size.
                if POLY == AES || dst.len() < 65_536 {
                    x86::gf8::mul_assign_gfni(crate::kernel::x86_v3_gfni_token(), dst, *coeff);
                } else {
                    x86::gf8::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, coeff.table);
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, coeff.table);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_assign_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => {
                aarch64::gf8::mul_assign_neon(crate::kernel::neon_token(), dst, coeff.table);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => {
                wasm32::gf8::mul_assign_simd128(crate::kernel::wasm128_token(), dst, coeff.table);
            }
            _ => mul_assign_nibble(dst, coeff.table),
        }
    }

    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                // Past the streaming threshold the destination will not be
                // read back soon: the AVX2 non-temporal body owns that shape.
                // Below it the 64-byte temporal form leads.
                if crate::kernel::x86::wants_nt_store(dst.len()) {
                    x86::gf8::mul_into_gfni(crate::kernel::x86_v3_gfni_token(), dst, *coeff, src);
                } else {
                    x86::gf8::mul_into_avx512(crate::kernel::x86_v4x_token(), dst, *coeff, src);
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::mul_into_gfni(crate::kernel::x86_v3_gfni_token(), dst, *coeff, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_into_avx2(crate::kernel::x86_v3_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_into_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => {
                aarch64::gf8::mul_into_neon(crate::kernel::neon_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => {
                wasm32::gf8::mul_into_simd128(
                    crate::kernel::wasm128_token(),
                    dst,
                    coeff.table,
                    src,
                );
            }
            _ => mul_into_nibble(dst, coeff.table, src),
        }
    }

    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<8, Poly<POLY>>],
        src: &[u8],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::gf8::mul_add_scatter_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_add_scatter_gfni(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            // The blocked shuffle scatter is measured for the AES field;
            // other polynomials compose the single-coefficient kernel per row.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if POLY == AES => x86::gf8::mul_add_scatter_avx2(
                crate::kernel::x86_v3_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if POLY == AES => x86::gf8::mul_add_scatter_ssse3(
                crate::kernel::x86_v2_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if POLY == AES => aarch64::gf8::mul_add_scatter_neon(
                crate::kernel::neon_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if POLY == AES => wasm32::gf8::mul_add_scatter_simd128(
                crate::kernel::wasm128_token(),
                rows,
                row_len,
                coeffs,
                src,
            ),
            Backend::Scalar => scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src),
            _ => {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    Self::mul_add(RawDispatch, row, &Prepared::new(coeff), src);
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn mul_add_scatter_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem<8, Poly<POLY>>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        let _ = coeffs;
        // Prepared affine scatter: the grouped rows read stored maps instead
        // of re-deriving them per row.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if POLY != AES && Self::backend() == Backend::V4x {
            return x86::gf8::mul_add_scatter_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                coeffs,
                src,
            );
        }
        Self::mul_add_scatter(RawDispatch, rows, row_len, values, src);
    }

    fn mul_add_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<8, Poly<POLY>>],
        srcs: &[&[u8]],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_add_gather_avx512(crate::kernel::x86_v4x_token(), dst, coeffs, srcs);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::mul_add_gather_gfni(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    coeffs,
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if POLY == AES => {
                x86::gf8::mul_add_gather_avx2(crate::kernel::x86_v3_token(), dst, coeffs, srcs);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if POLY == AES => {
                x86::gf8::mul_add_gather_ssse3(crate::kernel::x86_v2_token(), dst, coeffs, srcs);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if POLY == AES => {
                aarch64::gf8::mul_add_gather_neon(crate::kernel::neon_token(), dst, coeffs, srcs)
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if POLY == AES => wasm32::gf8::mul_add_gather_simd128(
                crate::kernel::wasm128_token(),
                dst,
                coeffs,
                srcs,
            ),
            Backend::Scalar => scalar::mul_add_gather::<Self>(dst, coeffs, srcs),
            _ => {
                for (&coeff, &src) in coeffs.iter().zip(srcs) {
                    Self::mul_add(RawDispatch, dst, &Prepared::new(coeff), src);
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn mul_add_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<8, Poly<POLY>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        let _ = coeffs;
        // Prepared affine gather: the blocked tile reads stored maps instead
        // of re-deriving them per tile.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if POLY != AES && Self::backend() == Backend::V4x {
            return x86::gf8::mul_add_gather_avx512(
                crate::kernel::x86_v4x_token(),
                dst,
                coeffs,
                srcs,
            );
        }
        Self::mul_add_gather(RawDispatch, dst, values, srcs);
    }

    fn mul_into_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<8, Poly<POLY>>],
        srcs: &[&[u8]],
    ) {
        // One destination pass instead of the fill-then-accumulate pair: the
        // one-row overwrite matrix holds the tile in registers from zero.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let terms = crate::kernel::FlatMatrix {
                coefficients: coeffs,
                nrows: 1,
                sources: srcs,
            };
            match Self::backend() {
                #[cfg(feature = "simd512")]
                Backend::V4x => {
                    return x86::gf8::mul_into_matrix_avx512_with(
                        crate::kernel::x86_v4x_token(),
                        dst,
                        dst.len(),
                        1,
                        &terms,
                    );
                }
                Backend::V3GfniCrypto if POLY != AES => {
                    return x86::gf8::mul_into_matrix_gfni_with(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        dst.len(),
                        1,
                        &terms,
                    );
                }
                _ => {}
            }
        }
        if POLY == AES {
            dst.fill(0);
            Self::mul_add_gather(RawDispatch, dst, coeffs, srcs);
            return;
        }
        let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        Self::mul_into(RawDispatch, dst, &Prepared::new(first), src);
        for (coeff, src) in pairs {
            Self::mul_add(RawDispatch, dst, &Prepared::new(coeff), src);
        }
    }

    #[cfg(feature = "alloc")]
    fn mul_into_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<8, Poly<POLY>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        if POLY == AES {
            return Self::mul_into_gather(RawDispatch, dst, values, srcs);
        }
        // Prepared overwrite gather: one destination pass through the
        // one-row overwrite matrix instead of the trait default's
        // into-then-AXPY shape.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        match Self::backend() {
            #[cfg(feature = "simd512")]
            Backend::V4x => {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows: 1,
                    sources: srcs,
                };
                return x86::gf8::mul_into_matrix_avx512_with(
                    crate::kernel::x86_v4x_token(),
                    dst,
                    dst.len(),
                    1,
                    &terms,
                );
            }
            Backend::V3GfniCrypto => {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: values,
                    nrows: 1,
                    sources: srcs,
                };
                return x86::gf8::mul_into_matrix_gfni_with(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    dst.len(),
                    1,
                    &terms,
                );
            }
            _ => {}
        }
        Self::mul_into_gather_with(RawDispatch, dst, coeffs, srcs);
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<8, Poly<POLY>>], &[u8])],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::gf8::mul_add_matrix_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_add_matrix_gfni(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if POLY == AES => x86::gf8::mul_add_matrix_avx2(
                crate::kernel::x86_v3_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if POLY == AES => x86::gf8::mul_add_matrix_ssse3(
                crate::kernel::x86_v2_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if POLY == AES => aarch64::gf8::mul_add_matrix_neon(
                crate::kernel::neon_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if POLY == AES => wasm32::gf8::mul_add_matrix_simd128(
                crate::kernel::wasm128_token(),
                rows,
                row_len,
                nrows,
                terms,
            ),
            Backend::Scalar => scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms),
            _ => {
                for &(coeffs, src) in terms {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                        Self::mul_add(RawDispatch, row, &Prepared::new(coeff), src);
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn mul_add_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<8, Poly<POLY>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = values;
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let terms = crate::kernel::FlatMatrix {
                coefficients: values,
                nrows,
                sources: srcs,
            };
            match Self::backend() {
                // The prepared tables carry their affine maps, so the blocked
                // 64-byte bodies read them directly.
                #[cfg(feature = "simd512")]
                Backend::V4x => {
                    let prepared = crate::kernel::FlatMatrix {
                        coefficients: coeffs,
                        nrows,
                        sources: srcs,
                    };
                    return x86::gf8::mul_add_matrix_avx512_with(
                        crate::kernel::x86_v4x_token(),
                        rows,
                        row_len,
                        nrows,
                        &prepared,
                    );
                }
                Backend::V3GfniCrypto => {
                    return x86::gf8::mul_add_matrix_gfni_with(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                Backend::V3 if POLY == AES => {
                    return x86::gf8::mul_add_matrix_avx2_with(
                        crate::kernel::x86_v3_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                Backend::V2 if POLY == AES => {
                    return x86::gf8::mul_add_matrix_ssse3_with(
                        crate::kernel::x86_v2_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                _ => {}
            }
        }
        for (term, &src) in srcs.iter().enumerate() {
            let start = term * nrows;
            for (row, coeff) in rows
                .chunks_exact_mut(row_len)
                .take(nrows)
                .zip(&coeffs[start..start + nrows])
            {
                Self::mul_add(RawDispatch, row, coeff, src);
            }
        }
    }

    fn mul_into_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<8, Poly<POLY>>], &[u8])],
    ) {
        // Overwrite seeds accumulators from zero in registers: one write pass,
        // no destination read, no separate fill.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V4x {
            return x86::gf8::mul_into_matrix_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                nrows,
                terms,
            );
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V3GfniCrypto {
            return x86::gf8::mul_into_matrix_gfni(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                nrows,
                terms,
            );
        }
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix(RawDispatch, rows, row_len, nrows, terms);
    }

    #[cfg(feature = "alloc")]
    fn mul_into_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<8, Poly<POLY>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V4x {
            let prepared = crate::kernel::FlatMatrix {
                coefficients: coeffs,
                nrows,
                sources: srcs,
            };
            return x86::gf8::mul_into_matrix_avx512_with(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                nrows,
                &prepared,
            );
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V3GfniCrypto {
            let terms = crate::kernel::FlatMatrix {
                coefficients: values,
                nrows,
                sources: srcs,
            };
            return x86::gf8::mul_into_matrix_gfni_with(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                nrows,
                &terms,
            );
        }
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix_plan(RawDispatch, rows, row_len, nrows, values, coeffs, srcs);
    }

    fn mul_add_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<8, Poly<POLY>>], &[u8])],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::gf8::mul_add_matrix_at_avx512(
                crate::kernel::x86_v4x_token(),
                dst,
                row_len,
                row_starts,
                terms,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_add_matrix_at_gfni(
                crate::kernel::x86_v3_gfni_token(),
                dst,
                row_len,
                row_starts,
                terms,
            ),
            // The shuffle and non-x86 backends have no blocked scattered
            // kernel.
            _ => Self::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms),
        }
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        match Self::backend() {
            // `GF2P8MULB` multiplies two vectors directly under the AES
            // polynomial; every other polynomial conjugates it by the field
            // isomorphism (two affine maps around one native multiply).
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_elementwise_avx512::<POLY>(crate::kernel::x86_v4x_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::mul_elementwise_gfni::<POLY>(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    a,
                    b,
                );
            }
            // No fixed coefficient means no nibble table, so the shuffle
            // backends use eight branchless shift/reduce rounds that thread
            // the polynomial's reduction byte.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_elementwise_avx2::<POLY>(crate::kernel::x86_v3_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_elementwise_ssse3::<POLY>(crate::kernel::x86_v2_token(), dst, a, b);
            }
            // Two varying operands are the one shape PMULL wins: two
            // `vmull_p8`s and a reduction network against eight bit-serial
            // rounds. The capability is cached in the backend, not probed per
            // call.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::NeonAes if POLY == AES => {
                aarch64::gf8::mul_elementwise_neon_aes(crate::kernel::neon_aes_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon if POLY == AES => {
                aarch64::gf8::mul_elementwise_neon(crate::kernel::neon_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if POLY == AES => {
                wasm32::gf8::mul_elementwise_simd128(crate::kernel::wasm128_token(), dst, a, b);
            }
            _ => scalar::mul_elementwise::<Self>(dst, a, b),
        }
    }

    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_elementwise_assign_avx512::<POLY>(
                    crate::kernel::x86_v4x_token(),
                    dst,
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::mul_elementwise_assign_gfni::<POLY>(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_elementwise_assign_avx2::<POLY>(
                    crate::kernel::x86_v3_token(),
                    dst,
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_elementwise_assign_ssse3::<POLY>(
                    crate::kernel::x86_v2_token(),
                    dst,
                    src,
                );
            }
            _ => scalar::mul_elementwise_assign::<Self>(dst, src),
        }
    }
}
