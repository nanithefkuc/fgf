//! GF(2^8) kernel dispatch.
//!
//! Owns the runtime backend selection for the field and provides the nibble
//! scalar path that every vector loop uses for its sub-lane tail.
//!
//! The prepared form of a `Gf8B` coefficient is simply a borrow of its entry
//! in the shared nibble table bank. Every backend can use it: the shuffle
//! backends index the tables directly, and GFNI reads back the coefficient
//! byte to broadcast. Preparation is therefore a single array index — unlike
//! GF(2^16), this field has nothing worth caching beyond what is already
//! precomputed in rodata. `Gf8D` prepares the same bank borrow plus the
//! coefficient's `VGF2P8AFFINEQB` matrix qword ([`Prepared8D`]), because
//! `GF2P8MULB` is the AES field and cannot multiply under `0x11D`.

use crate::field::gf8b::{Elem, Gf8B};
use crate::field::gf8d::{self, Gf8D};
use crate::kernel::tables::{ScaleTable, affine_8d, scale_table, scale_table_8d};
// `Backend` is referenced only from the SIMD dispatch arms, which cfg away
// entirely on a scalar-only build.
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, scalar};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::kernel::aarch64;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
use crate::kernel::wasm32;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

/// `dst ^= coeff * src` via split-nibble lookup.
///
/// Two table reads and two XORs per byte, versus a log/antilog pair with a
/// modular reduction. Also the tail handler for every vector kernel, which is
/// why it takes the already-built table rather than a coefficient.
#[cfg(feature = "internals")]
pub fn mul_add_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_add_nibble_impl(dst, table, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_add_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_add_nibble_impl(dst, table, src);
}

fn mul_add_nibble_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

/// `dst *= coeff` via split-nibble lookup.
#[cfg(feature = "internals")]
pub fn mul_assign_nibble(dst: &mut [u8], table: &ScaleTable) {
    mul_assign_nibble_impl(dst, table);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_assign_nibble(dst: &mut [u8], table: &ScaleTable) {
    mul_assign_nibble_impl(dst, table);
}

fn mul_assign_nibble_impl(dst: &mut [u8], table: &ScaleTable) {
    for d in dst.iter_mut() {
        *d = table.lo[(*d & 0x0f) as usize] ^ table.hi[(*d >> 4) as usize];
    }
}

/// `dst = coeff * src` via split-nibble lookup.
///
/// Tail handler for the fused out-of-place kernels, mirroring
/// [`mul_add_nibble`].
#[cfg(feature = "internals")]
pub fn mul_into_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_into_nibble_impl(dst, table, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_into_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_into_nibble_impl(dst, table, src);
}

fn mul_into_nibble_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

impl FieldKernels for Gf8B {
    type Prepared = &'static ScaleTable;

    #[inline]
    fn prepare(coeff: Elem) -> Self::Prepared {
        scale_table(coeff)
    }

    #[inline]
    fn prepared_coeff(prepared: &Self::Prepared) -> Elem {
        prepared.coeff
    }
    #[inline]
    fn active_backend() -> Backend {
        backend()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        matches!(
            backend(),
            Backend::V3GfniCrypto
                | Backend::V3
                | Backend::V2
                | Backend::NeonAes
                | Backend::Neon
                | Backend::Wasm128
        )
    }

    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_add_gfni(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_add_avx2(dst, coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_add_ssse3(dst, coeff, src),
            // PMULL is table-free but far slower than the nibble shuffle for
            // a fixed coefficient; see `aarch64::gf8` and BENCHMARKS.md.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_add_neon(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_add_simd128(dst, coeff, src),
            _ => mul_add_nibble(dst, coeff, src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_assign_gfni(dst, coeff.coeff),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_assign_avx2(dst, coeff),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_assign_ssse3(dst, coeff),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_assign_neon(dst, coeff),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_assign_simd128(dst, coeff),
            _ => mul_assign_nibble(dst, coeff),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_into_gfni(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_into_avx2(dst, coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_into_ssse3(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_into_neon(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_into_simd128(dst, coeff, src),
            _ => mul_into_nibble(dst, coeff, src),
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::scatter_gfni(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::scatter_avx2(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::scatter_ssse3(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => {
                aarch64::gf8::scatter_neon(rows, row_len, coeffs, src);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::scatter_simd128(rows, row_len, coeffs, src),
            _ => scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src),
        }
    }
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem],
        _coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter(rows, row_len, values, src);
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::gather_gfni(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::gather_avx2(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::gather_ssse3(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::gather_neon(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::gather_simd128(dst, coeffs, srcs),
            _ => scalar::mul_add_gather::<Self>(dst, coeffs, srcs),
        }
    }
    fn mul_add_gather_plan(
        dst: &mut [u8],
        values: &[Elem],
        _coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather(dst, values, srcs);
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::matrix_gfni(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::matrix_avx2(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::matrix_ssse3(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => {
                aarch64::gf8::matrix_neon(rows, row_len, nrows, terms);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::matrix_simd128(rows, row_len, nrows, terms),
            _ => scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms),
        }
    }
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem],
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
            match backend() {
                Backend::V3GfniCrypto => {
                    return x86::gf8::matrix_gfni_with(rows, row_len, nrows, &terms);
                }
                Backend::V3 => {
                    return x86::gf8::matrix_avx2_with(rows, row_len, nrows, &terms);
                }
                Backend::V2 => {
                    return x86::gf8::matrix_ssse3_with(rows, row_len, nrows, &terms);
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
                Self::mul_add(row, coeff, src);
            }
        }
    }

    fn mul_add_matrix_scattered(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem], &[u8])],
    ) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                x86::gf8::matrix_scattered_gfni(dst, row_len, row_starts, terms);
            }
            // The shuffle and non-x86 backends have no scattered kernel yet;
            // the portable path is correct and skips the same staging copy.
            _ => scalar::mul_add_matrix_scattered::<Self>(dst, row_len, row_starts, terms),
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            // `GF2P8MULB` multiplies two vectors directly: no broadcast, no
            // table, the same one instruction per 32 lanes.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::elementwise_gfni(dst, a, b),
            // Two varying operands are the one shape PMULL wins: two
            // `vmull_p8`s and a reduction network against eight bit-serial
            // rounds. The capability is cached in the backend, not probed per
            // call.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::NeonAes => aarch64::gf8::elementwise_pmull(dst, a, b),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::elementwise_neon(dst, a, b),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::elementwise_simd128(dst, a, b),
            // No fixed coefficient means no nibble table, so the shuffle
            // backends use the same eight branchless shift/reduce rounds as
            // baseline NEON and wasm.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::elementwise_avx2::<0x1b>(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::elementwise_ssse3::<0x1b>(dst, a, b),
            _ => scalar::mul_elementwise::<Gf8B>(dst, a, b),
        }
    }
}

/// The backend-ready form of a `Gf8D` coefficient.
///
/// One preparation gives every backend what it can consume: the GFNI kernels
/// read the `VGF2P8AFFINEQB` matrix qword, the shuffle backends and the
/// sub-lane scalar tails read the `0x11D` nibble tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prepared8D {
    table: &'static ScaleTable,
    affine: u64,
}

/// Reed–Solomon interop field, GF(2^8) under `0x11D`.
///
/// The split-nibble shuffle kernels are field-agnostic — they read only a
/// coefficient's `lo`/`hi` tables — so this field reuses [`Gf8B`]'s kernels
/// verbatim by handing them the `0x11D` bank ([`scale_table_8d`]). `GF2P8MULB`
/// is the AES field and MUST NOT be used; a GFNI host instead multiplies
/// through `VGF2P8AFFINEQB`, which is polynomial-independent, with the
/// const-derived `0x11D` affine bank ([`affine_8d`]). On a GFNI host the
/// register-blocked multi-row shapes (scatter/gather/matrix) fold rows in with
/// the affine map, holding a destination tile in registers across sources or
/// terms; other backends compose the single-coefficient shuffle per row.
/// Elementwise, whose two varying operands have no fixed table, runs the
/// branchless shift/reduce vector multiply threading the `0x11D` reduction
/// byte (portable scalar off x86).
impl FieldKernels for Gf8D {
    type Prepared = Prepared8D;

    #[inline]
    fn prepare(coeff: gf8d::Elem) -> Self::Prepared {
        Prepared8D {
            table: scale_table_8d(coeff),
            affine: affine_8d(coeff),
        }
    }

    #[inline]
    fn prepared_coeff(prepared: &Self::Prepared) -> gf8d::Elem {
        gf8d::Elem(prepared.table.coeff.0)
    }

    #[inline]
    fn active_backend() -> Backend {
        backend()
    }

    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_add_affine(dst, coeff.affine, coeff.table, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_add_avx2(dst, coeff.table, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_add_ssse3(dst, coeff.table, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_add_neon(dst, coeff.table, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_add_simd128(dst, coeff.table, src),
            _ => mul_add_nibble(dst, coeff.table, src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                // The one measured exception to "affine on GFNI": in-place
                // scaling is single-stream, and past 64 KiB the shuffle
                // outruns both single-instruction forms by ~15% on the
                // Core Ultra 7 258V (BENCHMARKS.md).
                if dst.len() < 65_536 {
                    x86::gf8::mul_assign_affine(dst, coeff.affine, coeff.table);
                } else {
                    x86::gf8::mul_assign_avx2(dst, coeff.table);
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_assign_avx2(dst, coeff.table),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_assign_ssse3(dst, coeff.table),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_assign_neon(dst, coeff.table),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_assign_simd128(dst, coeff.table),
            _ => mul_assign_nibble(dst, coeff.table),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => x86::gf8::mul_into_affine(dst, coeff.affine, coeff.table, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => x86::gf8::mul_into_avx2(dst, coeff.table, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::mul_into_ssse3(dst, coeff.table, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::gf8::mul_into_neon(dst, coeff.table, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::gf8::mul_into_simd128(dst, coeff.table, src),
            _ => mul_into_nibble(dst, coeff.table, src),
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[gf8d::Elem], src: &[u8]) {
        // Blocked affine on a GFNI host shares one source load across a row
        // group; other backends compose the single-coefficient kernel per row.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if backend() == Backend::V3GfniCrypto {
            return x86::gf8::scatter_affine(rows, row_len, coeffs, src);
        }
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &Self::prepare(coeff), src);
        }
    }
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        values: &[gf8d::Elem],
        _coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter(rows, row_len, values, src);
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[gf8d::Elem], srcs: &[&[u8]]) {
        // Blocked affine holds the destination tile in registers across every
        // source, so it is read and written once per tile, not once per source.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if backend() == Backend::V3GfniCrypto {
            return x86::gf8::gather_affine(dst, coeffs, srcs);
        }
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }
    fn mul_add_gather_plan(
        dst: &mut [u8],
        values: &[gf8d::Elem],
        _coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather(dst, values, srcs);
    }

    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf8d::Elem], &[u8])],
    ) {
        // Blocked affine holds a row-group tile in registers across all terms,
        // making destination traffic independent of the term count.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if backend() == Backend::V3GfniCrypto {
            return x86::gf8::matrix_affine(rows, row_len, nrows, terms);
        }
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &Self::prepare(coeff), src);
            }
        }
    }
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[gf8d::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = values;
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if backend() == Backend::V3GfniCrypto {
            let terms = crate::kernel::FlatMatrix {
                coefficients: values,
                nrows,
                sources: srcs,
            };
            return x86::gf8::matrix_affine_with(rows, row_len, nrows, &terms);
        }
        for (term, &src) in srcs.iter().enumerate() {
            let start = term * nrows;
            for (row, coeff) in rows
                .chunks_exact_mut(row_len)
                .take(nrows)
                .zip(&coeffs[start..start + nrows])
            {
                Self::mul_add(row, coeff, src);
            }
        }
    }

    fn mul_add_matrix_scattered(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[gf8d::Elem], &[u8])],
    ) {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if backend() == Backend::V3GfniCrypto {
            return x86::gf8::matrix_scattered_affine(dst, row_len, row_starts, terms);
        }
        // Shuffle and non-x86 backends have no scattered kernel; the portable
        // path is correct and skips the same staging copy.
        scalar::mul_add_matrix_scattered::<Self>(dst, row_len, row_starts, terms);
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

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            // `GF2P8MULB` is the AES field and cannot multiply under `0x11D`,
            // so even a GFNI host runs the branchless shift/reduce vector
            // multiply, threading the `0x11D` reduction byte (`0x1d`).
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto | Backend::V3 => x86::gf8::elementwise_avx2::<0x1d>(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => x86::gf8::elementwise_ssse3::<0x1d>(dst, a, b),
            _ => scalar::mul_elementwise::<Gf8D>(dst, a, b),
        }
    }
}
