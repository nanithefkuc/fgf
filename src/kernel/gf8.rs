//! GF(2^8) kernel dispatch for every degree-eight representation.
//!
//! Every degree-eight representation under [`Binary`] shares one dispatch: the
//! split-nibble scalar operations every architecture backend uses for
//! sub-lane tails, and one routing table that diverges by representation
//! only where the hardware does. [`Gf8Data`](crate::kernel::tables::Gf8Data)
//! carries what dispatch needs — AES nativity, the compile-time banks, the
//! isomorphism onto the AES field, and the reduction byte — and
//! [`Prepared`] is what survives below it: a nibble table and an affine map
//! qword, with no representation type attached. `GF2P8MULB` implements only
//! the AES field, so a native representation takes the `GF2P8MULB` bodies
//! and every other one its `VGF2P8AFFINEQB` map; each
//! `R::AES_NATIVE` comparison below is a constant that folds away in each
//! monomorphization, selecting between two non-generic kernel bodies.
//!
//! The raw-element operations resolve their coefficients against `R`'s banks
//! on access through the providers in this module, so they stay
//! allocation-free while no kernel body depends on `R`.
//!
//! # Backends
//!
//! Degree-eight towers join this dispatch through placeholder banks: they
//! report `Scalar` and route every bulk operation through the typed scalar
//! fallback. `AArch64` and Wasm use representation-specific nibble tables for
//! single-row scaling. Their blocked and elementwise routes serve only
//! AES-native representations; the others compose the single-row kernels.

use crate::field::{Binary, Elem};
use crate::kernel::tables::{Gf8Data, ScaleTable};
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

/// The backend-ready form of a GF(2^8) coefficient, representation-erased.
///
/// One preparation gives every backend what it can consume: the affine GFNI
/// and AVX-512 kernels read the `VGF2P8AFFINEQB` matrix qword, the native
/// AES kernels the coefficient byte, and the shuffle backends and sub-lane
/// scalar tails the nibble tables. Every entry comes from the
/// representation's compile-time banks, so no kernel recovers a field fact
/// from the coefficient byte — or from a representation type: `Prepared` is
/// the plain value the hot bodies see.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prepared {
    pub(crate) table: &'static ScaleTable,
    pub(crate) affine: u64,
}

impl Prepared {
    /// Resolve `coeff` against its representation's banks.
    ///
    /// The representation is inferred from the element; callers never name
    /// the crate-private bank facet.
    #[allow(private_bounds)]
    #[inline]
    #[must_use]
    pub fn new<R: Gf8Data>(coeff: Elem<Binary<8, R>>) -> Self {
        let raw = coeff.to_raw() as usize;
        Self {
            table: &R::SCALE[raw],
            affine: R::AFFINE[raw],
        }
    }

    /// The raw coefficient byte this was built from.
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub fn byte(self) -> u8 {
        self.table.coeff
    }

    /// The zero coefficient's prepared form: an all-zero map and table. The
    /// staging arrays the blocked scatter kernels fill group-wise start here;
    /// every slot is overwritten before it is read.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn zero() -> Self {
        Self {
            table: &crate::kernel::tables::ZERO_TABLE,
            affine: 0,
        }
    }

    /// Whether the coefficient is zero: the case the blocked scatter staging
    /// drops before grouping.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        self.table.coeff == 0
    }
}

/// Random access to resolved coefficients for the blocked scatter and gather
/// entries.
///
/// A prepared call holds its coefficients as a `[Prepared]` slice; the
/// dispatch layer's raw-element operations resolve each element against its
/// representation's banks on access, which keeps those one-shot forms
/// allocation-free without making any kernel body depend on the
/// representation. Hot loops that need one facet of a coefficient read it
/// through [`Coeffs::byte`], [`Coeffs::affine`] or [`Coeffs::table`], so a
/// raw-element provider answers from the element itself instead of building
/// a full [`Prepared`].
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
pub(crate) trait Coeffs {
    /// Number of coefficients.
    fn count(&self) -> usize;
    /// The prepared form of coefficient `index`.
    fn resolved(&self, index: usize) -> Prepared;
    /// The raw byte of coefficient `index`: the native AES multiplier and
    /// the zero test.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.resolved(index).byte()
    }
    /// The `VGF2P8AFFINEQB` matrix qword of coefficient `index`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn affine(&self, index: usize) -> u64 {
        self.resolved(index).affine
    }
    /// The nibble tables of coefficient `index`, for sub-lane scalar tails.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn table(&self, index: usize) -> &'static ScaleTable {
        self.resolved(index).table
    }
}

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
impl Coeffs for [Prepared] {
    #[inline]
    fn count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn resolved(&self, index: usize) -> Prepared {
        self[index]
    }
}

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
impl Coeffs for alloc::vec::Vec<Prepared> {
    #[inline]
    fn count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn resolved(&self, index: usize) -> Prepared {
        self[index]
    }
}

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
impl<const N: usize> Coeffs for [Prepared; N] {
    #[inline]
    fn count(&self) -> usize {
        N
    }

    #[inline]
    fn resolved(&self, index: usize) -> Prepared {
        self[index]
    }
}

/// Raw elements resolved against their representation's banks on access.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
struct Resolved<'a, R: Gf8Data>(&'a [Elem<Binary<8, R>>]);

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
impl<R: Gf8Data> Coeffs for Resolved<'_, R> {
    #[inline]
    fn count(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn resolved(&self, index: usize) -> Prepared {
        Prepared::new(self.0[index])
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.0[index].to_raw()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn affine(&self, index: usize) -> u64 {
        R::AFFINE[self.0[index].to_raw() as usize]
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn table(&self, index: usize) -> &'static ScaleTable {
        &R::SCALE[self.0[index].to_raw() as usize]
    }
}

/// Raw per-term elements as a [`Matrix`](crate::kernel::Matrix) of prepared
/// coefficients, resolved on access.
///
/// The public operation has already checked that every term supplies one
/// coefficient per destination row; an out-of-range access still panics
/// through slice indexing.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
// The unified element spelling nests over the term geometry here.
// Term slices stay slices so the dispatch layer serves raw-coefficient
// operations without allocating.
#[allow(clippy::type_complexity)]
struct TermsResolved<'a, R: Gf8Data>(&'a [(&'a [Elem<Binary<8, R>>], &'a [u8])]);

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
impl<R: Gf8Data> crate::kernel::Matrix<Prepared> for TermsResolved<'_, R> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> Prepared {
        Prepared::new(self.0[term].0[row])
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.0[term].1
    }
}

/// Raw flat row-major elements as a [`Matrix`](crate::kernel::Matrix) of
/// prepared coefficients, resolved on access.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
struct FlatResolved<'a, R: Gf8Data> {
    values: &'a [Elem<Binary<8, R>>],
    nrows: usize,
    sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<R: Gf8Data> crate::kernel::Matrix<Prepared> for FlatResolved<'_, R> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> Prepared {
        Prepared::new(self.values[term * self.nrows + row])
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

impl<R: Gf8Data> FieldKernels for Binary<8, R> {
    #[inline]
    fn backend() -> Backend {
        // Non-polynomial presentations — ordered bases and towers — never
        // touch the vector units: every bulk operation, including tails,
        // runs the typed scalar fallback, so polynomial arithmetic must not
        // observe their coordinates.
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return Backend::Scalar;
        }
        backend()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return false;
        }
        match backend() {
            Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => {
                cfg!(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))
            }
            Backend::NeonAes | Backend::Neon | Backend::Wasm128 => R::AES_NATIVE,
            _ => false,
        }
    }
}

impl<R: Gf8Data> KernelDispatch for Binary<8, R> {
    type Prepared = Prepared;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<Binary<8, R>>) -> Self::Prepared {
        Prepared::new(coeff)
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Binary<8, R>> {
        <Elem<Binary<8, R>>>::from_raw(prepared.table.coeff)
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
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            let coeff = <Elem<Binary<8, R>>>::from_raw(coeff.table.coeff);
            return scalar::mul_add::<Binary<8, R>>(dst, coeff, src);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_add_avx512(crate::kernel::x86_v4x_token(), dst, *coeff, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_add_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                        src,
                    );
                } else {
                    x86::gf8::mul_add_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                        src,
                    );
                }
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
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            let coeff = <Elem<Binary<8, R>>>::from_raw(coeff.table.coeff);
            return scalar::mul_assign::<Binary<8, R>>(dst, coeff);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_assign_avx512(crate::kernel::x86_v4x_token(), dst, *coeff);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                // The affine in-place form selects the nibble shuffle for
                // long buffers; the native AES multiply leads at every size.
                if R::AES_NATIVE {
                    x86::gf8::mul_assign_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                    );
                } else if dst.len() < 65_536 {
                    x86::gf8::mul_assign_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                    );
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
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            let coeff = <Elem<Binary<8, R>>>::from_raw(coeff.table.coeff);
            dst.copy_from_slice(src);
            return scalar::mul_assign::<Binary<8, R>>(dst, coeff);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                // Past the streaming threshold the destination will not be
                // read back soon: the AVX2 non-temporal body owns that shape.
                // Below it the 64-byte temporal form leads.
                if crate::kernel::x86::wants_nt_store(dst.len()) {
                    if R::AES_NATIVE {
                        x86::gf8::mul_into_gfni::<true>(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            *coeff,
                            src,
                        );
                    } else {
                        x86::gf8::mul_into_gfni::<false>(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            *coeff,
                            src,
                        );
                    }
                } else {
                    x86::gf8::mul_into_avx512(crate::kernel::x86_v4x_token(), dst, *coeff, src);
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_into_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                        src,
                    );
                } else {
                    x86::gf8::mul_into_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *coeff,
                        src,
                    );
                }
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
        coeffs: &[Elem<Binary<8, R>>],
        src: &[u8],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_add_scatter::<Binary<8, R>>(rows, row_len, coeffs, src);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::gf8::mul_add_scatter_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                &Resolved(coeffs),
                src,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_add_scatter_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        &Resolved(coeffs),
                        src,
                    );
                } else {
                    x86::gf8::mul_add_scatter_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        &Resolved(coeffs),
                        src,
                    );
                }
            }
            // The blocked shuffle scatter is measured for the AES encoding;
            // other representations compose the single-coefficient kernel
            // per row.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if R::AES_NATIVE => x86::gf8::mul_add_scatter_avx2(
                crate::kernel::x86_v3_token(),
                rows,
                row_len,
                &Resolved(coeffs),
                src,
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if R::AES_NATIVE => x86::gf8::mul_add_scatter_ssse3(
                crate::kernel::x86_v2_token(),
                rows,
                row_len,
                &Resolved(coeffs),
                src,
            ),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if R::AES_NATIVE => {
                aarch64::gf8::mul_add_scatter_neon(
                    crate::kernel::neon_token(),
                    rows,
                    row_len,
                    &Resolved(coeffs),
                    src,
                )
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if R::AES_NATIVE => wasm32::gf8::mul_add_scatter_simd128(
                crate::kernel::wasm128_token(),
                rows,
                row_len,
                &Resolved(coeffs),
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
        values: &[Elem<Binary<8, R>>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_add_scatter::<Binary<8, R>>(rows, row_len, values, src);
        }
        let _ = (values, coeffs);
        // Prepared affine scatter: the grouped rows read stored maps instead
        // of re-deriving them per row.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if !R::AES_NATIVE && Self::backend() == Backend::V4x {
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
        coeffs: &[Elem<Binary<8, R>>],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_add_gather::<Binary<8, R>>(dst, coeffs, srcs);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_add_gather_avx512(
                    crate::kernel::x86_v4x_token(),
                    dst,
                    &Resolved(coeffs),
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_add_gather_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        &Resolved(coeffs),
                        srcs,
                    );
                } else {
                    x86::gf8::mul_add_gather_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        &Resolved(coeffs),
                        srcs,
                    );
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if R::AES_NATIVE => {
                x86::gf8::mul_add_gather_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    &Resolved(coeffs),
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if R::AES_NATIVE => {
                x86::gf8::mul_add_gather_ssse3(
                    crate::kernel::x86_v2_token(),
                    dst,
                    &Resolved(coeffs),
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if R::AES_NATIVE => aarch64::gf8::mul_add_gather_neon(
                crate::kernel::neon_token(),
                dst,
                &Resolved(coeffs),
                srcs,
            ),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if R::AES_NATIVE => wasm32::gf8::mul_add_gather_simd128(
                crate::kernel::wasm128_token(),
                dst,
                &Resolved(coeffs),
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
        values: &[Elem<Binary<8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_add_gather::<Binary<8, R>>(dst, values, srcs);
        }
        let _ = (values, coeffs);
        // Prepared affine gather: the blocked tile reads stored maps instead
        // of re-deriving them per tile.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if !R::AES_NATIVE && Self::backend() == Backend::V4x {
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
        coeffs: &[Elem<Binary<8, R>>],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
            let Some((first, src)) = pairs.next() else {
                dst.fill(0);
                return;
            };
            Self::mul_into(RawDispatch, dst, &Prepared::new(first), src);
            for (coeff, src) in pairs {
                Self::mul_add(RawDispatch, dst, &Prepared::new(coeff), src);
            }
            return;
        }
        // One destination pass instead of the fill-then-accumulate pair: the
        // one-row overwrite matrix holds the tile in registers from zero.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let terms = FlatResolved {
                values: coeffs,
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
                Backend::V3GfniCrypto if !R::AES_NATIVE => {
                    return x86::gf8::mul_into_matrix_gfni_with::<false, _>(
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
        if R::AES_NATIVE {
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
        values: &[Elem<Binary<8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return Self::mul_into_gather_with(RawDispatch, dst, coeffs, srcs);
        }
        if R::AES_NATIVE {
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
                let terms = FlatResolved {
                    values,
                    nrows: 1,
                    sources: srcs,
                };
                return x86::gf8::mul_into_matrix_gfni_with::<false, _>(
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
        terms: &[(&[Elem<Binary<8, R>>], &[u8])],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_add_matrix::<Binary<8, R>>(rows, row_len, nrows, terms);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                x86::gf8::mul_add_matrix_avx512_with(
                    crate::kernel::x86_v4x_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &TermsResolved(terms),
                    );
                } else {
                    x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &TermsResolved(terms),
                    );
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 if R::AES_NATIVE => {
                x86::gf8::mul_add_matrix_avx2_with(
                    crate::kernel::x86_v3_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 if R::AES_NATIVE => {
                x86::gf8::mul_add_matrix_ssse3_with(
                    crate::kernel::x86_v2_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes if R::AES_NATIVE => {
                aarch64::gf8::mul_add_matrix_neon(
                    crate::kernel::neon_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if R::AES_NATIVE => {
                wasm32::gf8::mul_add_matrix_simd128(
                    crate::kernel::wasm128_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
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
        values: &[Elem<Binary<8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            for (term, &src) in srcs.iter().enumerate() {
                let start = term * nrows;
                for (row, &coeff) in rows
                    .chunks_exact_mut(row_len)
                    .take(nrows)
                    .zip(&values[start..start + nrows])
                {
                    scalar::mul_add::<Binary<8, R>>(row, coeff, src);
                }
            }
            return;
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = values;
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let terms = FlatResolved {
                values,
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
                    if R::AES_NATIVE {
                        return x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                            crate::kernel::x86_v3_gfni_token(),
                            rows,
                            row_len,
                            nrows,
                            &terms,
                        );
                    }
                    return x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                Backend::V3 if R::AES_NATIVE => {
                    return x86::gf8::mul_add_matrix_avx2_with(
                        crate::kernel::x86_v3_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                Backend::V2 if R::AES_NATIVE => {
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
        terms: &[(&[Elem<Binary<8, R>>], &[u8])],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            for row in rows.chunks_exact_mut(row_len).take(nrows) {
                row.fill(0);
            }
            return scalar::mul_add_matrix::<Binary<8, R>>(rows, row_len, nrows, terms);
        }
        // Overwrite seeds accumulators from zero in registers: one write pass,
        // no destination read, no separate fill.
        #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V4x {
            return x86::gf8::mul_into_matrix_avx512_with(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                nrows,
                &TermsResolved(terms),
            );
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if Self::backend() == Backend::V3GfniCrypto {
            if R::AES_NATIVE {
                return x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                    crate::kernel::x86_v3_gfni_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved(terms),
                );
            }
            return x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                nrows,
                &TermsResolved(terms),
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
        values: &[Elem<Binary<8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            for row in rows.chunks_exact_mut(row_len).take(nrows) {
                row.fill(0);
            }
            for (term, &src) in srcs.iter().enumerate() {
                let start = term * nrows;
                for (row, &coeff) in rows
                    .chunks_exact_mut(row_len)
                    .take(nrows)
                    .zip(&values[start..start + nrows])
                {
                    scalar::mul_add::<Binary<8, R>>(row, coeff, src);
                }
            }
            return;
        }
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
            let terms = FlatResolved {
                values,
                nrows,
                sources: srcs,
            };
            if R::AES_NATIVE {
                return x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                    crate::kernel::x86_v3_gfni_token(),
                    rows,
                    row_len,
                    nrows,
                    &terms,
                );
            }
            return x86::gf8::mul_into_matrix_gfni_with::<false, _>(
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
        terms: &[(&[Elem<Binary<8, R>>], &[u8])],
    ) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return Self::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => x86::gf8::mul_add_matrix_at_avx512_with(
                crate::kernel::x86_v4x_token(),
                dst,
                row_len,
                row_starts,
                &TermsResolved(terms),
            ),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        row_len,
                        row_starts,
                        &TermsResolved(terms),
                    );
                } else {
                    x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        row_len,
                        row_starts,
                        &TermsResolved(terms),
                    );
                }
            }
            // The shuffle and non-x86 backends have no blocked scattered
            // kernel.
            _ => Self::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms),
        }
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_elementwise::<Binary<8, R>>(dst, a, b);
        }
        match Self::backend() {
            // `GF2P8MULB` multiplies two vectors directly in the AES
            // encoding; every other representation conjugates it by the
            // isomorphism onto `0x11B` (two affine maps around one native
            // multiply).
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                if R::AES_NATIVE {
                    x86::gf8::mul_elementwise_avx512::<true>(
                        crate::kernel::x86_v4x_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    x86::gf8::mul_elementwise_avx512::<false>(
                        crate::kernel::x86_v4x_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_elementwise_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    x86::gf8::mul_elementwise_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                }
            }
            // No fixed coefficient means no nibble table, so the shuffle
            // backends use eight branchless shift/reduce rounds that thread
            // the representation's reduction byte.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_elementwise_avx2(
                    crate::kernel::x86_v3_token(),
                    R::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_elementwise_ssse3(
                    crate::kernel::x86_v2_token(),
                    R::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            }
            // Two varying operands are the one shape PMULL wins: two
            // `vmull_p8`s and a reduction network against eight bit-serial
            // rounds. The capability is cached in the backend, not probed per
            // call.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::NeonAes if R::AES_NATIVE => {
                aarch64::gf8::mul_elementwise_neon_aes(crate::kernel::neon_aes_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon if R::AES_NATIVE => {
                aarch64::gf8::mul_elementwise_neon(crate::kernel::neon_token(), dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 if R::AES_NATIVE => {
                wasm32::gf8::mul_elementwise_simd128(crate::kernel::wasm128_token(), dst, a, b);
            }
            _ => scalar::mul_elementwise::<Self>(dst, a, b),
        }
    }

    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        if const { !R::DESCRIPTION.is_polynomial_root() } {
            return scalar::mul_elementwise_assign::<Binary<8, R>>(dst, src);
        }
        match Self::backend() {
            #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V4x => {
                if R::AES_NATIVE {
                    x86::gf8::mul_elementwise_assign_avx512::<true>(
                        crate::kernel::x86_v4x_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    x86::gf8::mul_elementwise_assign_avx512::<false>(
                        crate::kernel::x86_v4x_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        src,
                    );
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3GfniCrypto => {
                if R::AES_NATIVE {
                    x86::gf8::mul_elementwise_assign_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    x86::gf8::mul_elementwise_assign_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        R::ISOMORPHISM,
                        R::REDUCTION_LOW,
                        dst,
                        src,
                    );
                }
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V3 => {
                x86::gf8::mul_elementwise_assign_avx2(
                    crate::kernel::x86_v3_token(),
                    R::REDUCTION_LOW,
                    dst,
                    src,
                );
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::V2 => {
                x86::gf8::mul_elementwise_assign_ssse3(
                    crate::kernel::x86_v2_token(),
                    R::REDUCTION_LOW,
                    dst,
                    src,
                );
            }
            _ => scalar::mul_elementwise_assign::<Self>(dst, src),
        }
    }
}
