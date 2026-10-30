//! GF(2^8) kernel dispatch for every byte-wide row.
//!
//! Every byte-wide row — each degree-eight representation
//! `FiniteField<2, 8, R>` and every `Packed<u8>` extension row — shares one
//! dispatch: the split-nibble scalar operations every architecture backend
//! uses for sub-lane tails, and one routing table, [`ByteDispatch`], that
//! diverges by row only where the hardware does.
//! [`Gf8Data`](crate::kernel::tables::Gf8Data) carries what dispatch needs —
//! AES nativity, the compile-time banks, the isomorphism onto the AES field,
//! and for polynomial rows the reduction byte — and [`Prepared`] is what
//! survives below it: a nibble table and an affine map qword, with no row
//! type attached. Multiplication by a fixed coefficient is a GF(2)-linear map
//! of the byte in every presentation, so the banks serve every row.
//! `GF2P8MULB` implements only the AES field, so a native row takes the
//! `GF2P8MULB` bodies and every other one its `VGF2P8AFFINEQB` map; each
//! `D::AES_NATIVE` comparison below is a constant that folds away in each
//! monomorphization, selecting between two non-generic kernel bodies.
//!
//! The raw-element operations resolve their coefficients against the row's
//! banks on access through the providers in this module, so they stay
//! allocation-free while no kernel body depends on the row.
//!
//! # Backends
//!
//! Elementwise products on the shuffle tiers reduce polynomial rows in their
//! own coordinates and conjugate every other row through its isomorphism onto
//! the AES field. Non-x86 targets run the portable scalar bodies for every
//! row.

use core::marker::PhantomData;

use crate::field::binary::BinaryRepr;
use crate::field::{Elem, Field, FiniteField};
use crate::kernel::tables::{Gf8Data, ScaleTable};
use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, scalar};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    // Reached through the `internals` facade and the kernel tests; dispatch
    // resolves raw bytes through `Prepared::of`.
    #[allow(private_bounds, dead_code)]
    #[inline]
    #[must_use]
    pub fn new<R: BinaryRepr<8> + Gf8Data>(coeff: Elem<FiniteField<2, 8, R>>) -> Self {
        Self::of::<R>(coeff.raw)
    }

    /// Resolve the raw coefficient byte `raw` against `D`'s banks.
    #[inline]
    #[must_use]
    pub(crate) fn of<D: Gf8Data>(raw: u8) -> Self {
        Self {
            table: &D::SCALE[raw as usize],
            affine: D::AFFINE[raw as usize],
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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) trait Coeffs {
    /// Number of coefficients.
    fn count(&self) -> usize;
    /// The prepared form of coefficient `index`.
    fn resolved(&self, index: usize) -> Prepared;
    /// The raw byte of coefficient `index`: the native AES multiplier and
    /// the zero test.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.resolved(index).byte()
    }
    /// The `VGF2P8AFFINEQB` matrix qword of coefficient `index`.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn affine(&self, index: usize) -> u64 {
        self.resolved(index).affine
    }
    /// The nibble tables of coefficient `index`, for sub-lane scalar tails.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn table(&self, index: usize) -> &'static ScaleTable {
        self.resolved(index).table
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

/// Raw elements resolved against their row's banks on access.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
struct Resolved<'a, F: Field, D>(&'a [Elem<F>], PhantomData<D>);

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<'a, F: Field, D> Resolved<'a, F, D> {
    #[inline]
    fn new(values: &'a [Elem<F>]) -> Self {
        Self(values, PhantomData)
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<F: Field<Raw = u8>, D: Gf8Data> Coeffs for Resolved<'_, F, D> {
    #[inline]
    fn count(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn resolved(&self, index: usize) -> Prepared {
        Prepared::of::<D>(self.0[index].raw)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.0[index].raw
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn affine(&self, index: usize) -> u64 {
        D::AFFINE[self.0[index].raw as usize]
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn table(&self, index: usize) -> &'static ScaleTable {
        &D::SCALE[self.0[index].raw as usize]
    }
}

/// Raw per-term elements as a [`Matrix`](crate::kernel::Matrix) of prepared
/// coefficients, resolved on access.
///
/// The public operation has already checked that every term supplies one
/// coefficient per destination row; an out-of-range access still panics
/// through slice indexing.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
// The unified element spelling nests over the term geometry here.
// Term slices stay slices so the dispatch layer serves raw-coefficient
// operations without allocating.
#[allow(clippy::type_complexity)]
struct TermsResolved<'a, F: Field, D>(&'a [(&'a [Elem<F>], &'a [u8])], PhantomData<D>);

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<'a, F: Field, D> TermsResolved<'a, F, D> {
    #[inline]
    fn new(terms: &'a [(&'a [Elem<F>], &'a [u8])]) -> Self {
        Self(terms, PhantomData)
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<F: Field<Raw = u8>, D: Gf8Data> crate::kernel::Matrix<Prepared> for TermsResolved<'_, F, D> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> Prepared {
        Prepared::of::<D>(self.0[term].0[row].raw)
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.0[term].1
    }
}

/// Raw flat row-major elements as a [`Matrix`](crate::kernel::Matrix) of
/// prepared coefficients, resolved on access.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
struct FlatResolved<'a, F: Field, D> {
    values: &'a [Elem<F>],
    nrows: usize,
    sources: &'a [&'a [u8]],
    data: PhantomData<D>,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<'a, F: Field, D> FlatResolved<'a, F, D> {
    #[inline]
    fn new(values: &'a [Elem<F>], nrows: usize, sources: &'a [&'a [u8]]) -> Self {
        Self {
            values,
            nrows,
            sources,
            data: PhantomData,
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<F: Field<Raw = u8>, D: Gf8Data> crate::kernel::Matrix<Prepared> for FlatResolved<'_, F, D> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> Prepared {
        Prepared::of::<D>(self.values[term * self.nrows + row].raw)
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

/// The process tier every byte-wide row's bulk operations dispatch on.
///
/// SIMD kernels are x86-64 only, so other targets run the portable scalar
/// bodies for every row.
#[inline]
pub(crate) fn byte_backend() -> Backend {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        crate::kernel::backend()
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        Backend::Scalar
    }
}

/// Whether every byte-wide row multiplies two varying vectors in vector
/// lanes on this process's tier.
#[inline]
pub(crate) fn byte_has_vector_elementwise() -> bool {
    matches!(
        byte_backend(),
        Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2
    )
}

impl<R: BinaryRepr<8> + Gf8Data> FieldKernels for FiniteField<2, 8, R> {
    #[inline]
    fn backend() -> Backend {
        byte_backend()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        byte_has_vector_elementwise()
    }
}

/// A byte-wide row the GF(2^8) dispatch serves: raw bytes as elements and
/// [`Prepared`] as its prepared coefficient.
pub(crate) trait ByteRow: KernelDispatch<Prepared = Prepared> + Field<Raw = u8> {}

impl<F: KernelDispatch<Prepared = Prepared> + Field<Raw = u8>> ByteRow for F {}

/// The GF(2^8) routing table for row `F` with byte facts `D`.
///
/// Each entry is the `KernelDispatch` entry of the same name; the
/// degree-eight representations and the `Packed<u8>` extension storage
/// forward to it.
pub(crate) struct ByteDispatch<F, D>(PhantomData<(F, D)>);

impl<F: ByteRow, D: Gf8Data> ByteDispatch<F, D> {
    /// The tier this row's bulk operations dispatch on.
    #[inline]
    fn backend() -> Backend {
        byte_backend()
    }

    #[inline]
    pub(crate) fn prepare(coeff: Elem<F>) -> Prepared {
        Prepared::of::<D>(coeff.raw)
    }

    #[inline]
    pub(crate) fn prepared_coeff(prepared: &Prepared) -> Elem<F> {
        Elem {
            raw: prepared.table.coeff,
        }
    }

    #[inline]
    pub(crate) fn add_assign(dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    #[inline]
    pub(crate) fn add_gather_offsets(region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        crate::kernel::xor_gather(region, dst, offsets);
    }

    #[inline]
    pub(crate) fn sub_assign(dst: &mut [u8], src: &[u8]) {
        crate::kernel::xor(dst, src);
    }

    pub(crate) fn mul_add(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                x86::gf8::mul_add_avx512(crate::kernel::x86_v4x_token(), dst, *coeff, src);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => {
                x86::gf8::mul_add_avx2(crate::kernel::x86_v3_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => {
                x86::gf8::mul_add_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table, src);
            }
            _ => mul_add_nibble(dst, coeff.table, src),
        }
    }

    pub(crate) fn mul_assign(dst: &mut [u8], coeff: &Prepared) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                x86::gf8::mul_assign_avx512(crate::kernel::x86_v4x_token(), dst, *coeff);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                // The affine in-place form selects the nibble shuffle for
                // long buffers; the native AES multiply leads at every size.
                if D::AES_NATIVE {
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => {
                x86::gf8::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, coeff.table);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => {
                x86::gf8::mul_assign_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table);
            }
            _ => mul_assign_nibble(dst, coeff.table),
        }
    }

    pub(crate) fn mul_into(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                // Past the streaming threshold the destination will not be
                // read back soon: the AVX2 non-temporal body owns that shape.
                // Below it the 64-byte temporal form leads.
                if crate::kernel::x86::wants_nt_store(dst.len()) {
                    if D::AES_NATIVE {
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => {
                x86::gf8::mul_into_avx2(crate::kernel::x86_v3_token(), dst, coeff.table, src);
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => {
                x86::gf8::mul_into_ssse3(crate::kernel::x86_v2_token(), dst, coeff.table, src);
            }
            _ => mul_into_nibble(dst, coeff.table, src),
        }
    }

    pub(crate) fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem<F>], src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => x86::gf8::mul_add_scatter_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                &Resolved::<F, D>::new(coeffs),
                src,
            ),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_add_scatter_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        &Resolved::<F, D>::new(coeffs),
                        src,
                    );
                } else {
                    x86::gf8::mul_add_scatter_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        &Resolved::<F, D>::new(coeffs),
                        src,
                    );
                }
            }
            // The blocked shuffle scatter is measured for the AES encoding;
            // other representations compose the single-coefficient kernel
            // per row.
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 if D::AES_NATIVE => x86::gf8::mul_add_scatter_avx2(
                crate::kernel::x86_v3_token(),
                rows,
                row_len,
                &Resolved::<F, D>::new(coeffs),
                src,
            ),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 if D::AES_NATIVE => x86::gf8::mul_add_scatter_ssse3(
                crate::kernel::x86_v2_token(),
                rows,
                row_len,
                &Resolved::<F, D>::new(coeffs),
                src,
            ),
            Backend::Scalar => scalar::mul_add_scatter::<F>(rows, row_len, coeffs, src),
            _ => {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    Self::mul_add(row, &Prepared::of::<D>(coeff.raw), src);
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem<F>],
        coeffs: &[Prepared],
        src: &[u8],
    ) {
        let _ = (values, coeffs);
        // Prepared affine scatter: the grouped rows read stored maps instead
        // of re-deriving them per row.
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        if !D::AES_NATIVE && Self::backend() == Backend::V4x {
            return x86::gf8::mul_add_scatter_avx512(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                coeffs,
                src,
            );
        }
        Self::mul_add_scatter(rows, row_len, values, src);
    }

    pub(crate) fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                x86::gf8::mul_add_gather_avx512(
                    crate::kernel::x86_v4x_token(),
                    dst,
                    &Resolved::<F, D>::new(coeffs),
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_add_gather_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        &Resolved::<F, D>::new(coeffs),
                        srcs,
                    );
                } else {
                    x86::gf8::mul_add_gather_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        &Resolved::<F, D>::new(coeffs),
                        srcs,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 if D::AES_NATIVE => {
                x86::gf8::mul_add_gather_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    &Resolved::<F, D>::new(coeffs),
                    srcs,
                );
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 if D::AES_NATIVE => {
                x86::gf8::mul_add_gather_ssse3(
                    crate::kernel::x86_v2_token(),
                    dst,
                    &Resolved::<F, D>::new(coeffs),
                    srcs,
                );
            }
            Backend::Scalar => scalar::mul_add_gather::<F>(dst, coeffs, srcs),
            _ => {
                for (&coeff, &src) in coeffs.iter().zip(srcs) {
                    Self::mul_add(dst, &Prepared::of::<D>(coeff.raw), src);
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_gather_plan(
        dst: &mut [u8],
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        let _ = (values, coeffs);
        // Prepared affine gather: the blocked tile reads stored maps instead
        // of re-deriving them per tile.
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        if !D::AES_NATIVE && Self::backend() == Backend::V4x {
            return x86::gf8::mul_add_gather_avx512(
                crate::kernel::x86_v4x_token(),
                dst,
                coeffs,
                srcs,
            );
        }
        Self::mul_add_gather(dst, values, srcs);
    }

    pub(crate) fn mul_into_gather(dst: &mut [u8], coeffs: &[Elem<F>], srcs: &[&[u8]]) {
        // One destination pass instead of the fill-then-accumulate pair: the
        // one-row overwrite matrix holds the tile in registers from zero.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let terms = FlatResolved::<F, D>::new(coeffs, 1, srcs);
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
                Backend::V3GfniCrypto if !D::AES_NATIVE => {
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
        if D::AES_NATIVE {
            dst.fill(0);
            Self::mul_add_gather(dst, coeffs, srcs);
            return;
        }
        crate::kernel::fallback::mul_into_gather_by_class::<F>(dst, coeffs, srcs);
    }

    #[cfg(any(feature = "alloc", test))]
    pub(crate) fn mul_into_gather_with(dst: &mut [u8], coeffs: &[Prepared], srcs: &[&[u8]]) {
        crate::kernel::fallback::mul_into_gather_by_class_with::<F>(dst, coeffs, srcs);
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_into_gather_plan(
        dst: &mut [u8],
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        if D::AES_NATIVE {
            return Self::mul_into_gather(dst, values, srcs);
        }
        // Prepared overwrite gather: one destination pass through the
        // one-row overwrite matrix instead of the trait default's
        // into-then-AXPY shape.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
                let terms = FlatResolved::<F, D>::new(values, 1, srcs);
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
        Self::mul_into_gather_with(dst, coeffs, srcs);
    }

    pub(crate) fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                x86::gf8::mul_add_matrix_avx512_with(
                    crate::kernel::x86_v4x_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved::<F, D>::new(terms),
                );
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &TermsResolved::<F, D>::new(terms),
                    );
                } else {
                    x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &TermsResolved::<F, D>::new(terms),
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 if D::AES_NATIVE => {
                x86::gf8::mul_add_matrix_avx2_with(
                    crate::kernel::x86_v3_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved::<F, D>::new(terms),
                );
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 if D::AES_NATIVE => {
                x86::gf8::mul_add_matrix_ssse3_with(
                    crate::kernel::x86_v2_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved::<F, D>::new(terms),
                );
            }
            Backend::Scalar => scalar::mul_add_matrix::<F>(rows, row_len, nrows, terms),
            _ => {
                for &(coeffs, src) in terms {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                        Self::mul_add(row, &Prepared::of::<D>(coeff.raw), src);
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let _ = values;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let terms = FlatResolved::<F, D>::new(values, nrows, srcs);
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
                    if D::AES_NATIVE {
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
                Backend::V3 if D::AES_NATIVE => {
                    return x86::gf8::mul_add_matrix_avx2_with(
                        crate::kernel::x86_v3_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                }
                Backend::V2 if D::AES_NATIVE => {
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
                Self::mul_add(row, coeff, src);
            }
        }
    }

    pub(crate) fn mul_into_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        // Overwrite seeds accumulators from zero in registers: one write pass,
        // no destination read, no separate fill.
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        if Self::backend() == Backend::V4x {
            return x86::gf8::mul_into_matrix_avx512_with(
                crate::kernel::x86_v4x_token(),
                rows,
                row_len,
                nrows,
                &TermsResolved::<F, D>::new(terms),
            );
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if Self::backend() == Backend::V3GfniCrypto {
            if D::AES_NATIVE {
                return x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                    crate::kernel::x86_v3_gfni_token(),
                    rows,
                    row_len,
                    nrows,
                    &TermsResolved::<F, D>::new(terms),
                );
            }
            return x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                crate::kernel::x86_v3_gfni_token(),
                rows,
                row_len,
                nrows,
                &TermsResolved::<F, D>::new(terms),
            );
        }
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix(rows, row_len, nrows, terms);
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_into_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if Self::backend() == Backend::V3GfniCrypto {
            let terms = FlatResolved::<F, D>::new(values, nrows, srcs);
            if D::AES_NATIVE {
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
        Self::mul_add_matrix_plan(rows, row_len, nrows, values, coeffs, srcs);
    }

    pub(crate) fn mul_add_matrix_at(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => x86::gf8::mul_add_matrix_at_avx512_with(
                crate::kernel::x86_v4x_token(),
                dst,
                row_len,
                row_starts,
                &TermsResolved::<F, D>::new(terms),
            ),
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        row_len,
                        row_starts,
                        &TermsResolved::<F, D>::new(terms),
                    );
                } else {
                    x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        row_len,
                        row_starts,
                        &TermsResolved::<F, D>::new(terms),
                    );
                }
            }
            // The shuffle and non-x86 backends have no blocked scattered
            // kernel.
            _ => F::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms),
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_matrix_at_plan(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let _ = values;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // The prepared tables carry their affine maps, so the blocked
            // scattered bodies read them directly.
            match Self::backend() {
                #[cfg(feature = "simd512")]
                Backend::V4x => {
                    let terms = crate::kernel::FlatMatrix {
                        coefficients: coeffs,
                        nrows: row_starts.len(),
                        sources: srcs,
                    };
                    return x86::gf8::mul_add_matrix_at_avx512_with(
                        crate::kernel::x86_v4x_token(),
                        dst,
                        row_len,
                        row_starts,
                        &terms,
                    );
                }
                Backend::V3GfniCrypto => {
                    let terms = FlatResolved::<F, D>::new(values, row_starts.len(), srcs);
                    if D::AES_NATIVE {
                        return x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            row_len,
                            row_starts,
                            &terms,
                        );
                    }
                    return x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        row_len,
                        row_starts,
                        &terms,
                    );
                }
                _ => {}
            }
        }
        crate::kernel::fallback::mul_add_matrix_at_plan::<F>(
            dst, row_len, row_starts, coeffs, srcs,
        );
    }

    pub(crate) fn mul_into_matrix_at(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        // Overwrite seeds accumulators from zero in registers: one write pass,
        // no destination read, no separate fill.
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        if Self::backend() == Backend::V4x {
            return x86::gf8::mul_into_matrix_at_avx512_with(
                crate::kernel::x86_v4x_token(),
                dst,
                row_len,
                row_starts,
                &TermsResolved::<F, D>::new(terms),
            );
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if Self::backend() == Backend::V3GfniCrypto {
            if D::AES_NATIVE {
                return x86::gf8::mul_into_matrix_at_gfni_with::<true, _>(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    row_len,
                    row_starts,
                    &TermsResolved::<F, D>::new(terms),
                );
            }
            return x86::gf8::mul_into_matrix_at_gfni_with::<false, _>(
                crate::kernel::x86_v3_gfni_token(),
                dst,
                row_len,
                row_starts,
                &TermsResolved::<F, D>::new(terms),
            );
        }
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        Self::mul_add_matrix_at(dst, row_len, row_starts, terms);
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn mul_into_matrix_at_plan(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<F>],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
        if Self::backend() == Backend::V4x {
            let terms = crate::kernel::FlatMatrix {
                coefficients: coeffs,
                nrows: row_starts.len(),
                sources: srcs,
            };
            return x86::gf8::mul_into_matrix_at_avx512_with(
                crate::kernel::x86_v4x_token(),
                dst,
                row_len,
                row_starts,
                &terms,
            );
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if Self::backend() == Backend::V3GfniCrypto {
            let terms = FlatResolved::<F, D>::new(values, row_starts.len(), srcs);
            if D::AES_NATIVE {
                return x86::gf8::mul_into_matrix_at_gfni_with::<true, _>(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    row_len,
                    row_starts,
                    &terms,
                );
            }
            return x86::gf8::mul_into_matrix_at_gfni_with::<false, _>(
                crate::kernel::x86_v3_gfni_token(),
                dst,
                row_len,
                row_starts,
                &terms,
            );
        }
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        Self::mul_add_matrix_at_plan(dst, row_len, row_starts, values, coeffs, srcs);
    }

    /// The prefix of a `len`-byte elementwise product that the affine
    /// GFNI and AVX-512 bodies serve. Polynomial rows give them everything,
    /// since their remainder reduces in the row's own coordinates. Every other
    /// row gives them whole 32-byte lanes and finishes on the conjugated
    /// shuffle body.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    fn affine_lanes(len: usize) -> usize {
        if D::POLYNOMIAL { len } else { len & !31 }
    }

    // One arm per tier, each splitting non-polynomial rows at the lane boundary.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match Self::backend() {
            // `GF2P8MULB` multiplies two vectors directly in the AES
            // encoding; every other row conjugates it by the isomorphism onto
            // `0x11B` (two affine maps around one native multiply).
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                if D::AES_NATIVE {
                    x86::gf8::mul_elementwise_avx512::<true>(
                        crate::kernel::x86_v4x_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    let lanes = Self::affine_lanes(dst.len());
                    let (dst, dst_rest) = dst.split_at_mut(lanes);
                    let (a, a_rest) = a.split_at(lanes);
                    let (b, b_rest) = b.split_at(lanes);
                    x86::gf8::mul_elementwise_avx512::<false>(
                        crate::kernel::x86_v4x_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                    x86::gf8::mul_elementwise_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst_rest,
                        a_rest,
                        b_rest,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_elementwise_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    let lanes = Self::affine_lanes(dst.len());
                    let (dst, dst_rest) = dst.split_at_mut(lanes);
                    let (a, a_rest) = a.split_at(lanes);
                    let (b, b_rest) = b.split_at(lanes);
                    x86::gf8::mul_elementwise_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                    x86::gf8::mul_elementwise_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst_rest,
                        a_rest,
                        b_rest,
                    );
                }
            }
            // No fixed coefficient means no nibble table, so the shuffle
            // backends use eight branchless shift/reduce rounds: in the row's
            // own coordinates for polynomial rows, conjugated through the AES
            // field for every other row.
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => {
                if D::POLYNOMIAL {
                    x86::gf8::mul_elementwise_avx2(
                        crate::kernel::x86_v3_token(),
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    x86::gf8::mul_elementwise_conjugated_avx2(
                        crate::kernel::x86_v3_token(),
                        &D::CONJUGATION,
                        dst,
                        a,
                        b,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => {
                if D::POLYNOMIAL {
                    x86::gf8::mul_elementwise_ssse3(
                        crate::kernel::x86_v2_token(),
                        D::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    x86::gf8::mul_elementwise_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst,
                        a,
                        b,
                    );
                }
            }
            _ => scalar::mul_elementwise::<F>(dst, a, b),
        }
    }

    pub(crate) fn mul_elementwise_assign(dst: &mut [u8], src: &[u8]) {
        match Self::backend() {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => {
                if D::AES_NATIVE {
                    x86::gf8::mul_elementwise_assign_avx512::<true>(
                        crate::kernel::x86_v4x_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    let lanes = Self::affine_lanes(dst.len());
                    let (dst, dst_rest) = dst.split_at_mut(lanes);
                    let (src, src_rest) = src.split_at(lanes);
                    x86::gf8::mul_elementwise_assign_avx512::<false>(
                        crate::kernel::x86_v4x_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                    x86::gf8::mul_elementwise_assign_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst_rest,
                        src_rest,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => {
                if D::AES_NATIVE {
                    x86::gf8::mul_elementwise_assign_gfni::<true>(
                        crate::kernel::x86_v3_gfni_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    let lanes = Self::affine_lanes(dst.len());
                    let (dst, dst_rest) = dst.split_at_mut(lanes);
                    let (src, src_rest) = src.split_at(lanes);
                    x86::gf8::mul_elementwise_assign_gfni::<false>(
                        crate::kernel::x86_v3_gfni_token(),
                        D::ISOMORPHISM,
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                    x86::gf8::mul_elementwise_assign_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst_rest,
                        src_rest,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => {
                if D::POLYNOMIAL {
                    x86::gf8::mul_elementwise_assign_avx2(
                        crate::kernel::x86_v3_token(),
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    x86::gf8::mul_elementwise_assign_conjugated_avx2(
                        crate::kernel::x86_v3_token(),
                        &D::CONJUGATION,
                        dst,
                        src,
                    );
                }
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => {
                if D::POLYNOMIAL {
                    x86::gf8::mul_elementwise_assign_ssse3(
                        crate::kernel::x86_v2_token(),
                        D::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    x86::gf8::mul_elementwise_assign_conjugated_ssse3(
                        crate::kernel::x86_v2_token(),
                        &D::CONJUGATION,
                        dst,
                        src,
                    );
                }
            }
            _ => scalar::mul_elementwise_assign::<F>(dst, src),
        }
    }
}

impl<R: BinaryRepr<8> + Gf8Data> KernelDispatch for FiniteField<2, 8, R> {
    type Prepared = Prepared;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<FiniteField<2, 8, R>>) -> Self::Prepared {
        ByteDispatch::<Self, R>::prepare(coeff)
    }

    #[inline]
    fn prepared_coeff(
        _proof: RawDispatch,
        prepared: &Self::Prepared,
    ) -> Elem<FiniteField<2, 8, R>> {
        ByteDispatch::<Self, R>::prepared_coeff(prepared)
    }

    #[inline]
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        ByteDispatch::<Self, R>::add_assign(dst, src);
    }

    #[inline]
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        ByteDispatch::<Self, R>::add_gather_offsets(region, dst, offsets);
    }

    #[inline]
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        ByteDispatch::<Self, R>::sub_assign(dst, src);
    }

    #[inline]
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        ByteDispatch::<Self, R>::mul_add(dst, coeff, src);
    }

    #[inline]
    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        ByteDispatch::<Self, R>::mul_assign(dst, coeff);
    }

    #[inline]
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        ByteDispatch::<Self, R>::mul_into(dst, coeff, src);
    }

    #[inline]
    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<FiniteField<2, 8, R>>],
        src: &[u8],
    ) {
        ByteDispatch::<Self, R>::mul_add_scatter(rows, row_len, coeffs, src);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_scatter_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        ByteDispatch::<Self, R>::mul_add_scatter_plan(rows, row_len, values, coeffs, src);
    }

    #[inline]
    fn mul_add_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<FiniteField<2, 8, R>>],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_add_gather(dst, coeffs, srcs);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_add_gather_plan(dst, values, coeffs, srcs);
    }

    #[inline]
    fn mul_into_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Elem<FiniteField<2, 8, R>>],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_into_gather(dst, coeffs, srcs);
    }

    #[cfg(any(feature = "alloc", test))]
    #[inline]
    fn mul_into_gather_with(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_into_gather_with(dst, coeffs, srcs);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_into_gather_plan(dst, values, coeffs, srcs);
    }

    #[inline]
    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<FiniteField<2, 8, R>>], &[u8])],
    ) {
        ByteDispatch::<Self, R>::mul_add_matrix(rows, row_len, nrows, terms);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_add_matrix_plan(rows, row_len, nrows, values, coeffs, srcs);
    }

    #[inline]
    fn mul_into_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<FiniteField<2, 8, R>>], &[u8])],
    ) {
        ByteDispatch::<Self, R>::mul_into_matrix(rows, row_len, nrows, terms);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_into_matrix_plan(rows, row_len, nrows, values, coeffs, srcs);
    }

    #[inline]
    fn mul_add_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<FiniteField<2, 8, R>>], &[u8])],
    ) {
        ByteDispatch::<Self, R>::mul_add_matrix_at(dst, row_len, row_starts, terms);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_add_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_add_matrix_at_plan(
            dst, row_len, row_starts, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_into_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<FiniteField<2, 8, R>>], &[u8])],
    ) {
        ByteDispatch::<Self, R>::mul_into_matrix_at(dst, row_len, row_starts, terms);
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul_into_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<FiniteField<2, 8, R>>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        ByteDispatch::<Self, R>::mul_into_matrix_at_plan(
            dst, row_len, row_starts, values, coeffs, srcs,
        );
    }

    #[inline]
    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        ByteDispatch::<Self, R>::mul_elementwise(dst, a, b);
    }

    #[inline]
    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        ByteDispatch::<Self, R>::mul_elementwise_assign(dst, src);
    }
}
