//! Vector kernels and backend selection.
//!
//! Layout of this module:
//!
//! - [`Backend`] — which instruction set the kernels will use, detected once
//!   per process.
//! - [`FieldKernels`] — the per-field kernel contract. GF(2^8) and GF(2^16)
//!   own hand-written SIMD dispatch; wider and Fan–Paar fields use the
//!   portable `scalar` kernels.
//! - `scalar` — the portable reference and fallback implementation. Every
//!   SIMD backend is differentially tested against it, and vector loops use
//!   it for sub-lane tails.
//! - `x86` / `aarch64` / `wasm32` — architecture-local intrinsics.
//!
//! Callers should use the safe, validated wrappers in [`crate::ops`] rather
//! than this module directly.

#[cfg(feature = "internals")]
pub mod fan_paar;
#[cfg(not(feature = "internals"))]
pub(crate) mod fan_paar;

#[cfg(feature = "internals")]
pub mod gf16;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf16;

#[cfg(feature = "internals")]
pub mod gf32;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf32;

#[cfg(feature = "internals")]
pub mod gf64;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf64;

#[cfg(feature = "internals")]
pub mod gf8;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf8;

#[cfg(feature = "internals")]
pub mod gf2;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf2;

#[cfg(feature = "internals")]
pub mod goldilocks;
#[cfg(not(feature = "internals"))]
pub(crate) mod goldilocks;

#[cfg(feature = "internals")]
pub mod mersenne31;
#[cfg(not(feature = "internals"))]
pub(crate) mod mersenne31;

#[cfg(feature = "internals")]
pub mod quad_mersenne31;
#[cfg(not(feature = "internals"))]
pub(crate) mod quad_mersenne31;

#[cfg(feature = "internals")]
pub mod prime;
#[cfg(not(feature = "internals"))]
pub(crate) mod prime;
#[cfg(feature = "internals")]
pub mod scalar;
#[cfg(not(feature = "internals"))]
pub(crate) mod scalar;

#[cfg(feature = "internals")]
pub mod tables;
#[cfg(not(feature = "internals"))]
pub(crate) mod tables;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[cfg(feature = "internals")]
pub mod aarch64;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[cfg(not(feature = "internals"))]
pub(crate) mod aarch64;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[cfg(feature = "internals")]
pub mod wasm32;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[cfg(not(feature = "internals"))]
pub(crate) mod wasm32;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
pub mod x86;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) mod x86;

#[cfg(test)]
mod tests;

use crate::field::Field;

// Only the SIMD-enabled resolve path consults the environment; under a
// std-less build `backend()` reports `Scalar` without touching `Selection`.
#[cfg(feature = "simd")]
use simdispatch::Selection;

mod private {
    pub trait Sealed {}
}
impl private::Sealed for crate::field::gf8b::Gf8B {}
impl private::Sealed for crate::field::gf8d::Gf8D {}
impl private::Sealed for crate::field::gf16::Gf16 {}
impl private::Sealed for crate::field::gf32::Gf32 {}
impl private::Sealed for crate::field::gf64::Gf64 {}
impl private::Sealed for crate::field::fan_paar::FanPaar8 {}
impl private::Sealed for crate::field::fan_paar::FanPaar16 {}
impl private::Sealed for crate::field::fan_paar::FanPaar32 {}
impl private::Sealed for crate::field::fan_paar::FanPaar64 {}
impl private::Sealed for crate::field::mersenne31::Mersenne31 {}
impl private::Sealed for crate::field::goldilocks::Goldilocks {}
impl private::Sealed for crate::field::quad_mersenne31::QuadMersenne31 {}

// The backend ladder is owned by `simdispatch` (the Level 0 single source for
// detection and ordering); FGF re-exports it so downstream consumers keep
// compiling, and builds its own `Selection` over the tiers it implements.
pub use simdispatch::{Backend, ParseBackendError};

/// The tiers FGF implements kernels for, in detection-preference order: the
/// common ladder minus the x86 `V1` floor (FGF's 16-byte multiply kernels
/// require SSSE3, not plain SSE2, so a V1-only host resolves to
/// [`Backend::Scalar`]) and minus the deferred 64-byte AVX-512 tier (V4x,
/// fgf's `avx512.rs` kernels are cross-compile-only and not validated).
pub const FGF_TIERS: &[Backend] = &[
    Backend::V3GfniCrypto,
    Backend::V3,
    Backend::V2,
    Backend::NeonAes,
    Backend::Neon,
    Backend::Wasm128,
    Backend::Scalar,
];

/// FGF's kernel-policy questions over the shared ladder.
///
/// `simdispatch::Backend` is capability; these two derive from the kernels
/// this crate actually implements, so they stay here (field-kernel policy,
/// not hardware facts). The ladder is owned upstream, so `Backend` gets these
/// only through this trait.
pub trait KernelBackend {
    /// Whether this backend has a native byte-wide field multiply (GFNI).
    fn has_native_mul(self) -> bool;

    /// Whether this backend implements the register-blocked multi-row
    /// kernels. Others decompose into repeated single-row AXPY.
    fn has_blocked_rows(self) -> bool;
}

impl KernelBackend for Backend {
    #[inline]
    fn has_native_mul(self) -> bool {
        matches!(self, Backend::V3GfniCrypto)
    }

    #[inline]
    fn has_blocked_rows(self) -> bool {
        matches!(
            self,
            Backend::V3GfniCrypto | Backend::Neon | Backend::NeonAes
        )
    }
}

/// The backend these kernels use, resolved once per process.
///
/// Runs [`Selection`] over [`FGF_TIERS`] — every tier this crate implements
/// kernels for, or [`Backend::Scalar`] when SIMD is compiled out — then
/// adjusted by the downgrade-only `SIMD_BACKEND` override. May be downgraded
/// at startup via `SIMD_BACKEND` (`v3_gfni_crypto`, `v3`, `v2`, `neon_aes`,
/// `neon`, `wasm128`, `scalar`); requests for a backend the host cannot run
/// are ignored. Detection itself is `simdispatch`'s `archmage` `summon()`
/// probe — the one probe in the stack.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "simd")]
    {
        *BACKEND
    }
    #[cfg(not(feature = "simd"))]
    {
        Backend::Scalar
    }
}

/// Memoized [`Selection`] over [`FGF_TIERS`], so dispatch never touches the
/// environment per call — a cache of the single-source resolve, not a second
/// resolver.
#[cfg(feature = "simd")]
static BACKEND: std::sync::LazyLock<Backend> =
    std::sync::LazyLock::new(|| Selection::new("SIMD_BACKEND").supports(FGF_TIERS).resolve());
/// The backend used for a particular field.
///
/// Wider polynomial towers and the Fan–Paar fields currently report
/// [`Backend::Scalar`] even when [`backend()`] selected a vector backend for
/// `Gf8B` and `Gf16`.
#[inline]
#[must_use]
pub fn backend_for<F: FieldKernels>() -> Backend {
    F::active_backend()
}

/// Whether elementwise multiplication is vectorized for `F` on this host.
#[inline]
#[must_use]
pub fn has_vector_elementwise<F: FieldKernels>() -> bool {
    F::has_vector_elementwise()
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
/// Matrix-like coefficient/source provider for the register-blocked x86 kernels.
#[allow(clippy::len_without_is_empty)]
pub trait Matrix<C> {
    /// Number of terms.
    fn len(&self) -> usize;
    /// The coefficient of `term` for destination row `row`.
    fn coefficient(&self, term: usize, row: usize) -> &C;
    /// The source buffer of `term`.
    fn source(&self, term: usize) -> &[u8];
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) trait Matrix<C> {
    fn len(&self) -> usize;
    fn coefficient(&self, term: usize, row: usize) -> &C;
    fn source(&self, term: usize) -> &[u8];
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<C> Matrix<C> for [(&[C], &[u8])] {
    #[inline]
    fn len(&self) -> usize {
        <[(&[C], &[u8])]>::len(self)
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self[term].0[row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self[term].1
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
/// Flat row-major coefficient matrix over borrowed sources.
pub struct FlatMatrix<'a, C> {
    /// Flat row-major coefficients, `terms * nrows` entries.
    pub coefficients: &'a [C],
    /// Destination row count.
    pub nrows: usize,
    /// Source buffers, one per term.
    pub sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) struct FlatMatrix<'a, C> {
    pub(crate) coefficients: &'a [C],
    pub(crate) nrows: usize,
    pub(crate) sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<C> Matrix<C> for FlatMatrix<'_, C> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self.coefficients[term * self.nrows + row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

/// The per-field vector kernel contract.
///
/// Implementations own runtime dispatch for their field. Every method's
/// preconditions are checked by the [`crate::ops`] wrappers, not here.
///
/// # Preconditions
///
/// All slice lengths are in **bytes** and must be whole multiples of
/// `Self::BYTES`. `dst` and `src` must have equal length.
// The seal stays private even under `internals`: implementors are fixed.
#[allow(private_interfaces)]
pub trait FieldKernels: Field + private::Sealed {
    /// The backend-ready form of one coefficient.
    ///
    /// Different backends want different things from a coefficient: GFNI
    /// wants a broadcast word, the shuffle backends want nibble tables, the
    /// scalar path wants the element itself. [`FieldKernels::prepare`]
    /// resolves that once — which is why the single-coefficient kernels below
    /// take a `Prepared` and not an `Elem`. The backend is fixed for the life
    /// of the process, so this moves the backend decision *out* of the hot
    /// call rather than adding one.
    type Prepared: Clone + Send + Sync + core::fmt::Debug;

    /// Resolve a coefficient into the form this host's backend wants.
    fn prepare(coeff: Self::Elem) -> Self::Prepared;

    /// Recover the coefficient a [`FieldKernels::Prepared`] was built from.
    fn prepared_coeff(prepared: &Self::Prepared) -> Self::Elem;

    /// Backend used by this field's kernels.
    #[inline]
    #[must_use]
    fn active_backend() -> Backend {
        Backend::Scalar
    }

    /// Whether [`FieldKernels::mul_elementwise`] uses a vector implementation.
    #[inline]
    #[must_use]
    fn has_vector_elementwise() -> bool {
        false
    }

    /// `dst += src`, elementwise field addition.
    ///
    /// Required, with no default: addition is XOR for the binary fields and a
    /// characteristic-`p` fold for the prime fields, so a wrong-by-default
    /// body would be a silent defect. Binary impls route to
    /// [`crate::kernel::xor`].
    fn add_assign(dst: &mut [u8], src: &[u8]);

    /// `dst ^= sum(srcs[i])`: fold byte-offset sources out of one backing
    /// region into one row, every coefficient implicitly one.
    ///
    /// The gather whose coefficients are all the multiplicative identity —
    /// the shape of back-substitution over solved rows. The default folds
    /// sources one at a time through [`FieldKernels::add_assign`], which is
    /// exact for every field; fields whose addition is XOR override it with
    /// the crate's blocked XOR gather, which holds the destination in
    /// registers across the whole source list.
    ///
    /// Callers pass offsets that satisfy `offset + dst.len() <=
    /// region.len()`; [`crate::ops::add_gather`] validates that.
    fn add_gather_offsets(region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        let live = dst.len();
        for &start in offsets {
            Self::add_assign(dst, &region[start as usize..start as usize + live]);
        }
    }

    /// `dst -= src`, elementwise field subtraction.
    ///
    /// Identical to [`FieldKernels::add_assign`] in characteristic two; prime
    /// fields subtract with a canonicalizing fold. Required, no default, for
    /// the same reason.
    fn sub_assign(dst: &mut [u8], src: &[u8]);

    /// Pairwise row addition over two equal flat row buffers.
    ///
    /// Both buffers hold the same whole number of contiguous `row_len`-byte
    /// rows, and every row pair adds fieldwise: `dst_row[j] += src_row[j]`.
    /// Row boundaries do not change elementwise addition, so running
    /// [`FieldKernels::add_assign`] over the whole buffers is exact for every
    /// field — that is the default, and on the reference host it is also the
    /// fastest known implementation at every measured geometry: a four-stream
    /// row-interleaved XOR candidate matched or trailed it from L1 to DRAM
    /// (BENCHMARKS.md, "Row-interleaved XOR"). The experimental interleaved
    /// kernels live behind `internals` for future evaluation; no backend
    /// override is wired until one measures a repeatable win. The prime
    /// fields additionally fold lanes rather than bytes, so their default is
    /// semantic, not just an optimization choice.
    fn add_assign_rows(dst: &mut [u8], src: &[u8], _row_len: usize) {
        Self::add_assign(dst, src);
    }

    /// `dst ^= coeff * src`. The workhorse AXPY.
    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// `dst = coeff * src`, out of place.
    ///
    /// The default copies `src` into `dst` and scales in place — two passes
    /// over the destination. Backends with a fused single-pass kernel
    /// override this. The override is worth roughly 2x on large buffers where
    /// the kernel is bandwidth-bound (x86 GF(2^8)/GF(2^16)); where it is
    /// compute-bound instead, as GF(2^16) is on NEON and wasm, halving
    /// destination traffic buys only a few percent (BENCHMARKS.md).
    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        dst.copy_from_slice(src);
        Self::mul_assign(dst, coeff);
    }

    /// `dst *= coeff`, in place.
    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared);

    /// One source into many rows: `rows[j] ^= coeffs[j] * src` for each `j`.
    ///
    /// `rows` is a flat buffer of `coeffs.len()` contiguous rows of
    /// `row_len` bytes. This is the systematic-encode shape; blocked
    /// backends load `src` once per tile and update several rows from it.
    ///
    /// The coefficients are raw elements, so every backend has to resolve
    /// each one into its own form before it can use it. Callers holding the
    /// resolved form should call [`FieldKernels::mul_add_scatter_with`].
    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Self::Elem], src: &[u8]);

    /// Many sources into one row: `dst ^= sum(coeffs[i] * srcs[i])`.
    ///
    /// The transpose of [`FieldKernels::mul_add_scatter`], and the shape that
    /// rebuilds a single lost symbol. Blocked backends hold the destination
    /// tile in registers while every source is folded in, so the destination
    /// is read and written once per tile rather than once per source.
    ///
    /// See [`FieldKernels::mul_add_gather_with`] for the prepared form.
    fn mul_add_gather(dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]);

    /// Many sources overwrite one row: `dst = sum(coeffs[i] * srcs[i])`.
    ///
    /// Unlike [`FieldKernels::mul_add_gather`], the previous destination is
    /// ignored. The default starts with a fused single-source [`FieldKernels::mul_into`],
    /// then accumulates the remaining prepared terms without allocation.
    fn dot_product(dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]) {
        let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        Self::mul_into(dst, &Self::prepare(first), src);
        for (coeff, src) in pairs {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }

    /// Many sources into many rows: for each `(coeffs, src)` term,
    /// `rows[j] ^= coeffs[j] * src` for every `j` in `0..nrows`.
    ///
    /// Equivalent to a [`FieldKernels::mul_add_scatter`] per term, but
    /// blocked backends hold a destination tile in registers across all
    /// terms, so destination memory traffic is independent of the term
    /// count. This is the decode/reconstruction shape.
    ///
    /// See [`FieldKernels::mul_add_matrix_with`] for the prepared form.
    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Elem], &[u8])],
    );

    /// [`FieldKernels::mul_add_scatter`] over already-prepared coefficients.
    ///
    /// The default keeps preparation out of the row loop by applying the
    /// single-coefficient kernel to each row. Fields may override this when a
    /// backend can retain several prepared coefficients in registers.
    fn mul_add_scatter_with(
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        for (row, coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, coeff, src);
        }
    }
    /// Prepared-plan scatter with access to both original and resolved
    /// coefficients.
    ///
    /// The default uses the prepared single-row path. Blocked backends may use
    /// `values` for representations whose preparation is already free.
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter_with(rows, row_len, coeffs, src);
    }

    /// [`FieldKernels::mul_add_gather`] over already-prepared coefficients.
    ///
    /// The default applies the single-coefficient kernel once per source.
    fn mul_add_gather_with(dst: &mut [u8], coeffs: &[Self::Prepared], srcs: &[&[u8]]) {
        for (coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, coeff, src);
        }
    }
    /// Prepared-plan gather with access to both original and resolved
    /// coefficients.
    ///
    /// The default applies prepared AXPY once per source.
    fn mul_add_gather_plan(
        dst: &mut [u8],
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather_with(dst, coeffs, srcs);
    }

    /// [`FieldKernels::dot_product`] over already-prepared coefficients.
    fn dot_product_with(dst: &mut [u8], coeffs: &[Self::Prepared], srcs: &[&[u8]]) {
        let mut pairs = coeffs.iter().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        Self::mul_into(dst, first, src);
        for (coeff, src) in pairs {
            Self::mul_add(dst, coeff, src);
        }
    }

    /// Prepared-plan dot product with original and resolved coefficients.
    fn dot_product_plan(
        dst: &mut [u8],
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::dot_product_with(dst, coeffs, srcs);
    }

    /// [`FieldKernels::mul_add_matrix`] over already-prepared coefficients.
    ///
    /// The default applies each term row by row. Fields may override this with
    /// a blocked implementation that retains destination tiles in registers.
    fn mul_add_matrix_with(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Prepared], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, coeff, src);
            }
        }
    }
    /// Prepared-plan matrix using flat row-major coefficients and source rows.
    ///
    /// The default is allocation-free repeated prepared AXPY. Register-blocked
    /// backends may override it and consume the same flat geometry directly.
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
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

    /// Many sources overwrite many rows: `rows[j] = sum_t coeffs[t][j] * src[t]`.
    ///
    /// The overwrite counterpart of [`FieldKernels::mul_add_matrix`] and the
    /// erasure-encode shape: the previous destination is ignored. The default
    /// zeroes the first `nrows` rows and accumulates; register-blocked backends
    /// override it to seed accumulators from zero in registers, writing each row
    /// once with no destination read and no separate fill.
    fn dot_product_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Elem], &[u8])],
    ) {
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix(rows, row_len, nrows, terms);
    }

    /// Prepared-plan overwrite matrix, the overwrite form of
    /// [`FieldKernels::mul_add_matrix_plan`].
    fn dot_product_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix_plan(rows, row_len, nrows, values, coeffs, srcs);
    }

    /// Many sources into many disjoint rows scattered through `dst`: for each
    /// `(coeffs, src)` term, `dst[row_starts[j]..][..row_len] ^= coeffs[j] *
    /// src` for every `j`.
    ///
    /// Like [`FieldKernels::mul_add_matrix`], but the destination rows are not
    /// contiguous — row `j` occupies `dst[row_starts[j] .. row_starts[j] +
    /// row_len]`. [`crate::ops::mul_add_matrix_scattered`] validates the
    /// offsets are in-bounds and pairwise disjoint before dispatch, so a
    /// blocked backend can write each recovered row to its final scattered
    /// slot and skip the staging copy a contiguous kernel forces on scattered
    /// decode outputs.
    ///
    /// The default applies each term row by row through the portable path.
    /// Register-blocked backends override it to retain destination tiles in
    /// registers across terms.
    fn mul_add_matrix_scattered(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Self::Elem], &[u8])],
    ) {
        scalar::mul_add_matrix_scattered::<Self>(dst, row_len, row_starts, terms);
    }

    /// `dst[i] = a[i] * b[i]`, elementwise over two full vectors.
    ///
    /// Both operands vary per lane, so there is no coefficient to broadcast
    /// and no table to index. GFNI multiplies vectors directly; `AArch64`,
    /// Wasm, and the shuffle-only x86 backends use a branchless
    /// shift/reduce vector multiply. The wider fields run the reference path.
    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]);
}

/// `dst ^= src` over raw bytes.
///
/// Field-independent: addition in every binary field is XOR, and XOR of a
/// packed element array is XOR of its bytes regardless of element width.
///
/// # Panics
/// Panics if the slices differ in length.
#[cfg(feature = "internals")]
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fgf::xor: length mismatch");
    xor_impl(dst, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fgf::xor: length mismatch");
    xor_impl(dst, src);
}

/// `dst ^= sum(srcs[i])` — a blocked XOR gather over byte offsets into one
/// backing region.
///
/// The field-independent shape of back-substitution: every coefficient is
/// one, so a many-source fold is pure XOR, and the sources are rows of one
/// buffer addressed by their start offsets. Addressing sources by offset
/// rather than slice lets a caller walk an index table without staging fat
/// pointers per source.
///
/// This is the XOR primitive under [`FieldKernels::add_gather_offsets`];
/// only fields whose addition is XOR dispatch to it.
///
/// # Panics
/// Panics if any offset plus `dst.len()` falls outside `region`.
pub(crate) fn xor_gather(region: &[u8], dst: &mut [u8], offsets: &[u32]) {
    let live = dst.len();
    for (index, &start) in offsets.iter().enumerate() {
        let start = start as usize;
        assert!(
            start + live <= region.len(),
            "xor_gather: offset {index} ({start}) + {live} exceeds region of {} bytes",
            region.len(),
        );
    }
    if offsets.is_empty() {
        return;
    }
    if live <= XOR_INLINE_MAX {
        for &start in offsets {
            scalar::xor(dst, &region[start as usize..start as usize + live]);
        }
        return;
    }
    match backend() {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V3GfniCrypto | Backend::V3 => x86::xor_gather_avx2(region, dst, offsets),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V2 => {
            for &start in offsets {
                x86::xor_sse2(dst, &region[start as usize..start as usize + live]);
            }
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Backend::Neon | Backend::NeonAes => {
            for &start in offsets {
                aarch64::xor_neon(dst, &region[start as usize..start as usize + live]);
            }
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32"))]
        Backend::Wasm128 => {
            for &start in offsets {
                wasm32::xor_simd128(dst, &region[start as usize..start as usize + live]);
            }
        }
        _ => {
            for &start in offsets {
                scalar::xor(dst, &region[start as usize..start as usize + live]);
            }
        }
    }
}

/// Buffers at most this long skip the dispatched SIMD XOR for the inline
/// portable one: below a single vector the `#[target_feature]` call
/// boundary costs more than the body saves, and short GF(2) rows (eight
/// elements per byte) sit almost entirely under it. Measured on the
/// reference host (BENCHMARKS.md, "Short-buffer inline XOR").
const XOR_INLINE_MAX: usize = 31;

fn xor_impl(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fgf::xor: length mismatch");

    if dst.len() <= XOR_INLINE_MAX {
        scalar::xor(dst, src);
        return;
    }
    match backend() {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V3GfniCrypto | Backend::V3 => x86::xor_avx2(dst, src),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V2 => x86::xor_sse2(dst, src),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Backend::Neon | Backend::NeonAes => aarch64::xor_neon(dst, src),
        #[cfg(all(feature = "simd", target_arch = "wasm32"))]
        Backend::Wasm128 => wasm32::xor_simd128(dst, src),
        _ => scalar::xor(dst, src),
    }
}
