//! Vector kernels and backend selection.
//!
//! Layout of this module:
//!
//! - [`Backend`] — which instruction set the kernels will use, detected once
//!   per process.
//! - [`FieldKernels`] — the sealed public bound over a field's kernels;
//!   process-selected entrypoints remain behind a crate-private
//!   `KernelDispatch` supertrait. The `internals` feature exposes direct
//!   token-bearing architecture entries for measurement and differential
//!   testing.
//! - `scalar` — the portable fallback and tail implementation. Vector kernels
//!   use it for sub-lane tails; tests compare the Fan–Paar family against the
//!   independent Wiedemann recurrence and other families against this module.
//! - `x86` — architecture-local intrinsics. Direct entries take exact
//!   `archmage` capability tokens and validate geometry; inner loops are
//!   `#[rite]` helpers inside the token-proven region. Other targets compile
//!   and run the portable scalar bodies.
//!
//! Callers should use the safe, validated wrappers in [`crate::ops`] rather
//! than this module directly.

pub(crate) mod tower;
// Reached only from the architecture kernels, which cfg away entirely on a
// scalar-only build.
#[allow(unused_imports)]
pub(crate) use tower::gf16;

pub(crate) mod bytes;
pub(crate) mod extension;
pub(crate) mod gf1;
pub(crate) mod gf8;
pub(crate) mod goldilocks;
pub(crate) mod mersenne31;
pub(crate) mod prime;
pub(crate) mod quad_mersenne31;
pub(crate) mod residue;
pub(crate) mod scalar;
pub(crate) mod tables;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) mod matrix_provider;
pub(crate) use bytes::{xor, xor_gather};
// Used by the simd512-gated blocked kernels and by tests (`just lint`
// builds `--all-targets`, where tests use it); a lib-only build without
// `simd512` has no other user.
#[allow(unused_imports)]
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) use matrix_provider::FlatMatrix;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) use matrix_provider::Matrix;

#[cfg(test)]
pub(crate) mod trace;

#[cfg(test)]
mod tests;

use crate::field::{FieldBuffer, PrimeIdentity, Residue};

// Only the SIMD-enabled resolve path consults the environment; under a
// std-less build `backend()` reports `Scalar` without touching `Selection`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::SimdToken;
#[cfg(feature = "simd")]
use simdispatch::Selection;

mod private {
    pub trait Sealed {}
}
impl<R: crate::kernel::tables::Gf8Data> private::Sealed for crate::field::FiniteField<2, 8, R> {}
impl private::Sealed for crate::field::mersenne31::Mersenne31 {}
impl private::Sealed for crate::field::goldilocks::Goldilocks {}
impl<const P: u64> private::Sealed for crate::field::FiniteField<P, 1, Residue> {}
impl<const P: u64, const N: u32, const D: usize, S: crate::field::extension::ExtensionSpec<D>>
    private::Sealed for crate::field::FiniteField<P, N, crate::field::extension::Extension<S, D>>
where
    <S as crate::field::extension::ExtensionSpec<D>>::Storage:
        crate::field::extension::storage::StorageRepr<
                <S as crate::field::extension::ExtensionSpec<D>>::Base,
                D,
            >,
{
}

// The backend ladder is owned by `simdispatch` (the Level 0 single source for
// detection and ordering); FGF re-exports it so downstream consumers keep
// compiling, and builds its own `Selection` over the tiers it implements.
pub use simdispatch::{Backend, ParseBackendError};

/// The tiers FGF implements kernels for, in detection-preference order: the
/// common ladder minus the x86 `V1` floor (FGF's 16-byte multiply kernels
/// require SSSE3, not plain SSE2, so a V1-only host resolves to
/// [`Backend::Scalar`]).
///
/// `V4x` resolves only when the `simd512` feature enables the 512-bit kernel
/// modules and `simdispatch/avx512` detection. Without `simd512` the variant
/// exists in the ladder but no `FGF_TIERS` entry can select it, so detection
/// falls through to `V3GfniCrypto` on AVX-512 hosts.
#[cfg(not(feature = "simd512"))]
pub const FGF_TIERS: &[Backend] = &[
    Backend::V3GfniCrypto,
    Backend::V3,
    Backend::V2,
    Backend::Scalar,
];
/// The tiers FGF implements kernels for, with the 64-byte AVX-512 byte-field,
/// GF(2^16) tower, prime-field, and byte-XOR kernels first. See the
/// non-`simd512` form for the rest.
#[cfg(feature = "simd512")]
pub const FGF_TIERS: &[Backend] = &[
    Backend::V4x,
    Backend::V3GfniCrypto,
    Backend::V3,
    Backend::V2,
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
        matches!(self, Backend::V4x | Backend::V3GfniCrypto)
    }

    #[inline]
    fn has_blocked_rows(self) -> bool {
        matches!(self, Backend::V4x | Backend::V3GfniCrypto)
    }
}

/// The backend these kernels use, resolved once per process.
///
/// Runs [`simdispatch::Selection`] over [`FGF_TIERS`] — every tier this crate implements
/// kernels for, or [`Backend::Scalar`] when SIMD is compiled out — then
/// adjusted by the downgrade-only `SIMD_BACKEND` override. May be downgraded
/// at startup via `SIMD_BACKEND` (`v4x`, `v3_gfni_crypto`, `v3`, `v2`,
/// `scalar`); requests for a backend the host
/// cannot run are ignored. `v4x` resolves only with the `simd512` feature on
/// an AVX-512+GFNI host; `SIMD_BACKEND=v4` is not a selected tier and behaves
/// like any unsupported request. Detection itself is `simdispatch`'s
/// `archmage` `summon()` probe — the one probe in the stack.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "simd")]
    {
        BACKEND.backend
    }
    #[cfg(not(feature = "simd"))]
    {
        Backend::Scalar
    }
}

/// Memoized [`Selection`] over [`FGF_TIERS`] plus the capability token for
/// that selection. Dispatch therefore resolves policy and materializes its
/// proof once.
#[cfg(feature = "simd")]
static BACKEND: std::sync::LazyLock<ResolvedBackend> =
    std::sync::LazyLock::new(ResolvedBackend::new);

#[cfg(feature = "simd")]
struct ResolvedBackend {
    backend: Backend,
    #[cfg(target_arch = "x86_64")]
    x86: X86Proof,
}

#[cfg(feature = "simd")]
impl ResolvedBackend {
    fn new() -> Self {
        let backend = Selection::new("SIMD_BACKEND").supports(FGF_TIERS).resolve();
        #[cfg(target_arch = "x86_64")]
        let x86 = match backend {
            #[cfg(feature = "simd512")]
            Backend::V4x => X86Proof::V4x(
                archmage::X64V4xToken::summon()
                    .expect("selected x86 V4x backend must have its capability token"),
            ),
            Backend::V3GfniCrypto => X86Proof::V3GfniCrypto(
                archmage::X64V3GfniCryptoToken::summon()
                    .expect("selected x86 GFNI backend must have its capability token"),
            ),
            Backend::V3 => X86Proof::V3(
                archmage::X64V3Token::summon()
                    .expect("selected x86 V3 backend must have its capability token"),
            ),
            Backend::V2 => X86Proof::V2(
                archmage::X64V2Token::summon()
                    .expect("selected x86 V2 backend must have its capability token"),
            ),
            _ => X86Proof::Scalar,
        };
        Self {
            backend,
            #[cfg(target_arch = "x86_64")]
            x86,
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub(crate) enum X86Proof {
    #[cfg(feature = "simd512")]
    V4x(archmage::X64V4xToken),
    V3GfniCrypto(archmage::X64V3GfniCryptoToken),
    V3(archmage::X64V3Token),
    V2(archmage::X64V2Token),
    Scalar,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_proof() -> X86Proof {
    BACKEND.x86
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_v3_gfni_token() -> archmage::X64V3GfniCryptoToken {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => token.v3_gfni_crypto(),
        X86Proof::V3GfniCrypto(token) => token,
        _ => unreachable!("GFNI kernel reached without the selected GFNI proof"),
    }
}

/// The selected V4x proof: the byte-field 512-bit kernels take it directly.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_v4x_token() -> archmage::X64V4xToken {
    match x86_proof() {
        X86Proof::V4x(token) => token,
        _ => unreachable!("AVX-512 kernel reached without the selected V4x proof"),
    }
}

/// The selected V4x proof narrowed to AVX-512F, the tier the prime-field
/// and byte-XOR 64-byte kernels require.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_v4_token() -> archmage::X64V4Token {
    x86_v4x_token().v4()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_v3_token() -> archmage::X64V3Token {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => token.v3(),
        X86Proof::V3GfniCrypto(token) => token.v3(),
        X86Proof::V3(token) => token,
        _ => unreachable!("V3 kernel reached without a selected V3 proof"),
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub(crate) fn x86_v2_token() -> archmage::X64V2Token {
    match x86_proof() {
        #[cfg(feature = "simd512")]
        X86Proof::V4x(token) => token.v3().v2(),
        X86Proof::V3GfniCrypto(token) => token.v3().v2(),
        X86Proof::V3(token) => token.v2(),
        X86Proof::V2(token) => token,
        X86Proof::Scalar => unreachable!("V2 kernel reached without a selected V2 proof"),
    }
}

/// The field's bulk-kernel strategy summary.
///
/// This is not a per-operation serving-tier guarantee: operation, coefficient,
/// and buffer geometry determine the executed body. Generic component rows
/// report their multiplication strategy, while addition, subtraction, and unit
/// accumulation reuse the base field's kernels. Residue rows summarize their
/// fixed-coefficient multiplication strategy; exact Mersenne31 and Goldilocks
/// moduli can reuse vector addition even when that strategy reports
/// [`Backend::Scalar`]. A scalar summary is not a blanket scalar-execution claim.
///
/// SIMD kernels are x86-64 only. On `AArch64` and `Wasm32` every field reports
/// [`Backend::Scalar`] and uses portable bodies; [`backend()`] still reports
/// the process selection.
#[inline]
#[must_use]
pub fn backend_for<F: FieldKernels>() -> Backend {
    F::backend()
}

/// Whether elementwise multiplication is vectorized for `F` on this host.
#[inline]
#[must_use]
pub fn has_vector_elementwise<F: FieldKernels>() -> bool {
    F::has_vector_elementwise()
}

/// Minimum row length in bytes at which `F` dispatches elementwise
/// multiplication to a vector kernel on this host; zero when every length
/// dispatches or no vector kernel serves `F`. Consumers choosing between
/// schedules use this beside [`has_vector_elementwise`] rather than
/// assuming the vector path serves sub-vector rows.
#[inline]
#[must_use]
pub fn vector_elementwise_min_bytes<F: FieldKernels>() -> usize {
    F::vector_elementwise_min_bytes()
}
/// The per-field vector kernel contract, as a public bound.
///
/// This trait is the *nameable* half of the kernel surface: generic code
/// writes `F: FieldKernels` and, through [`crate::ops`], gets that field's
/// dispatched kernels with every precondition checked. It is sealed — the
/// implementor set is fixed — and intentionally carries no methods beyond
/// two safe queries: the raw entry points live on the crate-private
/// `KernelDispatch` supertrait, so outside this crate they can be reached
/// only through the validated [`crate::ops`] facade (or, under the
/// unstable `internals` feature, through the token-gated architecture
/// modules — never as bare unchecked dispatch).
// Raw dispatch is sealed behind a private proof argument.
#[allow(private_bounds)]
pub trait FieldKernels: FieldBuffer + private::Sealed + KernelDispatch {
    /// Field-strategy summary, with the operation caveats of [`backend_for`].
    #[inline]
    #[must_use]
    fn backend() -> Backend {
        Backend::Scalar
    }

    /// Whether elementwise multiplication is vectorized for this field on
    /// this host.
    #[inline]
    #[must_use]
    fn has_vector_elementwise() -> bool {
        false
    }

    /// Minimum row length in bytes at which the vector elementwise kernels
    /// take over on this host; zero when no such threshold applies.
    #[inline]
    #[must_use]
    fn vector_elementwise_min_bytes() -> usize {
        0
    }
}

/// The raw per-field kernel entry points.
///
/// Implementations own runtime dispatch for their field. Every method's
/// preconditions are checked by the [`crate::ops`] wrappers, not here.
///
/// This trait is deliberately crate-private and a supertrait of
/// [`FieldKernels`]: external code can hold the `F: FieldKernels` bound but
/// cannot construct the private `RawDispatch` argument required by every
/// entry point. The seal stays private even under `internals`: implementors
/// are fixed.
///
/// # Preconditions
///
/// All slice lengths are in **bytes** and must be whole multiples of
/// `Self::BYTES`. `dst` and `src` must have equal length.
pub(crate) trait KernelDispatch: FieldBuffer {
    /// The backend-ready form of one coefficient.
    ///
    /// Different backends want different things from a coefficient: GFNI
    /// wants a broadcast word, the shuffle backends want nibble tables, the
    /// scalar path wants the element itself. [`KernelDispatch::prepare`]
    /// resolves that once — which is why the single-coefficient kernels below
    /// take a `Prepared` and not an `Elem`. The backend is fixed for the life
    /// of the process, so this moves the backend decision *out* of the hot
    /// call rather than adding one.
    type Prepared: Clone + Send + Sync + core::fmt::Debug;

    /// Resolve a coefficient into the form this host's backend wants.
    fn prepare(_proof: RawDispatch, coeff: crate::field::Elem<Self>) -> Self::Prepared;

    /// Recover the coefficient a [`KernelDispatch::Prepared`] was built from.
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> crate::field::Elem<Self>;

    /// `dst += src`, elementwise field addition.
    ///
    /// Required, with no default: addition is XOR for the binary fields and a
    /// characteristic-`p` fold for the prime fields, so a wrong-by-default
    /// body would be a silent defect. Binary impls route to
    /// [`crate::kernel::xor`].
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// `dst += sum(srcs[i])`: fold byte-offset sources out of one backing
    /// region into one row, every coefficient implicitly one.
    ///
    /// The gather whose coefficients are all the multiplicative identity —
    /// the shape of back-substitution over solved rows. The default folds
    /// sources one at a time through [`KernelDispatch::add_assign`], which is
    /// exact for every field; fields whose addition is XOR override it with
    /// the crate's blocked XOR gather, which holds the destination in
    /// registers across the whole source list.
    ///
    /// Callers pass offsets that satisfy `offset + dst.len() <=
    /// region.len()`; [`crate::ops::add_gather_offsets`] validates that.
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
        fallback::add_gather_offsets::<Self>(region, dst, offsets);
    }

    /// `dst -= src`, elementwise field subtraction.
    ///
    /// Identical to [`KernelDispatch::add_assign`] in characteristic two; prime
    /// fields subtract with a canonicalizing fold. Required, no default, for
    /// the same reason.
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// Pairwise row addition over two equal flat row buffers.
    ///
    /// Both buffers hold the same whole number of contiguous `row_len`-byte
    /// rows, and every row pair adds fieldwise: `dst_row[j] += src_row[j]`.
    /// Row boundaries do not change elementwise addition, so running
    /// [`KernelDispatch::add_assign`] over the whole buffers is exact for every
    /// field; that is the default. A backend overrides it only with a kernel
    /// that exploits the row geometry.
    fn add_assign_rows(_proof: RawDispatch, dst: &mut [u8], src: &[u8], _row_len: usize) {
        Self::add_assign(RawDispatch, dst, src);
    }

    /// Pairwise row subtraction over two equal flat row buffers.
    ///
    /// The row-geometry counterpart of [`KernelDispatch::sub_assign`],
    /// mirroring [`KernelDispatch::add_assign_rows`]: row boundaries do not
    /// change elementwise subtraction, so running
    /// [`KernelDispatch::sub_assign`] over the whole buffers is exact for
    /// every field; that is the default. A backend overrides it only with a
    /// kernel that exploits the row geometry.
    fn sub_assign_rows(_proof: RawDispatch, dst: &mut [u8], src: &[u8], _row_len: usize) {
        Self::sub_assign(RawDispatch, dst, src);
    }

    /// `dst += coeff * src`. The workhorse AXPY.
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// Applies a nonzero, non-unit value, preparing only the state its route needs.
    fn mul_add_unprepared(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: crate::field::Elem<Self>,
        src: &[u8],
    ) {
        Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
    }

    /// Applies one composed row through the field's portable row body.
    ///
    /// Compositions use it below [`fallback::COMPOSED_ROW_MIN_BYTES`].
    fn mul_add_portable(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeff: crate::field::Elem<Self>,
        src: &[u8],
    ) {
        prime::mul_add::<Self>(dst, coeff, src);
    }

    /// `dst = coeff * src`, out of place.
    ///
    /// The default copies `src` into `dst` and scales in place — two passes
    /// over the destination. Backends with a fused single-pass kernel
    /// override this to write the destination in one pass.
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        fallback::mul_into::<Self>(dst, coeff, src);
    }

    /// `dst *= coeff`, in place.
    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared);

    /// One source into many rows: `rows[j] += coeffs[j] * src` for each `j`.
    ///
    /// `rows` is a flat buffer of `coeffs.len()` contiguous rows of
    /// `row_len` bytes. This is the systematic-encode shape; blocked
    /// backends load `src` once per tile and update several rows from it.
    ///
    /// The coefficients are raw elements, so every backend has to resolve
    /// each one into its own form before it can use it. Callers holding the
    /// resolved form should call [`KernelDispatch::mul_add_scatter_with`].
    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[crate::field::Elem<Self>],
        src: &[u8],
    );

    /// One source overwrites many rows: `rows[j] = coeffs[j] * src` for each `j`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_scatter`]: the
    /// previous contents of the named rows are ignored, so callers need not
    /// zero them. The default applies the fused single-row
    /// [`KernelDispatch::mul_into`] per row without allocation;
    /// register-blocked backends may override it to seed accumulators from
    /// the source with no destination read.
    fn mul_into_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[crate::field::Elem<Self>],
        src: &[u8],
    ) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            fallback::mul_into_row::<Self>(row, coeff, src);
        }
    }

    /// Many sources into one row: `dst += sum(coeffs[i] * srcs[i])`.
    ///
    /// The transpose of [`KernelDispatch::mul_add_scatter`], and the shape that
    /// rebuilds a single lost symbol. Blocked backends hold the destination
    /// tile in registers while every source is folded in, so the destination
    /// is read and written once per tile rather than once per source.
    ///
    /// See [`KernelDispatch::mul_add_gather_with`] for the prepared form.
    fn mul_add_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[crate::field::Elem<Self>],
        srcs: &[&[u8]],
    );

    /// Many sources overwrite one row: `dst = sum(coeffs[i] * srcs[i])`.
    ///
    /// Unlike [`KernelDispatch::mul_add_gather`], the previous destination is
    /// ignored. The default starts with a fused single-source
    /// [`KernelDispatch::mul_into`], then accumulates the remaining prepared
    /// terms without allocation.
    fn mul_into_gather(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[crate::field::Elem<Self>],
        srcs: &[&[u8]],
    ) {
        fallback::mul_into_gather::<Self>(dst, coeffs, srcs);
    }

    /// Many sources into many rows: for each `(coeffs, src)` term,
    /// `rows[j] += coeffs[j] * src` for every `j` in `0..nrows`.
    ///
    /// Equivalent to a [`KernelDispatch::mul_add_scatter`] per term, but
    /// blocked backends hold a destination tile in registers across all
    /// terms, so destination memory traffic is independent of the term
    /// count. This is the decode/reconstruction shape.
    ///
    /// See [`KernelDispatch::mul_add_matrix_with`] for the prepared form.
    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::field::Elem<Self>], &[u8])],
    );

    /// [`KernelDispatch::mul_add_scatter`] over already-prepared coefficients.
    ///
    /// The default keeps preparation out of the row loop by applying the
    /// single-coefficient kernel to each row. Fields may override this when a
    /// backend can retain several prepared coefficients in registers.
    #[cfg(feature = "alloc")]
    fn mul_add_scatter_with(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        for (row, coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(RawDispatch, row, coeff, src);
        }
    }

    /// Prepared-plan scatter with access to both original and resolved
    /// coefficients.
    ///
    /// The default uses the prepared single-row path. Blocked backends may use
    /// `values` for representations whose preparation is already free.
    #[cfg(feature = "alloc")]
    fn mul_add_scatter_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        _values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src);
    }

    /// [`KernelDispatch::mul_add_gather`] over already-prepared coefficients.
    ///
    /// The default applies the single-coefficient kernel once per source.
    #[cfg(feature = "alloc")]
    fn mul_add_gather_with(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        for (coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(RawDispatch, dst, coeff, src);
        }
    }

    /// Prepared-plan gather with access to both original and resolved
    /// coefficients.
    ///
    /// The default applies prepared AXPY once per source.
    #[cfg(feature = "alloc")]
    fn mul_add_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather_with(RawDispatch, dst, coeffs, srcs);
    }

    /// [`KernelDispatch::mul_into_gather`] over already-prepared coefficients.
    #[cfg(any(feature = "alloc", test))]
    fn mul_into_gather_with(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        fallback::mul_into_gather_with::<Self>(dst, coeffs, srcs);
    }

    /// Prepared-plan dot product with original and resolved coefficients.
    #[cfg(feature = "alloc")]
    fn mul_into_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_into_gather_with(RawDispatch, dst, coeffs, srcs);
    }

    /// [`KernelDispatch::mul_add_matrix`] over already-prepared coefficients.
    ///
    /// The default applies each term row by row. Fields may override this with
    /// a blocked implementation that retains destination tiles in registers.
    #[cfg(test)]
    fn mul_add_matrix_with(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Prepared], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(RawDispatch, row, coeff, src);
            }
        }
    }

    /// Prepared-plan matrix using flat row-major coefficients and source rows.
    ///
    /// The default is allocation-free repeated prepared AXPY. Register-blocked
    /// backends may override it and consume the same flat geometry directly.
    #[cfg(feature = "alloc")]
    fn mul_add_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        _values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        fallback::mul_add_matrix_plan::<Self>(rows, row_len, nrows, coeffs, srcs);
    }

    /// Many sources overwrite many rows: `rows[j] = sum_t coeffs[t][j] * src[t]`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_matrix`] and the
    /// erasure-encode shape: the previous destination is ignored. The default
    /// zeroes the first `nrows` rows and accumulates; register-blocked backends
    /// override it to seed accumulators from zero in registers, writing each row
    /// once with no destination read and no separate fill.
    fn mul_into_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::field::Elem<Self>], &[u8])],
    ) {
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix(RawDispatch, rows, row_len, nrows, terms);
    }

    /// [`KernelDispatch::mul_add_matrix_plan`].
    #[cfg(feature = "alloc")]
    fn mul_into_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix_plan(RawDispatch, rows, row_len, nrows, values, coeffs, srcs);
    }

    /// Many sources into many disjoint rows scattered through `dst`: for each
    /// `(coeffs, src)` term, `dst[row_starts[j]..][..row_len] += coeffs[j] *
    /// src` for every `j`.
    ///
    /// Like [`KernelDispatch::mul_add_matrix`], but the destination rows are not
    /// contiguous — row `j` occupies `dst[row_starts[j] .. row_starts[j] +
    /// row_len]`. [`crate::ops::mul_add_matrix_at`] validates the
    /// offsets are in-bounds and pairwise disjoint before dispatch, so a
    /// blocked backend can write each recovered row to its final scattered
    /// slot and skip the staging copy a contiguous kernel forces on scattered
    /// decode outputs.
    ///
    /// The default is [`KernelDispatch::mul_add_matrix_at_rows`].
    /// Register-blocked backends override it to retain destination tiles in
    /// registers across terms.
    fn mul_add_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[crate::field::Elem<Self>], &[u8])],
    ) {
        Self::mul_add_matrix_at_rows(RawDispatch, dst, row_len, row_starts, terms);
    }

    /// [`KernelDispatch::mul_add_matrix_at`] as one dispatched
    /// [`KernelDispatch::mul_add`] per row and term, so it inherits the
    /// field's own arithmetic and single-row vector kernel. The fallback for
    /// every backend without a blocked scattered kernel.
    fn mul_add_matrix_at_rows(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[crate::field::Elem<Self>], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (&start, &coeff) in row_starts.iter().zip(coeffs) {
                Self::mul_add(
                    RawDispatch,
                    &mut dst[start..start + row_len],
                    &Self::prepare(RawDispatch, coeff),
                    src,
                );
            }
        }
    }

    /// Prepared-plan scattered matrix using flat source-major coefficients
    /// and source rows.
    ///
    /// The scattered counterpart of
    /// [`KernelDispatch::mul_add_matrix_plan`]. The default is
    /// allocation-free repeated prepared AXPY over the named rows.
    /// Register-blocked backends may override it and consume the same flat
    /// geometry directly.
    #[cfg(feature = "alloc")]
    fn mul_add_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        _values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        fallback::mul_add_matrix_at_plan::<Self>(dst, row_len, row_starts, coeffs, srcs);
    }

    /// Many sources overwrite many disjoint rows scattered through `dst`:
    /// for each output row `j`,
    /// `dst[row_starts[j]..][..row_len] = sum_t coeffs[t][j] * src[t]`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_matrix_at`].
    /// The default zeroes the named rows and accumulates; register-blocked
    /// backends override it to seed accumulators from zero in registers,
    /// writing each named row once with no destination read and no separate
    /// fill.
    fn mul_into_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[crate::field::Elem<Self>], &[u8])],
    ) {
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        Self::mul_add_matrix_at(RawDispatch, dst, row_len, row_starts, terms);
    }

    /// [`KernelDispatch::mul_add_matrix_at_plan`].
    #[cfg(feature = "alloc")]
    fn mul_into_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[crate::field::Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        for &start in row_starts {
            dst[start..start + row_len].fill(0);
        }
        Self::mul_add_matrix_at_plan(RawDispatch, dst, row_len, row_starts, values, coeffs, srcs);
    }

    /// `dst[i] = a[i] * b[i]`, elementwise over two full vectors.
    ///
    /// Both operands vary per lane, so there is no coefficient to broadcast
    /// and no table to index. GFNI multiplies vectors directly; `AArch64`,
    /// Wasm, and the shuffle-only x86 backends use a branchless
    /// shift/reduce vector multiply. The wider fields run the reference path.
    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]);

    /// `dst[i] *= src[i]`, elementwise, in place.
    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        scalar::mul_elementwise_assign::<Self>(dst, src);
    }

    /// `dst[i] += value`, one field element broadcast across every lane.
    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        fallback::add_assign_scalar::<Self>(dst, value);
    }

    /// `dst[i] -= value`, one field element broadcast across every lane.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        fallback::sub_assign_scalar::<Self>(dst, value);
    }
}

/// The default bodies of the [`KernelDispatch`] entries a storage-routed
/// extension row forwards, shared so the extension storages default to
/// exactly the trait's own defaults.
pub(crate) mod fallback {
    use super::{KernelDispatch, PrimeIdentity, RawDispatch, bytes, prime};
    use crate::field::{Elem, Field, FieldElem};

    /// Folds byte-offset sources one at a time through `add_assign`.
    pub(crate) fn add_gather_offsets<F: KernelDispatch>(
        region: &[u8],
        dst: &mut [u8],
        offsets: &[u32],
    ) {
        let live = dst.len();
        for &start in offsets {
            F::add_assign(
                RawDispatch,
                dst,
                &region[start as usize..start as usize + live],
            );
        }
    }

    /// Copies `src` into `dst`, then scales in place.
    pub(crate) fn mul_into<F: KernelDispatch>(dst: &mut [u8], coeff: &F::Prepared, src: &[u8]) {
        dst.copy_from_slice(src);
        F::mul_assign(RawDispatch, dst, coeff);
    }

    /// Smallest row, in bytes, that a composition dispatches row by row;
    /// shorter rows run the field's portable row body. Set by the paired
    /// composition record in `benchmarks/v3/compositions.md`.
    pub(crate) const COMPOSED_ROW_MIN_BYTES: usize = 128;

    /// Dispatches one row, preserving zero destinations byte-for-byte.
    #[inline]
    fn mul_add_row<F: KernelDispatch>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
        if coeff.is_zero() {
            return;
        }
        if dst.len() < COMPOSED_ROW_MIN_BYTES {
            F::mul_add_portable(RawDispatch, dst, coeff, src);
            return;
        }
        if coeff.is_one() {
            F::add_assign(RawDispatch, dst, src);
            return;
        }
        // Canonical zero and unit comparisons exclude both trivial coefficients.
        #[cfg(test)]
        let _general_trace = crate::kernel::trace::general_mul_add();
        F::mul_add_unprepared(RawDispatch, dst, coeff, src);
    }

    /// Accumulates one source into coefficient-addressed destination rows.
    pub(crate) fn mul_add_scatter<F: KernelDispatch>(
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<F>],
        src: &[u8],
    ) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            mul_add_row::<F>(row, coeff, src);
        }
    }

    /// Accumulates weighted sources into one destination row.
    pub(crate) fn mul_add_gather<F: KernelDispatch>(
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            mul_add_row::<F>(dst, coeff, src);
        }
    }

    /// Applies every coefficient/source term to the destination rows.
    pub(crate) fn mul_add_matrix<F: KernelDispatch>(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<F>], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                mul_add_row::<F>(row, coeff, src);
            }
        }
    }

    /// Overwrites one row, preserving source bytes for a unit coefficient.
    #[inline]
    pub(crate) fn mul_into_row<F: KernelDispatch>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
        if coeff.is_zero() {
            dst.fill(0);
        } else if coeff.is_one() {
            dst.copy_from_slice(src);
        } else {
            F::mul_into(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
        }
    }

    /// Overwrites one row with weighted sources: a fused single-source
    /// `mul_into` seeds the row, then each later term accumulates.
    pub(crate) fn mul_into_gather<F: KernelDispatch>(
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        F::mul_into(RawDispatch, dst, &F::prepare(RawDispatch, first), src);
        for (coeff, src) in pairs {
            F::mul_add(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
        }
    }

    /// [`mul_into_gather`] over prepared coefficients.
    #[cfg(any(feature = "alloc", test))]
    pub(crate) fn mul_into_gather_with<F: KernelDispatch>(
        dst: &mut [u8],
        coeffs: &[F::Prepared],
        srcs: &[&[u8]],
    ) {
        let mut pairs = coeffs.iter().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        F::mul_into(RawDispatch, dst, first, src);
        for (coeff, src) in pairs {
            F::mul_add(RawDispatch, dst, coeff, src);
        }
    }

    /// Overwrites one row with weighted sources term by term: the first term
    /// seeds the row through [`mul_into_row`], later zero terms are skipped,
    /// unit terms add their source, and only general terms multiply.
    pub(crate) fn mul_into_gather_checked<F: KernelDispatch>(
        dst: &mut [u8],
        coeffs: &[Elem<F>],
        srcs: &[&[u8]],
    ) {
        let mut terms = coeffs.iter().copied().zip(srcs.iter().copied());
        let Some((first, src)) = terms.next() else {
            dst.fill(0);
            return;
        };
        mul_into_row::<F>(dst, first, src);
        for (coeff, src) in terms {
            if coeff.is_zero() {
                continue;
            }
            if coeff.is_one() {
                F::add_assign(RawDispatch, dst, src);
                continue;
            }
            // Canonical zero and unit comparisons exclude both trivial coefficients.
            #[cfg(test)]
            let _general_trace = crate::kernel::trace::general_mul_add();
            F::mul_add_unprepared(RawDispatch, dst, coeff, src);
        }
    }

    /// [`mul_into_gather_checked`] over prepared coefficients, classified by
    /// the value each prepared form was built from.
    #[cfg(any(feature = "alloc", test))]
    pub(crate) fn mul_into_gather_checked_with<F: KernelDispatch>(
        dst: &mut [u8],
        coeffs: &[F::Prepared],
        srcs: &[&[u8]],
    ) {
        let mut terms = coeffs.iter().zip(srcs.iter().copied());
        let Some((first, src)) = terms.next() else {
            dst.fill(0);
            return;
        };
        let value = F::prepared_coeff(RawDispatch, first);
        if value.is_zero() {
            dst.fill(0);
        } else if value.is_one() {
            dst.copy_from_slice(src);
        } else {
            F::mul_into(RawDispatch, dst, first, src);
        }
        for (coeff, src) in terms {
            let value = F::prepared_coeff(RawDispatch, coeff);
            if value.is_zero() {
                continue;
            }
            if value.is_one() {
                F::add_assign(RawDispatch, dst, src);
                continue;
            }
            // Canonical zero and unit comparisons exclude both trivial coefficients.
            #[cfg(test)]
            let _general_trace = crate::kernel::trace::general_mul_add();
            F::mul_add(RawDispatch, dst, coeff, src);
        }
    }

    /// Allocation-free repeated prepared AXPY over flat row-major
    /// coefficients.
    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_matrix_plan<F: KernelDispatch>(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        coeffs: &[F::Prepared],
        srcs: &[&[u8]],
    ) {
        for (term, &src) in srcs.iter().enumerate() {
            let start = term * nrows;
            for (row, coeff) in rows
                .chunks_exact_mut(row_len)
                .take(nrows)
                .zip(&coeffs[start..start + nrows])
            {
                F::mul_add(RawDispatch, row, coeff, src);
            }
        }
    }

    /// Allocation-free repeated prepared AXPY over the named scattered rows.
    #[cfg(feature = "alloc")]
    pub(crate) fn mul_add_matrix_at_plan<F: KernelDispatch>(
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        coeffs: &[F::Prepared],
        srcs: &[&[u8]],
    ) {
        let nrows = row_starts.len();
        for (term, &src) in srcs.iter().enumerate() {
            let start = term * nrows;
            for (&at, coeff) in row_starts.iter().zip(&coeffs[start..start + nrows]) {
                F::mul_add(RawDispatch, &mut dst[at..at + row_len], coeff, src);
            }
        }
    }

    /// Broadcast addition: XOR of the encoded value in characteristic two,
    /// the modular path otherwise.
    pub(crate) fn add_assign_scalar<F: KernelDispatch>(dst: &mut [u8], value: &F::Prepared) {
        let elem = F::prepared_coeff(RawDispatch, value);
        if <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2 {
            let mut encoded = [0u8; 8];
            F::encode(&mut encoded[..F::BYTES], elem);
            bytes::xor_broadcast(dst, &encoded[..F::BYTES]);
        } else {
            prime::add_assign_scalar::<F>(dst, elem);
        }
    }

    /// Broadcast subtraction, as [`add_assign_scalar`].
    pub(crate) fn sub_assign_scalar<F: KernelDispatch>(dst: &mut [u8], value: &F::Prepared) {
        let elem = F::prepared_coeff(RawDispatch, value);
        if <<F as Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2 {
            let mut encoded = [0u8; 8];
            F::encode(&mut encoded[..F::BYTES], elem);
            bytes::xor_broadcast(dst, &encoded[..F::BYTES]);
        } else {
            prime::sub_assign_scalar::<F>(dst, elem);
        }
    }
}

/// Zero-sized proof of the right to call [`KernelDispatch`] entry points.
///
/// Private-supertrait sealing alone is not enough: bound elaboration lets
/// external generic code holding only the public [`FieldKernels`] bound name
/// `F::add_assign` and friends, trait-name privacy notwithstanding. Every
/// `KernelDispatch` method therefore also takes this argument, and the type
/// is `pub(crate)` — outside the crate it cannot be named, constructed, or
/// accepted as a parameter, so those elaborated paths have no argument to
/// call them with. Inside the crate it costs nothing: `RawDispatch` is a
/// zero-sized unit struct and the argument exists only in the type system.
pub(crate) struct RawDispatch;

/// Shared runtime geometry validation for direct architecture kernel entries.
///
/// Direct x86 entries own their validation through these helpers, which apply
/// the public geometry contract before entering the kernels.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) mod proven_checks {
    /// Paired length equality.
    #[inline]
    pub(crate) fn check_equal(
        name: &str,
        left_name: impl core::fmt::Display,
        left: usize,
        right_name: impl core::fmt::Display,
        right: usize,
    ) {
        assert_eq!(
            left, right,
            "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
        );
    }

    /// Whole-element lengths; only the x86 prime-field kernels check them.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) fn check_elem_multiple(name: &str, len: usize, elem_bytes: usize) {
        assert!(
            len.is_multiple_of(elem_bytes),
            "{name}: buffer of {len} bytes is not a whole number of {elem_bytes}-byte elements"
        );
    }

    /// Flat term geometry shared by the matrix entries: every term supplies
    /// `nrows` coefficients and a `row_len`-byte source.
    #[inline]
    pub(crate) fn check_terms<E>(
        name: &str,
        row_len: usize,
        nrows: usize,
        terms: &[(&[E], &[u8])],
    ) {
        for (t, &(coeffs, src)) in terms.iter().enumerate() {
            check_equal(
                name,
                format_args!("term {t} coefficients"),
                coeffs.len(),
                "rows",
                nrows,
            );
            check_equal(
                name,
                format_args!("term {t} source"),
                src.len(),
                "row_len",
                row_len,
            );
        }
    }
}
