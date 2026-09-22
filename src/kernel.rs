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
//! - `x86` / `aarch64` / `wasm32` — architecture-local intrinsics. Direct x86
//!   entries take exact `archmage` capability tokens and validate geometry.
//!   Wasm retains its token-proven compatibility facade.
//!
//! Callers should use the safe, validated wrappers in [`crate::ops`] rather
//! than this module directly.

pub(crate) mod fan_paar;
pub(crate) mod tower;
// The re-exported names are reached only from the architecture kernels,
// which cfg away entirely on a scalar-only build.
#[allow(unused_imports)]
pub(crate) use tower::{gf16, gf32, gf64};

pub(crate) mod gf2;
pub(crate) mod gf8;
pub(crate) mod goldilocks;
pub(crate) mod mersenne31;
pub(crate) mod prime;
pub(crate) mod quad_mersenne31;
pub(crate) mod scalar;
pub(crate) mod tables;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) mod aarch64;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub(crate) mod wasm32;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) mod x86;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) mod matrix_provider;
pub(crate) use byte_ops::xor;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) use matrix_provider::{FlatMatrix, Matrix};

#[cfg(test)]
mod tests;

use crate::field::Field;

// Only the SIMD-enabled resolve path consults the environment; under a
// std-less build `backend()` reports `Scalar` without touching `Selection`.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
use archmage::SimdToken;
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
/// Runs [`simdispatch::Selection`] over [`FGF_TIERS`] — every tier this crate implements
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    x86: X86Proof,
    #[cfg(target_arch = "aarch64")]
    neon: NeonProof,
    #[cfg(target_arch = "wasm32")]
    wasm128: Option<archmage::Wasm128Token>,
}

#[cfg(feature = "simd")]
impl ResolvedBackend {
    fn new() -> Self {
        let backend = Selection::new("SIMD_BACKEND").supports(FGF_TIERS).resolve();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86 = match backend {
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
        #[cfg(target_arch = "aarch64")]
        let neon = match backend {
            Backend::NeonAes => NeonProof::NeonAes(
                archmage::NeonAesToken::summon()
                    .expect("selected NEON-AES backend must have its capability token"),
            ),
            Backend::Neon => NeonProof::Neon(
                archmage::NeonToken::summon()
                    .expect("selected NEON backend must have its capability token"),
            ),
            _ => NeonProof::Scalar,
        };
        #[cfg(target_arch = "wasm32")]
        let wasm128 = match backend {
            Backend::Wasm128 => Some(
                archmage::Wasm128Token::summon()
                    .expect("selected wasm128 backend must have its capability token"),
            ),
            _ => None,
        };
        Self {
            backend,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86,
            #[cfg(target_arch = "aarch64")]
            neon,
            #[cfg(target_arch = "wasm32")]
            wasm128,
        }
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[derive(Clone, Copy)]
pub(crate) enum X86Proof {
    V3GfniCrypto(archmage::X64V3GfniCryptoToken),
    V3(archmage::X64V3Token),
    V2(archmage::X64V2Token),
    Scalar,
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub(crate) fn x86_proof() -> X86Proof {
    BACKEND.x86
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub(crate) fn x86_v3_gfni_token() -> archmage::X64V3GfniCryptoToken {
    match x86_proof() {
        X86Proof::V3GfniCrypto(token) => token,
        _ => unreachable!("GFNI kernel reached without the selected GFNI proof"),
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub(crate) fn x86_v3_token() -> archmage::X64V3Token {
    match x86_proof() {
        X86Proof::V3GfniCrypto(token) => token.v3(),
        X86Proof::V3(token) => token,
        _ => unreachable!("V3 kernel reached without a selected V3 proof"),
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub(crate) fn x86_v2_token() -> archmage::X64V2Token {
    match x86_proof() {
        X86Proof::V3GfniCrypto(token) => token.v3().v2(),
        X86Proof::V3(token) => token.v2(),
        X86Proof::V2(token) => token,
        X86Proof::Scalar => unreachable!("V2 kernel reached without a selected V2 proof"),
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
pub(crate) enum NeonProof {
    Neon(archmage::NeonToken),
    NeonAes(archmage::NeonAesToken),
    Scalar,
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub(crate) fn neon_proof() -> NeonProof {
    BACKEND.neon
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub(crate) fn neon_token() -> archmage::NeonToken {
    match neon_proof() {
        NeonProof::NeonAes(token) => token.neon(),
        NeonProof::Neon(token) => token,
        NeonProof::Scalar => unreachable!("NEON kernel reached without a selected NEON proof"),
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub(crate) fn neon_aes_token() -> archmage::NeonAesToken {
    match neon_proof() {
        NeonProof::NeonAes(token) => token,
        _ => unreachable!("PMULL kernel reached without a selected NEON-AES proof"),
    }
}

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[inline]
pub(crate) fn wasm128_token() -> archmage::Wasm128Token {
    BACKEND
        .wasm128
        .expect("Wasm kernel reached without a selected simd128 proof")
}

/// The backend used for a particular field.
///
/// Wider polynomial towers and the Fan–Paar fields currently report
/// [`Backend::Scalar`] even when [`backend()`] selected a vector backend for
/// `Gf8B` and `Gf16`.
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
pub trait FieldKernels: Field + private::Sealed + KernelDispatch {
    /// Backend used by this field's kernels.
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
pub(crate) trait KernelDispatch: Field {
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
    fn prepare(_proof: RawDispatch, coeff: Self::Elem) -> Self::Prepared;

    /// Recover the coefficient a [`KernelDispatch::Prepared`] was built from.
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Self::Elem;

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
        let live = dst.len();
        for &start in offsets {
            Self::add_assign(
                RawDispatch,
                dst,
                &region[start as usize..start as usize + live],
            );
        }
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
    /// field — that is the default, and on the reference host it is also the
    /// fastest known implementation at every measured geometry: a four-stream
    /// row-interleaved XOR candidate matched or trailed it from L1 to DRAM
    /// (BENCHMARKS.md, "Row-interleaved XOR"). The experimental interleaved
    /// kernels live behind `internals` for future evaluation; no backend
    /// override is wired until one measures a repeatable win. The prime
    /// fields additionally fold lanes rather than bytes, so their default is
    /// semantic, not just an optimization choice.
    fn add_assign_rows(_proof: RawDispatch, dst: &mut [u8], src: &[u8], _row_len: usize) {
        Self::add_assign(RawDispatch, dst, src);
    }

    /// `dst += coeff * src`. The workhorse AXPY.
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// `dst = coeff * src`, out of place.
    ///
    /// The default copies `src` into `dst` and scales in place — two passes
    /// over the destination. Backends with a fused single-pass kernel
    /// override this. The override is worth roughly 2x on large buffers where
    /// the kernel is bandwidth-bound (x86 GF(2^8)/GF(2^16)); where it is
    /// compute-bound instead, as GF(2^16) is on NEON and wasm, halving
    /// destination traffic buys only a few percent (BENCHMARKS.md).
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        dst.copy_from_slice(src);
        Self::mul_assign(RawDispatch, dst, coeff);
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
        coeffs: &[Self::Elem],
        src: &[u8],
    );

    /// Many sources into one row: `dst += sum(coeffs[i] * srcs[i])`.
    ///
    /// The transpose of [`KernelDispatch::mul_add_scatter`], and the shape that
    /// rebuilds a single lost symbol. Blocked backends hold the destination
    /// tile in registers while every source is folded in, so the destination
    /// is read and written once per tile rather than once per source.
    ///
    /// See [`KernelDispatch::mul_add_gather_with`] for the prepared form.
    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]);

    /// Many sources overwrite one row: `dst = sum(coeffs[i] * srcs[i])`.
    ///
    /// Unlike [`KernelDispatch::mul_add_gather`], the previous destination is
    /// ignored. The default starts with a fused single-source
    /// [`KernelDispatch::mul_into`], then accumulates the remaining prepared
    /// terms without allocation.
    fn mul_into_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]) {
        let mut pairs = coeffs.iter().copied().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        Self::mul_into(RawDispatch, dst, &Self::prepare(RawDispatch, first), src);
        for (coeff, src) in pairs {
            Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
        }
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
        terms: &[(&[Self::Elem], &[u8])],
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
        _values: &[Self::Elem],
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
        _values: &[Self::Elem],
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
        let mut pairs = coeffs.iter().zip(srcs.iter().copied());
        let Some((first, src)) = pairs.next() else {
            dst.fill(0);
            return;
        };
        Self::mul_into(RawDispatch, dst, first, src);
        for (coeff, src) in pairs {
            Self::mul_add(RawDispatch, dst, coeff, src);
        }
    }

    /// Prepared-plan dot product with original and resolved coefficients.
    #[cfg(feature = "alloc")]
    fn mul_into_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[Self::Elem],
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
                Self::mul_add(RawDispatch, row, coeff, src);
            }
        }
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
        terms: &[(&[Self::Elem], &[u8])],
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
        values: &[Self::Elem],
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
    /// The default applies each term row by row through the portable path.
    /// Register-blocked backends override it to retain destination tiles in
    /// registers across terms.
    fn mul_add_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Self::Elem], &[u8])],
    ) {
        scalar::mul_add_matrix_at::<Self>(dst, row_len, row_starts, terms);
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
        let elem = Self::prepared_coeff(RawDispatch, value);
        if Self::CHARACTERISTIC == 2 {
            scalar::xor_broadcast::<Self>(dst, elem);
        } else {
            prime::add_assign_scalar::<Self>(dst, elem);
        }
    }

    /// `dst[i] -= value`, one field element broadcast across every lane.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        let elem = Self::prepared_coeff(RawDispatch, value);
        if Self::CHARACTERISTIC == 2 {
            scalar::xor_broadcast::<Self>(dst, elem);
        } else {
            prime::sub_assign_scalar::<Self>(dst, elem);
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
/// Direct x86 entries own their validation; the `AArch64` and Wasm entries use
/// these same helpers to apply the public geometry contract before entering
/// their kernels, as do the deferred AVX-512 experiments.
// Compiled wherever an architecture kernel entry can use it.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
#[allow(dead_code)]
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

    /// Whole-element lengths.
    #[inline]
    pub(crate) fn check_elem_multiple(name: &str, len: usize, elem_bytes: usize) {
        assert!(
            len.is_multiple_of(elem_bytes),
            "{name}: buffer of {len} bytes is not a whole number of {elem_bytes}-byte elements"
        );
    }

    /// A flat row buffer must hold at least `count` rows of `row_len` bytes.
    #[inline]
    pub(crate) fn check_row_span(name: &str, buffer_len: usize, row_len: usize, count: usize) {
        let used = count
            .checked_mul(row_len)
            .unwrap_or_else(|| panic!("{name}: row geometry overflows"));
        assert!(
            buffer_len >= used,
            "{name}: rows is {buffer_len} bytes but {count} rows of {row_len} bytes need {used}"
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
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    match x86_proof() {
        X86Proof::V3GfniCrypto(token) => {
            x86::xor_gather_avx2(token.v3(), region, dst, offsets);
        }
        X86Proof::V3(token) => x86::xor_gather_avx2(token, region, dst, offsets),
        X86Proof::V2(token) => {
            for &start in offsets {
                x86::xor_sse2(
                    token.v1(),
                    dst,
                    &region[start as usize..start as usize + live],
                );
            }
        }
        X86Proof::Scalar => {
            for &start in offsets {
                scalar::xor(dst, &region[start as usize..start as usize + live]);
            }
        }
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    match backend() {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Backend::Neon | Backend::NeonAes => {
            for &start in offsets {
                aarch64::xor_neon(
                    neon_token(),
                    dst,
                    &region[start as usize..start as usize + live],
                );
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
/// portable one: below a single vector the out-of-line SIMD entry boundary
/// costs more than the body saves, and short GF(2) rows (eight
/// elements per byte) sit almost entirely under it. Measured on the
/// reference host (BENCHMARKS.md, "Short-buffer inline XOR").
const XOR_INLINE_MAX: usize = 31;

pub(crate) mod byte_ops {
    use super::*;

    /// `dst ^= src` over raw bytes.
    ///
    /// Field-independent: addition in every binary field is XOR, and XOR of a
    /// packed element array is XOR of its bytes regardless of element width.
    ///
    /// # Panics
    /// Panics before mutation if the slices differ in length.
    pub fn xor(dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len(), "fgf::xor: length mismatch");

        if dst.len() <= XOR_INLINE_MAX {
            scalar::xor(dst, src);
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        match x86_proof() {
            X86Proof::V3GfniCrypto(token) => x86::xor_avx2(token.v3(), dst, src),
            X86Proof::V3(token) => x86::xor_avx2(token, dst, src),
            X86Proof::V2(token) => x86::xor_sse2(token.v1(), dst, src),
            X86Proof::Scalar => scalar::xor(dst, src),
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        match backend() {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon | Backend::NeonAes => aarch64::xor_neon(neon_token(), dst, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Wasm128 => wasm32::xor_simd128(dst, src),
            _ => scalar::xor(dst, src),
        }
    }
}
