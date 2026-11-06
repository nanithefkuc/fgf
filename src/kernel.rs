//! Vector kernels and backend selection.
//!
//! Layout of this module:
//!
//! - [`Backend`] — which instruction set the kernels will use, detected once
//!   per process.
//! - the row and storage implementations of the field layer's
//!   `KernelDispatch` contract, in `gf8`, `goldilocks`, `mersenne31`,
//!   `residue`, and `extension` (whose storages the `tower` module
//!   serves); `fallback` holds the contract's default compositions. The
//!   `internals` feature exposes direct token-bearing architecture entries
//!   for measurement and differential testing.
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

#[macro_use]
pub(crate) mod fallback;

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
mod tests;

use crate::field::FieldBuffer;

// Only the SIMD-enabled resolve path consults the environment; under a
// std-less build `backend()` reports `Scalar` without touching `Selection`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::SimdToken;
#[cfg(feature = "simd")]
use simdispatch::Selection;

// The backend ladder is owned by `simdispatch` (the Level 0 single source for
// detection and ordering); FGF re-exports `Backend` for its public query
// surface and builds its own `Selection` over the tiers it implements.
pub use simdispatch::Backend;

/// The tiers FGF implements kernels for, in detection-preference order: the
/// common ladder minus the x86 `V1` floor (FGF's 16-byte multiply kernels
/// require SSSE3, not plain SSE2, so a V1-only host resolves to
/// [`Backend::Scalar`]).
///
/// `V4x` resolves only when the `simd512` feature enables the 512-bit kernel
/// modules and `simdispatch/avx512` detection. Without `simd512` the variant
/// exists in the ladder but no `FGF_TIERS` entry can select it, so detection
/// falls through to `V3GfniCrypto` on AVX-512 hosts.
// Re-exported through `internals`; the scalar-only configurations have no
// other reader.
#[cfg(not(feature = "simd512"))]
#[cfg_attr(not(feature = "simd"), allow(dead_code))]
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

// Re-exported through `internals`; production dispatch never calls it.
/// FGF's kernel-policy questions over the shared ladder.
///
/// `simdispatch::Backend` is capability; these two derive from the kernels
/// this crate actually implements, so they stay here (field-kernel policy,
/// not hardware facts). The ladder is owned upstream, so `Backend` gets these
/// only through this trait.
#[allow(dead_code)]
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
/// Runs [`simdispatch::Selection`] over `FGF_TIERS` — every tier this crate implements
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

/// The backend used for a particular field's bulk operations.
///
/// This is a field-strategy summary: it names the serving backend for the
/// field's dispatched bulk row operations and does not identify the serving
/// tier of every individual operation, which may route differently by shape
/// or length.
///
/// Presentations without a kernel strategy on the selected tier report
/// [`Backend::Scalar`] and run the typed scalar fallback: ordered-basis byte
/// fields, degree-eight towers, custom tower specs, and pinned towers on a
/// tier their kernels do not serve. SIMD kernels are x86-64 only, so on
/// `AArch64` and `Wasm32` every field reports [`Backend::Scalar`] and runs the
/// portable scalar bodies (`backend()` itself still reports `simdispatch`'s
/// process selection).
#[inline]
#[must_use]
pub fn backend_for<F: FieldBuffer>() -> Backend {
    let () = crate::field::Validate::<F>::OK;
    F::backend(crate::field::dispatch::RawDispatch)
}

// The free wrappers are re-exported through `internals`; the stable query
// surface is `backend_for` alone.
/// Whether elementwise multiplication is vectorized for `F` on this host.
#[allow(dead_code)]
#[inline]
#[must_use]
pub fn has_vector_elementwise<F: FieldBuffer>() -> bool {
    F::has_vector_elementwise(crate::field::dispatch::RawDispatch)
}

/// Minimum row length in bytes at which `F` dispatches elementwise
/// multiplication to a vector kernel on this host; zero when every length
/// dispatches or no vector kernel serves `F`. Consumers choosing between
/// schedules use this beside [`has_vector_elementwise`] rather than
/// assuming the vector path serves sub-vector rows.
#[allow(dead_code)]
#[inline]
#[must_use]
pub fn vector_elementwise_min_bytes<F: FieldBuffer>() -> usize {
    F::vector_elementwise_min_bytes(crate::field::dispatch::RawDispatch)
}

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
