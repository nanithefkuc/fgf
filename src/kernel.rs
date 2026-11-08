//! Vector kernels and backend selection.
//!
//! Layout of this module:
//!
//! - [`Backend`] — which instruction set the kernels will use, detected once
//!   per process.
//! - the per-family kernel strategies — one enum per kernel family, each
//!   resolved once beside the process backend and read by the family's
//!   dispatch arms instead of re-deriving the tier per call.
//! - the row and storage implementations of the field layer's
//!   `KernelDispatch` contract, in `gf8`, `goldilocks`, `mersenne31`,
//!   `residue`, and `extension` (whose storages the `tower` module
//!   serves); `fallback` holds the contract's default compositions, and
//!   the two shared impl generators below expand the prime-lane dispatch
//!   and the portable scaled-row loops. The `internals` feature exposes
//!   direct token-bearing architecture entries for measurement and
//!   differential testing.
//! - `routing` — every threshold and inline gate that selects between
//!   kernels, plus the residue geometry selectors and the record of where
//!   backend policy resolves.
//! - `scalar` — the portable fallback and tail implementation. Vector kernels
//!   use it for sub-lane tails; tests compare the Fan–Paar family against the
//!   independent Wiedemann recurrence and other families against this module.
//!   The scaled-row loops it shares with `prime` come from one generator.
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

/// Generates the portable scaled-row loops shared by one addition
/// convention.
///
/// `scalar` and `prime` decode, multiply, and encode through the same
/// field-generic bodies and share the same zero and unit dispositions;
/// only the unit shortcut differs, which is what keeps each convention's
/// byte-level behavior exact — XOR for the binary rows, the field's
/// modular add for the prime rows — and the elementwise-assign loop is
/// generated only where the convention exposes it. The summary docs name
/// the convention's addition and assignment.
macro_rules! portable_scaled_rows {
    (
        $mul_add_doc:literal,
        $mul_assign_doc:literal,
        $unit:path
        $(, $elementwise_assign:ident)?
        $(,)?
    ) => {
        #[doc = $mul_add_doc]
        pub fn mul_add<F: $crate::field::FieldBuffer>(
            dst: &mut [u8],
            coeff: $crate::field::Elem<F>,
            src: &[u8],
        ) {
            debug_assert_eq!(dst.len(), src.len());

            if coeff.is_zero() {
                return;
            }
            if coeff.is_one() {
                $unit(dst, src);
                return;
            }
            for (d, s) in dst
                .chunks_exact_mut(F::BYTES)
                .zip(src.chunks_exact(F::BYTES))
            {
                let value = F::decode(d).add(F::decode(s).mul(coeff));
                F::encode(d, value);
            }
        }

        #[doc = $mul_assign_doc]
        pub fn mul_assign<F: $crate::field::FieldBuffer>(
            dst: &mut [u8],
            coeff: $crate::field::Elem<F>,
        ) {
            if coeff.is_one() {
                return;
            }
            if coeff.is_zero() {
                dst.fill(0);
                return;
            }
            for d in dst.chunks_exact_mut(F::BYTES) {
                let value = F::decode(d).mul(coeff);
                F::encode(d, value);
            }
        }

        /// `rows[j] += coeffs[j] * src` for every row `j`.
        pub fn mul_add_scatter<F: $crate::field::FieldBuffer>(
            rows: &mut [u8],
            row_len: usize,
            coeffs: &[$crate::field::Elem<F>],
            src: &[u8],
        ) {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                mul_add::<F>(row, coeff, src);
            }
        }

        /// `dst += sum(coeffs[i] * srcs[i])`.
        pub fn mul_add_gather<F: $crate::field::FieldBuffer>(
            dst: &mut [u8],
            coeffs: &[$crate::field::Elem<F>],
            srcs: &[&[u8]],
        ) {
            for (&coeff, &src) in coeffs.iter().zip(srcs) {
                mul_add::<F>(dst, coeff, src);
            }
        }

        /// Apply every `(coeffs, src)` term to all `nrows` rows.
        pub fn mul_add_matrix<F: $crate::field::FieldBuffer>(
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[$crate::field::Elem<F>], &[u8])],
        ) {
            for &(coeffs, src) in terms {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                    mul_add::<F>(row, coeff, src);
                }
            }
        }

        /// `dst[i] = a[i] * b[i]`, elementwise.
        pub fn mul_elementwise<F: $crate::field::FieldBuffer>(
            dst: &mut [u8],
            a: &[u8],
            b: &[u8],
        ) {
            debug_assert_eq!(dst.len(), a.len());
            debug_assert_eq!(dst.len(), b.len());

            for ((d, x), y) in dst
                .chunks_exact_mut(F::BYTES)
                .zip(a.chunks_exact(F::BYTES))
                .zip(b.chunks_exact(F::BYTES))
            {
                F::encode(d, F::decode(x).mul(F::decode(y)));
            }
        }

        $(
            /// `dst[i] *= src[i]`, elementwise, in place.
            pub fn $elementwise_assign<F: $crate::field::FieldBuffer>(
                dst: &mut [u8],
                src: &[u8],
            ) {
                debug_assert_eq!(dst.len(), src.len());
                for (d, s) in dst
                    .chunks_exact_mut(F::BYTES)
                    .zip(src.chunks_exact(F::BYTES))
                {
                    F::encode(d, F::decode(d).mul(F::decode(s)));
                }
            }
        )?
    };
}

/// Generates the whole `KernelDispatch` implementation for one shipped
/// prime-lane field.
///
/// Mersenne31 and Goldilocks share the contract's shape: identity
/// preparation, the same fallback list, the same portable compositions for
/// the raw-element shapes, and one strategy match per vectorizable entry
/// that invokes their `$kernels` module's entry for the class. What stays
/// per-field is the field's own arithmetic, the lane width, and the module
/// docs. The overwrite fallback keeps the field's canonicalizing
/// `from_raw` loop rather than the portable prime body, so noncanonical
/// input lanes canonicalize exactly as before.
macro_rules! prime_dispatch {
    ($field:ident, $kernels:ident, $lane:ty $(,)?) => {
        impl $crate::field::dispatch::KernelDispatch for $field {
            /// The canonical lane word, broadcast by the kernels on entry.
            type Prepared = $crate::field::Elem<$field>;

            #[inline]
            fn backend(
                _proof: $crate::field::dispatch::RawDispatch,
            ) -> $crate::kernel::Backend {
                $crate::kernel::prime_strategy().report()
            }

            fallback_defaults!(
                row: add_gather_offsets, mul_add_scatter_plan, mul_add_gather_plan,
                mul_into_gather, mul_into_gather_plan, mul_add_matrix_plan,
                mul_into_matrix, mul_into_matrix_plan, mul_add_matrix_at,
                mul_add_matrix_at_plan, mul_into_matrix_at, mul_into_matrix_at_plan,
            );

            #[inline]
            fn prepare(
                _proof: $crate::field::dispatch::RawDispatch,
                coeff: $crate::field::Elem<$field>,
            ) -> Self::Prepared {
                coeff
            }

            #[inline]
            fn prepared_coeff(
                _proof: $crate::field::dispatch::RawDispatch,
                prepared: &Self::Prepared,
            ) -> $crate::field::Elem<$field> {
                *prepared
            }

            fn add_assign(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                src: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::add_assign_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::add_assign_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::add_assign_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                }
                $crate::kernel::prime::add_assign::<$field>(dst, src);
            }

            #[inline]
            fn sub_assign(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                src: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::sub_assign_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::sub_assign_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::sub_assign_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                }
                $crate::kernel::prime::sub_assign::<$field>(dst, src);
            }

            #[inline]
            fn mul_add(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                coeff: &Self::Prepared,
                src: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::mul_add_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::mul_add_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::mul_add_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                }
                $crate::kernel::prime::mul_add::<$field>(dst, *coeff, src);
            }

            #[inline]
            fn mul_assign(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                coeff: &Self::Prepared,
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::mul_assign_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            coeff.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::mul_assign_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            coeff.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::mul_assign_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            coeff.to_raw(),
                        );
                        return;
                    }
                }
                $crate::kernel::prime::mul_assign::<$field>(dst, *coeff);
            }

            fn mul_into(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                coeff: &Self::Prepared,
                src: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::mul_into_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::mul_into_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::mul_into_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            coeff.to_raw(),
                            src,
                        );
                        return;
                    }
                }
                for (d, s) in dst
                    .chunks_exact_mut(core::mem::size_of::<$lane>())
                    .zip(src.chunks_exact(core::mem::size_of::<$lane>()))
                {
                    let v = *coeff
                        * <$crate::field::Elem<$field> as $crate::field::FieldElem>::from_raw(
                            <$lane>::from_le_bytes(s.try_into().unwrap()),
                        );
                    d.copy_from_slice(&v.to_raw().to_le_bytes());
                }
            }

            fn mul_add_scatter(
                _proof: $crate::field::dispatch::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                coeffs: &[$crate::field::Elem<$field>],
                src: &[u8],
            ) {
                $crate::kernel::prime::mul_add_scatter::<$field>(rows, row_len, coeffs, src);
            }

            fn mul_add_gather(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                coeffs: &[$crate::field::Elem<$field>],
                srcs: &[&[u8]],
            ) {
                $crate::kernel::prime::mul_add_gather::<$field>(dst, coeffs, srcs);
            }

            fn mul_add_matrix(
                _proof: $crate::field::dispatch::RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[$crate::field::Elem<$field>], &[u8])],
            ) {
                $crate::kernel::prime::mul_add_matrix::<$field>(rows, row_len, nrows, terms);
            }

            fn mul_elementwise(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                a: &[u8],
                b: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::mul_elementwise_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            a,
                            b,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::mul_elementwise_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            a,
                            b,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::mul_elementwise_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            a,
                            b,
                        );
                        return;
                    }
                }
                $crate::kernel::prime::mul_elementwise::<$field>(dst, a, b);
            }

            fn mul_elementwise_assign(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                src: &[u8],
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::mul_elementwise_assign_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::mul_elementwise_assign_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::mul_elementwise_assign_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            src,
                        );
                        return;
                    }
                }
                $crate::kernel::scalar::mul_elementwise_assign::<$field>(dst, src);
            }

            fn add_assign_scalar(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                value: &Self::Prepared,
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::add_assign_scalar_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::add_assign_scalar_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::add_assign_scalar_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                }
                $crate::kernel::prime::add_assign_scalar::<$field>(dst, *value);
            }

            fn sub_assign_scalar(
                _proof: $crate::field::dispatch::RawDispatch,
                dst: &mut [u8],
                value: &Self::Prepared,
            ) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let strategy = $crate::kernel::prime_strategy();
                    #[cfg(feature = "simd512")]
                    if strategy == $crate::kernel::PrimeStrategy::Avx512 {
                        $crate::kernel::x86::$kernels::sub_assign_scalar_avx512(
                            $crate::kernel::x86_v4_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Avx2 {
                        $crate::kernel::x86::$kernels::sub_assign_scalar_avx2(
                            $crate::kernel::x86_v3_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                    if strategy == $crate::kernel::PrimeStrategy::Sse42 {
                        $crate::kernel::x86::$kernels::sub_assign_scalar_sse42(
                            $crate::kernel::x86_v2_token(),
                            dst,
                            value.to_raw(),
                        );
                        return;
                    }
                }
                $crate::kernel::prime::sub_assign_scalar::<$field>(dst, *value);
            }
        }
    };
}
pub(crate) mod bytes;
pub(crate) mod extension;
pub(crate) mod gf1;
pub(crate) mod gf8;
pub(crate) mod goldilocks;
pub(crate) mod mersenne31;
pub(crate) mod prime;
pub(crate) mod quad_mersenne31;
pub(crate) mod residue;
pub(crate) mod routing;
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

/// Memoized [`Selection`] over [`FGF_TIERS`] plus the capability token and
/// the per-family kernel strategies for that selection. Dispatch therefore
/// resolves policy and materializes its proof once; every dispatch arm
/// below reads one of the cached strategies instead of re-deriving a tier.
#[cfg(feature = "simd")]
static BACKEND: std::sync::LazyLock<ResolvedBackend> =
    std::sync::LazyLock::new(ResolvedBackend::new);

/// The kernel class serving the shipped prime-lane fields' rows.
///
/// Resolved once per process; the dispatch arms for Mersenne31 and
/// Goldilocks match this instead of the process backend, because the two
/// fields share one tier mapping.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrimeStrategy {
    /// 64-byte AVX-512 lanes.
    // The discriminants stay sparse: a dense set lets the compiler lower
    // the dispatch matches to a jump table, and the dependent table load
    // sits on every row operation's hot path.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512 = 4,
    /// 32-byte AVX2 lanes.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2 = 2,
    /// 16-byte SSE4.2 lanes.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Sse42 = 6,
    /// The portable `prime` bodies.
    Scalar = 0,
}

impl PrimeStrategy {
    /// The process tier label this strategy reports.
    #[must_use]
    pub(crate) fn report(self) -> Backend {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => Backend::V4x,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => Backend::V3,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Sse42 => Backend::V2,
            Self::Scalar => Backend::Scalar,
        }
    }
}

/// The kernel class serving GF(2^8) rows, before each representation's own
/// const descriptor facts narrow it.
///
/// Resolved once per process; the GF(2^8) dispatch folds its
/// representation's facts into this value rather than rematching the
/// process backend per call.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gf8Strategy {
    /// 64-byte AVX-512 byte-field kernels.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512,
    /// GFNI byte-multiply kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Gfni,
    /// 32-byte nibble-shuffle kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2,
    /// 16-byte nibble-shuffle kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Ssse3,
    /// The portable nibble-table bodies.
    Scalar,
}

impl Gf8Strategy {
    /// The process tier label this strategy reports.
    #[must_use]
    pub(crate) fn report(self) -> Backend {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => Backend::V4x,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Gfni => Backend::V3GfniCrypto,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => Backend::V3,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Ssse3 => Backend::V2,
            Self::Scalar => Backend::Scalar,
        }
    }
}

/// The kernel class serving GF(2^16) tower rows, before each row's pinned
/// instance narrows it.
///
/// Resolved once per process; the tower dispatch matches this and each
/// prepared coefficient's variant, which carries the choice its
/// preparation made.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tower16Strategy {
    /// 64-byte AVX-512 tower kernels.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512,
    /// GFNI byte-multiply tower kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Gfni,
    /// 32-byte nibble-shuffle tower kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2,
    /// 16-byte nibble-shuffle tower kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Ssse3,
    /// The portable tower bodies.
    Scalar,
}

impl Tower16Strategy {
    /// The process tier label this strategy reports.
    #[must_use]
    pub(crate) fn report(self) -> Backend {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => Backend::V4x,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Gfni => Backend::V3GfniCrypto,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => Backend::V3,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Ssse3 => Backend::V2,
            Self::Scalar => Backend::Scalar,
        }
    }
}

/// The kernel class serving GF(2^32) and GF(2^64) tower rows, before each
/// row's pinned instance narrows it.
///
/// The GFNI and AVX2 classes serve distinct instances at each width; the
/// `Avx512` class is the GFNI class selected on a `V4x` host, reported
/// separately because the public backend query names the process tier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WideTowerStrategy {
    /// The GFNI kernels on a `V4x` host.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512,
    /// GFNI lane-multiply kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Gfni,
    /// The Fan-Paar AVX2 lane scales.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2,
    /// The portable tower bodies.
    Scalar,
}

/// The kernel class serving `QuadMersenne31` component rows.
///
/// The multiplies take the AVX-512 kernels on an `Avx512` host; the adds,
/// the broadcasts, and the remaining operations take the AVX2 kernels
/// there, so an `Avx512` arm names the host class rather than one kernel
/// width.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuadMersenneStrategy {
    /// The AVX-512 host class.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512,
    /// The AVX2 host class.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2,
    /// The portable bodies.
    Scalar,
}

impl QuadMersenneStrategy {
    /// The process tier label this strategy reports.
    #[must_use]
    pub(crate) fn report(self) -> Backend {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => Backend::V4x,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => Backend::V3,
            Self::Scalar => Backend::Scalar,
        }
    }
}

/// The kernel class serving a residue row's vectorizable operations,
/// before the call's geometry narrows it.
///
/// The routing selectors fold length, width, and operation floors into
/// this value; the arms they feed invoke the AVX-512 or AVX2 kernels or
/// fall to the portable bodies. The dispatch reads it wherever SIMD
/// compiles; the `internals` facade exposes it for routing reports.
#[cfg_attr(not(all(feature = "simd", target_arch = "x86_64")), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResidueStrategy {
    /// The AVX-512 single-row and delayed kernels.
    #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
    Avx512,
    /// The AVX2 single-row and delayed kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Avx2,
    /// The portable prime bodies and repeated single-row compositions.
    Scalar,
}

#[cfg_attr(not(all(feature = "simd", target_arch = "x86_64")), allow(dead_code))]
impl ResidueStrategy {
    /// The process tier label this strategy reports.
    #[must_use]
    pub fn report(self) -> Backend {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => Backend::V4x,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => Backend::V3,
            Self::Scalar => Backend::Scalar,
        }
    }

    /// The label routing reports print for this class.
    // The bench routing report reads it through `internals`; no in-crate
    // caller exists.
    #[allow(dead_code)]
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Self::Avx512 => "avx512",
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Self::Avx2 => "avx2",
            Self::Scalar => "scalar",
        }
    }
}

/// The cached prime-lane strategy for this process.
#[inline]
pub(crate) fn prime_strategy() -> PrimeStrategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.prime
    }
    #[cfg(not(feature = "simd"))]
    {
        PrimeStrategy::Scalar
    }
}

/// The cached GF(2^8) strategy for this process.
#[inline]
pub(crate) fn gf8_strategy() -> Gf8Strategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.gf8
    }
    #[cfg(not(feature = "simd"))]
    {
        Gf8Strategy::Scalar
    }
}

/// The cached GF(2^16) tower strategy for this process.
#[inline]
pub(crate) fn tower16_strategy() -> Tower16Strategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.tower16
    }
    #[cfg(not(feature = "simd"))]
    {
        Tower16Strategy::Scalar
    }
}

/// The cached GF(2^32)/GF(2^64) tower strategy for this process.
#[inline]
pub(crate) fn wide_tower_strategy() -> WideTowerStrategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.wide_tower
    }
    #[cfg(not(feature = "simd"))]
    {
        WideTowerStrategy::Scalar
    }
}

/// The cached `QuadMersenne31` strategy for this process.
#[inline]
pub(crate) fn quad_mersenne_strategy() -> QuadMersenneStrategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.quad_mersenne
    }
    #[cfg(not(feature = "simd"))]
    {
        QuadMersenneStrategy::Scalar
    }
}

/// The cached residue strategy for this process.
#[cfg_attr(not(all(feature = "simd", target_arch = "x86_64")), allow(dead_code))]
#[inline]
pub fn residue_strategy() -> ResidueStrategy {
    #[cfg(feature = "simd")]
    {
        BACKEND.residue
    }
    #[cfg(not(feature = "simd"))]
    {
        ResidueStrategy::Scalar
    }
}

#[cfg(feature = "simd")]
struct ResolvedBackend {
    backend: Backend,
    #[cfg(target_arch = "x86_64")]
    x86: X86Proof,
    prime: PrimeStrategy,
    gf8: Gf8Strategy,
    tower16: Tower16Strategy,
    wide_tower: WideTowerStrategy,
    quad_mersenne: QuadMersenneStrategy,
    residue: ResidueStrategy,
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
        let prime = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => PrimeStrategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto | Backend::V3 => PrimeStrategy::Avx2,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => PrimeStrategy::Sse42,
            _ => PrimeStrategy::Scalar,
        };
        let gf8 = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => Gf8Strategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => Gf8Strategy::Gfni,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => Gf8Strategy::Avx2,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => Gf8Strategy::Ssse3,
            _ => Gf8Strategy::Scalar,
        };
        let tower16 = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => Tower16Strategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => Tower16Strategy::Gfni,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => Tower16Strategy::Avx2,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V2 => Tower16Strategy::Ssse3,
            _ => Tower16Strategy::Scalar,
        };
        let wide_tower = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => WideTowerStrategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto => WideTowerStrategy::Gfni,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3 => WideTowerStrategy::Avx2,
            _ => WideTowerStrategy::Scalar,
        };
        let quad_mersenne = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => QuadMersenneStrategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto | Backend::V3 => QuadMersenneStrategy::Avx2,
            _ => QuadMersenneStrategy::Scalar,
        };
        let residue = match backend {
            #[cfg(all(feature = "simd512", target_arch = "x86_64"))]
            Backend::V4x => ResidueStrategy::Avx512,
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            Backend::V3GfniCrypto | Backend::V3 => ResidueStrategy::Avx2,
            _ => ResidueStrategy::Scalar,
        };
        Self {
            backend,
            #[cfg(target_arch = "x86_64")]
            x86,
            prime,
            gf8,
            tower16,
            wide_tower,
            quad_mersenne,
            residue,
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
/// A presentation with no vector strategy on the selected tier reports
/// [`Backend::Scalar`]; [`backend()`] keeps reporting `simdispatch`'s
/// process selection. Which presentations and
/// tiers have a vector strategy is not part of this contract and may change
/// within 4.x; results are identical on every backend.
#[inline]
#[must_use]
pub fn backend_for<F: FieldBuffer>() -> Backend {
    let () = crate::field::Validate::<F>::OK;
    F::backend(crate::field::dispatch::RawDispatch)
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
