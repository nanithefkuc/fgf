//! GF(2^64 − 2^32 + 1) kernel dispatch.
//!
//! Prime-field arithmetic over 64-bit lanes. On x86 with AVX-512 (`V4x`,
//! under `simd512`) or AVX2 (`V3`) the multiply uses 32-bit-half `vpmuludq`
//! splitting and the Goldilocks split-fold (`2^64 ≡ 2^32 − 1`); SSE4.1
//! (`V2`) runs the same construction at half width. Every other target runs
//! the portable [`crate::kernel::prime`] path, so the backend query
//! reports `Scalar` there. The prepared form is the canonical lane word,
//! broadcast by the kernels on entry.
//!
//! The contract implementation is the shared prime-lane generator in
//! [`crate::kernel`], which matches the cached prime strategy once per call
//! and names this module's kernels; the field's own arithmetic, lane width,
//! and fallback loops stay here.

use crate::field::goldilocks::Goldilocks;

// The shared generator's fallback arms resolve `Elem` at this expansion
// site.
use crate::field::Elem;

prime_dispatch!(Goldilocks, goldilocks, u64);
