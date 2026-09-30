//! Field-independent byte kernels for x86 / `x86_64`.
//!
//! XOR is the additive operation of every binary field in the crate, so it
//! reads no field layout and needs no coefficient; the population count
//! behind bit-packed GF(2) weights is the other byte operation. Each file is
//! one instruction set: `avx512` under `simd512`, `avx2`, and `sse2`. Every
//! entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics.

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;
mod sse2;

pub use avx2::{xor_avx2, xor_broadcast_avx2, xor_gather_avx2};
#[cfg(feature = "simd512")]
pub(crate) use avx512::XOR_AVX512_MIN;
#[cfg(all(test, feature = "simd512"))]
pub(crate) use avx512::XOR_PEEL_MIN;
#[cfg(feature = "simd512")]
pub use avx512::{count_ones_avx512, xor_avx512};
pub use sse2::{xor_broadcast_sse2, xor_sse2};
