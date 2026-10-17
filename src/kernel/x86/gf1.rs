//! Bit-packed GF(2) interop kernels for x86 / `x86_64`.
//!
//! The portable `crate::kernel::gf1` seam expands packed bits into target
//! lanes and compresses lanes back into bitmaps. These entries vectorize
//! that seam: `sse42` over 16-byte lanes for `V2`, `avx2` over 32-byte
//! lanes for `V3` (and `V3GfniCrypto`, which reuses the AVX2 bodies), and
//! `avx512` over 64-byte lanes for `V4x` under `simd512`.
//!
//! Every public entry is a safe [`archmage`] capability-token function taking
//! the exact token its instructions require and asserting its own buffer
//! geometry. Vector bodies cover full vector units; any remainder lane
//! falls back to the entry's scalar word loop, which owns the bit indexing.

mod avx2;
mod sse42;

#[cfg(feature = "simd512")]
mod avx512;

pub use avx2::{compress_avx2, expand_avx2, restrict_avx2};
#[cfg(feature = "simd512")]
pub use avx512::{compress_avx512, expand_avx512, restrict_avx512};
pub use sse42::{compress_sse42, expand_sse42, restrict_sse42};
