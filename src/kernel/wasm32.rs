//! WebAssembly `simd128` kernels.
//!
//! Wasm's lane-local `i8x16.swizzle` has the same 16-entry lookup shape as
//! SSSE3 `PSHUFB` and NEON `TBL`, so the existing nibble tables are reused
//! without conversion. Every entry is a safe [`archmage`] capability-token
//! function taking [`archmage::Wasm128Token`]; inner loops are `#[rite]`
//! helpers inside the token-proven region.
//!
//! [`archmage`]: https://docs.rs/archmage

pub mod bytes;
pub mod gf16;
pub mod gf8;

pub(crate) use bytes::xor_simd128;
