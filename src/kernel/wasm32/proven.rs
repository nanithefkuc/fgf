//! Token-proven public entry points for the Wasm `simd128` kernels
//! (`internals`).
//!
//! On wasm32, `+simd128` is a compile-time capability choice. The genuine
//! [`Wasm128Token`](archmage::Wasm128Token) names that capability;
//! each public wrapper also validates geometry before entering its kernel.
//!
//! [`archmage`]: https://docs.rs/archmage
use crate::kernel::proven_checks::check_equal;

/// Field-independent byte kernels (`wasm32/mod.rs`).
pub mod bytes {
    use super::*;

    /// `dst ^= src` over `simd128` lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn xor_simd128(_proof: archmage::Wasm128Token, dst: &mut [u8], src: &[u8]) {
        check_equal("xor_simd128", "dst", dst.len(), "src", src.len());
        super::super::xor_simd128(dst, src);
    }
}

