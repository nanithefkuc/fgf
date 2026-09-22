//! Token-proven public entry points for the AArch64 byte kernels
//! (`internals`).
//!
//! The crate-private selected kernel runs after backend selection has proved
//! the CPU features; this wrapper takes a genuine [`archmage`] capability
//! token plus full geometry validation, so an `internals` consumer never
//! reaches a `#[target_feature]` body unchecked. NEON is baseline on AArch64
//! — the [`NeonToken`] still names the contract explicitly.
//!
//! Tokens come from [`archmage`]; summon with
//! [`SimdToken::summon`](archmage::SimdToken::summon).
//!
//! [`archmage`]: https://docs.rs/archmage
//! [`NeonToken`]: archmage::NeonToken

use crate::kernel::proven_checks::check_equal;

/// Field-independent byte kernels (`aarch64/mod.rs`).
pub mod bytes {
    use super::*;

    /// `dst ^= src` over NEON lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn xor_neon(_proof: archmage::NeonToken, dst: &mut [u8], src: &[u8]) {
        check_equal("xor_neon", "dst", dst.len(), "src", src.len());
        super::super::xor_neon(dst, src);
    }

    /// `dst ^= src` over rows, interleaved NEON streams.
    ///
    /// # Panics
    /// Panics if the slices differ in length or their length is not a whole
    /// number of `row_len`-byte rows.
    pub fn xor_rows_neon(_proof: archmage::NeonToken, dst: &mut [u8], src: &[u8], row_len: usize) {
        check_equal("xor_rows_neon", "dst", dst.len(), "src", src.len());
        assert_ne!(row_len, 0, "xor_rows_neon: row length must be nonzero");
        assert_eq!(
            dst.len() % row_len,
            0,
            "xor_rows_neon: partial trailing row"
        );
        super::super::xor_rows_neon(dst, src, row_len);
    }
}

