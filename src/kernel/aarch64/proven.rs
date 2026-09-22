//! Token-proven public entry points for the AArch64 kernels (`internals`).
//!
//! The crate-private selected kernels run after backend selection has
//! proved the CPU features; these wrappers take a genuine
//! [`archmage`] capability token plus full geometry validation, so an
//! `internals` consumer never reaches a `#[target_feature]` body unchecked.
//! NEON is baseline on AArch64 — the [`NeonToken`] still names the contract
//! explicitly — while `elementwise_pmull` needs the crypto extension
//! ([`NeonAesToken`]).
//!
//! Tokens come from [`archmage`]; summon with
//! [`SimdToken::summon`](archmage::SimdToken::summon).
//!
//! [`archmage`]: https://docs.rs/archmage
//! [`NeonToken`]: archmage::NeonToken
//! [`NeonAesToken`]: archmage::NeonAesToken

use crate::kernel::proven_checks::{check_elem_multiple, check_equal, check_row_span};

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

/// GF(2^16) tower kernels over interleaved element pairs.
pub mod gf16 {
    use super::super::gf16 as selected;
    use super::*;
    use crate::gf16::Elem;
    use crate::kernel::tables::TowerTables;

    /// `dst += coeff * src` over interleaved GF(2^16) elements.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_neon(
        _proof: archmage::NeonToken,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_add_neon", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_add_neon", dst.len(), 2);
        selected::mul_add_neon(dst, tables, src);
    }

    /// `dst *= coeff` over interleaved GF(2^16) elements.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_neon(_proof: archmage::NeonToken, dst: &mut [u8], tables: &TowerTables) {
        check_elem_multiple("gf16::mul_assign_neon", dst.len(), 2);
        selected::mul_assign_neon(dst, tables);
    }

    /// `dst = coeff * src`, fused single pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_neon(
        _proof: archmage::NeonToken,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_into_neon", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_into_neon", dst.len(), 2);
        selected::mul_into_neon(dst, tables, src);
    }

    /// One source into many rows, four rows per source load.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes (whole elements) and `row_len == src.len()`.
    pub fn scatter_neon(
        _proof: archmage::NeonToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem],
        src: &[u8],
    ) {
        check_equal("gf16::scatter_neon", "row_len", row_len, "src", src.len());
        check_elem_multiple("gf16::scatter_neon", row_len, 2);
        check_row_span("gf16::scatter_neon", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_neon(rows, row_len, coeffs, src);
    }

    /// Many sources into one row.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length (whole elements).
    pub fn gather_neon(
        _proof: archmage::NeonToken,
        dst: &mut [u8],
        coeffs: &[Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf16::gather_neon",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("gf16::gather_neon", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf16::gather_neon",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_neon(dst, coeffs, srcs);
    }

    /// Many sources into many rows.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
    /// (whole elements), every term supplies `nrows` coefficients, and every
    /// source is `row_len` bytes.
    pub fn matrix_neon(
        _proof: archmage::NeonToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem], &[u8])],
    ) {
        check_elem_multiple("gf16::matrix_neon", row_len, 2);
        check_row_span("gf16::matrix_neon", rows.len(), row_len, nrows);
        terms_geometry("gf16::matrix_neon", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_neon(rows, row_len, nrows, terms);
    }

    /// `dst[i] = a[i] * b[i]` over interleaved tower elements.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn elementwise_neon(_proof: archmage::NeonToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
        check_equal("gf16::elementwise_neon", "dst", dst.len(), "a", a.len());
        check_equal("gf16::elementwise_neon", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gf16::elementwise_neon", dst.len(), 2);
        selected::elementwise_neon(dst, a, b);
    }
}

/// Flat term geometry shared by the matrix wrappers: every term supplies
/// `nrows` coefficients and a `row_len`-byte source.
fn terms_geometry<E>(name: &str, row_len: usize, nrows: usize, terms: &[(&[E], &[u8])]) {
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
