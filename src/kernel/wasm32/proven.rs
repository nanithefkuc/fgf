//! Token-proven public entry points for the Wasm `simd128` kernels
//! (`internals`).
//!
//! On wasm32, `+simd128` is a compile-time capability choice. The genuine
//! [`Wasm128Token`](archmage::Wasm128Token) names that capability;
//! each public wrapper also validates geometry before entering its kernel.
//!
//! [`archmage`]: https://docs.rs/archmage
use crate::kernel::proven_checks::{check_elem_multiple, check_equal, check_row_span};

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

/// GF(2^8) kernels over the shared nibble tables.
pub mod gf8 {
    use super::super::gf8 as selected;
    use super::*;
    use crate::gf8b::Elem;
    use crate::kernel::tables::ScaleTable;

    /// `dst += coeff * src` over 16-byte SIMD lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_add_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        table: &ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_add_simd128", "dst", dst.len(), "src", src.len());
        selected::mul_add_simd128(dst, table, src);
    }

    /// `dst *= coeff` over 16-byte SIMD lanes.
    pub fn mul_assign_simd128(_proof: archmage::Wasm128Token, dst: &mut [u8], table: &ScaleTable) {
        selected::mul_assign_simd128(dst, table);
    }

    /// `dst = coeff * src`, fused single pass over 16-byte SIMD lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_into_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        table: &ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_into_simd128", "dst", dst.len(), "src", src.len());
        selected::mul_into_simd128(dst, table, src);
    }

    /// `dst[i] = a[i] * b[i]` over 16-byte SIMD lanes.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length.
    pub fn elementwise_simd128(_proof: archmage::Wasm128Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
        check_equal("gf8::elementwise_simd128", "dst", dst.len(), "a", a.len());
        check_equal("gf8::elementwise_simd128", "dst", dst.len(), "b", b.len());
        selected::elementwise_simd128(dst, a, b);
    }

    /// One source into many rows.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes and `row_len == src.len()`.
    pub fn scatter_simd128(
        _proof: archmage::Wasm128Token,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem],
        src: &[u8],
    ) {
        check_equal("gf8::scatter_simd128", "row_len", row_len, "src", src.len());
        check_row_span("gf8::scatter_simd128", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_simd128(rows, row_len, coeffs, src);
    }

    /// Many sources into one row.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        coeffs: &[Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_simd128",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_simd128",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_simd128(dst, coeffs, srcs);
    }

    /// Many sources into many rows.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
    /// every term supplies `nrows` coefficients, and every source is
    /// `row_len` bytes.
    pub fn matrix_simd128(
        _proof: archmage::Wasm128Token,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem], &[u8])],
    ) {
        check_row_span("gf8::matrix_simd128", rows.len(), row_len, nrows);
        super::terms_geometry("gf8::matrix_simd128", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_simd128(rows, row_len, nrows, terms);
    }
}

/// GF(2^16) tower kernels over interleaved element pairs.
pub mod gf16 {
    use super::super::gf16 as selected;
    use super::*;
    use crate::gf16::Elem;
    use crate::kernel::tables::TowerTables;

    /// `dst += coeff * src` over interleaved tower elements.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_add_simd128", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_add_simd128", dst.len(), 2);
        selected::mul_add_simd128(dst, tables, src);
    }

    /// `dst *= coeff` over interleaved tower elements.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        tables: &TowerTables,
    ) {
        check_elem_multiple("gf16::mul_assign_simd128", dst.len(), 2);
        selected::mul_assign_simd128(dst, tables);
    }

    /// `dst = coeff * src`, fused single pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_into_simd128", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_into_simd128", dst.len(), 2);
        selected::mul_into_simd128(dst, tables, src);
    }

    /// `dst[i] = a[i] * b[i]` over interleaved tower elements.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn elementwise_simd128(_proof: archmage::Wasm128Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
        check_equal("gf16::elementwise_simd128", "dst", dst.len(), "a", a.len());
        check_equal("gf16::elementwise_simd128", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gf16::elementwise_simd128", dst.len(), 2);
        selected::elementwise_simd128(dst, a, b);
    }

    /// One source into many rows.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes (whole elements) and `row_len == src.len()`.
    pub fn scatter_simd128(
        _proof: archmage::Wasm128Token,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem],
        src: &[u8],
    ) {
        check_equal(
            "gf16::scatter_simd128",
            "row_len",
            row_len,
            "src",
            src.len(),
        );
        check_elem_multiple("gf16::scatter_simd128", row_len, 2);
        check_row_span("gf16::scatter_simd128", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_simd128(rows, row_len, coeffs, src);
    }

    /// Many sources into one row.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length (whole elements).
    pub fn gather_simd128(
        _proof: archmage::Wasm128Token,
        dst: &mut [u8],
        coeffs: &[Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf16::gather_simd128",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("gf16::gather_simd128", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf16::gather_simd128",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_simd128(dst, coeffs, srcs);
    }

    /// Many sources into many rows.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
    /// (whole elements), every term supplies `nrows` coefficients, and every
    /// source is `row_len` bytes.
    pub fn matrix_simd128(
        _proof: archmage::Wasm128Token,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem], &[u8])],
    ) {
        check_elem_multiple("gf16::matrix_simd128", row_len, 2);
        check_row_span("gf16::matrix_simd128", rows.len(), row_len, nrows);
        super::terms_geometry("gf16::matrix_simd128", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_simd128(rows, row_len, nrows, terms);
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
