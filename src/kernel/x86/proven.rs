//! Token-proven public entry points for the x86 kernels (`internals`).
//!
//! The kernel functions one module up are crate-private *selected*
//! entrypoints: they run after the process-wide backend selection has already
//! proved the CPU features, so they carry only `debug_assert`s. This module
//! is the public `internals` surface: the same kernels behind a genuine
//! [`archmage`] capability token (the proof the selected dispatch would have
//! established) plus the full runtime geometry validation the
//! [`crate::ops`] facade performs — so an `internals` consumer can never
//! reach a target-feature body unchecked, on any host, in any build profile.
//!
//! Summon tokens with [`SimdToken::summon`](crate::kernel::SimdToken) and
//! skip the tier when it returns `None`:
//!
//! ```no_run
//! use fgf::kernel::{SimdToken, X64V3GfniCryptoToken, x86};
//!
//! let src = [0u8; 16];
//! let mut dst = [0u8; 16];
//! if let Some(token) = X64V3GfniCryptoToken::summon() {
//!     x86::proven::gf8::mul_add_gfni(token, &mut dst, fgf::gf8b::Elem::from_raw(7), &src);
//! }
//! ```
//!
//! Tokens prove the ISA only. Geometry (lengths, element boundaries, row
//! spans, scattered-row disjointness) is validated here, before dispatch and
//! outside the kernel loops; a violation panics exactly like the facade's.
//!
//! [`archmage`]: https://docs.rs/archmage

use crate::kernel::proven_checks::{
    check_elem_multiple, check_equal, check_row_span, check_scattered,
};

/// Field-independent byte kernels (`x86/mod.rs`), proven per tier.
pub mod bytes {
    use super::*;

    /// `dst ^= src` over 32-byte AVX2 lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn xor_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], src: &[u8]) {
        check_equal("xor_avx2", "dst", dst.len(), "src", src.len());
        super::super::xor_avx2(dst, src);
    }

    /// `dst ^= sum(srcs[i])` over byte-offset sources, AVX2 registers.
    ///
    /// # Panics
    /// Panics if any offset plus `dst.len()` exceeds the region length.
    pub fn xor_gather_avx2(
        _proof: crate::kernel::X64V3Token,
        region: &[u8],
        dst: &mut [u8],
        offsets: &[u32],
    ) {
        let live = dst.len();
        for (index, &start) in offsets.iter().enumerate() {
            let start = start as usize;
            assert!(
                start + live <= region.len(),
                "xor_gather_avx2: offset {index} ({start}) + {live} exceeds region of {} bytes",
                region.len(),
            );
        }
        super::super::xor_gather_avx2(region, dst, offsets);
    }

    /// `dst ^= src` over 16-byte SSE2 lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn xor_sse2(_proof: crate::kernel::X64V2Token, dst: &mut [u8], src: &[u8]) {
        check_equal("xor_sse2", "dst", dst.len(), "src", src.len());
        super::super::xor_sse2(dst, src);
    }

    /// `dst ^= src` over rows, four interleaved AVX2 streams.
    ///
    /// # Panics
    /// Panics if the slices differ in length or their length is not a whole
    /// number of `row_len`-byte rows.
    pub fn xor_rows_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        src: &[u8],
        row_len: usize,
    ) {
        check_equal("xor_rows_avx2", "dst", dst.len(), "src", src.len());
        assert_ne!(row_len, 0, "xor_rows_avx2: row length must be nonzero");
        assert_eq!(
            dst.len() % row_len,
            0,
            "xor_rows_avx2: partial trailing row"
        );
        super::super::xor_rows_avx2(dst, src, row_len);
    }

    /// `dst ^= src` over rows, four interleaved SSE2 streams.
    ///
    /// # Panics
    /// Panics if the slices differ in length or their length is not a whole
    /// number of `row_len`-byte rows.
    pub fn xor_rows_sse2(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        src: &[u8],
        row_len: usize,
    ) {
        check_equal("xor_rows_sse2", "dst", dst.len(), "src", src.len());
        assert_ne!(row_len, 0, "xor_rows_sse2: row length must be nonzero");
        assert_eq!(
            dst.len() % row_len,
            0,
            "xor_rows_sse2: partial trailing row"
        );
        super::super::xor_rows_sse2(dst, src, row_len);
    }
}

/// GF(2^8) kernels (both `0x11B` and the `0x11D` affine path).
pub mod gf8 {
    use super::super::gf8 as selected;
    use super::*;

    fn prepared_geometry(
        name: &str,
        buffer_len: usize,
        row_len: usize,
        nrows: usize,
        prepared_len: usize,
        srcs: &[&[u8]],
    ) {
        check_row_span(name, buffer_len, row_len, nrows);
        let expected = nrows
            .checked_mul(srcs.len())
            .expect("prepared matrix geometry overflows");
        check_equal(
            name,
            "coefficients",
            prepared_len,
            "required coefficients",
            expected,
        );
        for &src in srcs {
            check_equal(name, "source", src.len(), "row_len", row_len);
        }
    }

    /// Accumulate a source-major matrix of prepared `0x11D` affine coefficients.
    ///
    /// # Panics
    /// Panics on overflowing dimensions, insufficient output storage, a
    /// coefficient-count mismatch, or a source length different from `row_len`.
    pub fn matrix_affine_prepared_with(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        prepared: &[crate::kernel::gf8::Prepared8D],
        srcs: &[&[u8]],
    ) {
        prepared_geometry(
            "matrix_affine_prepared_with",
            rows.len(),
            row_len,
            nrows,
            prepared.len(),
            srcs,
        );
        if nrows == 0 || row_len == 0 {
            return;
        }
        selected::matrix_affine_prepared_with(rows, row_len, nrows, prepared, srcs);
    }

    /// Overwrite rows with a source-major matrix of prepared `0x11D` coefficients.
    ///
    /// # Panics
    /// Panics on overflowing dimensions, insufficient output storage, a
    /// coefficient-count mismatch, or a source length different from `row_len`.
    pub fn matrix_overwrite_affine_prepared_with(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        prepared: &[crate::kernel::gf8::Prepared8D],
        srcs: &[&[u8]],
    ) {
        prepared_geometry(
            "matrix_overwrite_affine_prepared_with",
            rows.len(),
            row_len,
            nrows,
            prepared.len(),
            srcs,
        );
        if nrows == 0 || row_len == 0 {
            return;
        }
        selected::matrix_overwrite_affine_prepared_with(rows, row_len, nrows, prepared, srcs);
    }

    /// `dst += coeff * src` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_add_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: crate::gf8b::Elem,
        src: &[u8],
    ) {
        check_equal("gf8::mul_add_gfni", "dst", dst.len(), "src", src.len());
        selected::mul_add_gfni(dst, coeff, src);
    }

    /// `dst *= coeff` with `GF2P8MULB` over 32-byte lanes.
    pub fn mul_assign_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: crate::gf8b::Elem,
    ) {
        selected::mul_assign_gfni(dst, coeff);
    }

    /// `dst = coeff * src` with `GF2P8MULB`, one write pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_into_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: crate::gf8b::Elem,
        src: &[u8],
    ) {
        check_equal("gf8::mul_into_gfni", "dst", dst.len(), "src", src.len());
        selected::mul_into_gfni(dst, coeff, src);
    }

    /// `dst += coeff * src` through `VGF2P8AFFINEQB` (any polynomial).
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_add_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        map: u64,
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_add_affine", "dst", dst.len(), "src", src.len());
        selected::mul_add_affine(dst, map, table, src);
    }

    /// `dst *= coeff` through `VGF2P8AFFINEQB`.
    pub fn mul_assign_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        map: u64,
        table: &crate::kernel::tables::ScaleTable,
    ) {
        selected::mul_assign_affine(dst, map, table);
    }

    /// `dst = coeff * src` through `VGF2P8AFFINEQB`, one write pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_into_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        map: u64,
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_into_affine", "dst", dst.len(), "src", src.len());
        selected::mul_into_affine(dst, map, table, src);
    }

    /// `dst += coeff * src` by nibble shuffle over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_add_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_add_avx2", "dst", dst.len(), "src", src.len());
        selected::mul_add_avx2(dst, table, src);
    }

    /// `dst *= coeff` by nibble shuffle over 32-byte lanes.
    pub fn mul_assign_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
    ) {
        selected::mul_assign_avx2(dst, table);
    }

    /// `dst = coeff * src` by nibble shuffle over 32-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_into_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_into_avx2", "dst", dst.len(), "src", src.len());
        selected::mul_into_avx2(dst, table, src);
    }

    /// `dst += coeff * src` by nibble shuffle over 16-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_add_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_add_ssse3", "dst", dst.len(), "src", src.len());
        selected::mul_add_ssse3(dst, table, src);
    }

    /// `dst *= coeff` by nibble shuffle over 16-byte lanes.
    pub fn mul_assign_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
    ) {
        selected::mul_assign_ssse3(dst, table);
    }

    /// `dst = coeff * src` by nibble shuffle over 16-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn mul_into_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        table: &crate::kernel::tables::ScaleTable,
        src: &[u8],
    ) {
        check_equal("gf8::mul_into_ssse3", "dst", dst.len(), "src", src.len());
        selected::mul_into_ssse3(dst, table, src);
    }

    /// One source into many rows, GFNI, sharing loads across a row group.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes and `row_len == src.len()`.
    pub fn scatter_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[crate::gf8b::Elem],
        src: &[u8],
    ) {
        check_equal("gf8::scatter_gfni", "row_len", row_len, "src", src.len());
        check_row_span("gf8::scatter_gfni", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_gfni(rows, row_len, coeffs, src);
    }

    /// One source into many rows through the affine map (`0x11D`).
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes and `row_len == src.len()`.
    pub fn scatter_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[crate::gf8d::Elem],
        src: &[u8],
    ) {
        check_equal("gf8::scatter_affine", "row_len", row_len, "src", src.len());
        check_row_span("gf8::scatter_affine", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_affine(rows, row_len, coeffs, src);
    }

    /// Many sources into one row, GFNI, destination held in registers.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[crate::gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_gfni",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_gfni",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_gfni(dst, coeffs, srcs);
    }

    /// Benchmark candidate: repeated single-source GFNI AXPY with the fused
    /// sub-tile tail — the control the blocked gathers are measured against.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_gfni_axpy_tail(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[crate::gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_gfni_axpy_tail",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_gfni_axpy_tail",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_gfni_axpy_tail(dst, coeffs, srcs);
    }

    /// Benchmark candidate: blocked GFNI gather over `TILE_LANES` 32-byte
    /// accumulators.
    ///
    /// # Panics
    /// Panics unless `TILE_LANES` is 1–4, `coeffs.len() == srcs.len()`, and
    /// every source matches `dst` in length.
    pub fn gather_gfni_tile<const TILE_LANES: usize>(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[crate::gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        assert!(
            matches!(TILE_LANES, 1..=4),
            "gf8::gather_gfni_tile: TILE_LANES must be 1-4"
        );
        check_equal(
            "gf8::gather_gfni_tile",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_gfni_tile",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_gfni_tile::<TILE_LANES>(dst, coeffs, srcs);
    }

    /// Benchmark candidate: split-accumulator GFNI gather over 128-byte
    /// tiles.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_gfni_split(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[crate::gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_gfni_split",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_gfni_split",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_gfni_split(dst, coeffs, srcs);
    }

    /// Many sources into one row through the affine map (`0x11D`), blocked
    /// over 128-byte tiles with the destination held in registers.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[crate::gf8d::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_affine",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_affine",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_affine(dst, coeffs, srcs);
    }

    /// Benchmark candidate: gather over already-prepared affine factors
    /// (one [`Affine8BFactor`] per source).
    ///
    /// # Panics
    /// Panics unless `factors.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gather_affine_8b(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        factors: &[super::Affine8BFactor],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf8::gather_affine_8b",
            "factors",
            factors.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf8::gather_affine_8b",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_affine_8b(dst, factors, srcs);
    }

    /// Many sources into many rows, GFNI, destination tiles in registers.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
    /// every term supplies `nrows` coefficients, and every source is
    /// `row_len` bytes.
    pub fn matrix_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::gf8b::Elem], &[u8])],
    ) {
        check_row_span("gf8::matrix_gfni", rows.len(), row_len, nrows);
        for (t, &(coeffs, src)) in terms.iter().enumerate() {
            check_equal(
                "gf8::matrix_gfni",
                format_args!("term {t} coefficients"),
                coeffs.len(),
                "rows",
                nrows,
            );
            check_equal(
                "gf8::matrix_gfni",
                format_args!("term {t} source"),
                src.len(),
                "row_len",
                row_len,
            );
        }
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_gfni(rows, row_len, nrows, terms);
    }

    /// [`matrix_gfni`] over a generic [`Matrix`](crate::kernel::Matrix)
    /// coefficient/source provider.
    ///
    /// `Matrix` is a safe, externally implementable trait, and the kernel
    /// walks the provider repeatedly; a provider whose `len`/`source`
    /// answers change between this call's validation and the kernel's
    /// iteration can break the raw-pointer geometry the blocked kernel
    /// relies on. The token proves the ISA only, so the provider invariant
    /// is the caller's to uphold.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that, for the whole call, `terms.len()` is
    /// stable and every `terms.source(t < len())` returns the same
    /// `row_len`-byte slice, with coefficient geometry consistent with the
    /// provider's own contract. The crate's
    /// [`FlatMatrix`](crate::kernel::FlatMatrix) and slice-of-tuples
    /// providers satisfy this by construction.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
    /// (checked even for a well-behaved provider).
    pub unsafe fn matrix_gfni_with<M: crate::kernel::Matrix<crate::gf8b::Elem> + ?Sized>(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &M,
    ) {
        check_row_span("gf8::matrix_gfni_with", rows.len(), row_len, nrows);
        if row_len == 0 || nrows == 0 || terms.len() == 0 {
            return;
        }
        selected::matrix_gfni_with(rows, row_len, nrows, terms);
    }

    /// [`matrix_gfni`] under the `0x11D` polynomial, via the affine map.
    ///
    /// # Panics
    /// As [`matrix_gfni`].
    pub fn matrix_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
    ) {
        check_row_span("gf8::matrix_affine", rows.len(), row_len, nrows);
        for (t, &(coeffs, src)) in terms.iter().enumerate() {
            check_equal(
                "gf8::matrix_affine",
                format_args!("term {t} coefficients"),
                coeffs.len(),
                "rows",
                nrows,
            );
            check_equal(
                "gf8::matrix_affine",
                format_args!("term {t} source"),
                src.len(),
                "row_len",
                row_len,
            );
        }
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_affine(rows, row_len, nrows, terms);
    }

    /// [`matrix_affine`] over a generic [`Matrix`](crate::kernel::Matrix)
    /// provider.
    ///
    /// # Safety
    ///
    /// As [`matrix_gfni_with`]: the provider must return stable geometry for
    /// the whole call. The token proves the ISA, not the provider.
    ///
    /// # Panics
    /// As [`matrix_gfni_with`].
    pub unsafe fn matrix_affine_with<M: crate::kernel::Matrix<crate::gf8d::Elem> + ?Sized>(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &M,
    ) {
        check_row_span("gf8::matrix_affine_with", rows.len(), row_len, nrows);
        if row_len == 0 || nrows == 0 || terms.len() == 0 {
            return;
        }
        selected::matrix_affine_with(rows, row_len, nrows, terms);
    }

    /// Overwrite counterpart of [`matrix_gfni`]: seeds from zero, writes once.
    ///
    /// # Panics
    /// As [`matrix_gfni`].
    pub fn matrix_overwrite_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::gf8b::Elem], &[u8])],
    ) {
        check_row_span("gf8::matrix_overwrite_gfni", rows.len(), row_len, nrows);
        scattered_terms("gf8::matrix_overwrite_gfni", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 {
            return;
        }
        selected::matrix_overwrite_gfni(rows, row_len, nrows, terms);
    }

    /// [`matrix_overwrite_gfni`] over a generic
    /// [`Matrix`](crate::kernel::Matrix) provider.
    ///
    /// # Safety
    ///
    /// As [`matrix_gfni_with`]: the provider must return stable geometry for
    /// the whole call. The token proves the ISA, not the provider.
    ///
    /// # Panics
    /// As [`matrix_gfni_with`].
    pub unsafe fn matrix_overwrite_gfni_with<
        M: crate::kernel::Matrix<crate::gf8b::Elem> + ?Sized,
    >(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &M,
    ) {
        check_row_span(
            "gf8::matrix_overwrite_gfni_with",
            rows.len(),
            row_len,
            nrows,
        );
        // An empty term list still writes: the overwrite sum is zero over
        // the addressed rows, exactly like the slice twin (whose tile loop
        // stores zeroed accumulators with no terms).
        if row_len == 0 || nrows == 0 {
            return;
        }
        selected::matrix_overwrite_gfni_with(rows, row_len, nrows, terms);
    }

    /// Overwrite counterpart of [`matrix_affine`].
    ///
    /// # Panics
    /// As [`matrix_affine`].
    pub fn matrix_overwrite_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
    ) {
        check_row_span("gf8::matrix_overwrite_affine", rows.len(), row_len, nrows);
        scattered_terms("gf8::matrix_overwrite_affine", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 {
            return;
        }
        selected::matrix_overwrite_affine(rows, row_len, nrows, terms);
    }

    /// [`matrix_overwrite_affine`] over a generic
    /// [`Matrix`](crate::kernel::Matrix) provider.
    ///
    /// # Safety
    ///
    /// As [`matrix_gfni_with`]: the provider must return stable geometry for
    /// the whole call. The token proves the ISA, not the provider.
    ///
    /// # Panics
    /// As [`matrix_gfni_with`].
    pub unsafe fn matrix_overwrite_affine_with<
        M: crate::kernel::Matrix<crate::gf8d::Elem> + ?Sized,
    >(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &M,
    ) {
        check_row_span(
            "gf8::matrix_overwrite_affine_with",
            rows.len(),
            row_len,
            nrows,
        );
        // As `matrix_overwrite_gfni_with`: empty terms zero the addressed
        // rows instead of returning early.
        if row_len == 0 || nrows == 0 {
            return;
        }
        selected::matrix_overwrite_affine_with(rows, row_len, nrows, terms);
    }

    /// Many sources into disjoint scattered rows, GFNI.
    ///
    /// # Panics
    /// Panics unless every row is in-bounds and pairwise disjoint, and every
    /// term supplies `row_starts.len()` coefficients and `row_len`-byte
    /// sources.
    pub fn matrix_scattered_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[crate::gf8b::Elem], &[u8])],
    ) {
        check_scattered(
            "gf8::matrix_scattered_gfni",
            dst.len(),
            row_len,
            row_starts,
            1,
        );
        super::scattered_terms(
            "gf8::matrix_scattered_gfni",
            row_len,
            row_starts.len(),
            terms,
        );
        if row_len == 0 || row_starts.is_empty() || terms.is_empty() {
            return;
        }
        selected::matrix_scattered_gfni(dst, row_len, row_starts, terms);
    }

    /// [`matrix_scattered_gfni`] under `0x11D`, via the affine map.
    ///
    /// # Panics
    /// As [`matrix_scattered_gfni`].
    pub fn matrix_scattered_affine(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[crate::gf8d::Elem], &[u8])],
    ) {
        check_scattered(
            "gf8::matrix_scattered_affine",
            dst.len(),
            row_len,
            row_starts,
            1,
        );
        super::scattered_terms(
            "gf8::matrix_scattered_affine",
            row_len,
            row_starts.len(),
            terms,
        );
        if row_len == 0 || row_starts.is_empty() || terms.is_empty() {
            return;
        }
        selected::matrix_scattered_affine(dst, row_len, row_starts, terms);
    }

    /// `dst[i] = a[i] * b[i]` with `GF2P8MULB`.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length.
    pub fn elementwise_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gf8::elementwise_gfni", "dst", dst.len(), "a", a.len());
        check_equal("gf8::elementwise_gfni", "dst", dst.len(), "b", b.len());
        selected::elementwise_gfni(dst, a, b);
    }

    /// `dst[i] = a[i] * b[i]` by branchless shift/reduce, AVX2 width.
    ///
    /// `RED` selects the reduction byte (`0x1b` for `0x11B`, `0x1d` for
    /// `0x11D`).
    ///
    /// # Panics
    /// Panics unless all three buffers match in length.
    pub fn elementwise_avx2<const RED: u8>(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gf8::elementwise_avx2", "dst", dst.len(), "a", a.len());
        check_equal("gf8::elementwise_avx2", "dst", dst.len(), "b", b.len());
        selected::elementwise_avx2::<RED>(dst, a, b);
    }

    /// `dst[i] = a[i] * b[i]` by branchless shift/reduce, SSE width.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length.
    pub fn elementwise_ssse3<const RED: u8>(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gf8::elementwise_ssse3", "dst", dst.len(), "a", a.len());
        check_equal("gf8::elementwise_ssse3", "dst", dst.len(), "b", b.len());
        selected::elementwise_ssse3::<RED>(dst, a, b);
    }

    // --- Experimental two-row overwrite shapes (perf_p2 candidates). All
    // --- forward to entries that already carry full runtime asserts.

    /// Experimental: two-row overwrite matrix, GFNI (`0x11B`).
    ///
    /// `nt` selects non-temporal stores once `rows` is large enough; `lanes`
    /// the unroll (1 or 2).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite2_8b(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8b::Elem], &[u8])],
        nt: bool,
        lanes: usize,
    ) {
        selected::matrix_overwrite2_8b(rows, row_len, terms, nt, lanes);
    }

    /// Experimental: two-row overwrite matrix, affine (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite2_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
        nt: bool,
        lanes: usize,
    ) {
        selected::matrix_overwrite2_8d(rows, row_len, terms, nt, lanes);
    }

    /// Experimental: six-row overwrite matrix, shuffle kernels (`0x11B`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite6_shuffle_8b(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8b::Elem], &[u8])],
    ) {
        selected::matrix_overwrite6_shuffle_8b(rows, row_len, terms);
    }

    /// Experimental: six-row overwrite matrix, shuffle kernels (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite6_shuffle_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
    ) {
        selected::matrix_overwrite6_shuffle_8d(rows, row_len, terms);
    }

    /// Experimental: six-row overwrite over pre-derived 32-byte table rows
    /// (`0x11B`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite6_shuffle_packed_8b(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        tables: &[[u8; 32]],
        srcs: &[&[u8]],
    ) {
        selected::matrix_overwrite6_shuffle_packed_8b(rows, row_len, tables, srcs);
    }

    /// Experimental: six-row overwrite over pre-derived 32-byte table rows
    /// (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite6_shuffle_packed_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        tables: &[[u8; 32]],
        srcs: &[&[u8]],
    ) {
        selected::matrix_overwrite6_shuffle_packed_8d(rows, row_len, tables, srcs);
    }

    /// Experimental: two-row overwrite over pre-derived 32-byte table rows.
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite2_shuffle_packed(
        _proof: crate::kernel::X64V3Token,
        rows: &mut [u8],
        row_len: usize,
        tables: &[[u8; 32]],
        srcs: &[&[u8]],
        lanes: usize,
    ) {
        selected::matrix_overwrite2_shuffle_packed(rows, row_len, tables, srcs, lanes);
    }

    /// Experimental: single-row overwrite, one coefficient per term (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite1_tile_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
        lanes: usize,
    ) {
        selected::matrix_overwrite1_tile_8d(rows, row_len, terms, lanes);
    }

    /// Experimental: single-row overwrite from pre-derived affine maps.
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite1_external_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        maps: &[u64],
        srcs: &[&[u8]],
        lanes: usize,
    ) {
        selected::matrix_overwrite1_external_8d(rows, row_len, maps, srcs, lanes);
    }

    /// Experimental: single-row overwrite, table init per term (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite1_fullinit_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
        lanes: usize,
    ) {
        selected::matrix_overwrite1_fullinit_8d(rows, row_len, terms, lanes);
    }

    /// Experimental: single-row overwrite, chunked table init (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite1_chunk_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
        lanes: usize,
        chunk: usize,
    ) {
        selected::matrix_overwrite1_chunk_8d(rows, row_len, terms, lanes, chunk);
    }

    /// Experimental: single-row overwrite, chunked external maps.
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite1_external_chunk_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        maps: &[u64],
        srcs: &[&[u8]],
        lanes: usize,
        chunk: usize,
    ) {
        selected::matrix_overwrite1_external_chunk_8d(rows, row_len, maps, srcs, lanes, chunk);
    }

    /// Experimental: multi-row overwrite, chunked (`0x11D`).
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite_chunk_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[crate::gf8d::Elem], &[u8])],
        chunk: usize,
    ) {
        selected::matrix_overwrite_chunk_8d(rows, row_len, nrows, terms, chunk);
    }

    /// Experimental: multi-row overwrite from grouped external maps.
    ///
    /// # Panics
    /// As the selected kernel's own geometry asserts.
    pub fn matrix_overwrite_external_grouped_8d(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        maps: &[u64],
        srcs: &[&[u8]],
        chunk: usize,
    ) {
        selected::matrix_overwrite_external_grouped_8d(rows, row_len, nrows, maps, srcs, chunk);
    }
}

/// Term geometry for the scattered wrappers: every term supplies one
/// coefficient per row and a `row_len`-byte source.
fn scattered_terms<E>(name: &str, row_len: usize, nrows: usize, terms: &[(&[E], &[u8])]) {
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

/// GF(2^16) tower kernels over interleaved element pairs.
pub mod gf16 {
    use super::super::gf16 as selected;
    use super::*;
    use crate::gf16;
    use crate::kernel::tables::{TowerCoeff, TowerTables};

    /// `dst += coeff * src` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: TowerCoeff,
        src: &[u8],
    ) {
        check_equal("gf16::mul_add_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_add_gfni", dst.len(), 2);
        selected::mul_add_gfni(dst, coeff, src);
    }

    /// `dst *= coeff` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: TowerCoeff,
    ) {
        check_elem_multiple("gf16::mul_assign_gfni", dst.len(), 2);
        selected::mul_assign_gfni(dst, coeff);
    }

    /// `dst = coeff * src` with `GF2P8MULB`, one write pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: TowerCoeff,
        src: &[u8],
    ) {
        check_equal("gf16::mul_into_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_into_gfni", dst.len(), 2);
        selected::mul_into_gfni(dst, coeff, src);
    }

    /// `dst += coeff * src` by nibble shuffle over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_add_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_add_avx2", dst.len(), 2);
        selected::mul_add_avx2(dst, tables, src);
    }

    /// `dst *= coeff` by nibble shuffle over 32-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &TowerTables,
    ) {
        check_elem_multiple("gf16::mul_assign_avx2", dst.len(), 2);
        selected::mul_assign_avx2(dst, tables);
    }

    /// `dst = coeff * src` by nibble shuffle over 32-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_into_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_into_avx2", dst.len(), 2);
        selected::mul_into_avx2(dst, tables, src);
    }

    /// `dst += coeff * src` by nibble shuffle over 16-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_add_ssse3", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_add_ssse3", dst.len(), 2);
        selected::mul_add_ssse3(dst, tables, src);
    }

    /// `dst *= coeff` by nibble shuffle over 16-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &TowerTables,
    ) {
        check_elem_multiple("gf16::mul_assign_ssse3", dst.len(), 2);
        selected::mul_assign_ssse3(dst, tables);
    }

    /// `dst = coeff * src` by nibble shuffle over 16-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &TowerTables,
        src: &[u8],
    ) {
        check_equal("gf16::mul_into_ssse3", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf16::mul_into_ssse3", dst.len(), 2);
        selected::mul_into_ssse3(dst, tables, src);
    }

    /// One source into many rows, GFNI.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
    /// bytes and `row_len == src.len()` (whole elements).
    pub fn scatter_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[gf16::Elem],
        src: &[u8],
    ) {
        check_equal("gf16::scatter_gfni", "row_len", row_len, "src", src.len());
        check_elem_multiple("gf16::scatter_gfni", row_len, 2);
        check_row_span("gf16::scatter_gfni", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_gfni(rows, row_len, coeffs, src);
    }

    /// One source into many rows by nibble shuffle over 32-byte lanes.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// As [`scatter_gfni`].
    pub fn scatter_avx2<C: TableCoefficient>(
        _proof: crate::kernel::X64V3Token,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[C],
        src: &[u8],
    ) {
        check_equal("gf16::scatter_avx2", "row_len", row_len, "src", src.len());
        check_elem_multiple("gf16::scatter_avx2", row_len, 2);
        check_row_span("gf16::scatter_avx2", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_avx2(rows, row_len, coeffs, src);
    }

    /// One source into many rows by nibble shuffle over 16-byte lanes.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// As [`scatter_gfni`].
    pub fn scatter_ssse3<C: TableCoefficient>(
        _proof: crate::kernel::X64V2Token,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[C],
        src: &[u8],
    ) {
        check_equal("gf16::scatter_ssse3", "row_len", row_len, "src", src.len());
        check_elem_multiple("gf16::scatter_ssse3", row_len, 2);
        check_row_span("gf16::scatter_ssse3", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::scatter_ssse3(rows, row_len, coeffs, src);
    }

    /// Many sources into one row, GFNI.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length (whole elements).
    pub fn gather_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeffs: &[gf16::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf16::gather_gfni",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("gf16::gather_gfni", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf16::gather_gfni",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_gfni(dst, coeffs, srcs);
    }

    /// Many sources into one row by nibble shuffle over 32-byte lanes.
    ///
    /// # Panics
    /// As [`gather_gfni`].
    pub fn gather_avx2<C: TableCoefficient>(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeffs: &[C],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf16::gather_avx2",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("gf16::gather_avx2", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf16::gather_avx2",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_avx2(dst, coeffs, srcs);
    }

    /// Many sources into one row by nibble shuffle over 16-byte lanes.
    ///
    /// # Panics
    /// As [`gather_gfni`].
    pub fn gather_ssse3<C: TableCoefficient>(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        coeffs: &[C],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "gf16::gather_ssse3",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("gf16::gather_ssse3", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "gf16::gather_ssse3",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gather_ssse3(dst, coeffs, srcs);
    }

    /// Many sources into many rows, GFNI.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
    /// every term supplies `nrows` coefficients, and every source is
    /// `row_len` bytes.
    pub fn matrix_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf16::Elem], &[u8])],
    ) {
        check_elem_multiple("gf16::matrix_gfni", row_len, 2);
        check_row_span("gf16::matrix_gfni", rows.len(), row_len, nrows);
        super::scattered_free_terms("gf16::matrix_gfni", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_gfni(rows, row_len, nrows, terms);
    }

    /// Many sources into many rows by nibble shuffle over 32-byte lanes.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// As [`matrix_gfni`].
    pub fn matrix_avx2(
        _proof: crate::kernel::X64V3Token,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf16::Elem], &[u8])],
    ) {
        check_elem_multiple("gf16::matrix_avx2", row_len, 2);
        check_row_span("gf16::matrix_avx2", rows.len(), row_len, nrows);
        super::scattered_free_terms("gf16::matrix_avx2", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_avx2(rows, row_len, nrows, terms);
    }

    /// Many sources into many rows by nibble shuffle over 16-byte lanes.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// As [`matrix_gfni`].
    pub fn matrix_ssse3(
        _proof: crate::kernel::X64V2Token,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf16::Elem], &[u8])],
    ) {
        check_elem_multiple("gf16::matrix_ssse3", row_len, 2);
        check_row_span("gf16::matrix_ssse3", rows.len(), row_len, nrows);
        super::scattered_free_terms("gf16::matrix_ssse3", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::matrix_ssse3(rows, row_len, nrows, terms);
    }

    /// `dst[i] = a[i] * b[i]` with `GF2P8MULB`.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn elementwise_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gf16::elementwise_gfni", "dst", dst.len(), "a", a.len());
        check_equal("gf16::elementwise_gfni", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gf16::elementwise_gfni", dst.len(), 2);
        selected::elementwise_gfni(dst, a, b);
    }

    /// `dst[i] = a[i] * b[i]` by branchless shift/reduce, AVX2 width.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn elementwise_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
        check_equal("gf16::elementwise_avx2", "dst", dst.len(), "a", a.len());
        check_equal("gf16::elementwise_avx2", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gf16::elementwise_avx2", dst.len(), 2);
        selected::elementwise_avx2(dst, a, b);
    }

    /// `dst[i] = a[i] * b[i]` by branchless shift/reduce, SSE width.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn elementwise_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gf16::elementwise_ssse3", "dst", dst.len(), "a", a.len());
        check_equal("gf16::elementwise_ssse3", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gf16::elementwise_ssse3", dst.len(), 2);
        selected::elementwise_ssse3(dst, a, b);
    }
}

/// GF(2^32) tower kernels over GFNI.
pub mod gf32 {
    use super::super::gf32 as selected;
    use super::*;
    use crate::gf32;

    /// `dst += coeff * src` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf32::Elem,
        tiles: [u32; 4],
        src: &[u8],
    ) {
        check_equal("gf32::mul_add_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf32::mul_add_gfni", dst.len(), 4);
        selected::mul_add_gfni(dst, coeff, tiles, src);
    }

    /// `dst *= coeff` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf32::Elem,
        tiles: [u32; 4],
    ) {
        check_elem_multiple("gf32::mul_assign_gfni", dst.len(), 4);
        selected::mul_assign_gfni(dst, coeff, tiles);
    }

    /// `dst = coeff * src` with `GF2P8MULB`, one write pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf32::Elem,
        tiles: [u32; 4],
        src: &[u8],
    ) {
        check_equal("gf32::mul_into_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf32::mul_into_gfni", dst.len(), 4);
        selected::mul_into_gfni(dst, coeff, tiles, src);
    }
}

/// GF(2^64) tower kernels over GFNI.
pub mod gf64 {
    use super::super::gf64 as selected;
    use super::*;
    use crate::gf64;

    /// `dst += coeff * src` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf64::Elem,
        tiles: [u64; 8],
        src: &[u8],
    ) {
        check_equal("gf64::mul_add_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf64::mul_add_gfni", dst.len(), 8);
        selected::mul_add_gfni(dst, coeff, tiles, src);
    }

    /// `dst *= coeff` with `GF2P8MULB` over 32-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf64::Elem,
        tiles: [u64; 8],
    ) {
        check_elem_multiple("gf64::mul_assign_gfni", dst.len(), 8);
        selected::mul_assign_gfni(dst, coeff, tiles);
    }

    /// `dst = coeff * src` with `GF2P8MULB`, one write pass.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_gfni(
        _proof: crate::kernel::X64V3GfniCryptoToken,
        dst: &mut [u8],
        coeff: gf64::Elem,
        tiles: [u64; 8],
        src: &[u8],
    ) {
        check_equal("gf64::mul_into_gfni", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gf64::mul_into_gfni", dst.len(), 8);
        selected::mul_into_gfni(dst, coeff, tiles, src);
    }
}

/// Canonical Fan–Paar tower kernels.
pub mod fan_paar {
    use super::super::fan_paar as selected;
    use super::*;
    use crate::fan_paar::{fp32, fp64};
    use crate::kernel::tables::FpTowerTables;

    /// `dst += coeff * src` with `PSHUFB` lookups over 32-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
        src: &[u8],
    ) {
        check_equal("fan_paar::mul_add_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("fan_paar::mul_add_avx2", dst.len(), 2);
        selected::mul_add_avx2(dst, tables, src);
    }

    /// `dst *= coeff` with `PSHUFB` lookups over 32-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
    ) {
        check_elem_multiple("fan_paar::mul_assign_avx2", dst.len(), 2);
        selected::mul_assign_avx2(dst, tables);
    }

    /// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_into_avx2",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_into_avx2", dst.len(), 2);
        selected::mul_into_avx2(dst, tables, src);
    }

    /// `dst += coeff * src` with `PSHUFB` lookups over 16-byte lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_add_ssse3",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_add_ssse3", dst.len(), 2);
        selected::mul_add_ssse3(dst, tables, src);
    }

    /// `dst *= coeff` with `PSHUFB` lookups over 16-byte lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
    ) {
        check_elem_multiple("fan_paar::mul_assign_ssse3", dst.len(), 2);
        selected::mul_assign_ssse3(dst, tables);
    }

    /// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes, fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_ssse3(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        tables: &FpTowerTables,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_into_ssse3",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_into_ssse3", dst.len(), 2);
        selected::mul_into_ssse3(dst, tables, src);
    }

    /// `dst += coeff * src` (Fan–Paar GF(2^32), AVX2).
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_fp32_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp32::Elem,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_add_fp32_avx2",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_add_fp32_avx2", dst.len(), 4);
        selected::mul_add_fp32_avx2(dst, coeff, src);
    }

    /// `dst *= coeff` (Fan–Paar GF(2^32), AVX2).
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_fp32_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp32::Elem,
    ) {
        check_elem_multiple("fan_paar::mul_assign_fp32_avx2", dst.len(), 4);
        selected::mul_assign_fp32_avx2(dst, coeff);
    }

    /// `dst = coeff * src` (Fan–Paar GF(2^32), AVX2), fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_fp32_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp32::Elem,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_into_fp32_avx2",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_into_fp32_avx2", dst.len(), 4);
        selected::mul_into_fp32_avx2(dst, coeff, src);
    }

    /// `dst += coeff * src` (Fan–Paar GF(2^64), AVX2).
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_add_fp64_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp64::Elem,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_add_fp64_avx2",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_add_fp64_avx2", dst.len(), 8);
        selected::mul_add_fp64_avx2(dst, coeff, src);
    }

    /// `dst *= coeff` (Fan–Paar GF(2^64), AVX2).
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn mul_assign_fp64_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp64::Elem,
    ) {
        check_elem_multiple("fan_paar::mul_assign_fp64_avx2", dst.len(), 8);
        selected::mul_assign_fp64_avx2(dst, coeff);
    }

    /// `dst = coeff * src` (Fan–Paar GF(2^64), AVX2), fused.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn mul_into_fp64_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: fp64::Elem,
        src: &[u8],
    ) {
        check_equal(
            "fan_paar::mul_into_fp64_avx2",
            "dst",
            dst.len(),
            "src",
            src.len(),
        );
        check_elem_multiple("fan_paar::mul_into_fp64_avx2", dst.len(), 8);
        selected::mul_into_fp64_avx2(dst, coeff, src);
    }
}

/// Prime-field integer-SIMD kernels (Mersenne31 and Goldilocks).
///
/// These kernels compute field-correct results for **canonical** lanes and
/// coefficients; arbitrary raw bit patterns stay memory-safe but compute
/// unspecified field values (see the `ops` module docs).
pub mod prime {
    use super::super::prime as selected;
    use super::*;

    /// `dst += src (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn add_assign_m31_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], src: &[u8]) {
        check_equal("m31::add_assign_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::add_assign_avx2", dst.len(), 4);
        selected::add_assign_m31_avx2(dst, src);
    }

    /// `dst -= src (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn sub_assign_m31_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], src: &[u8]) {
        check_equal("m31::sub_assign_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::sub_assign_avx2", dst.len(), 4);
        selected::sub_assign_m31_avx2(dst, src);
    }

    /// `dst += coeff * src (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_add_m31_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: u32,
        src: &[u8],
    ) {
        check_equal("m31::mul_add_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::mul_add_avx2", dst.len(), 4);
        selected::mul_add_m31_avx2(dst, coeff, src);
    }

    /// `dst *= coeff (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics on a partial trailing lane.
    pub fn mul_assign_m31_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], coeff: u32) {
        check_elem_multiple("m31::mul_assign_avx2", dst.len(), 4);
        selected::mul_assign_m31_avx2(dst, coeff);
    }

    /// `dst = coeff * src (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_into_m31_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: u32,
        src: &[u8],
    ) {
        check_equal("m31::mul_into_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::mul_into_avx2", dst.len(), 4);
        selected::mul_into_m31_avx2(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, AVX2.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole lanes).
    pub fn mul_elementwise_m31_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("m31::elementwise_avx2", "dst", dst.len(), "a", a.len());
        check_equal("m31::elementwise_avx2", "dst", dst.len(), "b", b.len());
        check_elem_multiple("m31::elementwise_avx2", dst.len(), 4);
        selected::mul_elementwise_m31_avx2(dst, a, b);
    }

    /// `dst += src (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn add_assign_m31_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], src: &[u8]) {
        check_equal("m31::add_assign_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::add_assign_sse41", dst.len(), 4);
        selected::add_assign_m31_sse41(dst, src);
    }

    /// `dst -= src (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn sub_assign_m31_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], src: &[u8]) {
        check_equal("m31::sub_assign_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::sub_assign_sse41", dst.len(), 4);
        selected::sub_assign_m31_sse41(dst, src);
    }

    /// `dst += coeff * src (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_add_m31_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        coeff: u32,
        src: &[u8],
    ) {
        check_equal("m31::mul_add_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::mul_add_sse41", dst.len(), 4);
        selected::mul_add_m31_sse41(dst, coeff, src);
    }

    /// `dst *= coeff (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics on a partial trailing lane.
    pub fn mul_assign_m31_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], coeff: u32) {
        check_elem_multiple("m31::mul_assign_sse41", dst.len(), 4);
        selected::mul_assign_m31_sse41(dst, coeff);
    }

    /// `dst = coeff * src (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_into_m31_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        coeff: u32,
        src: &[u8],
    ) {
        check_equal("m31::mul_into_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("m31::mul_into_sse41", dst.len(), 4);
        selected::mul_into_m31_sse41(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i] (mod p)`, Mersenne31, SSE4.2.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole lanes).
    pub fn mul_elementwise_m31_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("m31::elementwise_sse41", "dst", dst.len(), "a", a.len());
        check_equal("m31::elementwise_sse41", "dst", dst.len(), "b", b.len());
        check_elem_multiple("m31::elementwise_sse41", dst.len(), 4);
        selected::mul_elementwise_m31_sse41(dst, a, b);
    }

    /// `dst += src (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn add_assign_gld_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], src: &[u8]) {
        check_equal("gld::add_assign_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::add_assign_avx2", dst.len(), 8);
        selected::add_assign_gld_avx2(dst, src);
    }

    /// `dst -= src (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn sub_assign_gld_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], src: &[u8]) {
        check_equal("gld::sub_assign_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::sub_assign_avx2", dst.len(), 8);
        selected::sub_assign_gld_avx2(dst, src);
    }

    /// `dst += coeff * src (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_add_gld_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: u64,
        src: &[u8],
    ) {
        check_equal("gld::mul_add_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::mul_add_avx2", dst.len(), 8);
        selected::mul_add_gld_avx2(dst, coeff, src);
    }

    /// `dst *= coeff (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics on a partial trailing lane.
    pub fn mul_assign_gld_avx2(_proof: crate::kernel::X64V3Token, dst: &mut [u8], coeff: u64) {
        check_elem_multiple("gld::mul_assign_avx2", dst.len(), 8);
        selected::mul_assign_gld_avx2(dst, coeff);
    }

    /// `dst = coeff * src (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_into_gld_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        coeff: u64,
        src: &[u8],
    ) {
        check_equal("gld::mul_into_avx2", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::mul_into_avx2", dst.len(), 8);
        selected::mul_into_gld_avx2(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, AVX2.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole lanes).
    pub fn mul_elementwise_gld_avx2(
        _proof: crate::kernel::X64V3Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gld::elementwise_avx2", "dst", dst.len(), "a", a.len());
        check_equal("gld::elementwise_avx2", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gld::elementwise_avx2", dst.len(), 8);
        selected::mul_elementwise_gld_avx2(dst, a, b);
    }

    /// `dst += src (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn add_assign_gld_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], src: &[u8]) {
        check_equal("gld::add_assign_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::add_assign_sse41", dst.len(), 8);
        selected::add_assign_gld_sse41(dst, src);
    }

    /// `dst -= src (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn sub_assign_gld_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], src: &[u8]) {
        check_equal("gld::sub_assign_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::sub_assign_sse41", dst.len(), 8);
        selected::sub_assign_gld_sse41(dst, src);
    }

    /// `dst += coeff * src (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_add_gld_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        coeff: u64,
        src: &[u8],
    ) {
        check_equal("gld::mul_add_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::mul_add_sse41", dst.len(), 8);
        selected::mul_add_gld_sse41(dst, coeff, src);
    }

    /// `dst *= coeff (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics on a partial trailing lane.
    pub fn mul_assign_gld_sse41(_proof: crate::kernel::X64V2Token, dst: &mut [u8], coeff: u64) {
        check_elem_multiple("gld::mul_assign_sse41", dst.len(), 8);
        selected::mul_assign_gld_sse41(dst, coeff);
    }

    /// `dst = coeff * src (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial lane.
    pub fn mul_into_gld_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        coeff: u64,
        src: &[u8],
    ) {
        check_equal("gld::mul_into_sse41", "dst", dst.len(), "src", src.len());
        check_elem_multiple("gld::mul_into_sse41", dst.len(), 8);
        selected::mul_into_gld_sse41(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i] (mod p)`, Goldilocks, SSE4.2.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole lanes).
    pub fn mul_elementwise_gld_sse41(
        _proof: crate::kernel::X64V2Token,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("gld::elementwise_sse41", "dst", dst.len(), "a", a.len());
        check_equal("gld::elementwise_sse41", "dst", dst.len(), "b", b.len());
        check_elem_multiple("gld::elementwise_sse41", dst.len(), 8);
        selected::mul_elementwise_gld_sse41(dst, a, b);
    }
}

/// Experimental 64-byte AVX-512 kernels.
///
/// The AVX-512 tier is deferred (cross-compile-only, not on the dispatch
/// ladder), and the archmage build here does not enable its `avx512`
/// intrinsics feature — the tokens type-check as proofs, but summoning them
/// on this build is not an executable-coverage claim. Multiply kernels take
/// [`X64V4xToken`](crate::kernel::X64V4xToken) because `X64V4Token` alone
/// does not prove GFNI; the AVX-512F-only `xor` takes
/// [`X64V4Token`](crate::kernel::X64V4Token).
pub mod avx512 {
    use super::super::avx512 as selected;
    use super::*;
    use crate::gf8b;
    use crate::gf16;
    use crate::kernel::tables::TowerCoeff;

    /// `dst ^= src` over 64-byte AVX-512 lanes (AVX-512F only).
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn xor(_proof: crate::kernel::X64V4Token, dst: &mut [u8], src: &[u8]) {
        check_equal("avx512::xor", "dst", dst.len(), "src", src.len());
        selected::xor(dst, src);
    }

    /// `dst += coeff * src` over 64-byte GFNI lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn gf8_mul_add(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeff: gf8b::Elem,
        src: &[u8],
    ) {
        check_equal("avx512::gf8_mul_add", "dst", dst.len(), "src", src.len());
        selected::gf8_mul_add(dst, coeff, src);
    }

    /// `dst *= coeff` over 64-byte GFNI lanes.
    pub fn gf8_mul_assign(_proof: crate::kernel::X64V4xToken, dst: &mut [u8], coeff: gf8b::Elem) {
        selected::gf8_mul_assign(dst, coeff);
    }

    /// `dst = coeff * src` over 64-byte GFNI lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length.
    pub fn gf8_mul_into(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeff: gf8b::Elem,
        src: &[u8],
    ) {
        check_equal("avx512::gf8_mul_into", "dst", dst.len(), "src", src.len());
        selected::gf8_mul_into(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i]` over 64-byte GFNI lanes.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length.
    pub fn gf8_elementwise(_proof: crate::kernel::X64V4xToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
        check_equal("avx512::gf8_elementwise", "dst", dst.len(), "a", a.len());
        check_equal("avx512::gf8_elementwise", "dst", dst.len(), "b", b.len());
        selected::gf8_elementwise(dst, a, b);
    }

    /// `dst += coeff * src` over 64-byte tower-field lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn gf16_mul_add(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeff: TowerCoeff,
        src: &[u8],
    ) {
        check_equal("avx512::gf16_mul_add", "dst", dst.len(), "src", src.len());
        check_elem_multiple("avx512::gf16_mul_add", dst.len(), 2);
        selected::gf16_mul_add(dst, coeff, src);
    }

    /// `dst *= coeff` over 64-byte tower-field lanes.
    ///
    /// # Panics
    /// Panics on a partial trailing element.
    pub fn gf16_mul_assign(_proof: crate::kernel::X64V4xToken, dst: &mut [u8], coeff: TowerCoeff) {
        check_elem_multiple("avx512::gf16_mul_assign", dst.len(), 2);
        selected::gf16_mul_assign(dst, coeff);
    }

    /// `dst = coeff * src` over 64-byte tower-field lanes.
    ///
    /// # Panics
    /// Panics if the slices differ in length or hold a partial element.
    pub fn gf16_mul_into(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeff: TowerCoeff,
        src: &[u8],
    ) {
        check_equal("avx512::gf16_mul_into", "dst", dst.len(), "src", src.len());
        check_elem_multiple("avx512::gf16_mul_into", dst.len(), 2);
        selected::gf16_mul_into(dst, coeff, src);
    }

    /// `dst[i] = a[i] * b[i]` over interleaved tower elements.
    ///
    /// # Panics
    /// Panics unless all three buffers match in length (whole elements).
    pub fn gf16_elementwise(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        a: &[u8],
        b: &[u8],
    ) {
        check_equal("avx512::gf16_elementwise", "dst", dst.len(), "a", a.len());
        check_equal("avx512::gf16_elementwise", "dst", dst.len(), "b", b.len());
        check_elem_multiple("avx512::gf16_elementwise", dst.len(), 2);
        selected::gf16_elementwise(dst, a, b);
    }

    /// One GF(2^8) source into many rows, eight rows per source load.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds exactly `coeffs.len()` rows of `row_len`
    /// bytes and `row_len == src.len()` (the kernel's own assert).
    pub fn gf8_scatter(
        _proof: crate::kernel::X64V4xToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[gf8b::Elem],
        src: &[u8],
    ) {
        check_equal("avx512::gf8_scatter", "row_len", row_len, "src", src.len());
        check_row_span("avx512::gf8_scatter", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::gf8_scatter(rows, row_len, coeffs, src);
    }

    /// Many GF(2^8) sources into one destination.
    ///
    /// # Panics
    /// Panics unless `coeffs.len() == srcs.len()` and every source matches
    /// `dst` in length.
    pub fn gf8_gather(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "avx512::gf8_gather",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "avx512::gf8_gather",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gf8_gather(dst, coeffs, srcs);
    }

    /// Many sources into many GF(2^8) rows.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
    /// and every term supplies `nrows` coefficients and a `row_len`-byte
    /// source.
    pub fn gf8_matrix(
        _proof: crate::kernel::X64V4xToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf8b::Elem], &[u8])],
    ) {
        check_row_span("avx512::gf8_matrix", rows.len(), row_len, nrows);
        super::scattered_free_terms("avx512::gf8_matrix", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::gf8_matrix(rows, row_len, nrows, terms);
    }

    /// One tower-field source into many rows, eight rows per source load.
    ///
    /// A zero-length row with a zero-length source is a no-op.
    ///
    /// # Panics
    /// As [`gf8_scatter`], with whole 2-byte elements.
    pub fn gf16_scatter(
        _proof: crate::kernel::X64V4xToken,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[gf16::Elem],
        src: &[u8],
    ) {
        check_equal("avx512::gf16_scatter", "row_len", row_len, "src", src.len());
        check_elem_multiple("avx512::gf16_scatter", row_len, 2);
        check_row_span("avx512::gf16_scatter", rows.len(), row_len, coeffs.len());
        if row_len == 0 || coeffs.is_empty() {
            return;
        }
        selected::gf16_scatter(rows, row_len, coeffs, src);
    }

    /// Many tower-field sources into one destination.
    ///
    /// # Panics
    /// As [`gf8_gather`], with whole 2-byte elements.
    pub fn gf16_gather(
        _proof: crate::kernel::X64V4xToken,
        dst: &mut [u8],
        coeffs: &[gf16::Elem],
        srcs: &[&[u8]],
    ) {
        check_equal(
            "avx512::gf16_gather",
            "coefficients",
            coeffs.len(),
            "sources",
            srcs.len(),
        );
        check_elem_multiple("avx512::gf16_gather", dst.len(), 2);
        for (index, &src) in srcs.iter().enumerate() {
            check_equal(
                "avx512::gf16_gather",
                "dst",
                dst.len(),
                format_args!("source {index}"),
                src.len(),
            );
        }
        if dst.is_empty() || srcs.is_empty() {
            return;
        }
        selected::gf16_gather(dst, coeffs, srcs);
    }

    /// Many sources into many tower-field rows.
    ///
    /// A zero-length row with zero-length sources is a no-op.
    ///
    /// # Panics
    /// As [`gf8_matrix`], with whole 2-byte elements.
    pub fn gf16_matrix(
        _proof: crate::kernel::X64V4xToken,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[gf16::Elem], &[u8])],
    ) {
        check_elem_multiple("avx512::gf16_matrix", row_len, 2);
        check_row_span("avx512::gf16_matrix", rows.len(), row_len, nrows);
        super::scattered_free_terms("avx512::gf16_matrix", row_len, nrows, terms);
        if row_len == 0 || nrows == 0 || terms.is_empty() {
            return;
        }
        selected::gf16_matrix(rows, row_len, nrows, terms);
    }
}

/// Flat term geometry shared by the matrix wrappers: every term supplies
/// `nrows` coefficients and a `row_len`-byte source.
fn scattered_free_terms<E>(name: &str, row_len: usize, nrows: usize, terms: &[(&[E], &[u8])]) {
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

/// Pure (CPU-independent) helpers from the crate-private kernel modules,
/// re-exported so `internals` consumers keep their paths.
pub use super::gf8::{Affine8BFactor, prepare_affine_8b, resolve_probe_8d};
pub use super::gf16::TableCoefficient;
pub use super::gf32::gf32_tiles;
pub use super::gf64::gf64_tiles;

// The unsafe subset of the proven surface (see the module docs): wrappers
// generic over an externally implementable `Matrix` provider cannot validate
// provider stability, so their contracts carry it. Everything else in this
// module is safe.
