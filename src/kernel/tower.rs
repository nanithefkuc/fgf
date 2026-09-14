//! Kernel dispatch for the Rijndael-rooted quadratic tower.
//!
//! [`gf16`] carries hand-written SIMD kernels on every backend, because its
//! two-byte lane is what the byte-wide hardware multiply operates on
//! directly. The wider levels reach vector hardware only through the GFNI
//! tower identity on x86, and reduce to the portable scalar kernel
//! everywhere else; that shared shape is one macro, instantiated by [`gf32`]
//! and [`gf64`]. The trait defaults compose every multi-row and prepared
//! operation from the single-row multiply, so the GFNI win reaches
//! scatter/gather/matrix over a prepared [`crate::ops::CoeffVec`] without a
//! per-shape kernel.

/// Emit one GFNI-only tower dispatch level into the calling module.
///
/// Parameters: the marker identifier, its field module name, the `x86` tile
/// derivation function and the tile array type it returns, the field name,
/// and the sentence describing the tile set.
macro_rules! gfni_tower_dispatch {
    ($marker:ident, $module:ident, $tiles_fn:ident, $tiles:ty, $name:literal, $tiles_doc:literal) => {
        use crate::field::tower::$module::{Elem, $marker};
        #[allow(unused_imports)]
        use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        use crate::kernel::x86;

        #[doc = concat!("A ", $name, " coefficient resolved into the form this host's backend wants.")]
        ///
        #[doc = concat!("`Compact` carries ", $tiles_doc, ", derived once in coefficient")]
        /// preparation, so a prepared [`crate::ops::Coeff`] or
        /// [`crate::ops::CoeffVec`] reuses them across the whole buffer.
        /// `Plain` hands the element to the portable scalar kernel — the path
        /// every non-GFNI backend and the sub-lane tail take.
        #[derive(Clone, Debug)]
        pub enum Prepared {
            #[doc = concat!("GFNI: ", $tiles_doc, ", plus the element for the scalar tail.")]
            #[cfg(any(
                all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")),
                feature = "internals"
            ))]
            Compact {
                #[doc = concat!("The ", $name, " coefficient, for the portable tail.")]
                coeff: Elem,
                #[doc = concat!("The tiles from [`x86::", stringify!($module), "::", stringify!($tiles_fn), "`].")]
                tiles: $tiles,
            },
            /// No GFNI backend: the element itself, for the portable scalar
            /// kernel.
            Plain(Elem),
        }

        impl Prepared {
            /// The coefficient this was built from.
            #[inline]
            #[must_use]
            pub const fn coeff(&self) -> Elem {
                match self {
                    Self::Plain(coeff) => *coeff,
                    #[cfg(any(
                        all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")),
                        feature = "internals"
                    ))]
                    Self::Compact { coeff, .. } => *coeff,
                }
            }
        }

        impl FieldKernels for $marker {
            #[inline]
            fn backend() -> Backend {
                match backend() {
                    Backend::V3GfniCrypto => backend(),
                    _ => Backend::Scalar,
                }
            }

            #[inline]
            fn has_vector_elementwise() -> bool {
                false
            }
        }

        impl KernelDispatch for $marker {
            type Prepared = Prepared;

            fn prepare(_proof: RawDispatch, coeff: Elem) -> Prepared {
                match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V3GfniCrypto => Prepared::Compact {
                        coeff,
                        tiles: x86::$module::$tiles_fn(coeff),
                    },
                    _ => Prepared::Plain(coeff),
                }
            }

            #[inline]
            fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem {
                prepared.coeff()
            }

            #[inline]
            fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
                crate::kernel::xor(dst, src);
            }

            #[inline]
            fn add_gather_offsets(
                _proof: RawDispatch,
                region: &[u8],
                dst: &mut [u8],
                offsets: &[u32],
            ) {
                crate::kernel::xor_gather(region, dst, offsets);
            }

            #[inline]
            fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
                crate::kernel::xor(dst, src);
            }

            fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
                match coeff {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Prepared::Compact { coeff, tiles } => {
                        x86::$module::mul_add_gfni(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            *coeff,
                            *tiles,
                            src,
                        );
                    }
                    other => scalar::mul_add::<$marker>(dst, other.coeff(), src),
                }
            }

            fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared) {
                match coeff {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Prepared::Compact { coeff, tiles } => {
                        x86::$module::mul_assign_gfni(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            *coeff,
                            *tiles,
                        );
                    }
                    other => scalar::mul_assign::<$marker>(dst, other.coeff()),
                }
            }

            fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
                match coeff {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Prepared::Compact { coeff, tiles } => {
                        x86::$module::mul_into_gfni(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            *coeff,
                            *tiles,
                            src,
                        );
                    }
                    other => {
                        dst.copy_from_slice(src);
                        Self::mul_assign(RawDispatch, dst, other);
                    }
                }
            }

            fn mul_add_scatter(
                _proof: RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                coeffs: &[Elem],
                src: &[u8],
            ) {
                // The trait default `mul_add_scatter_with` would do the same
                // per-row `mul_add`, but it takes already-prepared
                // coefficients. The base path receives raw elements, so each
                // row prepares its own; the cost is one tile derivation per
                // row, amortized over `row_len`.
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                }
            }

            fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
                for (&coeff, &src) in coeffs.iter().zip(srcs) {
                    Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
                }
            }

            fn mul_add_matrix(
                _proof: RawDispatch,
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[Elem], &[u8])],
            ) {
                for &(coeffs, src) in terms {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                        Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                    }
                }
            }

            fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
                // Both operands vary per lane, so there is no fixed
                // coefficient to broadcast; the GFNI elementwise path is not
                // implemented.
                scalar::mul_elementwise::<$marker>(dst, a, b);
            }
        }
    };
}

pub mod gf16 {
    //! GF(2^16) kernel dispatch.
    //!
    //! Every backend here exploits the same tower identity: a 16-bit multiply is
    //! two byte-wide multiplies, one of the source and one of the source with
    //! adjacent bytes swapped, under the alternating coefficient pair in
    //! [`TowerCoeff`]. Hardware that can multiply bytes in the field — GFNI —
    //! needs no table; hardware that cannot emulates each of the four base-field
    //! factors with a nibble shuffle.

    use crate::field::gf16::{Elem, Gf16};
    use crate::kernel::tables::{ScaleTable, TowerCoeff, scale_table};
    // `Backend` and `TowerTables` are referenced only from the SIMD dispatch
    // arms, which cfg away entirely on a scalar-only build or on an architecture
    // without the corresponding backend.
    #[allow(unused_imports)]
    use crate::kernel::tables::TowerTables;
    #[allow(unused_imports)]
    use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    use crate::kernel::aarch64;
    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    use crate::kernel::wasm32;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::kernel::x86;

    /// A GF(2^16) coefficient resolved into the form this host's backend wants.
    ///
    /// The three variants are not interchangeable representations of the same
    /// cost. `Compact` is two base multiplies; `Tables` is four nibble tables and
    /// ~140 bytes to copy. Choosing between them at *preparation* time is the
    /// point: a GFNI host never pays for tables it will not read, and a shuffle
    /// host builds them once instead of on every call.
    #[derive(Clone, Debug)]
    pub enum Prepared {
        /// Native byte multiply (GFNI) or `PMULL`: a pair of broadcast words.
        Compact(TowerCoeff),
        /// Shuffle backends (AVX2, SSSE3, NEON): four nibble tables.
        Tables(TowerTables),
        /// No vector unit: the coefficient itself.
        Plain(Elem),
    }

    impl Prepared {
        /// The coefficient this was built from.
        #[inline]
        #[must_use]
        pub const fn coeff(&self) -> Elem {
            match self {
                Self::Compact(compact) => compact.coeff,
                Self::Tables(tables) => tables.coeff,
                Self::Plain(coeff) => *coeff,
            }
        }
    }

    /// The four base-field nibble tables for `coeff`, borrowed from the shared
    /// bank.
    ///
    /// Deliberately not a [`TowerTables`]: that copies 136 bytes of table onto the
    /// stack, and the tail handlers below run on as little as a single two-byte
    /// element. Borrowing costs the two base multiplies of [`TowerCoeff::new`]
    /// plus four rodata indexes, so the blocked backend kernels resolve their
    /// coefficients through here too rather than duplicating the derivation.
    #[inline]
    pub(crate) fn factor_tables(coeff: Elem) -> [&'static ScaleTable; 4] {
        let f = TowerCoeff::new(coeff).factors();
        [
            scale_table(f[0]),
            scale_table(f[1]),
            scale_table(f[2]),
            scale_table(f[3]),
        ]
    }

    /// `dst ^= coeff * src` over interleaved elements, one element at a time.
    ///
    /// The tail handler for the vector kernels.
    #[cfg(feature = "internals")]
    pub fn mul_add_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        mul_add_scalar_impl(dst, coeff, src);
    }

    #[cfg(not(feature = "internals"))]
    pub(crate) fn mul_add_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        mul_add_scalar_impl(dst, coeff, src);
    }

    /// Per element this is 8 nibble lookups and 7 XORs against the 3 base-field
    /// multiplies a Karatsuba [`Elem::mul`] performs — each of those a
    /// carry-less product plus reduction, and none of them shared between
    /// elements. Since `coeff` is fixed for the whole buffer its four base
    /// factors collapse into split-nibble tables once, so the loop body becomes
    /// pure loads.
    fn mul_add_scalar_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        if coeff == Elem::ZERO {
            return;
        }
        if coeff == Elem::ONE {
            scalar::xor(dst, src);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(coeff);
        for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
            let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
            d[0] ^= f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
            d[1] ^= f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
        }
    }

    /// `dst *= coeff` over interleaved elements, one element at a time.
    #[cfg(feature = "internals")]
    pub fn mul_assign_scalar(dst: &mut [u8], coeff: Elem) {
        mul_assign_scalar_impl(dst, coeff);
    }

    #[cfg(not(feature = "internals"))]
    pub(crate) fn mul_assign_scalar(dst: &mut [u8], coeff: Elem) {
        mul_assign_scalar_impl(dst, coeff);
    }

    /// Table-driven for the same reason as [`mul_add_scalar_impl`], reading the
    /// destination as its own source.
    fn mul_assign_scalar_impl(dst: &mut [u8], coeff: Elem) {
        if coeff == Elem::ONE {
            return;
        }
        if coeff == Elem::ZERO {
            dst.fill(0);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(coeff);
        for d in dst.chunks_exact_mut(2) {
            let (a_lo, a_hi) = ((d[0] & 0x0f) as usize, (d[0] >> 4) as usize);
            let (b_lo, b_hi) = ((d[1] & 0x0f) as usize, (d[1] >> 4) as usize);
            d[0] = f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
            d[1] = f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
        }
    }

    /// `dst = coeff * src` over interleaved elements, one element at a time.
    ///
    /// Tail handler for the fused out-of-place vector kernels, mirroring
    /// [`mul_add_scalar`] without the destination read.
    #[cfg(feature = "internals")]
    pub fn mul_into_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        mul_into_scalar_impl(dst, coeff, src);
    }

    #[cfg(not(feature = "internals"))]
    #[allow(dead_code)]
    pub(crate) fn mul_into_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        mul_into_scalar_impl(dst, coeff, src);
    }

    /// Table-driven for the same reason as [`mul_add_scalar_impl`], storing the
    /// product instead of accumulating it.
    fn mul_into_scalar_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        if coeff == Elem::ZERO {
            dst.fill(0);
            return;
        }
        if coeff == Elem::ONE {
            dst.copy_from_slice(src);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(coeff);
        for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
            let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
            d[0] = f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
            d[1] = f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
        }
    }

    /// Repeated AXPY, used where blocking a GF(2^16) gather does not pay.
    ///
    /// AVX2 has enough width but not enough registers to retain several
    /// four-table coefficient sets, and a gather has nothing to share in any
    /// case: coefficients are one-to-one with sources, so a source's nibble split
    /// feeds exactly one coefficient. It measured at parity or behind repeated
    /// AXPY, so AXPY stays wired. SSSE3's smaller table vectors do block
    /// profitably and are wired to the blocked kernel. See "Crossover and
    /// dispatch decisions" in BENCHMARKS.md.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn gather_avx2_axpy(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            x86::gf16::mul_add_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                &TowerTables::new(coeff),
                src,
            );
        }
    }

    /// Repeated AXPY, used where blocking a GF(2^16) matrix does not pay.
    ///
    /// The blocked AVX2 matrix folds every term into a register-resident
    /// destination tile, which should beat re-reading the destination per term.
    /// It does not, quite: it wins on short rows and sits at parity or slightly
    /// behind on long ones, which is not enough to justify a row-length branch in
    /// dispatch (BENCHMARKS.md). Revisit if the tile ever widens past two
    /// accumulators.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn matrix_avx2_axpy(rows: &mut [u8], row_len: usize, terms: &[(&[Elem], &[u8])]) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                x86::gf16::mul_add_avx2(
                    crate::kernel::x86_v3_token(),
                    row,
                    &TowerTables::new(coeff),
                    src,
                );
            }
        }
    }

    impl FieldKernels for Gf16 {
        #[inline]
        fn backend() -> Backend {
            backend()
        }

        #[inline]
        fn has_vector_elementwise() -> bool {
            matches!(backend(), |Backend::V3GfniCrypto| Backend::V3
                | Backend::V2
                | Backend::NeonAes
                | Backend::Neon
                | Backend::Wasm128)
        }
    }

    impl KernelDispatch for Gf16 {
        type Prepared = Prepared;

        fn prepare(_proof: RawDispatch, coeff: Elem) -> Prepared {
            match backend() {
                Backend::V3GfniCrypto => Prepared::Compact(TowerCoeff::new(coeff)),
                // PMULL is table-free, so the broadcast-word form looks like the
                // natural fit here. It is not: it measured far behind these four
                // nibble tables (see `aarch64::gf16`), so PMULL hosts prepare and
                // shuffle exactly like baseline NEON.
                Backend::V3 | Backend::V2 | Backend::NeonAes | Backend::Neon | Backend::Wasm128 => {
                    Prepared::Tables(TowerTables::new(coeff))
                }
                // Portable fallback: Scalar and any future tier FGF does not
                // vectorize prepare the plain element form (`Backend` is
                // `#[non_exhaustive]`).
                _ => Prepared::Plain(coeff),
            }
        }

        #[inline]
        fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem {
            prepared.coeff()
        }

        #[inline]
        fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
            crate::kernel::xor(dst, src);
        }

        #[inline]
        fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]) {
            crate::kernel::xor_gather(region, dst, offsets);
        }

        #[inline]
        fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
            crate::kernel::xor(dst, src);
        }

        fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                // `Prepared::Compact` is only produced on a GFNI host, so the
                // backend dispatch collapses to the GFNI kernel directly (the
                // deferred 64-byte Avx512 tier is not in the ladder).
                Prepared::Compact(compact) => {
                    x86::gf16::mul_add_gfni(crate::kernel::x86_v3_gfni_token(), dst, *compact, src);
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_add_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                    }
                    _ => x86::gf16::mul_add_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => aarch64::gf16::mul_add_neon(dst, tables, src),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => wasm32::gf16::mul_add_simd128(dst, tables, src),
                other => mul_add_scalar(dst, other.coeff(), src),
            }
        }

        fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact(compact) => {
                    x86::gf16::mul_assign_gfni(crate::kernel::x86_v3_gfni_token(), dst, *compact);
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_assign_ssse3(crate::kernel::x86_v2_token(), dst, tables);
                    }
                    _ => x86::gf16::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, tables),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => aarch64::gf16::mul_assign_neon(dst, tables),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => wasm32::gf16::mul_assign_simd128(dst, tables),
                other => mul_assign_scalar(dst, other.coeff()),
            }
        }

        fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact(compact) => {
                    x86::gf16::mul_into_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *compact,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_into_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                    }
                    _ => x86::gf16::mul_into_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => aarch64::gf16::mul_into_neon(dst, tables, src),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => wasm32::gf16::mul_into_simd128(dst, tables, src),
                // Every other prepared form is a scalar coefficient: copying and
                // scaling in place is one pass either way.
                other => {
                    dst.copy_from_slice(src);
                    Self::mul_assign(RawDispatch, dst, other);
                }
            }
        }

        fn mul_add_scatter(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            coeffs: &[Elem],
            src: &[u8],
        ) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto => {
                    x86::gf16::mul_add_scatter_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        coeffs,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => x86::gf16::mul_add_scatter_avx2(
                    crate::kernel::x86_v3_token(),
                    rows,
                    row_len,
                    coeffs,
                    src,
                ),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_scatter_ssse3(
                    crate::kernel::x86_v2_token(),
                    rows,
                    row_len,
                    coeffs,
                    src,
                ),
                // Multi-row shapes keep the nibble tables: they derive them once
                // per coefficient and then amortize the cheaper byte loop over
                // every row of the group, which is the trade PMULL loses.
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => {
                    aarch64::gf16::scatter_neon(rows, row_len, coeffs, src);
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::scatter_simd128(rows, row_len, coeffs, src),
                // Not `scalar::mul_add_scatter`: that would re-derive a full
                // Karatsuba multiply per element. `mul_add_scalar` amortizes one
                // table resolve over each row.
                _ => {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                        mul_add_scalar(row, coeff, src);
                    }
                }
            }
        }
        #[cfg(feature = "alloc")]
        fn mul_add_scatter_plan(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            values: &[Elem],
            coeffs: &[Prepared],
            src: &[u8],
        ) {
            match backend() {
                Backend::V3GfniCrypto => {
                    Self::mul_add_scatter(RawDispatch, rows, row_len, values, src);
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => x86::gf16::mul_add_scatter_avx2(
                    crate::kernel::x86_v3_token(),
                    rows,
                    row_len,
                    coeffs,
                    src,
                ),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_scatter_ssse3(
                    crate::kernel::x86_v2_token(),
                    rows,
                    row_len,
                    coeffs,
                    src,
                ),
                _ => Self::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src),
            }
        }

        fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                // Blocked: the four-source group derives its broadcasts once and
                // keeps them live, so it reads the destination once per group
                // instead of once per source. That hoist is what makes it beat
                // repeated GFNI AXPY; with the broadcasts still inside the byte
                // loop the same kernel lost badly, which is why dispatch used to
                // avoid it. Numbers in BENCHMARKS.md.
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto => x86::gf16::mul_add_gather_gfni(
                    crate::kernel::x86_v3_gfni_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => gather_avx2_axpy(dst, coeffs, srcs),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_gather_ssse3(
                    crate::kernel::x86_v2_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => aarch64::gf16::gather_neon(dst, coeffs, srcs),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::gather_simd128(dst, coeffs, srcs),
                // See `mul_add_scatter`: one table resolve per term beats the
                // generic oracle's per-element multiply.
                _ => {
                    for (&coeff, &src) in coeffs.iter().zip(srcs) {
                        mul_add_scalar(dst, coeff, src);
                    }
                }
            }
        }

        #[cfg(feature = "alloc")]
        fn mul_add_gather_plan(
            _proof: RawDispatch,
            dst: &mut [u8],
            values: &[Elem],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            match backend() {
                Backend::V3GfniCrypto => Self::mul_add_gather(RawDispatch, dst, values, srcs),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_gather_ssse3(
                    crate::kernel::x86_v2_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                _ => Self::mul_add_gather_with(RawDispatch, dst, coeffs, srcs),
            }
        }

        fn mul_add_matrix(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem], &[u8])],
        ) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto => {
                    x86::gf16::mul_add_matrix_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        terms,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => matrix_avx2_axpy(rows, row_len, terms),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_matrix_ssse3(
                    crate::kernel::x86_v2_token(),
                    rows,
                    row_len,
                    nrows,
                    terms,
                ),
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => {
                    aarch64::gf16::matrix_neon(rows, row_len, nrows, terms);
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::matrix_simd128(rows, row_len, nrows, terms),
                // See `mul_add_scatter`: one table resolve per (term, row) beats
                // the generic oracle's per-element multiply.
                _ => {
                    for &(coeffs, src) in terms {
                        let blocks = rows.chunks_exact_mut(row_len).take(nrows);
                        for (row, &coeff) in blocks.zip(coeffs) {
                            mul_add_scalar(row, coeff, src);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "alloc")]
        fn mul_add_matrix_plan(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            values: &[Elem],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let _ = values;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            match backend() {
                Backend::V3GfniCrypto => {
                    let terms = crate::kernel::FlatMatrix {
                        coefficients: values,
                        nrows,
                        sources: srcs,
                    };
                    x86::gf16::mul_add_matrix_gfni_with(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                    return;
                }
                Backend::V2 => {
                    let terms = crate::kernel::FlatMatrix {
                        coefficients: coeffs,
                        nrows,
                        sources: srcs,
                    };
                    x86::gf16::mul_add_matrix_ssse3_with(
                        crate::kernel::x86_v2_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                    return;
                }
                _ => {}
            }
            for (term, &src) in srcs.iter().enumerate() {
                let start = term * nrows;
                for (row, coeff) in rows
                    .chunks_exact_mut(row_len)
                    .take(nrows)
                    .zip(&coeffs[start..start + nrows])
                {
                    Self::mul_add(RawDispatch, row, coeff, src);
                }
            }
        }

        fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto => {
                    x86::gf16::mul_elementwise_gfni(crate::kernel::x86_v3_gfni_token(), dst, a, b);
                }
                // Unlike GF(2^8), the tower elementwise product does *not* prefer
                // PMULL: the same three-multiply identity over `vmull_p8`
                // measured behind these bit-serial rounds, so that kernel was
                // removed rather than wired. GF(2^8) elementwise, where PMULL
                // replaces eight bit-serial rounds with two multiplies, does win
                // — see `Gf8B::mul_elementwise` and BENCHMARKS.md.
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => aarch64::gf16::elementwise_neon(dst, a, b),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::elementwise_simd128(dst, a, b),
                // See `Gf8B::mul_elementwise`: no fixed coefficient, so the
                // shuffle backends multiply the two varying base-field operands
                // bit-serially and keep a nibble table only for constant `DELTA`.
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => {
                    x86::gf16::mul_elementwise_avx2(crate::kernel::x86_v3_token(), dst, a, b);
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => {
                    x86::gf16::mul_elementwise_ssse3(crate::kernel::x86_v2_token(), dst, a, b);
                }
                _ => scalar::mul_elementwise::<Gf16>(dst, a, b),
            }
        }
    }
}

pub mod gf32 {
    //! GF(2^32) kernel dispatch.
    //!
    //! On GFNI x86 a GF(2^32) multiply is the level-2 tower identity of
    //! [`crate::kernel::gf16`]: two GF(2^16) lane multiplies under a
    //! period-2 coefficient, each itself two byte-wide multiplies — four
    //! `GF2P8MULB` per 32-byte lane. Everywhere else the portable scalar
    //! kernel applies.

    gfni_tower_dispatch!(
        Gf32,
        gf32,
        gf32_tiles,
        [u32; 4],
        "GF(2^32)",
        "the four 4-byte GFNI broadcast tiles"
    );
}

pub mod gf64 {
    //! GF(2^64) kernel dispatch.
    //!
    //! On GFNI x86 a GF(2^64) multiply is the level-3 tower identity — two
    //! GF(2^32) lane multiplies under a period-2 coefficient, each itself
    //! the four-`GF2P8MULB` [`crate::kernel::gf32`] scale, so eight
    //! `GF2P8MULB` per 32-byte lane. Everywhere else the portable scalar
    //! kernel applies.

    gfni_tower_dispatch!(
        Gf64,
        gf64,
        gf64_tiles,
        [u64; 8],
        "GF(2^64)",
        "the eight 8-byte GFNI broadcast tiles"
    );
}
