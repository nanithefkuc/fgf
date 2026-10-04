//! Kernel dispatch for the quadratic tower fields.
//!
//! [`gf16`] is representation-generic — every tower spec shares one dispatch
//! — and carries hand-written SIMD kernels on every backend, because the
//! two-byte lane is what the byte-wide hardware multiply operates on
//! directly. The wider levels are fixed Rijndael-rooted towers that reach
//! vector hardware only through the GFNI tower identity on x86, and reduce
//! to the portable scalar kernel everywhere else; that shared shape is one
//! macro, instantiated by [`gf32`] and [`gf64`]. The trait defaults compose
//! every multi-row and prepared operation from the single-row multiply, so
//! the GFNI win reaches scatter/gather/matrix over a prepared
//! [`crate::ops::CoeffVec`] without a per-shape kernel.

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
        #[derive(Clone, Copy, Debug)]
        pub enum Prepared {
            #[doc = concat!("GFNI: ", $tiles_doc, ", plus the element for the scalar tail.")]
            #[allow(dead_code)]
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
                    Self::Compact { coeff, .. } => *coeff,
                }
            }
        }

        impl FieldKernels for $marker {
            #[inline]
            fn backend() -> Backend {
                match backend() {
                    Backend::V4x | Backend::V3GfniCrypto => backend(),
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
                    Backend::V4x | Backend::V3GfniCrypto => Prepared::Compact {
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
    //! GF(2^16) kernel dispatch for every tower spec.
    //!
    //! Every backend here exploits the same tower identity: a 16-bit multiply
    //! is two byte-wide multiplies, one of the source and one of the source
    //! with adjacent bytes swapped, under the alternating coefficient pair in
    //! [`TowerCoeff`]. Hardware that can multiply bytes in the base field —
    //! GFNI, and only for an AES-native base — needs no table; hardware that
    //! cannot emulates each of the four base-field factors with a nibble
    //! shuffle from [`TowerTables`]. Coefficients erase to those two raw
    //! forms at the dispatch boundary: no kernel body below sees the spec,
    //! the base representation, or an element.
    //!
    //! A non-AES base changes only the GFNI tiers: `GF2P8MULB` multiplies AES
    //! bytes alone, so they route to the AVX2 tables kernels instead, and
    //! [`FieldKernels::backend`](crate::kernel::FieldKernels) reports the
    //! tier that actually serves the field. The vector elementwise formulas
    //! fold the relation's linear term into the Karatsuba cross sum, which
    //! holds only when that coefficient is the base's multiplicative
    //! identity, so any other spec takes the scalar elementwise path on
    //! every tier.

    use crate::field::tower::{Tower, TowerSpec, spec_a_is_one};
    use crate::field::{Elem, Gf};
    use crate::kernel::tables::{ByteBanks, ScaleTable, TowerCoeff, TowerTables};
    // `Backend` is referenced only from the SIMD dispatch arms, which cfg
    // away entirely on a scalar-only build or on an architecture without the
    // corresponding backend.
    #[allow(unused_imports)]
    use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    use crate::kernel::Matrix;
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    use crate::kernel::aarch64;
    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    use crate::kernel::wasm32;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::kernel::x86;

    /// A GF(2^16) coefficient resolved into the form this host's backend
    /// wants.
    ///
    /// The three variants are not interchangeable representations of the same
    /// cost. `Compact` is two base multiplies; `Tables` is four nibble tables
    /// and ~140 bytes to copy. Choosing between them at *preparation* time is
    /// the point: a GFNI host never pays for tables it will not read, and a
    /// shuffle host builds them once instead of on every call. Every payload
    /// is raw data — no spec or representation type survives here.
    #[derive(Clone, Copy, Debug)]
    pub enum Prepared {
        /// Native byte multiply (GFNI) or `PMULL`: a pair of broadcast words.
        Compact(TowerCoeff),
        /// Shuffle backends (AVX2, SSSE3, NEON): four nibble tables.
        Tables(TowerTables),
        /// No vector unit: the raw coefficient word.
        Plain(u16),
    }

    impl Prepared {
        /// The raw coefficient word this was built from.
        #[inline]
        #[must_use]
        pub const fn coeff(&self) -> u16 {
            match self {
                Self::Compact(compact) => compact.coeff,
                Self::Tables(tables) => tables.coeff,
                Self::Plain(coeff) => *coeff,
            }
        }
    }

    /// Random access to resolved GF(2^16) coefficients for the blocked
    /// scatter, gather, and matrix entries.
    ///
    /// A prepared call holds its coefficients as materialized
    /// [`TowerTables`] values; the dispatch layer's raw-element operations
    /// resolve each element against its spec's base bank on access, which
    /// keeps those one-shot forms allocation-free without making any kernel
    /// body depend on the spec.
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    pub(crate) trait Coeffs {
        /// Number of coefficients.
        fn count(&self) -> usize;
        /// The degenerate class of coefficient `index`: the cases the
        /// blocked loops special-case before any table work.
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        fn kind(&self, index: usize) -> CoeffKind;
        /// The nibble-table form of coefficient `index`: the shuffle and
        /// scalar tails.
        fn resolved(&self, index: usize) -> TowerTables;
        /// The broadcast-pair form of coefficient `index`, derived without
        /// building the nibble tables: the GFNI and AVX-512 blocked routes.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        fn compact(&self, index: usize) -> TowerCoeff;
    }

    /// The degenerate coefficient classes the blocked loops special-case:
    /// zero contributes nothing, one is a plain XOR.
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CoeffKind {
        /// The coefficient is zero: nothing to fold in.
        Zero,
        /// The coefficient is one: a plain XOR, no lookup.
        One,
        /// Anything else: the full tower multiply.
        General,
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl Coeffs for [TowerTables] {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            CoeffKind::of(&self[index])
        }
        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            self[index]
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            self[index].compact()
        }
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl Coeffs for alloc::vec::Vec<TowerTables> {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            CoeffKind::of(&self[index])
        }
        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            self[index]
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            self[index].compact()
        }
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<const N: usize> Coeffs for [TowerTables; N] {
        #[inline]
        fn count(&self) -> usize {
            N
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            CoeffKind::of(&self[index])
        }
        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            self[index]
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            self[index].compact()
        }
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec> Coeffs for alloc::vec::Vec<Elem<Gf<16, Tower<S>>>>
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let coeff = self[index];
            if coeff == Elem::<Gf<16, Tower<S>>>::ZERO {
                CoeffKind::Zero
            } else if coeff == Elem::<Gf<16, Tower<S>>>::ONE {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }

        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            TowerTables::new::<S>(self[index])
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            TowerCoeff::new::<S>(self[index])
        }
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec, const N: usize> Coeffs for [Elem<Gf<16, Tower<S>>>; N]
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn count(&self) -> usize {
            N
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let coeff = self[index];
            if coeff == Elem::<Gf<16, Tower<S>>>::ZERO {
                CoeffKind::Zero
            } else if coeff == Elem::<Gf<16, Tower<S>>>::ONE {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }

        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            TowerTables::new::<S>(self[index])
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            TowerCoeff::new::<S>(self[index])
        }
    }

    /// Raw tower elements resolved against their spec's base bank on access.
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec> Coeffs for [Elem<Gf<16, Tower<S>>>]
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let coeff = self[index];
            if coeff == Elem::<Gf<16, Tower<S>>>::ZERO {
                CoeffKind::Zero
            } else if coeff == Elem::<Gf<16, Tower<S>>>::ONE {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }
        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            TowerTables::new::<S>(self[index])
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            TowerCoeff::new::<S>(self[index])
        }
    }

    /// Raw per-term tower elements as a [`Matrix`] of resolved tables.
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec> Matrix<TowerTables> for [(&[Elem<Gf<16, Tower<S>>>], &[u8])]
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn len(&self) -> usize {
            <[(&[Elem<Gf<16, Tower<S>>>], &[u8])]>::len(self)
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerTables {
            TowerTables::new::<S>(self[term].0[row])
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self[term].1
        }
    }

    /// Raw per-term tower elements as a [`Matrix`] of broadcast pairs: the
    /// GFNI and AVX-512 matrix routes, which read no nibble table.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    impl<S: TowerSpec> Matrix<TowerCoeff> for [(&[Elem<Gf<16, Tower<S>>>], &[u8])]
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn len(&self) -> usize {
            <[(&[Elem<Gf<16, Tower<S>>>], &[u8])]>::len(self)
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerCoeff {
            TowerCoeff::new::<S>(self[term].0[row])
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self[term].1
        }
    }

    /// Raw flat row-major tower elements as a [`Matrix`], resolved on
    /// access into nibble tables or broadcast pairs.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[allow(private_bounds)]
    pub struct FlatResolved<'a, S: TowerSpec>
    where
        S::Base: ByteBanks,
    {
        values: &'a [Elem<Gf<16, Tower<S>>>],
        nrows: usize,
        sources: &'a [&'a [u8]],
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[allow(private_bounds)]
    impl<S: TowerSpec> FlatResolved<'_, S>
    where
        S::Base: ByteBanks,
    {
        /// Wrap flat row-major coefficients (`terms * nrows` values) with
        /// their per-term sources.
        #[inline]
        #[must_use]
        pub fn new<'a>(
            values: &'a [Elem<Gf<16, Tower<S>>>],
            nrows: usize,
            sources: &'a [&'a [u8]],
        ) -> FlatResolved<'a, S> {
            FlatResolved {
                values,
                nrows,
                sources,
            }
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    impl<S: TowerSpec> Matrix<TowerTables> for FlatResolved<'_, S>
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn len(&self) -> usize {
            self.sources.len()
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerTables {
            TowerTables::new::<S>(self.values[term * self.nrows + row])
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self.sources[term]
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    impl<S: TowerSpec> Matrix<TowerCoeff> for FlatResolved<'_, S>
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn len(&self) -> usize {
            self.sources.len()
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerCoeff {
            TowerCoeff::new::<S>(self.values[term * self.nrows + row])
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self.sources[term]
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
    impl CoeffKind {
        /// The class of a resolved coefficient.
        #[inline]
        pub(crate) fn of(tables: &TowerTables) -> Self {
            if tables.is_zero() {
                Self::Zero
            } else if tables.is_one() {
                Self::One
            } else {
                Self::General
            }
        }
    }

    /// The nibble-table form of a prepared coefficient, resolving through
    /// the spec's bank when the prepared form carries none.
    #[inline]
    fn prepared_tables<S: TowerSpec>(prepared: &Prepared) -> TowerTables
    where
        S::Base: ByteBanks,
    {
        match prepared {
            Prepared::Tables(tables) => *tables,
            other => TowerTables::new::<S>(Elem::<Gf<16, Tower<S>>>::from_raw(other.coeff())),
        }
    }

    /// The four nibble tables of a resolved coefficient.
    ///
    /// Deliberately borrowed, not copied: the tail handlers below run on as
    /// little as a single two-byte element.
    #[inline]
    pub(crate) fn factor_tables(tables: &TowerTables) -> [&ScaleTable; 4] {
        [
            &tables.factors[0],
            &tables.factors[1],
            &tables.factors[2],
            &tables.factors[3],
        ]
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    /// One base-field byte multiply of two varying bytes, by shift and reduce.
    ///
    /// The elementwise tails have no vector lane left and no fixed
    /// coefficient to table, so the few remaining products reduce
    /// carrylessly against the base reduction byte the entry receives.
    #[inline]
    pub(crate) const fn mul_reduce(mut a: u8, b: u8, reduction: u8) -> u8 {
        let mut product = 0u8;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 1 {
                product ^= a;
            }
            let overflow = a & 0x80 != 0;
            a <<= 1;
            if overflow {
                a ^= reduction;
            }
            i += 1;
        }
        product
    }

    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    /// The two components of one elementwise tail product under a unit
    /// linear coefficient: `constant = a*c ^ B*(b*d)` and
    /// `extension = a*d ^ b*c ^ b*d` for the interleaved bytes `[a, b]` and
    /// `[c, d]`.
    #[inline]
    pub(crate) fn tail_product(x: [u8; 2], y: [u8; 2], b_raw: u8, reduction: u8) -> [u8; 2] {
        let ac = mul_reduce(x[0], y[0], reduction);
        let bd = mul_reduce(x[1], y[1], reduction);
        let ad = mul_reduce(x[0], y[1], reduction);
        let bc = mul_reduce(x[1], y[0], reduction);
        let b_bd = mul_reduce(b_raw, bd, reduction);
        [ac ^ b_bd, ad ^ bc ^ bd]
    }

    /// `dst ^= coeff * src` over interleaved elements, one element at a time.
    ///
    /// The tail handler for the vector kernels.
    pub fn mul_add_scalar(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        if tables.is_zero() {
            return;
        }
        if tables.is_one() {
            scalar::xor(dst, src);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(tables);
        for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
            let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
            d[0] ^= f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
            d[1] ^= f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
        }
    }

    /// `dst *= coeff` over interleaved elements, one element at a time.
    pub fn mul_assign_scalar(dst: &mut [u8], tables: &TowerTables) {
        if tables.is_one() {
            return;
        }
        if tables.is_zero() {
            dst.fill(0);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(tables);
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
    #[allow(dead_code)]
    pub fn mul_into_scalar(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len());
        if tables.is_zero() {
            dst.fill(0);
            return;
        }
        if tables.is_one() {
            dst.copy_from_slice(src);
            return;
        }
        let [f0, f1, f2, f3] = factor_tables(tables);
        for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
            let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
            d[0] = f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
            d[1] = f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
        }
    }

    /// The AVX2 GF(2^16) gather: one AXPY per source.
    ///
    /// AVX2 has enough width but not enough registers to retain several
    /// four-table coefficient sets, and a gather's coefficients are
    /// one-to-one with sources, so a source's nibble split feeds exactly one
    /// coefficient and blocking has nothing to share.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn gather_axpy_avx2(dst: &mut [u8], coeffs: &(impl Coeffs + ?Sized), srcs: &[&[u8]]) {
        let count = coeffs.count().min(srcs.len());
        for (index, src) in (0..count).zip(srcs) {
            x86::gf16::mul_add_avx2(
                crate::kernel::x86_v3_token(),
                dst,
                &coeffs.resolved(index),
                src,
            );
        }
    }

    /// The AVX2 GF(2^16) matrix: one AXPY per term and row.
    ///
    /// With four nibble tables per coefficient, AVX2's register file holds
    /// too few destination accumulators for a blocked tile to repay itself.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn matrix_axpy_avx2(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &(impl Matrix<TowerTables> + ?Sized),
    ) {
        for term in 0..terms.len() {
            let src = terms.source(term);
            for (row, r) in rows.chunks_exact_mut(row_len).take(nrows).enumerate() {
                x86::gf16::mul_add_avx2(
                    crate::kernel::x86_v3_token(),
                    r,
                    &terms.coefficient(term, row),
                    src,
                );
            }
        }
    }

    impl<S: TowerSpec> FieldKernels for Gf<16, Tower<S>>
    where
        S::Base: ByteBanks,
    {
        #[inline]
        fn backend() -> Backend {
            match backend() {
                // The GFNI byte multiply serves AES bytes alone, so a
                // non-AES base takes the AVX2 tables path on those tiers.
                Backend::V4x | Backend::V3GfniCrypto if !S::Base::AES_NATIVE => Backend::V3,
                selected => selected,
            }
        }

        #[inline]
        fn has_vector_elementwise() -> bool {
            // The vector elementwise formulas fold the relation's linear
            // term into the Karatsuba cross sum, which holds only when that
            // coefficient is the base's multiplicative identity; every
            // other spec runs the scalar path.
            if !spec_a_is_one::<S>() {
                return false;
            }
            match backend() {
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => cfg!(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                )),
                Backend::NeonAes | Backend::Neon | Backend::Wasm128 => S::Base::AES_NATIVE,
                _ => false,
            }
        }
    }

    impl<S: TowerSpec> KernelDispatch for Gf<16, Tower<S>>
    where
        S::Base: ByteBanks,
    {
        type Prepared = Prepared;

        fn prepare(_proof: RawDispatch, coeff: Elem<Gf<16, Tower<S>>>) -> Prepared {
            match backend() {
                Backend::V4x | Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    Prepared::Compact(TowerCoeff::new::<S>(coeff))
                }
                // Every other vector tier runs the four nibble tables: the
                // shuffle hosts by construction, and the GFNI tiers with a
                // non-AES base have no `GF2P8MULB` byte multiply, so their
                // fixed-coefficient operations take the AVX2 tables kernels.
                // PMULL measured far behind these tables (see
                // `aarch64::gf16`), so PMULL hosts prepare and shuffle
                // exactly like baseline NEON.
                Backend::V4x
                | Backend::V3GfniCrypto
                | Backend::V3
                | Backend::V2
                | Backend::NeonAes
                | Backend::Neon
                | Backend::Wasm128 => Prepared::Tables(TowerTables::new::<S>(coeff)),
                // Portable fallback: Scalar and any future tier FGF does not
                // vectorize. `Backend` is `#[non_exhaustive]`.
                _ => Prepared::Plain(coeff.to_raw()),
            }
        }

        #[inline]
        fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem<Gf<16, Tower<S>>> {
            Elem::<Gf<16, Tower<S>>>::from_raw(prepared.coeff())
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
                Prepared::Compact(compact) => match backend() {
                    // Rows shorter than one 64-byte lane go straight to the
                    // 32-byte kernel `mul_add_avx512` would hand them to.
                    #[cfg(feature = "simd512")]
                    Backend::V4x if dst.len() >= 64 => {
                        x86::gf16::mul_add_avx512(
                            crate::kernel::x86_v4x_token(),
                            dst,
                            *compact,
                            src,
                        );
                    }
                    _ => x86::gf16::mul_add_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *compact,
                        src,
                    ),
                },
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_add_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                    }
                    _ => x86::gf16::mul_add_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => {
                    aarch64::gf16::mul_add_neon(crate::kernel::neon_token(), dst, tables, src)
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => {
                    wasm32::gf16::mul_add_simd128(crate::kernel::wasm128_token(), dst, tables, src)
                }
                other => mul_add_scalar(dst, &prepared_tables::<S>(other), src),
            }
        }

        fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact(compact) => match backend() {
                    #[cfg(feature = "simd512")]
                    Backend::V4x => {
                        x86::gf16::mul_assign_avx512(crate::kernel::x86_v4x_token(), dst, *compact);
                    }
                    _ => x86::gf16::mul_assign_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *compact,
                    ),
                },
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_assign_ssse3(crate::kernel::x86_v2_token(), dst, tables);
                    }
                    _ => x86::gf16::mul_assign_avx2(crate::kernel::x86_v3_token(), dst, tables),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => {
                    aarch64::gf16::mul_assign_neon(crate::kernel::neon_token(), dst, tables)
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => {
                    wasm32::gf16::mul_assign_simd128(crate::kernel::wasm128_token(), dst, tables)
                }
                other => mul_assign_scalar(dst, &prepared_tables::<S>(other)),
            }
        }

        fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact(compact) => match backend() {
                    #[cfg(feature = "simd512")]
                    Backend::V4x => {
                        x86::gf16::mul_into_avx512(
                            crate::kernel::x86_v4x_token(),
                            dst,
                            *compact,
                            src,
                        );
                    }
                    _ => x86::gf16::mul_into_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        *compact,
                        src,
                    ),
                },
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Tables(tables) => match backend() {
                    Backend::V2 => {
                        x86::gf16::mul_into_ssse3(crate::kernel::x86_v2_token(), dst, tables, src);
                    }
                    _ => x86::gf16::mul_into_avx2(crate::kernel::x86_v3_token(), dst, tables, src),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => {
                    aarch64::gf16::mul_into_neon(crate::kernel::neon_token(), dst, tables, src)
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => {
                    wasm32::gf16::mul_into_simd128(crate::kernel::wasm128_token(), dst, tables, src)
                }
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
            coeffs: &[Elem<Gf<16, Tower<S>>>],
            src: &[u8],
        ) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    x86::gf16::mul_add_scatter_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        coeffs,
                        src,
                    );
                }
                // A non-AES base has no GFNI byte multiply; its blocked
                // shapes take the AVX2 tables kernels.
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto => {
                    x86::gf16::mul_add_scatter_avx2(
                        crate::kernel::x86_v3_token(),
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
                    aarch64::gf16::mul_add_scatter_neon(
                        crate::kernel::neon_token(),
                        rows,
                        row_len,
                        coeffs,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::mul_add_scatter_simd128(
                    crate::kernel::wasm128_token(),
                    rows,
                    row_len,
                    coeffs,
                    src,
                ),
                // Not `scalar::mul_add_scatter`: that would re-derive a full
                // Karatsuba multiply per element. `mul_add_scalar` amortizes one
                // table resolve over each row.
                _ => {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                        mul_add_scalar(row, &TowerTables::new::<S>(coeff), src);
                    }
                }
            }
        }
        #[cfg(feature = "alloc")]
        fn mul_add_scatter_plan(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            values: &[Elem<Gf<16, Tower<S>>>],
            coeffs: &[Prepared],
            src: &[u8],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let _ = values;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            if matches!(
                backend(),
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2
            ) {
                // The element slice resolves against the base bank on
                // access, which is the same work the prepared form holds.
                return Self::mul_add_scatter(RawDispatch, rows, row_len, values, src);
            }
            Self::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src);
        }

        fn mul_add_gather(
            _proof: RawDispatch,
            dst: &mut [u8],
            coeffs: &[Elem<Gf<16, Tower<S>>>],
            srcs: &[&[u8]],
        ) {
            match backend() {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                // Blocked: the four-source group derives its broadcasts once and
                // keeps them live, so it reads the destination once per group
                // instead of once per source.
                Backend::V4x | Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    x86::gf16::mul_add_gather_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        coeffs,
                        srcs,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto => gather_axpy_avx2(dst, coeffs, srcs),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => gather_axpy_avx2(dst, coeffs, srcs),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => x86::gf16::mul_add_gather_ssse3(
                    crate::kernel::x86_v2_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => aarch64::gf16::mul_add_gather_neon(
                    crate::kernel::neon_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::mul_add_gather_simd128(
                    crate::kernel::wasm128_token(),
                    dst,
                    coeffs,
                    srcs,
                ),
                // See `mul_add_scatter`: one table resolve per term beats the
                // generic oracle's per-element multiply.
                _ => {
                    for (&coeff, &src) in coeffs.iter().zip(srcs) {
                        mul_add_scalar(dst, &TowerTables::new::<S>(coeff), src);
                    }
                }
            }
        }

        #[cfg(feature = "alloc")]
        fn mul_add_gather_plan(
            _proof: RawDispatch,
            dst: &mut [u8],
            values: &[Elem<Gf<16, Tower<S>>>],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let _ = values;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            match backend() {
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => {
                    return Self::mul_add_gather(RawDispatch, dst, values, srcs);
                }
                Backend::V2 => {
                    return x86::gf16::mul_add_gather_ssse3(
                        crate::kernel::x86_v2_token(),
                        dst,
                        values,
                        srcs,
                    );
                }
                _ => {}
            }
            Self::mul_add_gather_with(RawDispatch, dst, coeffs, srcs);
        }

        fn mul_add_matrix(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem<Gf<16, Tower<S>>>], &[u8])],
        ) {
            match backend() {
                #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x if S::Base::AES_NATIVE => x86::gf16::mul_add_matrix_avx512(
                    crate::kernel::x86_v4x_token(),
                    rows,
                    row_len,
                    nrows,
                    terms,
                ),
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    x86::gf16::mul_add_matrix_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        rows,
                        row_len,
                        nrows,
                        terms,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => {
                    matrix_axpy_avx2(rows, row_len, nrows, terms);
                }
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
                    aarch64::gf16::mul_add_matrix_neon(
                        crate::kernel::neon_token(),
                        rows,
                        row_len,
                        nrows,
                        terms,
                    );
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 => wasm32::gf16::mul_add_matrix_simd128(
                    crate::kernel::wasm128_token(),
                    rows,
                    row_len,
                    nrows,
                    terms,
                ),
                // See `mul_add_scatter`: one table resolve per (term, row) beats
                // the generic oracle's per-element multiply.
                _ => {
                    for &(coeffs, src) in terms {
                        let blocks = rows.chunks_exact_mut(row_len).take(nrows);
                        for (row, &coeff) in blocks.zip(coeffs) {
                            mul_add_scalar(row, &TowerTables::new::<S>(coeff), src);
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
            values: &[Elem<Gf<16, Tower<S>>>],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                let _ = (values, coeffs, srcs);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            match backend() {
                #[cfg(feature = "simd512")]
                Backend::V4x if S::Base::AES_NATIVE => {
                    let terms = FlatResolved::<S>::new(values, nrows, srcs);
                    x86::gf16::mul_add_matrix_avx512_with(
                        crate::kernel::x86_v4x_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                    );
                    return;
                }
                Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    let terms = FlatResolved::<S>::new(values, nrows, srcs);
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
                    let terms = FlatResolved::<S>::new(values, nrows, srcs);
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
            // The vector bodies below fold the relation's linear term into
            // the Karatsuba cross sum, which holds only when that
            // coefficient is the base's multiplicative identity; every other
            // spec runs the portable scalar path.
            if !spec_a_is_one::<S>() {
                return scalar::mul_elementwise::<Self>(dst, a, b);
            }
            #[cfg(feature = "simd")]
            let b_raw = S::B;
            #[cfg(all(
                feature = "simd",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )
            ))]
            let reduction = S::Base::REDUCTION_LOW;
            match backend() {
                #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x if S::Base::AES_NATIVE => {
                    x86::gf16::mul_elementwise_avx512(
                        crate::kernel::x86_v4x_token(),
                        dst,
                        a,
                        b,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    x86::gf16::mul_elementwise_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        a,
                        b,
                        b_raw,
                        reduction,
                    );
                }
                // No GFNI byte multiply for a non-AES base: the bit-serial
                // AVX2 kernel serves any base through its reduction byte.
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => {
                    x86::gf16::mul_elementwise_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        a,
                        b,
                        &S::Base::SCALE[b_raw as usize],
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => {
                    x86::gf16::mul_elementwise_ssse3(
                        crate::kernel::x86_v2_token(),
                        dst,
                        a,
                        b,
                        &S::Base::SCALE[b_raw as usize],
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes if S::Base::AES_NATIVE => {
                    aarch64::gf16::mul_elementwise_neon(
                        crate::kernel::neon_token(),
                        dst,
                        a,
                        b,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Backend::Wasm128 if S::Base::AES_NATIVE => {
                    wasm32::gf16::mul_elementwise_simd128(
                        crate::kernel::wasm128_token(),
                        dst,
                        a,
                        b,
                        b_raw,
                        reduction,
                    );
                }
                _ => scalar::mul_elementwise::<Self>(dst, a, b),
            }
        }

        fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
            // As `mul_elementwise`: the vector assign formulas hold only
            // when the linear coefficient is the base's one.
            if !spec_a_is_one::<S>() {
                return scalar::mul_elementwise_assign::<Self>(dst, src);
            }
            // The vector assign kernels are x86-only; the other tiers take
            // the scalar default.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let b_raw = S::B;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let reduction = S::Base::REDUCTION_LOW;
            match backend() {
                #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x if S::Base::AES_NATIVE => {
                    x86::gf16::mul_elementwise_assign_avx512(
                        crate::kernel::x86_v4x_token(),
                        dst,
                        src,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto if S::Base::AES_NATIVE => {
                    x86::gf16::mul_elementwise_assign_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        src,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => {
                    x86::gf16::mul_elementwise_assign_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        src,
                        &S::Base::SCALE[b_raw as usize],
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => {
                    x86::gf16::mul_elementwise_assign_ssse3(
                        crate::kernel::x86_v2_token(),
                        dst,
                        src,
                        &S::Base::SCALE[b_raw as usize],
                        reduction,
                    );
                }
                _ => scalar::mul_elementwise_assign::<Self>(dst, src),
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
    //! the four-`GF2P8MULB` [`crate::kernel::tower::gf32`] scale, so eight
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
