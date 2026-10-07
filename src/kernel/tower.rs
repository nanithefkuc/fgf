//! Kernel dispatch for the recursive quadratic tower fields.
//!
//! One dispatch implementation per degree serves every representation,
//! branching on the field description: presentations structurally equal to
//! the pinned Rijndael towers route to their byte-multiply kernels, those
//! equal to the pinned Fan-Paar towers to the Fan-Paar shuffle and lane
//! kernels, and every other tower — custom specs and towers over basis
//! bases — routes every bulk operation and tail to the typed scalar
//! fallback, reporting [`Backend::Scalar`]. The trait defaults compose
//! multi-row and prepared operations from the single-row multiply, so the
//! vector wins reach scatter/gather/matrix over a prepared
//! [`crate::ops::CoeffVec`] without a per-shape kernel.

pub mod gf16 {
    //! GF(2^16) kernel dispatch for every tower presentation.
    //!
    //! The Rijndael tower exploits the tower identity on every backend: a
    //! 16-bit multiply is two byte-wide multiplies, one of the source and
    //! one of the source with adjacent bytes swapped, under the alternating
    //! coefficient pair in [`TowerCoeff`]. Hardware that can multiply AES
    //! bytes — GFNI — needs no table; hardware that cannot emulates each of
    //! the four base-field factors with a nibble shuffle from
    //! [`TowerTables`]. The Fan-Paar tower multiplies through its own four
    //! nibble tables ([`FpTowerTables`]), built from the Fan-Paar byte
    //! subfield. Coefficients erase to those raw forms at the dispatch
    //! boundary: no kernel body below sees the spec, the base
    //! representation, or an element.

    use crate::field::Elem;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::field::binary::tower::FanPaar16;
    use crate::field::binary::tower::{
        FANPAAR16_DESC, RIJNDAEL16_DESC, Rijndael16, Tower, TowerSpec,
    };
    use crate::field::binary::{Binary, BinaryDegree, BinaryRepr};
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    use crate::field::{AES, Polynomial};
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::kernel::tables::AES_BANK;
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    use crate::kernel::tables::{ByteBanks, RIJNDAEL16_B};
    use crate::kernel::tables::{FpTowerTables, ScaleTable, TowerCoeff, TowerTables};
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

    /// Whether the tower's description is the pinned Rijndael16
    /// presentation, whose arithmetic the byte-multiply kernels implement.
    ///
    /// The body evaluates at compile time, so every caller's branch on it
    /// folds in each monomorphization instead of walking the description
    /// at run time.
    #[inline]
    pub(crate) const fn rijndael16<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<8>,
    {
        const { <Tower<S> as BinaryRepr<16>>::DESCRIPTION.same_structure(RIJNDAEL16_DESC) }
    }

    /// Whether the tower's description is the pinned Fan-Paar GF(2^16)
    /// presentation, whose arithmetic the Fan-Paar kernels implement.
    #[inline]
    pub(crate) const fn fanpaar16<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<8>,
    {
        const { <Tower<S> as BinaryRepr<16>>::DESCRIPTION.same_structure(FANPAAR16_DESC) }
    }

    /// A GF(2^16) coefficient resolved into the form this host's backend
    /// wants.
    ///
    /// The variants are not interchangeable representations of the same
    /// cost. `Compact` is two base multiplies; `Tables` is four nibble tables
    /// and ~140 bytes to copy. Choosing between them at *preparation* time is
    /// the point: a GFNI host never pays for tables it will not read, and a
    /// shuffle host builds them once instead of on every call. Every payload
    /// is raw data — no spec or representation type survives here.
    #[derive(Clone, Copy, Debug)]
    pub enum Prepared {
        /// Native byte multiply (GFNI) for the Rijndael tower: a pair of
        /// broadcast words.
        Compact(TowerCoeff),
        /// Shuffle backends (AVX2, SSSE3, NEON, Wasm) for the Rijndael
        /// tower: four nibble tables.
        Tables(TowerTables),
        /// Shuffle backends (AVX2, SSSE3) for the Fan-Paar tower: its four
        /// nibble tables over the Fan-Paar byte subfield.
        #[allow(dead_code)]
        Fp(FpTowerTables),
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
                Self::Fp(tables) => tables.coeff.to_raw(),
                Self::Plain(coeff) => *coeff,
            }
        }
    }

    /// Random access to resolved GF(2^16) coefficients for the blocked
    /// scatter, gather, and matrix entries.
    ///
    /// A prepared call holds its coefficients as materialized
    /// [`TowerTables`] values; the dispatch layer's raw-element operations
    /// resolve each element on access, which keeps those one-shot forms
    /// allocation-free without making any kernel body depend on the spec.
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

    /// Resolve one raw tower word into the Rijndael nibble tables.
    ///
    /// Only reached for descriptions structurally equal to the pinned
    /// Rijndael16 presentation, whose arithmetic those tables encode.
    #[inline]
    fn rijndael_tables(raw: u16) -> TowerTables {
        TowerTables::new(Elem::<Binary<16, Tower<Rijndael16>>> { raw })
    }

    /// Resolve one raw tower word into the Rijndael broadcast pair.
    #[inline]
    fn rijndael_compact(raw: u16) -> TowerCoeff {
        TowerCoeff::new(Elem::<Binary<16, Tower<Rijndael16>>> { raw })
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
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec> Coeffs for alloc::vec::Vec<Elem<Binary<16, Tower<S>>>>
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let raw = self[index].to_raw();
            if raw == 0 {
                CoeffKind::Zero
            } else if raw == Elem::<Binary<16, Tower<S>>>::ONE.to_raw() {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }

        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            rijndael_tables(self[index].to_raw())
        }

        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            rijndael_compact(self[index].to_raw())
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
    impl<S: TowerSpec, const N: usize> Coeffs for [Elem<Binary<16, Tower<S>>>; N]
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn count(&self) -> usize {
            N
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let raw = self[index].to_raw();
            if raw == 0 {
                CoeffKind::Zero
            } else if raw == Elem::<Binary<16, Tower<S>>>::ONE.to_raw() {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }

        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            rijndael_tables(self[index].to_raw())
        }

        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            rijndael_compact(self[index].to_raw())
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
    impl<S: TowerSpec> Coeffs for [Elem<Binary<16, Tower<S>>>]
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn count(&self) -> usize {
            self.len()
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "wasm32")))]
        #[inline]
        fn kind(&self, index: usize) -> CoeffKind {
            let raw = self[index].to_raw();
            if raw == 0 {
                CoeffKind::Zero
            } else if raw == Elem::<Binary<16, Tower<S>>>::ONE.to_raw() {
                CoeffKind::One
            } else {
                CoeffKind::General
            }
        }

        #[inline]
        fn resolved(&self, index: usize) -> TowerTables {
            rijndael_tables(self[index].to_raw())
        }

        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        #[inline]
        fn compact(&self, index: usize) -> TowerCoeff {
            rijndael_compact(self[index].to_raw())
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
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )
    ))]
    impl<S: TowerSpec> Matrix<TowerTables> for [(&[Elem<Binary<16, Tower<S>>>], &[u8])]
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn len(&self) -> usize {
            <[(&[Elem<Binary<16, Tower<S>>>], &[u8])]>::len(self)
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerTables {
            rijndael_tables(self[term].0[row].to_raw())
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self[term].1
        }
    }

    /// Raw per-term tower elements as a [`Matrix`] of broadcast pairs: the
    /// GFNI and AVX-512 matrix routes, which read no nibble table.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    impl<S: TowerSpec> Matrix<TowerCoeff> for [(&[Elem<Binary<16, Tower<S>>>], &[u8])]
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn len(&self) -> usize {
            <[(&[Elem<Binary<16, Tower<S>>>], &[u8])]>::len(self)
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerCoeff {
            rijndael_compact(self[term].0[row].to_raw())
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
        S::Base: BinaryDegree<8>,
    {
        values: &'a [Elem<Binary<16, Tower<S>>>],
        nrows: usize,
        sources: &'a [&'a [u8]],
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[allow(private_bounds)]
    impl<S: TowerSpec> FlatResolved<'_, S>
    where
        S::Base: BinaryDegree<8>,
    {
        /// Wrap flat row-major coefficients (`terms * nrows` values) with
        /// their per-term sources.
        #[inline]
        #[must_use]
        pub fn new<'a>(
            values: &'a [Elem<Binary<16, Tower<S>>>],
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
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn len(&self) -> usize {
            self.sources.len()
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerTables {
            rijndael_tables(self.values[term * self.nrows + row].to_raw())
        }

        #[inline]
        fn source(&self, term: usize) -> &[u8] {
            self.sources[term]
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    impl<S: TowerSpec> Matrix<TowerCoeff> for FlatResolved<'_, S>
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn len(&self) -> usize {
            self.sources.len()
        }

        #[inline]
        fn coefficient(&self, term: usize, row: usize) -> TowerCoeff {
            rijndael_compact(self.values[term * self.nrows + row].to_raw())
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

    impl<S: TowerSpec> FieldKernels for Binary<16, Tower<S>>
    where
        S::Base: BinaryDegree<8>,
    {
        #[inline]
        fn backend() -> Backend {
            if rijndael16::<S>() {
                return backend();
            }
            if fanpaar16::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => backend(),
                    _ => Backend::Scalar,
                };
            }
            Backend::Scalar
        }

        #[inline]
        fn has_vector_elementwise() -> bool {
            // The vector elementwise formulas fold the relation's linear
            // term into the Karatsuba cross sum, which holds only when that
            // coefficient is the base's multiplicative identity: the
            // Rijndael relation's `A = 1`. Every other tower runs the
            // scalar path.
            if !rijndael16::<S>() {
                return false;
            }
            matches!(
                backend(),
                Backend::V4x
                    | Backend::V3GfniCrypto
                    | Backend::V3
                    | Backend::V2
                    | Backend::NeonAes
                    | Backend::Neon
                    | Backend::Wasm128
            ) && cfg!(all(
                feature = "simd",
                any(target_arch = "x86", target_arch = "x86_64")
            ))
        }
    }

    impl<S: TowerSpec> KernelDispatch for Binary<16, Tower<S>>
    where
        S::Base: BinaryDegree<8>,
    {
        type Prepared = Prepared;

        fn prepare(_proof: RawDispatch, coeff: Elem<Binary<16, Tower<S>>>) -> Prepared {
            let raw = coeff.to_raw();
            if rijndael16::<S>() {
                return match backend() {
                    Backend::V4x | Backend::V3GfniCrypto => {
                        Prepared::Compact(rijndael_compact(raw))
                    }
                    // Every other vector tier runs the four nibble tables: the
                    // shuffle hosts by construction. PMULL measured far behind
                    // these tables (see `aarch64::gf16`), so PMULL hosts
                    // prepare and shuffle exactly like baseline NEON.
                    Backend::V3
                    | Backend::V2
                    | Backend::NeonAes
                    | Backend::Neon
                    | Backend::Wasm128 => Prepared::Tables(rijndael_tables(raw)),
                    // Portable fallback: Scalar and any future tier FGF does
                    // not vectorize. `Backend` is `#[non_exhaustive]`.
                    _ => Prepared::Plain(raw),
                };
            }
            if fanpaar16::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2 => {
                        Prepared::Fp(FpTowerTables::new(Elem::<Binary<16, Tower<FanPaar16>>> {
                            raw,
                        }))
                    }
                    _ => Prepared::Plain(raw),
                };
            }
            Prepared::Plain(raw)
        }

        #[inline]
        fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem<Binary<16, Tower<S>>> {
            Elem::<Binary<16, Tower<S>>>::from_raw(prepared.coeff())
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
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(tables) => match backend() {
                    Backend::V2 => {
                        x86::fan_paar::mul_add_ssse3_fp16(
                            crate::kernel::x86_v2_token(),
                            dst,
                            tables,
                            src,
                        );
                    }
                    _ => x86::fan_paar::mul_add_avx2_fp16(
                        crate::kernel::x86_v3_token(),
                        dst,
                        tables,
                        src,
                    ),
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => {
                    aarch64::gf16::mul_add_neon(crate::kernel::neon_token(), dst, tables, src)
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => {
                    wasm32::gf16::mul_add_simd128(crate::kernel::wasm128_token(), dst, tables, src)
                }
                other => {
                    let coeff = Elem::<Self>::from_raw(other.coeff());
                    if rijndael16::<S>() {
                        // Not `scalar::mul_add`: the nibble tables amortize
                        // one resolve over the whole buffer.
                        mul_add_scalar(dst, &rijndael_tables(other.coeff()), src);
                    } else {
                        scalar::mul_add::<Self>(dst, coeff, src);
                    }
                }
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
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(tables) => match backend() {
                    Backend::V2 => {
                        x86::fan_paar::mul_assign_ssse3_fp16(
                            crate::kernel::x86_v2_token(),
                            dst,
                            tables,
                        );
                    }
                    _ => {
                        x86::fan_paar::mul_assign_avx2_fp16(
                            crate::kernel::x86_v3_token(),
                            dst,
                            tables,
                        );
                    }
                },
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Prepared::Tables(tables) => {
                    aarch64::gf16::mul_assign_neon(crate::kernel::neon_token(), dst, tables)
                }
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                Prepared::Tables(tables) => {
                    wasm32::gf16::mul_assign_simd128(crate::kernel::wasm128_token(), dst, tables)
                }
                other => {
                    let coeff = Elem::<Self>::from_raw(other.coeff());
                    if rijndael16::<S>() {
                        mul_assign_scalar(dst, &rijndael_tables(other.coeff()));
                    } else {
                        scalar::mul_assign::<Self>(dst, coeff);
                    }
                }
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
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(tables) => match backend() {
                    Backend::V2 => {
                        x86::fan_paar::mul_into_ssse3_fp16(
                            crate::kernel::x86_v2_token(),
                            dst,
                            tables,
                            src,
                        );
                    }
                    _ => x86::fan_paar::mul_into_avx2_fp16(
                        crate::kernel::x86_v3_token(),
                        dst,
                        tables,
                        src,
                    ),
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
            coeffs: &[Elem<Binary<16, Tower<S>>>],
            src: &[u8],
        ) {
            if rijndael16::<S>() {
                match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto => {
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
                            mul_add_scalar(row, &rijndael_tables(coeff.to_raw()), src);
                        }
                    }
                }
                return;
            }
            if fanpaar16::<S>() {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                }
                return;
            }
            scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
        }

        #[cfg(feature = "alloc")]
        fn mul_add_scatter_plan(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            values: &[Elem<Binary<16, Tower<S>>>],
            coeffs: &[Prepared],
            src: &[u8],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let _ = values;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            if rijndael16::<S>()
                && matches!(
                    backend(),
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 | Backend::V2
                )
            {
                // The element slice resolves against the base bank on
                // access, which is the same work the prepared form holds.
                return Self::mul_add_scatter(RawDispatch, rows, row_len, values, src);
            }
            Self::mul_add_scatter_with(RawDispatch, rows, row_len, coeffs, src);
        }

        fn mul_add_gather(
            _proof: RawDispatch,
            dst: &mut [u8],
            coeffs: &[Elem<Binary<16, Tower<S>>>],
            srcs: &[&[u8]],
        ) {
            if rijndael16::<S>() {
                match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    // Blocked: the four-source group derives its broadcasts once and
                    // keeps them live, so it reads the destination once per group
                    // instead of once per source.
                    Backend::V4x | Backend::V3GfniCrypto => {
                        x86::gf16::mul_add_gather_gfni(
                            crate::kernel::x86_v3_gfni_token(),
                            dst,
                            coeffs,
                            srcs,
                        );
                    }
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
                            mul_add_scalar(dst, &rijndael_tables(coeff.to_raw()), src);
                        }
                    }
                }
                return;
            }
            if fanpaar16::<S>() {
                for (&coeff, &src) in coeffs.iter().zip(srcs) {
                    Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
                }
                return;
            }
            scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
        }

        #[cfg(feature = "alloc")]
        fn mul_add_gather_plan(
            _proof: RawDispatch,
            dst: &mut [u8],
            values: &[Elem<Binary<16, Tower<S>>>],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let _ = values;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            if rijndael16::<S>() {
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
            }
            Self::mul_add_gather_with(RawDispatch, dst, coeffs, srcs);
        }

        fn mul_add_matrix(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem<Binary<16, Tower<S>>>], &[u8])],
        ) {
            if rijndael16::<S>() {
                match backend() {
                    #[cfg(all(
                        feature = "simd512",
                        any(target_arch = "x86", target_arch = "x86_64")
                    ))]
                    Backend::V4x => x86::gf16::mul_add_matrix_avx512(
                        crate::kernel::x86_v4x_token(),
                        rows,
                        row_len,
                        nrows,
                        terms,
                    ),
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
                    Backend::V3 => matrix_axpy_avx2(rows, row_len, nrows, terms),
                    // Without the 512-bit kernels the V4x tier cannot
                    // resolve; the arm keeps the match exhaustive anyway.
                    #[cfg(all(
                        not(feature = "simd512"),
                        feature = "simd",
                        any(target_arch = "x86", target_arch = "x86_64")
                    ))]
                    Backend::V4x => matrix_axpy_avx2(rows, row_len, nrows, terms),
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
                                mul_add_scalar(row, &rijndael_tables(coeff.to_raw()), src);
                            }
                        }
                    }
                }
                return;
            }
            if fanpaar16::<S>() {
                for &(coeffs, src) in terms {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                        Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                    }
                }
                return;
            }
            scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
        }

        #[cfg(feature = "alloc")]
        fn mul_add_matrix_plan(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            values: &[Elem<Binary<16, Tower<S>>>],
            coeffs: &[Prepared],
            srcs: &[&[u8]],
        ) {
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                let _ = (values, coeffs, srcs);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            if rijndael16::<S>() {
                match backend() {
                    #[cfg(feature = "simd512")]
                    Backend::V4x => {
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
                    Backend::V3GfniCrypto => {
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

        // One dispatch body over every backend arm; the split keeps the
        // tier routing readable in one place.
        #[allow(clippy::too_many_lines)]
        fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
            // The vector bodies below fold the relation's linear term into
            // the Karatsuba cross sum, which holds only for the Rijndael
            // relation (`A = 1`); every other tower runs the portable
            // scalar path.
            if !rijndael16::<S>() {
                return scalar::mul_elementwise::<Self>(dst, a, b);
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
            let b_raw = RIJNDAEL16_B;
            #[cfg(all(
                feature = "simd",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )
            ))]
            let reduction = <Polynomial<AES> as ByteBanks>::REDUCTION_LOW;
            match backend() {
                #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x => {
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
                Backend::V3GfniCrypto => {
                    x86::gf16::mul_elementwise_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        a,
                        b,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => {
                    x86::gf16::mul_elementwise_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        a,
                        b,
                        &AES_BANK[RIJNDAEL16_B as usize],
                        reduction,
                    );
                }
                // Without the 512-bit kernels the V4x tier cannot resolve;
                // the arm keeps the match exhaustive anyway.
                #[cfg(all(
                    not(feature = "simd512"),
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                Backend::V4x => {
                    x86::gf16::mul_elementwise_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        a,
                        b,
                        &AES_BANK[RIJNDAEL16_B as usize],
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
                        &AES_BANK[RIJNDAEL16_B as usize],
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                Backend::Neon | Backend::NeonAes => {
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
                Backend::Wasm128 => {
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
            // for the Rijndael relation; the other tiers take the scalar
            // default.
            if !rijndael16::<S>() {
                return scalar::mul_elementwise_assign::<Self>(dst, src);
            }
            // The vector assign kernels are x86-only; the other tiers take
            // the scalar default.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let b_raw = RIJNDAEL16_B;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let reduction = <Polynomial<AES> as ByteBanks>::REDUCTION_LOW;
            match backend() {
                #[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V4x => {
                    x86::gf16::mul_elementwise_assign_avx512(
                        crate::kernel::x86_v4x_token(),
                        dst,
                        src,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3GfniCrypto => {
                    x86::gf16::mul_elementwise_assign_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        src,
                        b_raw,
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V3 => {
                    x86::gf16::mul_elementwise_assign_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        src,
                        &AES_BANK[RIJNDAEL16_B as usize],
                        reduction,
                    );
                }
                // Without the 512-bit kernels the V4x tier cannot resolve;
                // the arm keeps the match exhaustive anyway.
                #[cfg(all(
                    not(feature = "simd512"),
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                Backend::V4x => {
                    x86::gf16::mul_elementwise_assign_avx2(
                        crate::kernel::x86_v3_token(),
                        dst,
                        src,
                        &AES_BANK[RIJNDAEL16_B as usize],
                        reduction,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::V2 => {
                    x86::gf16::mul_elementwise_assign_ssse3(
                        crate::kernel::x86_v2_token(),
                        dst,
                        src,
                        &AES_BANK[RIJNDAEL16_B as usize],
                        reduction,
                    );
                }
                _ => scalar::mul_elementwise_assign::<Self>(dst, src),
            }
        }
    }
}

pub mod gf32 {
    //! GF(2^32) kernel dispatch for every tower presentation.
    //!
    //! On GFNI x86 the Rijndael tower's multiply is the level-2 tower
    //! identity of the GF(2^16) dispatch: two GF(2^16) lane multiplies
    //! under a period-2 coefficient, each itself two byte-wide multiplies —
    //! four `GF2P8MULB` per 32-byte lane. The Fan-Paar tower runs its AVX2
    //! lane scales. Everywhere else, and for every other tower, the
    //! portable scalar kernel applies.

    use crate::field::Elem;
    use crate::field::binary::tower::{FANPAAR32_DESC, RIJNDAEL32_DESC, Tower, TowerSpec};
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::field::binary::tower::{FanPaar32, Rijndael32};
    use crate::field::binary::{Binary, BinaryDegree, BinaryRepr};
    #[allow(unused_imports)]
    use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::kernel::x86;

    /// Whether the tower's description is the pinned Rijndael32
    /// presentation.
    ///
    /// The body evaluates at compile time; see [`tower::gf16::rijndael16`].
    #[inline]
    pub(crate) const fn rijndael32<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<16>,
    {
        const { <Tower<S> as BinaryRepr<32>>::DESCRIPTION.same_structure(RIJNDAEL32_DESC) }
    }

    /// Whether the tower's description is the pinned Fan-Paar GF(2^32)
    /// presentation.
    #[inline]
    pub(crate) const fn fanpaar32<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<16>,
    {
        const { <Tower<S> as BinaryRepr<32>>::DESCRIPTION.same_structure(FANPAAR32_DESC) }
    }

    /// A GF(2^32) coefficient resolved into the form this host's backend
    /// wants.
    ///
    /// `Compact` carries the four 4-byte GFNI broadcast tiles, derived once
    /// in coefficient preparation, so a prepared [`crate::ops::Coeff`] or
    /// [`crate::ops::CoeffVec`] reuses them across the whole buffer. `Fp`
    /// carries the raw word for the Fan-Paar AVX2 lane scales. `Plain`
    /// hands the element to the portable scalar kernel.
    #[derive(Clone, Copy, Debug)]
    pub enum Prepared {
        /// GFNI: the four 4-byte broadcast tiles, plus the raw word for the
        /// scalar tail.
        #[allow(dead_code)]
        Compact {
            /// The GF(2^32) coefficient's raw word, for the portable tail.
            coeff: u32,
            /// The tiles from [`x86::gf32::gf32_tiles`].
            tiles: [u32; 4],
        },
        /// AVX2 Fan-Paar lane scales: the raw word.
        #[allow(dead_code)]
        Fp(u32),
        /// No vector backend: the raw word, for the portable scalar kernel.
        Plain(u32),
    }

    impl Prepared {
        /// The raw coefficient word this was built from.
        #[inline]
        #[must_use]
        pub const fn coeff(&self) -> u32 {
            match self {
                Self::Compact { coeff, .. } | Self::Fp(coeff) | Self::Plain(coeff) => *coeff,
            }
        }
    }

    impl<S: TowerSpec> FieldKernels for Binary<32, Tower<S>>
    where
        S::Base: BinaryDegree<16>,
    {
        #[inline]
        fn backend() -> Backend {
            if rijndael32::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto => backend(),
                    _ => Backend::Scalar,
                };
            }
            if fanpaar32::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => backend(),
                    _ => Backend::Scalar,
                };
            }
            Backend::Scalar
        }

        #[inline]
        fn has_vector_elementwise() -> bool {
            false
        }
    }

    impl<S: TowerSpec> KernelDispatch for Binary<32, Tower<S>>
    where
        S::Base: BinaryDegree<16>,
    {
        type Prepared = Prepared;

        fn prepare(_proof: RawDispatch, coeff: Elem<Binary<32, Tower<S>>>) -> Prepared {
            let raw = coeff.to_raw();
            if rijndael32::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto => Prepared::Compact {
                        coeff: raw,
                        tiles: x86::gf32::gf32_tiles(Elem::<Binary<32, Tower<Rijndael32>>> { raw }),
                    },
                    _ => Prepared::Plain(raw),
                };
            }
            if fanpaar32::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => Prepared::Fp(raw),
                    _ => Prepared::Plain(raw),
                };
            }
            Prepared::Plain(raw)
        }

        #[inline]
        fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem<Binary<32, Tower<S>>> {
            Elem::<Binary<32, Tower<S>>>::from_raw(prepared.coeff())
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
                Prepared::Compact { coeff, tiles } => {
                    x86::gf32::mul_add_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<32, Tower<Rijndael32>>> { raw: *coeff },
                        *tiles,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_add_avx2_fp32(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<32, Tower<FanPaar32>>> { raw: *coeff },
                        src,
                    );
                }
                other => scalar::mul_add::<Self>(dst, Elem::<Self>::from_raw(other.coeff()), src),
            }
        }

        fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact { coeff, tiles } => {
                    x86::gf32::mul_assign_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<32, Tower<Rijndael32>>> { raw: *coeff },
                        *tiles,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_assign_avx2_fp32(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<32, Tower<FanPaar32>>> { raw: *coeff },
                    );
                }
                other => scalar::mul_assign::<Self>(dst, Elem::<Self>::from_raw(other.coeff())),
            }
        }

        fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact { coeff, tiles } => {
                    x86::gf32::mul_into_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<32, Tower<Rijndael32>>> { raw: *coeff },
                        *tiles,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_into_avx2_fp32(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<32, Tower<FanPaar32>>> { raw: *coeff },
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
            coeffs: &[Elem<Binary<32, Tower<S>>>],
            src: &[u8],
        ) {
            // The trait default `mul_add_scatter_with` would do the same
            // per-row `mul_add`, but it takes already-prepared coefficients.
            // The base path receives raw elements, so each row prepares its
            // own; the cost is one tile derivation per row, amortized over
            // `row_len`.
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
            }
        }

        fn mul_add_gather(
            _proof: RawDispatch,
            dst: &mut [u8],
            coeffs: &[Elem<Binary<32, Tower<S>>>],
            srcs: &[&[u8]],
        ) {
            for (&coeff, &src) in coeffs.iter().zip(srcs) {
                Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
            }
        }

        fn mul_add_matrix(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem<Binary<32, Tower<S>>>], &[u8])],
        ) {
            for &(coeffs, src) in terms {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                    Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                }
            }
        }

        fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
            // Both operands vary per lane, so there is no fixed
            // coefficient to broadcast; no elementwise vector path serves
            // the degree-32 towers.
            scalar::mul_elementwise::<Self>(dst, a, b);
        }
    }
}

pub mod gf64 {
    //! GF(2^64) kernel dispatch for every tower presentation.
    //!
    //! On GFNI x86 the Rijndael tower's multiply is the level-3 tower
    //! identity — two GF(2^32) lane multiplies under a period-2
    //! coefficient, each itself the four-`GF2P8MULB`
    //! [`crate::kernel::tower::gf32`] scale, so eight `GF2P8MULB` per 32-byte
    //! lane. The Fan-Paar tower runs its AVX2 lane scales. Everywhere
    //! else, and for every other tower, the portable scalar kernel
    //! applies.

    use crate::field::Elem;
    use crate::field::binary::tower::{FANPAAR64_DESC, RIJNDAEL64_DESC, Tower, TowerSpec};
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::field::binary::tower::{FanPaar64, Rijndael64};
    use crate::field::binary::{Binary, BinaryDegree, BinaryRepr};
    #[allow(unused_imports)]
    use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, backend, scalar};

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use crate::kernel::x86;

    /// Whether the tower's description is the pinned Rijndael64
    /// presentation.
    ///
    /// The body evaluates at compile time; see [`tower::gf16::rijndael16`].
    #[inline]
    pub(crate) const fn rijndael64<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<32>,
    {
        const { <Tower<S> as BinaryRepr<64>>::DESCRIPTION.same_structure(RIJNDAEL64_DESC) }
    }

    /// Whether the tower's description is the pinned Fan-Paar GF(2^64)
    /// presentation.
    #[inline]
    pub(crate) const fn fanpaar64<S: TowerSpec>() -> bool
    where
        S::Base: BinaryDegree<32>,
    {
        const { <Tower<S> as BinaryRepr<64>>::DESCRIPTION.same_structure(FANPAAR64_DESC) }
    }

    /// A GF(2^64) coefficient resolved into the form this host's backend
    /// wants.
    ///
    /// `Compact` carries the eight 8-byte GFNI broadcast tiles, `Fp` the
    /// raw word for the Fan-Paar AVX2 lane scales, and `Plain` the raw word
    /// for the portable scalar kernel.
    #[derive(Clone, Copy, Debug)]
    pub enum Prepared {
        /// GFNI: the eight 8-byte broadcast tiles, plus the raw word for
        /// the scalar tail.
        #[allow(dead_code)]
        Compact {
            /// The GF(2^64) coefficient's raw word, for the portable tail.
            coeff: u64,
            /// The tiles from [`x86::gf64::gf64_tiles`].
            tiles: [u64; 8],
        },
        /// AVX2 Fan-Paar lane scales: the raw word.
        #[allow(dead_code)]
        Fp(u64),
        /// No vector backend: the raw word, for the portable scalar kernel.
        Plain(u64),
    }

    impl Prepared {
        /// The raw coefficient word this was built from.
        #[inline]
        #[must_use]
        pub const fn coeff(&self) -> u64 {
            match self {
                Self::Compact { coeff, .. } | Self::Fp(coeff) | Self::Plain(coeff) => *coeff,
            }
        }
    }

    impl<S: TowerSpec> FieldKernels for Binary<64, Tower<S>>
    where
        S::Base: BinaryDegree<32>,
    {
        #[inline]
        fn backend() -> Backend {
            if rijndael64::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto => backend(),
                    _ => Backend::Scalar,
                };
            }
            if fanpaar64::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => backend(),
                    _ => Backend::Scalar,
                };
            }
            Backend::Scalar
        }

        #[inline]
        fn has_vector_elementwise() -> bool {
            false
        }
    }

    impl<S: TowerSpec> KernelDispatch for Binary<64, Tower<S>>
    where
        S::Base: BinaryDegree<32>,
    {
        type Prepared = Prepared;

        fn prepare(_proof: RawDispatch, coeff: Elem<Binary<64, Tower<S>>>) -> Prepared {
            let raw = coeff.to_raw();
            if rijndael64::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto => Prepared::Compact {
                        coeff: raw,
                        tiles: x86::gf64::gf64_tiles(Elem::<Binary<64, Tower<Rijndael64>>> { raw }),
                    },
                    _ => Prepared::Plain(raw),
                };
            }
            if fanpaar64::<S>() {
                return match backend() {
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    Backend::V4x | Backend::V3GfniCrypto | Backend::V3 => Prepared::Fp(raw),
                    _ => Prepared::Plain(raw),
                };
            }
            Prepared::Plain(raw)
        }

        #[inline]
        fn prepared_coeff(_proof: RawDispatch, prepared: &Prepared) -> Elem<Binary<64, Tower<S>>> {
            Elem::<Binary<64, Tower<S>>>::from_raw(prepared.coeff())
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
                Prepared::Compact { coeff, tiles } => {
                    x86::gf64::mul_add_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<64, Tower<Rijndael64>>> { raw: *coeff },
                        *tiles,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_add_avx2_fp64(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<64, Tower<FanPaar64>>> { raw: *coeff },
                        src,
                    );
                }
                other => scalar::mul_add::<Self>(dst, Elem::<Self>::from_raw(other.coeff()), src),
            }
        }

        fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact { coeff, tiles } => {
                    x86::gf64::mul_assign_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<64, Tower<Rijndael64>>> { raw: *coeff },
                        *tiles,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_assign_avx2_fp64(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<64, Tower<FanPaar64>>> { raw: *coeff },
                    );
                }
                other => scalar::mul_assign::<Self>(dst, Elem::<Self>::from_raw(other.coeff())),
            }
        }

        fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
            match coeff {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Compact { coeff, tiles } => {
                    x86::gf64::mul_into_gfni(
                        crate::kernel::x86_v3_gfni_token(),
                        dst,
                        Elem::<Binary<64, Tower<Rijndael64>>> { raw: *coeff },
                        *tiles,
                        src,
                    );
                }
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                Prepared::Fp(coeff) => {
                    x86::fan_paar::mul_into_avx2_fp64(
                        crate::kernel::x86_v3_token(),
                        dst,
                        Elem::<Binary<64, Tower<FanPaar64>>> { raw: *coeff },
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
            coeffs: &[Elem<Binary<64, Tower<S>>>],
            src: &[u8],
        ) {
            // As the degree-32 dispatch: each row prepares its own
            // coefficient, amortizing the derivation over `row_len`.
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
            }
        }

        fn mul_add_gather(
            _proof: RawDispatch,
            dst: &mut [u8],
            coeffs: &[Elem<Binary<64, Tower<S>>>],
            srcs: &[&[u8]],
        ) {
            for (&coeff, &src) in coeffs.iter().zip(srcs) {
                Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
            }
        }

        fn mul_add_matrix(
            _proof: RawDispatch,
            rows: &mut [u8],
            row_len: usize,
            nrows: usize,
            terms: &[(&[Elem<Binary<64, Tower<S>>>], &[u8])],
        ) {
            for &(coeffs, src) in terms {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                    Self::mul_add(RawDispatch, row, &Self::prepare(RawDispatch, coeff), src);
                }
            }
        }

        fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
            // Both operands vary per lane, so there is no fixed
            // coefficient to broadcast; no elementwise vector path serves
            // the degree-64 towers.
            scalar::mul_elementwise::<Self>(dst, a, b);
        }
    }
}
