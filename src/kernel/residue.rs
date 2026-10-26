//! Residue prime-field kernel dispatch.
//!
//! Every `FiniteField<P, 1, Residue>` row runs the portable
//! [`crate::kernel::prime`] path, except the vectorized shapes on
//! AVX2-capable x86 hosts. What each row class prepares and which
//! operations vectorize:
//!
//! - One-byte rows with `P <= 127` prepare the canonical coefficient and
//!   build two nibble tables per call; `mul_into`, `mul_assign`, and
//!   `mul_add` vectorize on `V4x`/`V3GfniCrypto`/`V3` through
//!   [`single_row_backend`].
//! - Four-byte rows prepare the Shoup pair
//!   `(c, floor(c * 2^32 / P))`; `mul_into`, `mul_add`, and `mul_assign`
//!   vectorize on `V4x`/`V3GfniCrypto`/`V3` through [`single_row_backend`].
//! - Eight-byte rows with odd `P` prepare the Montgomery word
//!   `c * 2^64 mod P`; fixed-coefficient operations route through Montgomery
//!   REDC on `V4x`/`V3GfniCrypto`/`V3` through [`single_row_backend`].
//! - Four-byte rows delay reduction across shared sources in prepared
//!   four-output matrices at or above the delayed source count in
//!   [`DELAYED_MIN_SOURCES`] and rows of at least 256 bytes on
//!   `V4x`/`V3GfniCrypto`/`V3` at or above each disposition's depth floor
//!   in [`DELAYED_INTO_MIN_DEPTH`] and [`DELAYED_ADD_MIN_DEPTH`]; every
//!   other shape uses the repeated single-row path.
//! - Four-byte rows vectorize `mul_elementwise` and
//!   `mul_elementwise_assign` through the narrow Barrett form on
//!   `V4x`/`V3GfniCrypto`/`V3`, selected by [`single_row_backend`].
//!
//! [`FieldKernels::backend`] reports the highest implemented vector tier for
//! rows whose prepared fixed-coefficient operations vectorize: `V4x` with
//! `simd512` on capable hosts, otherwise `V3` on AVX2-capable hosts. Other
//! widths report [`Backend::Scalar`]. Short rows use the measured AVX2 or
//! portable paths; routing evidence is recorded in `benchmarks/v3/residue.md`.
//! Preparation computes one `u64` value with `u128` arithmetic, allocates
//! nothing, and needs no `alloc`; every packed application runs
//! allocation-free, and no workspace carries meaning between calls.
//!
//! Byte-preserving shortcuts: a zero coefficient fills zero lanes in
//! overwrite and leaves the destination untouched in accumulation; a unit
//! coefficient copies the source bytes unchanged in overwrite, adds them
//! in accumulation, and leaves the destination untouched in place.
//! Zero accumulation, unit overwrite, and unit in-place execution preserve
//! existing or copied noncanonical bytes. Additive operations, including
//! unit accumulation, canonicalize the lanes they write. General coefficients
//! decode, multiply, and encode canonical lanes. Vector arms preserve these
//! same rules.
//!
//! Broadcast add/sub route through the modular path for every
//! characteristic. The characteristic-two XOR default is a binary-field
//! convention; a residue row's lanes hold canonical residues at the
//! presentation's byte width, and XOR broadcast over them would leave
//! noncanonical lane words behind at `P = 2`.

use crate::field::Elem;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::field::FieldBuffer;
use crate::field::finite_field::{FiniteField, Residue};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::backend;
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
use crate::kernel::x86::residue::avx512;
use crate::kernel::{Backend, FieldKernels, KernelDispatch, RawDispatch, prime, scalar};

/// The backend-ready form of one residue coefficient: exactly one `u64`
/// word, so `Coeff<F>` costs what `Elem<F>` costs.
///
/// Contents are decided at compile time from the row: the canonical `c`
/// for byte and two-byte rows and `P = 2`, the Shoup pair
/// `c | (floor(c * 2^32 / P) << 32)` for four-byte rows, and the
/// Montgomery word `c * 2^64 mod P` for eight-byte odd rows.
///
/// `CoeffVec`/`CoeffMatrix` store `Box<[PreparedResidue]>`, so this
/// one-word proof covers their entries too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedResidue {
    pub(crate) word: u64,
}

/// The prepared entry is exactly one word on every row class.
const _: () = assert!(core::mem::size_of::<PreparedResidue>() == core::mem::size_of::<u64>());

impl PreparedResidue {
    /// Rebuild the prepared word from its raw `u64`: the Shoup pair
    /// `c | (floor(c * 2^32 / P) << 32)` on four-byte rows. Direct-kernel
    /// oracle construction only; production paths resolve through the
    /// dispatch preparation instead.
    // A lib-only build constructs it nowhere else: the direct-kernel
    // oracle tests build their terms through it.
    #[allow(dead_code)]
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        Self { word }
    }

    /// Resolve the canonical `c` into this row's word, with `u128`
    /// arithmetic only. Every truncation is exact: Montgomery words sit
    /// below `P` and Shoup quotients below `2^32`.
    #[allow(clippy::cast_possible_truncation)]
    const fn prepare<const P: u64>(bytes: usize, c: u64) -> Self {
        if bytes == 4 {
            let q = (c as u128 * (1u128 << 32) / P as u128) as u64;
            Self {
                word: c | (q << 32),
            }
        } else if bytes == 8 && P % 2 == 1 {
            Self {
                word: ((c as u128 * (1u128 << 64) % P as u128) as u64),
            }
        } else {
            Self { word: c }
        }
    }

    /// Recover the canonical `c`: the REDC of the doubleword Montgomery
    /// word, or the low half of the Shoup pair.
    const fn coeff<const P: u64>(self, bytes: usize) -> u64 {
        if bytes == 8 && P % 2 == 1 {
            // REDC of `word * 1` in the doubleword form.
            redc64_word(self.word as u128, P)
        } else if bytes == 4 {
            self.word & 0xFFFF_FFFF
        } else if bytes == 8 {
            self.word
        } else if bytes == 2 {
            self.word & 0xFFFF
        } else {
            self.word & 0xFF
        }
    }
}

/// Scalar word REDC64 of a numerator below `P * 2^64`: the one-word
/// form of the doubleword vector kernel's reduction.
#[allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]
const fn redc64_word(t: u128, p: u64) -> u64 {
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        i += 1;
    }
    let ninv = inv.wrapping_neg();
    let m = (t as u64).wrapping_mul(ninv);
    let (num, carry) = t.overflowing_add(m as u128 * p as u128);
    let u = (num >> 64) as u64;
    if carry || u >= p {
        u.wrapping_sub(p)
    } else {
        u
    }
}

impl<const P: u64> FieldKernels for FiniteField<P, 1, Residue> {
    #[inline]
    fn backend() -> Backend {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // Reports the fixed-coefficient multiplication strategy. Exact
            // prime add/sub can vectorize while this summary is Scalar.
            if routes_vector(<Self as FieldBuffer>::BYTES, P) {
                vector_tier()
            } else {
                Backend::Scalar
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            Backend::Scalar
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            Self::BYTES == 4 && vector_backend()
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            false
        }
    }
}

/// Whether this row's prepared fixed-coefficient path can vectorize:
/// byte rows at `P <= 127`, any four-byte row, eight-byte odd rows.
/// This predicate only names the classes the kernels cover. The routing
/// evidence is recorded under `benchmarks/v3/residue.md`. The benchmark
/// routing report calls this instead of copying the class list.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn routes_vector(bytes: usize, p: u64) -> bool {
    (bytes == 1 && p <= 127) || bytes == 4 || (bytes == 8 && p % 2 == 1)
}

/// Shortest AVX2 overwrite row, in lanes, for this width class.
/// The paired measurement is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[allow(clippy::match_same_arms)]
pub const fn vector_floor_into_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 64,
        8 => 8,
        _ => 8,
    }
}
/// Shortest AVX2 in-place-scale row, in lanes, for this width class.
/// The paired measurement is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn vector_floor_assign_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 64,
        8 => 16,
        _ => 8,
    }
}
/// Shortest AVX2 accumulation row, in lanes, for this width class.
/// The paired measurement is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[allow(clippy::match_same_arms)]
pub const fn vector_floor_add_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 8,
        8 => 32,
        _ => 8,
    }
}
/// The single-row and elementwise operation the routing floor selects.
/// `Assign` is the in-place scale; `Elementwise` and `ElementwiseAssign`
/// are the raw-word element products. Dispatch matches on this instead of
/// threading positional flags through every selector call.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorOp {
    /// Overwrite: `mul_into`.
    Into,
    /// Accumulation: `mul_add`.
    Add,
    /// In-place scale: `mul_assign`.
    Assign,
    /// Raw-word product: `mul_elementwise`.
    Elementwise,
    /// In-place raw-word product: `mul_elementwise_assign`.
    ElementwiseAssign,
}
/// Shortest AVX2 elementwise row, in lanes, for this operation.
/// The paired measurement is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn vector_floor_elementwise_lanes(op: VectorOp) -> usize {
    match op {
        VectorOp::Elementwise => 16,
        VectorOp::ElementwiseAssign => 8,
        VectorOp::Into | VectorOp::Add | VectorOp::Assign => {
            panic!("elementwise lanes have no single-row operation")
        }
    }
}
/// Shortest AVX2 row, in bytes, for this width class and operation.
/// The paired measurement is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn avx2_floor_op_len(bytes: usize, op: VectorOp) -> usize {
    let lanes = match op {
        VectorOp::Add => vector_floor_add_lanes(bytes),
        VectorOp::Assign => vector_floor_assign_lanes(bytes),
        VectorOp::Into => vector_floor_into_lanes(bytes),
        VectorOp::Elementwise | VectorOp::ElementwiseAssign => vector_floor_elementwise_lanes(op),
    };
    lanes * bytes
}
/// Whether this host's process backend serves the residue vector bodies:
/// one of the AVX2-capable x86 tiers. Dispatch and the benchmark routing
/// report share this instead of repeating the tier list.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub fn vector_backend() -> bool {
    vector_tier() != Backend::Scalar
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn vector_tier() -> Backend {
    match backend() {
        #[cfg(feature = "simd512")]
        Backend::V4x => Backend::V4x,
        Backend::V3GfniCrypto | Backend::V3 => Backend::V3,
        _ => Backend::Scalar,
    }
}
/// Select the actual single-row kernel tier for this geometry and operation.
///
/// AVX-512 requires `simd512` and a capable process backend. Shorter rows
/// retain their AVX2 or portable route. Elementwise vectors require u32
/// lanes. Coefficient zero/unit shortcuts apply before this selector.
/// Routing evidence is recorded in `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[inline]
pub fn single_row_backend(bytes: usize, p: u64, len: usize, op: VectorOp) -> Backend {
    if !routes_vector(bytes, p)
        || (matches!(op, VectorOp::Elementwise | VectorOp::ElementwiseAssign) && bytes != 4)
    {
        return Backend::Scalar;
    }
    let tier = vector_tier();
    if tier == Backend::Scalar {
        return Backend::Scalar;
    }
    #[cfg(feature = "simd512")]
    {
        let floor = match (bytes, op) {
            (4, VectorOp::Assign) => 1024,
            (8, VectorOp::Assign) => 256,
            _ => 64,
        };
        if tier == Backend::V4x && len >= floor {
            return Backend::V4x;
        }
    }
    if len >= avx2_floor_op_len(bytes, op) {
        Backend::V3
    } else {
        Backend::Scalar
    }
}

/// Sources per delayed group: `K = floor(2^32 / P)`, the largest count
/// whose 64-bit products stay below `2^32 * P`. Four-byte rows hold
/// `P < 2^32`, so the quotient fits the delay depth. The benchmark
/// routing report calls this instead of recomputing the depth.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub const fn delay_depth<const P: u64>() -> u64 {
    ((1u128 << 32) / P as u128) as u64
}

/// Whether delayed accumulation reproduces the repeated single-row
/// composition byte for byte for this coefficient matrix.
///
/// The fallback composes one dispatched `mul_add` per (term, row) pair.
/// Zero coefficients preserve incoming bytes, and nonzero coefficients
/// produce canonical lanes, including unit coefficients routed through addition.
/// An output row whose coefficients are all zero therefore keeps
/// its incoming bytes exactly — aliases included — but the delayed
/// accumulate kernel normalizes the destination up front and always
/// stores canonical lanes. The two agree on every input iff no output
/// row is all-zero, i.e. every output row has at least one nonzero
/// coefficient. The scan reads the original elements only and allocates
/// nothing.
///
/// Overwrite needs no such check: its composition zeroes the rows first,
/// so every all-zero row holds canonical zeros and both paths store the
/// same canonical sums without reading the destination.
///
/// The caller has already required a nonempty row count.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn delayed_add_matches_composition<const P: u64>(
    nrows: usize,
    values: &[Elem<FiniteField<P, 1, Residue>>],
) -> bool {
    debug_assert!(nrows > 0);
    let nterms = values.len() / nrows;
    (0..nrows).all(|j| (0..nterms).any(|t| !values[t * nrows + j].is_zero()))
}

/// Whether the delayed tile covers this call: exactly four outputs, rows
/// of at least 256 bytes, a depth at or above the disposition's floor in
/// [`DELAYED_INTO_MIN_DEPTH`] or [`DELAYED_ADD_MIN_DEPTH`], and at least
/// [`DELAYED_MIN_SOURCES`] sources, with the accumulation composition
/// check at the call site. Shallower depths, fewer sources, and every
/// other shape stay on the repeated single-row path. The routing evidence
/// is recorded under `benchmarks/v3/residue.md`. The benchmark routing
/// report calls this instead of copying the floors.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn routes_delayed(
    nrows: usize,
    k: u64,
    row_len: usize,
    nterms: usize,
    add: bool,
) -> bool {
    let floor = if add {
        DELAYED_ADD_MIN_DEPTH
    } else {
        DELAYED_INTO_MIN_DEPTH
    };
    nrows == 4 && k >= floor && row_len >= 256 && nterms >= DELAYED_MIN_SOURCES
}

/// Fewest sources routed into the delayed tile, set by the paired
/// measurement recorded under `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_MIN_SOURCES: usize = 8;

/// Smallest delay depth routed for overwrite, set by the paired measurement
/// recorded under `benchmarks/v3/residue.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_INTO_MIN_DEPTH: u64 = 4;

/// Smallest delay depth routed for accumulation, set by the same record.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_ADD_MIN_DEPTH: u64 = 4;

/// Select the delayed-matrix kernel tier for this geometry and disposition.
///
/// `Scalar` selects the repeated single-row composition instead of a delayed
/// tile. Accumulation additionally checks byte-preserving zero coefficients
/// at the call site. The benchmark report uses this same selector.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub fn delayed_matrix_backend(
    nrows: usize,
    k: u64,
    row_len: usize,
    nterms: usize,
    add: bool,
) -> Backend {
    if routes_delayed(nrows, k, row_len, nterms, add) {
        vector_tier()
    } else {
        Backend::Scalar
    }
}

/// Smallest Mersenne31-modulus row, in bytes, whose addition reuses the
/// Mersenne31 vector kernels. Set by the paired composition record in
/// `benchmarks/v3/compositions.md`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const MERSENNE31_ADD_MIN_BYTES: usize = 60;

/// Smallest Goldilocks-modulus row, in bytes, whose addition reuses the
/// Goldilocks AVX2 and AVX-512 kernels. Set by the same record.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const GOLDILOCKS_ADD_MIN_BYTES: usize = 248;

/// Whether an exact-modulus row of `len` bytes reuses the specialized
/// prime field's addition kernels. Scalar tiers keep the residue loop;
/// Goldilocks-modulus rows also keep it on SSE4.2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn reuses_prime_addition<const P: u64>(len: usize) -> bool {
    match P {
        0x7fff_ffff => backend() != Backend::Scalar && len >= MERSENNE31_ADD_MIN_BYTES,
        0xffff_ffff_0000_0001 => {
            matches!(
                backend(),
                Backend::V4x | Backend::V3GfniCrypto | Backend::V3
            ) && len >= GOLDILOCKS_ADD_MIN_BYTES
        }
        _ => false,
    }
}

/// Portable builds keep the residue loop for every modulus.
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
const fn reuses_prime_addition<const P: u64>(_len: usize) -> bool {
    false
}

impl<const P: u64> KernelDispatch for FiniteField<P, 1, Residue> {
    type Prepared = PreparedResidue;

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared {
        PreparedResidue::prepare::<P>(<Self as crate::field::FieldBuffer>::BYTES, coeff.to_raw())
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self> {
        Elem::<Self>::from_raw(prepared.coeff::<P>(<Self as crate::field::FieldBuffer>::BYTES))
    }

    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match P {
            0x7fff_ffff if reuses_prime_addition::<P>(dst.len()) => {
                <crate::Mersenne31 as KernelDispatch>::add_assign(RawDispatch, dst, src);
            }
            0xffff_ffff_0000_0001 if reuses_prime_addition::<P>(dst.len()) => {
                <crate::Goldilocks as KernelDispatch>::add_assign(RawDispatch, dst, src);
            }
            _ => prime::add_assign::<Self>(dst, src),
        }
    }

    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        match P {
            0x7fff_ffff if reuses_prime_addition::<P>(dst.len()) => {
                <crate::Mersenne31 as KernelDispatch>::sub_assign(RawDispatch, dst, src);
            }
            0xffff_ffff_0000_0001 if reuses_prime_addition::<P>(dst.len()) => {
                <crate::Goldilocks as KernelDispatch>::sub_assign(RawDispatch, dst, src);
            }
            _ => prime::sub_assign::<Self>(dst, src),
        }
    }

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        mul_add_prepared::<P>(dst, *coeff, src);
    }

    fn mul_add_unprepared(_proof: RawDispatch, dst: &mut [u8], coeff: Elem<Self>, src: &[u8]) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if single_row_backend(
            <Self as crate::field::FieldBuffer>::BYTES,
            P,
            dst.len(),
            VectorOp::Add,
        ) != Backend::Scalar
        {
            Self::mul_add(RawDispatch, dst, &Self::prepare(RawDispatch, coeff), src);
            return;
        }
        prime::mul_add::<Self>(dst, coeff, src);
    }

    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        mul_into_prepared::<P>(dst, *coeff, src);
    }

    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared) {
        mul_assign_prepared::<P>(dst, *coeff);
    }

    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    ) {
        crate::kernel::fallback::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        crate::kernel::fallback::mul_add_gather::<Self>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        crate::kernel::fallback::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
    }
    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_possible_truncation)]
    fn mul_add_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let _ = values;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let selected =
                delayed_matrix_backend(nrows, delay_depth::<P>(), row_len, srcs.len(), true);
            if Self::BYTES == 4
                && selected != Backend::Scalar
                && delayed_add_matches_composition::<P>(nrows, values)
            {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows,
                    sources: srcs,
                };
                match selected {
                    #[cfg(feature = "simd512")]
                    Backend::V4x => {
                        avx512::matrix::mul_add_with::<P>(rows, row_len, nrows, &terms);
                    }
                    _ => crate::kernel::x86::residue::mul_add_matrix_u32_avx2_with(
                        crate::kernel::x86_v3_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                        P as u32,
                    ),
                }
                return;
            }
        }
        crate::kernel::fallback::mul_add_matrix_plan::<Self>(rows, row_len, nrows, coeffs, srcs);
    }

    #[cfg(feature = "alloc")]
    #[allow(clippy::cast_possible_truncation)]
    fn mul_into_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let selected =
                delayed_matrix_backend(nrows, delay_depth::<P>(), row_len, srcs.len(), false);
            if Self::BYTES == 4 && selected != Backend::Scalar {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows,
                    sources: srcs,
                };
                match selected {
                    #[cfg(feature = "simd512")]
                    Backend::V4x => {
                        avx512::matrix::mul_into_with::<P>(rows, row_len, nrows, &terms);
                    }
                    _ => crate::kernel::x86::residue::mul_into_matrix_u32_avx2_with(
                        crate::kernel::x86_v3_token(),
                        rows,
                        row_len,
                        nrows,
                        &terms,
                        P as u32,
                    ),
                }
                return;
            }
        }
        for row in rows.chunks_exact_mut(row_len).take(nrows) {
            row.fill(0);
        }
        Self::mul_add_matrix_plan(RawDispatch, rows, row_len, nrows, values, coeffs, srcs);
    }

    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        match single_row_backend(Self::BYTES, P, dst.len(), VectorOp::Elementwise) {
            #[cfg(feature = "simd512")]
            Backend::V4x => return avx512::single::mul_elementwise::<P>(dst, a, b),
            Backend::V3 => {
                crate::kernel::x86::residue::mul_elementwise_u32_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    a,
                    b,
                    Self::U32_PARAMS,
                );
                return;
            }
            _ => {}
        }
        prime::mul_elementwise::<Self>(dst, a, b);
    }

    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        match single_row_backend(Self::BYTES, P, dst.len(), VectorOp::ElementwiseAssign) {
            #[cfg(feature = "simd512")]
            Backend::V4x => return avx512::single::mul_elementwise_assign::<P>(dst, src),
            Backend::V3 => {
                crate::kernel::x86::residue::mul_elementwise_assign_u32_avx2(
                    crate::kernel::x86_v3_token(),
                    dst,
                    src,
                    Self::U32_PARAMS,
                );
                return;
            }
            _ => {}
        }
        scalar::mul_elementwise_assign::<Self>(dst, src);
    }

    /// Modular broadcast addition. The explicit override keeps every
    /// characteristic, `P = 2` included, off the binary XOR default.
    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::add_assign_scalar::<Self>(dst, Self::prepared_coeff(RawDispatch, value));
    }

    /// Modular broadcast subtraction, for the same reason.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared) {
        prime::sub_assign_scalar::<Self>(dst, Self::prepared_coeff(RawDispatch, value));
    }
}

/// Route one prepared fixed-coefficient overwrite through the vector path
/// when this row and host qualify, else the portable path.
#[allow(clippy::cast_possible_truncation, clippy::needless_return)]
fn mul_into_prepared<const P: u64>(dst: &mut [u8], coeff: PreparedResidue, src: &[u8]) {
    type Row<const Q: u64> = FiniteField<Q, 1, Residue>;
    let c = Row::<P>::prepared_coeff(RawDispatch, &coeff);
    if c.is_zero() {
        dst.fill(0);
        return;
    }
    if c.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        VectorOp::Into,
    ) {
        #[cfg(feature = "simd512")]
        Backend::V4x => return avx512::single::mul_into_with::<P>(dst, coeff, src),
        Backend::V3 => {
            let token = crate::kernel::x86_v3_token();
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 1 && P <= 127 {
                crate::kernel::x86::residue::mul_into_u8_avx2(
                    token,
                    dst,
                    src,
                    c.to_raw() as u8,
                    P as u8,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 4 {
                crate::kernel::x86::residue::mul_into_u32_avx2(
                    token,
                    dst,
                    src,
                    (coeff.word & 0xFFFF_FFFF) as u32,
                    (coeff.word >> 32) as u32,
                    P as u32,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 8 && P % 2 == 1 {
                crate::kernel::x86::residue::mul_into_u64_avx2(token, dst, src, coeff.word, P);
                return;
            }
        }
        _ => {}
    }
    prime::mul_into::<Row<P>>(dst, c, src);
}

/// Route one prepared AXPY through the vector path when qualified.
#[allow(clippy::cast_possible_truncation, clippy::needless_return)]
fn mul_add_prepared<const P: u64>(dst: &mut [u8], coeff: PreparedResidue, src: &[u8]) {
    type Row<const Q: u64> = FiniteField<Q, 1, Residue>;
    let c = Row::<P>::prepared_coeff(RawDispatch, &coeff);
    if c.is_zero() {
        return;
    }
    if c.is_one() {
        Row::<P>::add_assign(RawDispatch, dst, src);
        return;
    }
    // Canonical zero and unit comparisons exclude both trivial coefficients.
    #[cfg(test)]
    let _general_trace = crate::kernel::trace::term(crate::kernel::trace::Term::General);
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        VectorOp::Add,
    ) {
        #[cfg(feature = "simd512")]
        Backend::V4x => return avx512::single::mul_add_with::<P>(dst, coeff, src),
        Backend::V3 => {
            let token = crate::kernel::x86_v3_token();
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 1 && P <= 127 {
                crate::kernel::x86::residue::mul_add_u8_avx2(
                    token,
                    dst,
                    src,
                    c.to_raw() as u8,
                    P as u8,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 4 {
                crate::kernel::x86::residue::mul_add_u32_avx2(
                    token,
                    dst,
                    src,
                    (coeff.word & 0xFFFF_FFFF) as u32,
                    (coeff.word >> 32) as u32,
                    P as u32,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 8 && P % 2 == 1 {
                crate::kernel::x86::residue::mul_add_u64_avx2(token, dst, src, coeff.word, P);
                return;
            }
        }
        _ => {}
    }
    prime::mul_add::<Row<P>>(dst, c, src);
}

/// Route one prepared in-place scale through the vector path when qualified.
#[allow(clippy::cast_possible_truncation, clippy::needless_return)]
fn mul_assign_prepared<const P: u64>(dst: &mut [u8], coeff: PreparedResidue) {
    type Row<const Q: u64> = FiniteField<Q, 1, Residue>;
    let c = Row::<P>::prepared_coeff(RawDispatch, &coeff);
    if c.is_one() {
        return;
    }
    if c.is_zero() {
        dst.fill(0);
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        VectorOp::Assign,
    ) {
        #[cfg(feature = "simd512")]
        Backend::V4x => return avx512::single::mul_assign_with::<P>(dst, coeff),
        Backend::V3 => {
            let token = crate::kernel::x86_v3_token();
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 1 && P <= 127 {
                crate::kernel::x86::residue::mul_assign_u8_avx2(
                    token,
                    dst,
                    c.to_raw() as u8,
                    P as u8,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 4 {
                crate::kernel::x86::residue::mul_assign_u32_avx2(
                    token,
                    dst,
                    (coeff.word & 0xFFFF_FFFF) as u32,
                    (coeff.word >> 32) as u32,
                    P as u32,
                );
                return;
            }
            if <Row<P> as crate::field::FieldBuffer>::BYTES == 8 && P % 2 == 1 {
                crate::kernel::x86::residue::mul_assign_u64_avx2(token, dst, coeff.word, P);
                return;
            }
        }
        _ => {}
    }
    prime::mul_assign::<Row<P>>(dst, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDE: u64 = 18_446_744_073_709_551_557;

    /// Lane bytes encoding `P + (i mod (lane_max - P + 1))` per width, so
    /// every tested width holds genuine noncanonical lanes.
    fn alias_src<const P: u64>(bytes: usize) -> [u8; 64] {
        let lane_max = u64::MAX >> (64 - 8 * bytes);
        let span = lane_max - P + 1;
        let mut out = [0u8; 64];
        let lanes = 64 / bytes;
        let mut i = 0;
        while i < lanes {
            let v = P + (i as u64 % span);
            out[i * bytes..(i + 1) * bytes].copy_from_slice(&v.to_le_bytes()[..bytes]);
            i += 1;
        }
        out
    }

    /// The routed single-row entries apply the portable zero and unit
    /// shortcuts byte for byte, on every width class, before any vector
    /// dispatch.
    #[test]
    fn routed_single_row_shortcuts_match_the_portable_bytes() {
        fn check<const P: u64>() {
            type Row<const Q: u64> = FiniteField<Q, 1, Residue>;
            let bytes = <Row<P> as crate::field::FieldBuffer>::BYTES;
            let len = 8 * bytes;
            let full = alias_src::<P>(bytes);
            let src = &full[..len];
            // Destinations start from the alias source rotated by one lane:
            // every width class keeps noncanonical lanes in place.
            let mut start = [0u8; 64];
            for (i, lane) in start[..len].chunks_exact_mut(bytes).enumerate() {
                let s = ((i + 1) % (len / bytes)) * bytes;
                lane.copy_from_slice(&src[s..s + bytes]);
            }
            for c in [Elem::<Row<P>>::ZERO, Elem::<Row<P>>::ONE] {
                let prepared = Row::<P>::prepare(RawDispatch, c);

                let (mut got, mut want) = (start, start);
                mul_into_prepared::<P>(&mut got[..len], prepared, src);
                prime::mul_into::<Row<P>>(&mut want[..len], c, src);
                assert_eq!(got, want, "mul_into P={P} c={}", c.to_raw());

                let (mut got, mut want) = (start, start);
                mul_add_prepared::<P>(&mut got[..len], prepared, src);
                prime::mul_add::<Row<P>>(&mut want[..len], c, src);
                assert_eq!(got, want, "mul_add P={P} c={}", c.to_raw());

                let (mut got, mut want) = (start, start);
                mul_assign_prepared::<P>(&mut got[..len], prepared);
                prime::mul_assign::<Row<P>>(&mut want[..len], c);
                assert_eq!(got, want, "mul_assign P={P} c={}", c.to_raw());
            }
        }
        check::<101>();
        check::<65_521>();
        check::<1_000_000_007>();
        check::<WIDE>();
    }

    /// `redc64_word` returns `t * 2^-64 mod P`, canonical, across the whole
    /// accepted numerator range, including numerators whose reduction
    /// needs the final subtraction.
    #[test]
    fn redc64_word_is_the_canonical_montgomery_reduction() {
        let p = u128::from(WIDE);
        let top = p << 64;
        let numerators = [
            0,
            1,
            p - 1,
            p,
            1u128 << 64,
            (p - 1) << 64,
            top - 1,
            top - p,
            top / 2,
            top / 3 + 12_345,
        ];
        for t in numerators {
            let u = redc64_word(t, WIDE);
            assert!(u < WIDE, "t={t}");
            // u * 2^64 ≡ t (mod P), checked without the reduction under test.
            let lhs = (u128::from(u) % p) * ((1u128 << 64) % p) % p;
            let rhs = t % p;
            assert_eq!(lhs, rhs, "t={t}");
        }
    }

    /// Every row class recovers the exact coefficient it prepared.
    #[test]
    fn prepared_word_round_trips_the_coefficient() {
        fn check<const P: u64>(bytes: usize) {
            for c in [0, 1, 2, P / 2, P - 1] {
                let prepared = PreparedResidue::prepare::<P>(bytes, c);
                assert_eq!(prepared.coeff::<P>(bytes), c, "P={P} c={c}");
            }
        }
        check::<2>(8);
        check::<251>(1);
        check::<65_521>(2);
        check::<1_000_000_007>(4);
        check::<4_294_967_291>(4);
        check::<WIDE>(8);
        check::<4_294_967_311>(8);
    }
}
