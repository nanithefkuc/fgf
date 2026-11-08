//! Residue prime-field kernel dispatch.
//!
//! Every `FiniteField<P, 1, Residue>` row runs the portable
//! [`crate::kernel::prime`] path, except the vectorized shapes on
//! AVX2-capable x86 hosts. What each row class prepares and which
//! operations vectorize:
//!
//! - One-byte rows with `P <= 127` prepare the canonical coefficient and
//!   build two nibble tables per call; `mul_into`, `mul_assign`, and
//!   `mul_add` vectorize through the single-row selector in
//!   [`crate::kernel::routing`].
//! - Four-byte rows prepare the Shoup pair
//!   `(c, floor(c * 2^32 / P))`; `mul_into`, `mul_add`, and `mul_assign`
//!   vectorize through the same selector.
//! - Eight-byte rows with odd `P` prepare the Montgomery word
//!   `c * 2^64 mod P`; fixed-coefficient operations route through Montgomery
//!   REDC through the same selector.
//! - Four-byte rows delay reduction across shared sources in prepared
//!   four-output matrices at or above the delayed source count and row
//!   floor in [`crate::kernel::routing`], at or above each disposition's
//!   depth floor; every other shape uses the repeated single-row path.
//! - Four-byte rows vectorize `mul_elementwise` and
//!   `mul_elementwise_assign` through the narrow Barrett form, selected by
//!   the same selector.
//!
//! the backend query reports the highest implemented vector tier for
//! rows whose prepared fixed-coefficient operations vectorize: the AVX-512
//! class with `simd512` on capable hosts, otherwise the AVX2 class on
//! AVX2-capable hosts. Other widths report [`Backend::Scalar`]. Short rows
//! use the measured AVX2 or portable paths; routing evidence is the 3.0.2
//! record at
//! <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
//! Preparation computes one `u64` value with `u128` arithmetic, allocates
//! nothing, and needs no `alloc`; every packed application runs
//! allocation-free, and no workspace carries meaning between calls.
//!
//! Byte-preserving shortcuts: a zero coefficient fills zero lanes in
//! overwrite and leaves the destination untouched in accumulation; a unit
//! coefficient copies the source bytes unchanged in overwrite, adds them
//! in accumulation, and leaves the destination untouched in place. Those
//! three paths preserve noncanonical lane bytes instead of canonicalizing
//! them; every other coefficient decodes, multiplies, and encodes
//! canonical lanes. The vector arms apply the same shortcuts before
//! dispatch, so routed output matches the portable composition byte for
//! byte on every input.
//!
//! Broadcast add/sub route through the modular path for every
//! characteristic. The characteristic-two XOR default is a binary-field
//! convention; a residue row's lanes hold canonical residues at the
//! presentation's byte width, and XOR broadcast over them would leave
//! noncanonical lane words behind at `P = 2`.
//!
//! Every dispatch arm matches a selector from [`crate::kernel::routing`],
//! which folds the call's geometry into the cached residue strategy; no
//! arm reads the process backend.
use crate::field::Elem;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::field::FieldBuffer;
use crate::field::dispatch::{KernelDispatch, RawDispatch};
use crate::field::finite_field::{FiniteField, Residue};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::ResidueStrategy;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::routing;
use crate::kernel::{Backend, prime, scalar};

#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
use crate::kernel::x86::residue::avx512;

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

impl<const P: u64> KernelDispatch for FiniteField<P, 1, Residue> {
    type Prepared = PreparedResidue;

    #[inline]
    fn backend(_proof: RawDispatch) -> Backend {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // Reports the implemented tier for nibble-table u8 rows,
            // every u32 row, and u64 rows at odd moduli.
            if routing::routes_vector(<Self as crate::field::FieldBuffer>::BYTES, P) {
                crate::kernel::residue_strategy().report()
            } else {
                Backend::Scalar
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            Backend::Scalar
        }
    }

    fallback_defaults!(
        row: add_gather_offsets, mul_add_scatter_plan, mul_add_gather_plan,
        mul_into_gather, mul_into_gather_plan, mul_into_matrix,
        mul_add_matrix_at, mul_add_matrix_at_plan, mul_into_matrix_at,
        mul_into_matrix_at_plan,
    );

    #[inline]
    fn prepare(_proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared {
        PreparedResidue::prepare::<P>(<Self as crate::field::FieldBuffer>::BYTES, coeff.to_raw())
    }

    #[inline]
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self> {
        Elem::<Self>::from_raw(prepared.coeff::<P>(<Self as crate::field::FieldBuffer>::BYTES))
    }

    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        prime::add_assign::<Self>(dst, src);
    }

    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]) {
        prime::sub_assign::<Self>(dst, src);
    }

    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        mul_add_prepared::<P>(dst, *coeff, src);
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
        prime::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
    }

    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]) {
        prime::mul_add_gather::<Self>(dst, coeffs, srcs);
    }

    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    ) {
        prime::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
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
            let selected = routing::delayed_matrix_backend(
                nrows,
                routing::delay_depth::<P>(),
                row_len,
                srcs.len(),
                true,
            );
            if Self::BYTES == 4
                && selected != ResidueStrategy::Scalar
                && delayed_add_matches_composition::<P>(nrows, values)
            {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows,
                    sources: srcs,
                };
                match selected {
                    #[cfg(feature = "simd512")]
                    ResidueStrategy::Avx512 => {
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
        crate::kernel::fallback::mul_add_matrix_plan::<Self>(
            rows, row_len, nrows, values, coeffs, srcs,
        );
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
            let selected = routing::delayed_matrix_backend(
                nrows,
                routing::delay_depth::<P>(),
                row_len,
                srcs.len(),
                false,
            );
            if Self::BYTES == 4 && selected != ResidueStrategy::Scalar {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows,
                    sources: srcs,
                };
                match selected {
                    #[cfg(feature = "simd512")]
                    ResidueStrategy::Avx512 => {
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
        match routing::single_row_backend(Self::BYTES, P, dst.len(), routing::VectorOp::Elementwise)
        {
            #[cfg(feature = "simd512")]
            ResidueStrategy::Avx512 => return avx512::single::mul_elementwise::<P>(dst, a, b),
            ResidueStrategy::Avx2 => {
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
        match routing::single_row_backend(
            Self::BYTES,
            P,
            dst.len(),
            routing::VectorOp::ElementwiseAssign,
        ) {
            #[cfg(feature = "simd512")]
            ResidueStrategy::Avx512 => {
                return avx512::single::mul_elementwise_assign::<P>(dst, src);
            }
            ResidueStrategy::Avx2 => {
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

/// Whether delayed accumulation reproduces the repeated single-row
/// composition byte for byte for this coefficient matrix.
///
/// The fallback composes one portable `mul_add` per (term, row) pair, and
/// the portable `mul_add` leaves a row's bytes untouched when its
/// coefficient is zero while every other coefficient stores canonical
/// lanes. An output row whose coefficients are all zero therefore keeps
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
    match routing::single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        routing::VectorOp::Into,
    ) {
        #[cfg(feature = "simd512")]
        ResidueStrategy::Avx512 => return avx512::single::mul_into_with::<P>(dst, coeff, src),
        ResidueStrategy::Avx2 => {
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
        prime::add_assign::<Row<P>>(dst, src);
        return;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match routing::single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        routing::VectorOp::Add,
    ) {
        #[cfg(feature = "simd512")]
        ResidueStrategy::Avx512 => return avx512::single::mul_add_with::<P>(dst, coeff, src),
        ResidueStrategy::Avx2 => {
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
    match routing::single_row_backend(
        <Row<P> as crate::field::FieldBuffer>::BYTES,
        P,
        dst.len(),
        routing::VectorOp::Assign,
    ) {
        #[cfg(feature = "simd512")]
        ResidueStrategy::Avx512 => return avx512::single::mul_assign_with::<P>(dst, coeff),
        ResidueStrategy::Avx2 => {
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
