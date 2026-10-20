//! End-to-end differential tests for the token-gated `internals` surface.
//!
//! The in-crate kernel tests drive process-selected entries through dispatch.
//! These tests instead call each direct architecture entry as an external
//! `internals` consumer would: summon the exact capability token, skip a tier
//! when the host cannot prove it, and compare every observable byte against an
//! independent scalar oracle or per-lane field arithmetic.
//!
//! Destinations are seeded with nonzero noise before every overwrite
//! operation so an accumulation-into-destination bug cannot agree with the
//! overwrite oracle.

#![cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
#![allow(clippy::cast_possible_truncation)]
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use fgf::internals::field::fan_paar as wiedemann;
#[cfg(feature = "simd512")]
use fgf::internals::kernel::X64V4xToken;
use fgf::internals::kernel::gf8::Prepared;
use fgf::internals::kernel::scalar;
use fgf::internals::kernel::tables::{
    FpTowerTables, TowerCoeff, TowerTables, isomorphism_to_aes, scale_table,
};
use fgf::internals::kernel::{
    FlatMatrix, SimdToken, X64V2Token, X64V3GfniCryptoToken, X64V3Token, x86,
};
use fgf::poly::{AES, RS};
use fgf::{Elem, FiniteField, Goldilocks, Mersenne31, Polynomial, goldilocks};

type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf32 = fgf::FiniteField<2, 32, fgf::Extension<fgf::Rijndael32, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;
type FanPaar16Field = fgf::FiniteField<2, 16, fgf::Extension<fgf::FanPaar16, 2>>;
type FanPaar32Field = fgf::FiniteField<2, 32, fgf::Extension<fgf::FanPaar32, 2>>;
type FanPaar64Field = fgf::FiniteField<2, 64, fgf::Extension<fgf::FanPaar64, 2>>;

/// Deterministic LCG noise, the crate's shared convention.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Byte lengths straddling the 16/32-byte lane boundaries and tails.
#[cfg(not(miri))]
const LENGTHS8: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 64, 96, 130, 300];
/// Truncated under Miri to empty, tiny, lane-minus-one, lane, lane-plus-one,
/// and two lanes with a tail; the long rows only multiply interpreted bytes.
#[cfg(miri)]
const LENGTHS8: &[usize] = &[0, 1, 15, 16, 17, 32, 33];
/// Even lengths for GF(2^16)-shaped kernels.
#[cfg(not(miri))]
const LENGTHS16: &[usize] = &[0, 2, 16, 30, 32, 34, 64, 66, 128, 130, 300];
/// Truncated under Miri on the same terms as `LENGTHS8`.
#[cfg(miri)]
const LENGTHS16: &[usize] = &[0, 2, 30, 32, 34];
/// Multiples of four for GF(2^32) kernels.
#[cfg(not(miri))]
const LENGTHS32: &[usize] = &[0, 4, 16, 28, 32, 36, 64, 68, 128, 132, 300];
/// Truncated under Miri on the same terms as `LENGTHS8`.
#[cfg(miri)]
const LENGTHS32: &[usize] = &[0, 4, 28, 32, 36];
/// Multiples of eight for GF(2^64) kernels.
#[cfg(not(miri))]
const LENGTHS64: &[usize] = &[0, 8, 24, 32, 40, 64, 72, 128, 136, 296];
/// Truncated under Miri on the same terms as `LENGTHS8`.
#[cfg(miri)]
const LENGTHS64: &[usize] = &[0, 8, 24, 32, 40];

/// Row counts straddling the blocked kernels' four/two/one row groups.
#[cfg(not(miri))]
const ROW_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 8, 9];
/// Truncated under Miri to single, pair, full group, and group-plus-one: the
/// walk-transition counters are safe control flow, while every row window
/// keeps the offset-row residue.
#[cfg(miri)]
const ROW_COUNTS: &[usize] = &[1, 2, 4, 5];
#[cfg(not(miri))]
/// 617 bytes: six 96-byte three-lane tiles, one whole 32-byte lane, and a
/// sub-lane tail.
const ROW_LENS: &[usize] = &[2, 32, 34, 300, 617];
/// Truncated under Miri: the long row only multiplies interpreted bytes.
#[cfg(miri)]
const ROW_LENS: &[usize] = &[2, 32, 34];
#[cfg(not(miri))]
const EVEN_ROW_LENS: &[usize] = &[2, 30, 32, 34];
/// Truncated under Miri on the same terms as `ROW_LENS`.
#[cfg(miri)]
const EVEN_ROW_LENS: &[usize] = &[2, 32, 34];
#[cfg(not(miri))]
const NTERMS: &[usize] = &[1, 2, 3, 8, 9, 17];
/// Truncated under Miri to single, pair, and one count past the resolve
/// chunk, which keeps the scratch-staging seam.
#[cfg(miri)]
const NTERMS: &[usize] = &[1, 2, 9];

/// GF(2^8) coefficients: both short-circuits, the extremes, a spread.
fn gf8_coeffs() -> Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>> {
    #[cfg(not(miri))]
    let raw = [0u8, 1, 2, 3, 7, 0x53, 0xd3, 0xff];
    // Short-circuits, one ordinary value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u8, 1, 0x53, 0xff];
    raw.iter()
        .map(|&c| Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(c))
        .collect()
}

/// The Reed–Solomon field's coefficients, same byte spread under `0x11D`.
fn gf8_rs_coeffs() -> Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>> {
    #[cfg(not(miri))]
    let raw = [0u8, 1, 2, 3, 7, 0x53, 0xd3, 0xff];
    // Short-circuits, one ordinary value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u8, 1, 0x53, 0xff];
    raw.iter()
        .map(|&c| Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(c))
        .collect()
}

/// GF(2^16) coefficients: short-circuits, pure components, mixed, extremes.
fn gf16_coeffs() -> Vec<Elem<Gf16>> {
    #[cfg(not(miri))]
    let raw = [0u16, 1, 0x0100, 0x00ff, 0x1234, 0xbeef, 0x7411, 0xffff];
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u16, 1, 0x1234, 0xffff];
    raw.iter().map(|&c| Elem::<Gf16>::from_raw(c)).collect()
}

fn gf32_coeffs() -> Vec<Elem<Gf32>> {
    #[cfg(not(miri))]
    let raw = [0u32, 1, 0x0001_0000, 0x0000_ffff, 0x1234_5678, u32::MAX];
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u32, 1, 0x1234_5678, u32::MAX];
    raw.iter().map(|&c| Elem::<Gf32>::from_raw(c)).collect()
}

fn gf64_coeffs() -> Vec<Elem<Gf64>> {
    #[cfg(not(miri))]
    let raw = [
        0u64,
        1,
        0x0000_0001_0000_0000,
        0x0000_0000_ffff_ffff,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    ];
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u64, 1, 0x0123_4567_89ab_cdef, u64::MAX];
    raw.iter().map(|&c| Elem::<Gf64>::from_raw(c)).collect()
}

// ---------------------------------------------------------------------------
// Scalar oracles.
// ---------------------------------------------------------------------------

fn gf8_ref<const POLY: u128>(
    dst: &mut [u8],
    coeff: Elem<FiniteField<2, 8, Polynomial<POLY>>>,
    src: &[u8],
) {
    scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(dst, coeff, src);
}
fn gf16_ref(dst: &mut [u8], coeff: Elem<Gf16>, src: &[u8]) {
    scalar::mul_add::<Gf16>(dst, coeff, src);
}
fn gf32_ref(dst: &mut [u8], coeff: Elem<Gf32>, src: &[u8]) {
    scalar::mul_add::<Gf32>(dst, coeff, src);
}
fn gf64_ref(dst: &mut [u8], coeff: Elem<Gf64>, src: &[u8]) {
    scalar::mul_add::<Gf64>(dst, coeff, src);
}
/// `dst ^= coeff * src` through the independent Fan-Paar recurrence.
fn fp_buffer_mul_add(dst: &mut [u8], coeff: u64, src: &[u8], bits: u32) {
    let width = bits as usize / 8;
    for (d, s) in dst.chunks_exact_mut(width).zip(src.chunks_exact(width)) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(s);
        let product = wiedemann::fp_multiply(u64::from_le_bytes(word), coeff, bits);
        let mut acc = [0u8; 8];
        acc[..width].copy_from_slice(d);
        let summed = u64::from_le_bytes(acc) ^ product;
        d.copy_from_slice(&summed.to_le_bytes()[..width]);
    }
}

/// `dst *= coeff` through the independent Fan-Paar recurrence.
fn fp_buffer_mul_assign(dst: &mut [u8], coeff: u64, bits: u32) {
    let width = bits as usize / 8;
    for d in dst.chunks_exact_mut(width) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(d);
        let scaled = wiedemann::fp_multiply(u64::from_le_bytes(word), coeff, bits);
        d.copy_from_slice(&scaled.to_le_bytes()[..width]);
    }
}

fn fp16_ref(dst: &mut [u8], coeff: Elem<FanPaar16Field>, src: &[u8]) {
    fp_buffer_mul_add(dst, u64::from(coeff.to_raw()), src, 16);
}
fn fp32_ref(dst: &mut [u8], coeff: Elem<FanPaar32Field>, src: &[u8]) {
    fp_buffer_mul_add(dst, u64::from(coeff.to_raw()), src, 32);
}
fn fp64_ref(dst: &mut [u8], coeff: Elem<FanPaar64Field>, src: &[u8]) {
    fp_buffer_mul_add(dst, coeff.to_raw(), src, 64);
}
fn m31_ref(dst: &mut [u8], coeff: Elem<Mersenne31>, src: &[u8]) {
    scalar::mul_add::<Mersenne31>(dst, coeff, src);
}
fn gld_ref(dst: &mut [u8], coeff: Elem<Goldilocks>, src: &[u8]) {
    scalar::mul_add::<Goldilocks>(dst, coeff, src);
}

// ---------------------------------------------------------------------------
// Generic differential drivers.
// ---------------------------------------------------------------------------

fn check_mul_add<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        let src = noise(len, 0x51);
        for &coeff in coeffs {
            let mut got = noise(len, 0x62);
            let mut want = got.clone();
            kernel(&mut got, coeff, &src);
            reference(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_mul_assign<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E),
    kernel: impl Fn(&mut [u8], E),
) {
    for &len in lengths {
        for &coeff in coeffs {
            let mut got = noise(len, 0x73);
            let mut want = got.clone();
            kernel(&mut got, coeff);
            reference(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_mul_into<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        let src = noise(len, 0x84);
        for &coeff in coeffs {
            // Noise in the destination: a fused kernel must overwrite, not
            // accumulate. The oracle accumulates from zero, which the AXPY
            // reference expresses exactly.
            let mut got = noise(len, 0x95);
            let mut want = vec![0u8; len];
            kernel(&mut got, coeff, &src);
            reference(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_elementwise<F: fgf::FieldBuffer>(
    name: &str,
    lengths: &[usize],
    kernel: impl Fn(&mut [u8], &[u8], &[u8]),
) {
    for &len in lengths {
        let a = noise(len, 0x21);
        let b = noise(len, 0x32);
        let mut got = noise(len, 0x43);
        let mut want = got.clone();
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<F>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Scatter driver with surplus preservation: the rows buffer carries extra
/// trailing bytes and they must survive untouched.
fn check_scatter<E: Copy + core::fmt::Debug>(
    name: &str,
    row_lens: &[usize],
    reference: impl Fn(&mut [u8], E, &[u8]),
    coeff_at: impl Fn(usize) -> E,
    kernel: impl Fn(&mut [u8], usize, &[E], &[u8]),
) {
    const SURPLUS: usize = 19;
    for &row_len in row_lens {
        for &nrows in ROW_COUNTS {
            let src = noise(row_len, 0xb7);
            let coeffs: Vec<E> = (0..nrows).map(&coeff_at).collect();
            let mut got = noise(nrows * row_len + SURPLUS, 0xc8);
            let mut want = got.clone();

            kernel(&mut got, row_len, &coeffs, &src);
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                reference(row, coeff, &src);
            }
            assert_eq!(got, want, "{name}: row_len {row_len}, nrows {nrows}");
        }
    }
}

/// Gather driver: `dst ^= sum(coeffs[i] * srcs[i])` from a noise seed.
fn check_gather<E: Copy + core::fmt::Debug>(
    name: &str,
    row_len: usize,
    reference: impl Fn(&mut [u8], E, &[u8]),
    coeff_at: impl Fn(usize) -> E,
    kernel: impl Fn(&mut [u8], &[E], &[&[u8]]),
) {
    for &nterms in NTERMS {
        let sources: Vec<Vec<u8>> = (0..nterms)
            .map(|t| noise(row_len, 0x400 + t as u64))
            .collect();
        let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        let coeffs: Vec<E> = (0..nterms).map(&coeff_at).collect();

        let mut got = noise(row_len, 0xd9);
        let mut want = got.clone();
        kernel(&mut got, &coeffs, &srcs);
        for (&coeff, &src) in coeffs.iter().zip(&srcs) {
            reference(&mut want, coeff, src);
        }
        assert_eq!(got, want, "{name}: row_len {row_len}, terms {nterms}");
    }
}

/// Build one matrix case: identically seeded destination-with-surplus for
/// kernel and oracle, plus the term list.
fn matrix_case<E: Copy, R>(
    row_len: usize,
    nrows: usize,
    nterms: usize,
    coeff_at: &impl Fn(usize, usize) -> E,
    body: impl FnOnce(Vec<u8>, Vec<u8>, &[(&[E], &[u8])]) -> R,
) -> R {
    const SURPLUS: usize = 19;
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x500 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<E>> = (0..nterms)
        .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
        .collect();
    let terms: Vec<(&[E], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    body(
        noise(nrows * row_len + SURPLUS, 0xd9),
        noise(nrows * row_len + SURPLUS, 0xd9),
        &terms,
    )
}

/// Fold every term into the oracle with the scalar reference AXPY.
fn fold_terms<E: Copy>(
    want: &mut [u8],
    row_len: usize,
    terms: &[(&[E], &[u8])],
    reference: &impl Fn(&mut [u8], E, &[u8]),
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
            reference(row, coeff, src);
        }
    }
}

/// Accumulate matrix driver with surplus preservation.
fn check_matrix_accumulate<E: Copy>(
    name: &str,
    row_lens: &[usize],
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
) {
    for &row_len in row_lens {
        for &nrows in ROW_COUNTS {
            for &nterms in NTERMS {
                matrix_case(
                    row_len,
                    nrows,
                    nterms,
                    &coeff_at,
                    |mut got, mut want, terms| {
                        kernel(&mut got, row_len, nrows, terms);
                        fold_terms(&mut want, row_len, terms, &reference);
                        assert_eq!(
                            got, want,
                            "{name}: row_len {row_len}, nrows {nrows}, terms {nterms}"
                        );
                    },
                );
            }
        }
    }
}

/// Overwrite matrix driver: the destination is nonzero noise, the oracle
/// accumulates from zero over exactly the addressed rows, and the trailing
/// surplus must survive untouched.
fn check_matrix_overwrite<E: Copy>(
    name: &str,
    row_lens: &[usize],
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
) {
    for &row_len in row_lens {
        for &nrows in ROW_COUNTS {
            for &nterms in NTERMS {
                matrix_case(
                    row_len,
                    nrows,
                    nterms,
                    &coeff_at,
                    |mut got, mut want, terms| {
                        // The oracle accumulates from zero over the addressed
                        // rows and keeps the surplus noise.
                        want[..nrows * row_len].fill(0);
                        kernel(&mut got, row_len, nrows, terms);
                        fold_terms(&mut want, row_len, terms, &reference);
                        assert_eq!(
                            got, want,
                            "{name}: row_len {row_len}, nrows {nrows}, terms {nterms}"
                        );
                    },
                );
            }
        }
    }
}

fn gf8_coeff_at2<const POLY: u128>(
    t: usize,
    j: usize,
) -> Elem<FiniteField<2, 8, Polynomial<POLY>>> {
    Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw(((t * 31 + j * 29) % 256) as u8)
}

/// The low byte of a polynomial, the reduction the shift/reduce elementwise
/// entries thread, through the public polynomial constant.
fn reduction_low<const POLY: u128>() -> u8 {
    Polynomial::<POLY>::POLY.to_le_bytes()[0]
}

/// Resolve element coefficients to the prepared form the blocked entries take.
fn prepared<const POLY: u128>(
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
) -> Vec<Prepared> {
    coeffs.iter().map(|&c| Prepared::new(c)).collect()
}

/// Resolve raw GF(2^8) terms into the prepared coefficient pairs the blocked
/// matrix entries take; the sets stay alive across `body`.
macro_rules! prepared_pairs {
    ($terms:expr, |$pairs:ident| $body:expr) => {{
        let sets: Vec<Vec<Prepared>> = $terms.iter().map(|(coeffs, _)| prepared(coeffs)).collect();
        let $pairs: Vec<(&[Prepared], &[u8])> = sets
            .iter()
            .zip($terms.iter().map(|&(_, src)| src))
            .map(|(set, src)| (set.as_slice(), src))
            .collect();
        $body
    }};
}
fn gf16_coeff_at2(t: usize, j: usize) -> Elem<Gf16> {
    Elem::<Gf16>::from_raw(((t * 7919 + j * 613) % 65536) as u16)
}

// ---------------------------------------------------------------------------
// Field-independent byte kernels.
// ---------------------------------------------------------------------------

#[test]
fn proven_bytes_xor_matches_scalar() {
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let Some(v2) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE2 on this host");
        return;
    };

    for &len in LENGTHS8 {
        let src = noise(len, 0x11);
        let mut got = noise(len, 0x22);
        let mut want = got.clone();
        x86::bytes::xor_avx2(v3, &mut got, &src);
        scalar::xor(&mut want, &src);
        assert_eq!(got, want, "xor_avx2: len {len}");

        let mut got = noise(len, 0x33);
        let mut want = got.clone();
        x86::bytes::xor_sse2(v2.v1(), &mut got, &src);
        scalar::xor(&mut want, &src);
        assert_eq!(got, want, "xor_sse2: len {len}");
    }
}

#[test]
fn proven_bytes_xor_gather_matches_offsets() {
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let region = noise(512, 0x77);
    let live = 100;
    for offsets in [
        vec![0u32],
        vec![0u32, 32, 412],
        vec![7u32, 131, 260, 411],
        vec![],
    ] {
        let mut got = noise(live, 0x88);
        let mut want = got.clone();
        x86::bytes::xor_gather_avx2(v3, &region, &mut got, &offsets);
        for &start in &offsets {
            let start = start as usize;
            scalar::xor(&mut want, &region[start..start + live]);
        }
        assert_eq!(got, want, "xor_gather_avx2: offsets {offsets:?}");
    }
}

// ---------------------------------------------------------------------------
// GF(2^8) multiply kernels across tiers.
// ---------------------------------------------------------------------------

#[test]
fn proven_gf8_mul_kernels_match_scalar() {
    let Some(v3gfni) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let Some(v2) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE2 on this host");
        return;
    };

    let coeffs = gf8_coeffs();
    let rs_coeffs = gf8_rs_coeffs();
    check_mul_add(
        "gf8<0x11b>::mul_add_gfni",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_add_gfni::<true>(v3gfni, dst, Prepared::new(c), src),
    );
    check_mul_assign(
        "gf8<0x11b>::mul_assign_gfni",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>,
        |dst, c| x86::gf8::mul_assign_gfni::<true>(v3gfni, dst, Prepared::new(c)),
    );
    check_mul_into(
        "gf8<0x11b>::mul_into_gfni",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_into_gfni::<true>(v3gfni, dst, Prepared::new(c), src),
    );
    check_mul_add(
        "gf8<0x11b>::mul_add_avx2",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_add_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8<0x11b>::mul_assign_avx2",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>,
        |dst, c| x86::gf8::mul_assign_avx2(v3, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8<0x11b>::mul_into_avx2",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_into_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_add(
        "gf8<0x11b>::mul_add_ssse3",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_add_ssse3(v2, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8<0x11b>::mul_assign_ssse3",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>,
        |dst, c| x86::gf8::mul_assign_ssse3(v2, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8<0x11b>::mul_into_ssse3",
        LENGTHS8,
        &coeffs,
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_into_ssse3(v2, dst, scale_table(c), src),
    );

    check_elementwise::<FiniteField<2, 8, Polynomial<AES>>>(
        "gf8::mul_elementwise_gfni",
        LENGTHS8,
        |dst, a, b| {
            x86::gf8::mul_elementwise_gfni::<true>(
                v3gfni,
                isomorphism_to_aes::<AES>(),
                reduction_low::<AES>(),
                dst,
                a,
                b,
            )
        },
    );
    check_elementwise::<FiniteField<2, 8, Polynomial<AES>>>(
        "gf8::mul_elementwise_avx2",
        LENGTHS8,
        |dst, a, b| x86::gf8::mul_elementwise_avx2(v3, reduction_low::<AES>(), dst, a, b),
    );
    check_elementwise::<FiniteField<2, 8, Polynomial<AES>>>(
        "gf8::mul_elementwise_ssse3",
        LENGTHS8,
        |dst, a, b| x86::gf8::mul_elementwise_ssse3(v2, reduction_low::<AES>(), dst, a, b),
    );

    // The shuffle single-row kernels are polynomial-agnostic: the
    // coefficient's own bank serves the Reed–Solomon field too.
    check_mul_add(
        "gf8<0x11d>::mul_add_avx2",
        LENGTHS8,
        &rs_coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_add_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8<0x11d>::mul_assign_avx2",
        LENGTHS8,
        &rs_coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>,
        |dst, c| x86::gf8::mul_assign_avx2(v3, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8<0x11d>::mul_into_avx2",
        LENGTHS8,
        &rs_coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_into_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_add(
        "gf8<0x11d>::mul_add_ssse3",
        LENGTHS8,
        &rs_coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_add_ssse3(v2, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8<0x11d>::mul_assign_ssse3",
        LENGTHS8,
        &rs_coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>,
        |dst, c| x86::gf8::mul_assign_ssse3(v2, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8<0x11d>::mul_into_ssse3",
        LENGTHS8,
        &rs_coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_into_ssse3(v2, dst, scale_table(c), src),
    );
    check_elementwise::<FiniteField<2, 8, Polynomial<RS>>>(
        "gf8::mul_elementwise_avx2 rs",
        LENGTHS8,
        |dst, a, b| x86::gf8::mul_elementwise_avx2(v3, reduction_low::<RS>(), dst, a, b),
    );
    check_elementwise::<FiniteField<2, 8, Polynomial<RS>>>(
        "gf8::mul_elementwise_ssse3 rs",
        LENGTHS8,
        |dst, a, b| x86::gf8::mul_elementwise_ssse3(v2, reduction_low::<RS>(), dst, a, b),
    );
}

#[test]
fn proven_gf8_rs_gfni_kernels_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let coeffs = gf8_rs_coeffs();
    check_mul_add(
        "gf8<0x11d>::mul_add_gfni",
        LENGTHS8,
        &coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_add_gfni::<false>(token, dst, Prepared::new(c), src),
    );
    check_mul_assign(
        "gf8<0x11d>::mul_assign_gfni",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>,
        |dst, c| x86::gf8::mul_assign_gfni::<false>(token, dst, Prepared::new(c)),
    );
    check_mul_into(
        "gf8<0x11d>::mul_into_gfni",
        LENGTHS8,
        &coeffs,
        gf8_ref::<RS>,
        |dst, c, src| x86::gf8::mul_into_gfni::<false>(token, dst, Prepared::new(c), src),
    );

    // The affine body on AES data: the selection every non-AES-native
    // representation takes, exercised on the native field.
    check_mul_add(
        "gf8<0x11b>::mul_add_gfni affine",
        LENGTHS8,
        &gf8_coeffs(),
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_add_gfni::<false>(token, dst, Prepared::new(c), src),
    );
    check_mul_assign(
        "gf8<0x11b>::mul_assign_gfni affine",
        LENGTHS8,
        &gf8_coeffs(),
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>,
        |dst, c| x86::gf8::mul_assign_gfni::<false>(token, dst, Prepared::new(c)),
    );
    check_mul_into(
        "gf8<0x11b>::mul_into_gfni affine",
        LENGTHS8,
        &gf8_coeffs(),
        gf8_ref::<AES>,
        |dst, c, src| x86::gf8::mul_into_gfni::<false>(token, dst, Prepared::new(c), src),
    );

    check_elementwise::<FiniteField<2, 8, Polynomial<RS>>>(
        "gf8::mul_elementwise_gfni rs",
        LENGTHS8,
        |dst, a, b| {
            x86::gf8::mul_elementwise_gfni::<false>(
                token,
                isomorphism_to_aes::<RS>(),
                reduction_low::<RS>(),
                dst,
                a,
                b,
            )
        },
    );
}

// ---------------------------------------------------------------------------
// GF(2^8) scatter / gather / matrix / scattered.
// ---------------------------------------------------------------------------

#[test]
fn proven_gf8_scatter_gather_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };

    check_scatter(
        "gf8::mul_add_scatter_gfni",
        ROW_LENS,
        gf8_ref::<AES>,
        |j| Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw((j as u8).wrapping_mul(29)),
        |rows, row_len, coeffs, src| {
            x86::gf8::mul_add_scatter_gfni::<true>(token, rows, row_len, &prepared(coeffs), src)
        },
    );
    check_scatter(
        "gf8::mul_add_scatter_gfni rs",
        ROW_LENS,
        gf8_ref::<RS>,
        |j| Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw((j as u8).wrapping_mul(29)),
        |rows, row_len, coeffs, src| {
            x86::gf8::mul_add_scatter_gfni::<false>(token, rows, row_len, &prepared(coeffs), src)
        },
    );

    #[cfg(not(miri))]
    let row_lens: &[usize] = &[32usize, 130, 300];
    // The long row only multiplies interpreted bytes under Miri.
    #[cfg(miri)]
    let row_lens: &[usize] = &[32usize, 130];
    for &row_len in row_lens {
        check_gather(
            "gf8::mul_add_gather_gfni",
            row_len,
            gf8_ref::<AES>,
            |i| {
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                    (i as u8).wrapping_mul(37).wrapping_add(3),
                )
            },
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_gfni::<true>(token, dst, &prepared(coeffs), srcs)
            },
        );
        check_gather(
            "gf8::mul_add_gather_gfni rs",
            row_len,
            gf8_ref::<RS>,
            |i| {
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(
                    (i as u8).wrapping_mul(53).wrapping_add(13),
                )
            },
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_gfni::<false>(token, dst, &prepared(coeffs), srcs)
            },
        );
    }
}

#[test]
fn proven_gf8_matrix_kernels_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };

    check_matrix_accumulate(
        "gf8::mul_add_matrix_gfni",
        ROW_LENS,
        gf8_coeff_at2::<AES>,
        gf8_ref::<AES>,
        |rows, row_len, nrows, terms| {
            prepared_pairs!(terms, |pairs| {
                x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &pairs[..],
                )
            })
        },
    );
    check_matrix_accumulate(
        "gf8::mul_add_matrix_gfni rs",
        ROW_LENS,
        gf8_coeff_at2::<RS>,
        gf8_ref::<RS>,
        |rows, row_len, nrows, terms| {
            prepared_pairs!(terms, |pairs| {
                x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &pairs[..],
                )
            })
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni",
        ROW_LENS,
        gf8_coeff_at2::<AES>,
        gf8_ref::<AES>,
        |rows, row_len, nrows, terms| {
            prepared_pairs!(terms, |pairs| {
                x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &pairs[..],
                )
            })
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni rs",
        ROW_LENS,
        gf8_coeff_at2::<RS>,
        gf8_ref::<RS>,
        |rows, row_len, nrows, terms| {
            prepared_pairs!(terms, |pairs| {
                x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &pairs[..],
                )
            })
        },
    );

    // The unsafe generic-provider twins, driven through the crate's own
    // stable FlatMatrix provider.
    check_matrix_accumulate(
        "gf8::mul_add_matrix_gfni_with",
        ROW_LENS,
        gf8_coeff_at2::<AES>,
        gf8_ref::<AES>,
        |rows, row_len, nrows, terms| {
            flat_gf8(terms, nrows, |matrix| {
                x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni_with",
        ROW_LENS,
        gf8_coeff_at2::<AES>,
        gf8_ref::<AES>,
        |rows, row_len, nrows, terms| {
            flat_gf8(terms, nrows, |matrix| {
                x86::gf8::mul_into_matrix_gfni_with::<true, _>(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_accumulate(
        "gf8::mul_add_matrix_gfni_with rs",
        ROW_LENS,
        gf8_coeff_at2::<RS>,
        gf8_ref::<RS>,
        |rows, row_len, nrows, terms| {
            flat_gf8(terms, nrows, |matrix| {
                x86::gf8::mul_add_matrix_gfni_with::<false, _>(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni_with rs",
        ROW_LENS,
        gf8_coeff_at2::<RS>,
        gf8_ref::<RS>,
        |rows, row_len, nrows, terms| {
            flat_gf8(terms, nrows, |matrix| {
                x86::gf8::mul_into_matrix_gfni_with::<false, _>(token, rows, row_len, nrows, matrix)
            });
        },
    );
}

/// Build a stable FlatMatrix of prepared coefficients over a term list and
/// hand it to `body`.
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn flat_gf8<const POLY: u128>(
    terms: &[(&[Elem<FiniteField<2, 8, Polynomial<POLY>>>], &[u8])],
    nrows: usize,
    body: impl FnOnce(&FlatMatrix<'_, Prepared>),
) {
    let mut flat = Vec::with_capacity(terms.len() * nrows);
    for &(coeffs, _) in terms {
        flat.extend(prepared(coeffs));
    }
    let srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
    body(&FlatMatrix {
        coefficients: &flat,
        nrows,
        sources: &srcs,
    });
}

#[test]
fn proven_gf8_matrix_scattered_matches_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    // Disjoint rows at mixed byte offsets, including zero rows and a
    // tight fit; GF(2^8) rows only need byte alignment.
    let starts = [0usize, 40, 96, 200, 260, 320, 400];
    for &row_len in &[2usize, 32, 34] {
        for nrows in [0usize, 1, 3, starts.len()] {
            scattered_case_gf8::<AES>(token, "mul_add_matrix_at_gfni", row_len, &starts[..nrows]);
            scattered_case_gf8::<RS>(
                token,
                "mul_add_matrix_at_gfni rs",
                row_len,
                &starts[..nrows],
            );
        }
    }
}

// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn scattered_case_gf8<const POLY: u128>(
    token: X64V3GfniCryptoToken,
    name: &str,
    row_len: usize,
    starts: &[usize],
) {
    const TOTAL: usize = 512;
    let nterms = 3;
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x600 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<POLY>>>>> = (0..nterms)
        .map(|t| {
            (0..starts.len())
                .map(|j| gf8_coeff_at2::<POLY>(t, j))
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<POLY>>>], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut got = noise(TOTAL, 0xe1);
    let mut want = got.clone();
    prepared_pairs!(terms, |pairs| {
        if POLY == AES {
            x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                token,
                &mut got,
                row_len,
                starts,
                &pairs[..],
            );
        } else {
            x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                token,
                &mut got,
                row_len,
                starts,
                &pairs[..],
            );
        }
    });
    for &(coeffs, src) in &terms {
        for (&start, &coeff) in starts.iter().zip(coeffs) {
            gf8_ref::<POLY>(&mut want[start..start + row_len], coeff, src);
        }
    }
    assert_eq!(got, want, "{name}: row_len {row_len}");
}

// ---------------------------------------------------------------------------
// Degenerate boundaries: empty terms/sources and zero rows.
// ---------------------------------------------------------------------------
#[test]
fn proven_overwrite_wrappers_zero_addressed_rows_on_empty_terms() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let (row_len, nrows, surplus) = (96usize, 6, 17);

    // Empty term list: the overwrite sum is zero over the addressed rows;
    // the surplus stays untouched.
    let empty: [(&[Prepared], &[u8]); 0] = [];
    let mut got = noise(nrows * row_len + surplus, 0xf1);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_gfni_with::<true, _>(token, &mut got, row_len, nrows, &empty[..]);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_gfni: empty terms");

    // The FlatMatrix twins must agree with their slice twins.
    let srcs: Vec<&[u8]> = Vec::new();
    let flat: Vec<Prepared> = Vec::new();
    let matrix = FlatMatrix {
        coefficients: &flat,
        nrows,
        sources: &srcs,
    };
    let mut got = noise(nrows * row_len + surplus, 0xf3);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_gfni_with::<true, _>(token, &mut got, row_len, nrows, &matrix);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_gfni_with: empty terms");

    let mut got = noise(nrows * row_len + surplus, 0xf4);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_gfni_with::<false, _>(token, &mut got, row_len, nrows, &matrix);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_gfni_with rs: empty terms");

    // Accumulation with no terms is a no-op: nothing changes.
    let mut got = noise(nrows * row_len + surplus, 0xf5);
    let want = got.clone();
    x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut got, row_len, nrows, &empty[..]);
    assert_eq!(got, want, "mul_add_matrix_gfni: empty terms is a no-op");

    // Zero rows: nothing is addressed, nothing changes (every term supplies
    // zero coefficients, matching the zero row count).
    let mut got = noise(row_len + surplus, 0xf6);
    let want = got.clone();
    let src = noise(row_len, 0xf7);
    let no_coeffs: [Prepared; 0] = [];
    let terms: Vec<(&[Prepared], &[u8])> = vec![(no_coeffs.as_slice(), src.as_slice())];
    x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut got, row_len, 0, &terms[..]);
    x86::gf8::mul_into_matrix_gfni_with::<true, _>(token, &mut got, row_len, 0, &terms[..]);
    assert_eq!(got, want, "zero rows address nothing");

    // Zero-length rows with a zero-length source are a documented no-op;
    // an empty gather with no sources must also leave the destination be.
    let mut got = noise(surplus, 0xf8);
    let want = got.clone();
    let coeffs = [Prepared::new(
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(7),
    )];
    let no_coeffs: [Prepared; 0] = [];
    let no_srcs: [&[u8]; 0] = [];
    x86::gf8::mul_add_scatter_gfni::<true>(token, &mut got, 0, &coeffs, &[]);
    x86::gf8::mul_add_gather_gfni::<true>(token, &mut got, &no_coeffs, &no_srcs);
    assert_eq!(got, want, "zero-length rows are a no-op");
}

#[test]
fn proven_gf16_degenerate_boundaries() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let (row_len, nrows, surplus) = (96usize, 5, 18);

    // Accumulate with no terms: no-op.
    let mut got = noise(nrows * row_len + surplus, 0x171);
    let want = got.clone();
    let no_terms: [(&[Elem<Gf16>], &[u8]); 0] = [];
    x86::gf16::mul_add_matrix_gfni(token, &mut got, row_len, nrows, &no_terms);
    assert_eq!(
        got, want,
        "gf16::mul_add_matrix_gfni: empty terms is a no-op"
    );

    // Zero rows and empty gather sources.
    let mut got = noise(row_len + surplus, 0x172);
    let want = got.clone();
    let coeffs = [Elem::<Gf16>::from_raw(0xbeef)];
    let no_coeffs: [Elem<Gf16>; 0] = [];
    x86::gf16::mul_add_gather_gfni(token, &mut got, &no_coeffs, &[]);
    x86::gf16::mul_add_scatter_gfni(token, &mut got, 0, &coeffs, &[]);
    assert_eq!(got, want, "gf16: degenerate geometry is a no-op");
}

// ---------------------------------------------------------------------------
// Geometry rejection — real panics from the entry validation.
// ---------------------------------------------------------------------------

/// Asserts that `body` panics, skipping when the host cannot summon the
/// capability token the kernel demands.
///
/// `#[should_panic]` cannot express the skip: a token-less early return is
/// reported as "did not panic as expected" on every host without the feature.
fn rejects_geometry<T: SimdToken>(what: &str, body: impl FnOnce(T)) {
    let Some(token) = T::summon() else {
        eprintln!("skipping {what}: host cannot summon the required token");
        return;
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(token)));
    assert!(outcome.is_err(), "{what}: expected a geometry panic");
}

#[test]
fn proven_gf8_mul_add_rejects_length_mismatch() {
    rejects_geometry("gf8 mul_add", |token| {
        let mut dst = vec![0u8; 16];
        let src = vec![0u8; 17];
        x86::gf8::mul_add_gfni::<true>(
            token,
            &mut dst,
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(3)),
            &src,
        );
    });
}

#[test]
fn proven_gf16_rejects_partial_element() {
    rejects_geometry("gf16 mul_add", |token| {
        let mut dst = vec![0u8; 3];
        let src = vec![0u8; 3];
        x86::gf16::mul_add_gfni(
            token,
            &mut dst,
            TowerCoeff::new(Elem::<Gf16>::from_raw(7)),
            &src,
        );
    });
}

#[test]
fn proven_scatter_rejects_short_rows_buffer() {
    rejects_geometry("gf8 scatter", |token| {
        let mut rows = vec![0u8; 16];
        let src = vec![0u8; 16];
        let coeffs = [
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1)),
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(2)),
        ];
        x86::gf8::mul_add_scatter_gfni::<true>(token, &mut rows, 16, &coeffs, &src);
    });
}

#[test]
fn proven_matrix_rejects_term_coefficient_mismatch() {
    rejects_geometry("gf8 matrix", |token| {
        let mut rows = vec![0u8; 64];
        let src = vec![0u8; 32];
        let short = [Prepared::new(
            Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1),
        )];
        let terms: Vec<(&[Prepared], &[u8])> = vec![(short.as_slice(), src.as_slice())];
        x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut rows, 32, 2, &terms[..]);
    });
}

#[test]
fn proven_matrix_scattered_rejects_overlapping_rows() {
    rejects_geometry("gf8 matrix_at overlap", |token| {
        let mut dst = vec![0u8; 128];
        let src = vec![0u8; 64];
        let coeffs = [Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1)); 2];
        let terms: Vec<(&[Prepared], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
        let starts = [0usize, 32];
        x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(token, &mut dst, 64, &starts, &terms[..]);
    });
}

#[test]
fn proven_matrix_scattered_rejects_out_of_bounds_row() {
    rejects_geometry("gf8 matrix_at bounds", |token| {
        let mut dst = vec![0u8; 64];
        let src = vec![0u8; 64];
        let coeffs = [Prepared::new(
            Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1),
        )];
        let terms: Vec<(&[Prepared], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
        let starts = [16usize];
        x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(token, &mut dst, 64, &starts, &terms[..]);
    });
}

/// A safe matrix provider whose first `source` call returns a full row and
/// every later call an empty slice: a provider the entry's geometry check
/// cannot pin, since it answers differently each time it is asked.
struct ShrinkingSource<'a, C> {
    coefficients: &'a [C],
    full: &'a [u8],
    calls: std::cell::Cell<usize>,
}

impl<C: Copy> fgf::internals::kernel::Matrix<C> for ShrinkingSource<'_, C> {
    fn len(&self) -> usize {
        1
    }
    fn coefficient(&self, _term: usize, row: usize) -> C {
        self.coefficients[row]
    }
    fn source(&self, _term: usize) -> &[u8] {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        if calls == 0 { self.full } else { &[] }
    }
}

#[test]
fn proven_matrix_with_rejects_source_that_shrinks_after_validation() {
    rejects_geometry("gf8 matrix_with shrinking source", |token| {
        let coeffs = [Prepared::new(
            Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(3),
        )];
        let src = vec![1u8; 256];
        let terms = ShrinkingSource {
            coefficients: &coeffs,
            full: &src,
            calls: std::cell::Cell::new(0),
        };
        let mut rows = vec![0u8; 256];
        x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut rows, 256, 1, &terms);
    });
    #[cfg(feature = "simd512")]
    for row_len in [256, 512, 576] {
        rejects_geometry(
            "gf8 avx512 matrix_with shrinking source",
            |token: X64V4xToken| {
                let coeffs = [Prepared::new(
                    Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(3),
                )];
                let src = vec![1u8; row_len];
                let terms = ShrinkingSource {
                    coefficients: &coeffs,
                    full: &src,
                    calls: std::cell::Cell::new(0),
                };
                let mut rows = vec![0u8; row_len];
                x86::gf8::mul_add_matrix_avx512_with(token, &mut rows, row_len, 1, &terms);
            },
        );
    }
}

#[test]
fn proven_gather_rejects_count_mismatch() {
    rejects_geometry("gf8 gather", |token| {
        let mut dst = vec![0u8; 32];
        let src = vec![0u8; 32];
        let coeffs = [
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1)),
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(2)),
        ];
        let srcs: Vec<&[u8]> = vec![src.as_slice()];
        x86::gf8::mul_add_gather_gfni::<true>(token, &mut dst, &coeffs, &srcs);
    });
}

#[cfg(feature = "simd512")]
#[test]
fn proven_gf8_rs512_rejects_bad_geometry() {
    rejects_geometry("gf8<0x11d>512 mul_add", |token: X64V4xToken| {
        let mut dst = vec![0u8; 16];
        let src = vec![0u8; 17];
        let coeff = Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(3));
        x86::gf8::mul_add_avx512(token, &mut dst, coeff, &src);
    });
    rejects_geometry("gf8<0x11d>512 scatter", |token: X64V4xToken| {
        let mut rows = vec![0u8; 16];
        let src = vec![0u8; 16];
        let coeffs = [
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(1)),
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(2)),
        ];
        x86::gf8::mul_add_scatter_avx512(token, &mut rows, 16, &coeffs, &src);
    });
    rejects_geometry("gf8<0x11d>512 matrix_at overlap", |token: X64V4xToken| {
        let mut dst = vec![0u8; 128];
        let src = vec![0u8; 64];
        let coeffs = [
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(1)),
            Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(2)),
        ];
        let terms: Vec<(&[Prepared], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
        x86::gf8::mul_add_matrix_at_avx512_with(token, &mut dst, 64, &[0, 32], &terms[..]);
    });
}

#[test]
fn proven_xor_gather_rejects_out_of_bounds_offset() {
    rejects_geometry("xor_gather avx2", |v3: X64V3Token| {
        let region = vec![0u8; 64];
        let mut dst = vec![0u8; 32];
        x86::bytes::xor_gather_avx2(v3, &region, &mut dst, &[40]);
    });
}

#[test]
fn proven_prime_rejects_partial_lane() {
    rejects_geometry("m31 add_assign avx2", |v3: X64V3Token| {
        let mut dst = vec![0u8; 7];
        let src = vec![0u8; 7];
        x86::mersenne31::add_assign_avx2(v3, &mut dst, &src);
    });
}

// ---------------------------------------------------------------------------
// GF(2^16) / GF(2^32) / GF(2^64) tower kernels.
// ---------------------------------------------------------------------------

#[test]
fn proven_gf16_kernels_match_scalar() {
    let Some(v3gfni) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let Some(v2) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE2 on this host");
        return;
    };
    let coeffs = gf16_coeffs();

    check_mul_add(
        "gf16::mul_add_gfni",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_add_gfni(v3gfni, dst, TowerCoeff::new(c), src),
    );
    check_mul_assign(
        "gf16::mul_assign_gfni",
        LENGTHS16,
        &coeffs,
        scalar::mul_assign::<Gf16>,
        |dst, c| x86::gf16::mul_assign_gfni(v3gfni, dst, TowerCoeff::new(c)),
    );
    check_mul_into(
        "gf16::mul_into_gfni",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_into_gfni(v3gfni, dst, TowerCoeff::new(c), src),
    );
    check_mul_add(
        "gf16::mul_add_avx2",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_add_avx2(v3, dst, &TowerTables::new(c), src),
    );
    check_mul_assign(
        "gf16::mul_assign_avx2",
        LENGTHS16,
        &coeffs,
        scalar::mul_assign::<Gf16>,
        |dst, c| x86::gf16::mul_assign_avx2(v3, dst, &TowerTables::new(c)),
    );
    check_mul_into(
        "gf16::mul_into_avx2",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_into_avx2(v3, dst, &TowerTables::new(c), src),
    );
    check_mul_add(
        "gf16::mul_add_ssse3",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_add_ssse3(v2, dst, &TowerTables::new(c), src),
    );
    check_mul_assign(
        "gf16::mul_assign_ssse3",
        LENGTHS16,
        &coeffs,
        scalar::mul_assign::<Gf16>,
        |dst, c| x86::gf16::mul_assign_ssse3(v2, dst, &TowerTables::new(c)),
    );
    check_mul_into(
        "gf16::mul_into_ssse3",
        LENGTHS16,
        &coeffs,
        gf16_ref,
        |dst, c, src| x86::gf16::mul_into_ssse3(v2, dst, &TowerTables::new(c), src),
    );

    let delta_byte: u8 = 0x20;
    let reduction = fgf::poly::Polynomial::<AES>::POLY.to_le_bytes()[0];
    let delta_table = scale_table(fgf::Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
        0x20,
    ));
    check_elementwise::<Gf16>("gf16::mul_elementwise_gfni", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_gfni(v3gfni, dst, a, b, delta_byte, reduction)
    });
    check_elementwise::<Gf16>("gf16::mul_elementwise_avx2", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_avx2(v3, dst, a, b, delta_table, reduction)
    });
    check_elementwise::<Gf16>("gf16::mul_elementwise_ssse3", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_ssse3(v2, dst, a, b, delta_table, reduction)
    });

    // Scatter/gather/matrix across all three tiers.
    check_scatter(
        "gf16::mul_add_scatter_gfni",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| Elem::<Gf16>::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_gfni(v3gfni, rows, row_len, coeffs, src)
        },
    );
    check_scatter(
        "gf16::mul_add_scatter_avx2",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| Elem::<Gf16>::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_avx2(v3, rows, row_len, coeffs, src)
        },
    );
    check_scatter(
        "gf16::mul_add_scatter_ssse3",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| Elem::<Gf16>::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_ssse3(v2, rows, row_len, coeffs, src)
        },
    );
    for &row_len in &[32usize, 130] {
        check_gather(
            "gf16::mul_add_gather_gfni",
            row_len,
            gf16_ref,
            |i| Elem::<Gf16>::from_raw((i as u16).wrapping_mul(613)),
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_gfni(v3gfni, dst, coeffs, srcs),
        );
        check_gather(
            "gf16::mul_add_gather_ssse3",
            row_len,
            gf16_ref,
            |i| Elem::<Gf16>::from_raw((i as u16).wrapping_mul(613)),
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_ssse3(v2, dst, coeffs, srcs),
        );
    }
    check_matrix_accumulate(
        "gf16::mul_add_matrix_gfni",
        EVEN_ROW_LENS,
        gf16_coeff_at2,
        gf16_ref,
        |rows, row_len, nrows, terms| {
            x86::gf16::mul_add_matrix_gfni(v3gfni, rows, row_len, nrows, terms)
        },
    );
    check_matrix_accumulate(
        "gf16::mul_add_matrix_ssse3",
        EVEN_ROW_LENS,
        gf16_coeff_at2,
        gf16_ref,
        |rows, row_len, nrows, terms| {
            x86::gf16::mul_add_matrix_ssse3(v2, rows, row_len, nrows, terms)
        },
    );
}

#[test]
fn proven_gf32_gf64_kernels_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let coeffs32 = gf32_coeffs();
    check_mul_add(
        "gf32::mul_add_gfni",
        LENGTHS32,
        &coeffs32,
        gf32_ref,
        |dst, c, src| x86::gf32::mul_add_gfni(token, dst, c, x86::gf32::gf32_tiles(c), src),
    );
    check_mul_assign(
        "gf32::mul_assign_gfni",
        LENGTHS32,
        &coeffs32,
        scalar::mul_assign::<Gf32>,
        |dst, c| x86::gf32::mul_assign_gfni(token, dst, c, x86::gf32::gf32_tiles(c)),
    );
    check_mul_into(
        "gf32::mul_into_gfni",
        LENGTHS32,
        &coeffs32,
        gf32_ref,
        |dst, c, src| x86::gf32::mul_into_gfni(token, dst, c, x86::gf32::gf32_tiles(c), src),
    );

    let coeffs64 = gf64_coeffs();
    check_mul_add(
        "gf64::mul_add_gfni",
        LENGTHS64,
        &coeffs64,
        gf64_ref,
        |dst, c, src| x86::gf64::mul_add_gfni(token, dst, c, x86::gf64::gf64_tiles(c), src),
    );
    check_mul_assign(
        "gf64::mul_assign_gfni",
        LENGTHS64,
        &coeffs64,
        scalar::mul_assign::<Gf64>,
        |dst, c| x86::gf64::mul_assign_gfni(token, dst, c, x86::gf64::gf64_tiles(c)),
    );
    check_mul_into(
        "gf64::mul_into_gfni",
        LENGTHS64,
        &coeffs64,
        gf64_ref,
        |dst, c, src| x86::gf64::mul_into_gfni(token, dst, c, x86::gf64::gf64_tiles(c), src),
    );
}

// ---------------------------------------------------------------------------
// Fan–Paar towers.
// ---------------------------------------------------------------------------

#[test]
fn proven_fan_paar_kernels_match_scalar() {
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let Some(v2) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE2 on this host");
        return;
    };

    #[cfg(not(miri))]
    let fp16_coeffs: Vec<Elem<FanPaar16Field>> =
        [0u16, 1, 0x0100, 0x00ff, 0xbeef, 0x7411, u16::MAX]
            .iter()
            .map(|&c| Elem::<FanPaar16Field>::from_raw(c))
            .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp16_coeffs: Vec<Elem<FanPaar16Field>> = [0u16, 1, 0xbeef, u16::MAX]
        .iter()
        .map(|&c| Elem::<FanPaar16Field>::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_avx2_fp16",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_add_avx2_fp16(v3, dst, &FpTowerTables::new(c), src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_avx2_fp16",
        LENGTHS16,
        &fp16_coeffs,
        |dst, c| fp_buffer_mul_assign(dst, u64::from(c.to_raw()), 16),
        |dst, c| x86::fan_paar::mul_assign_avx2_fp16(v3, dst, &FpTowerTables::new(c)),
    );
    check_mul_into(
        "fan_paar::mul_into_avx2_fp16",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_into_avx2_fp16(v3, dst, &FpTowerTables::new(c), src),
    );
    check_mul_add(
        "fan_paar::mul_add_ssse3_fp16",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_add_ssse3_fp16(v2, dst, &FpTowerTables::new(c), src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_ssse3_fp16",
        LENGTHS16,
        &fp16_coeffs,
        |dst, c| fp_buffer_mul_assign(dst, u64::from(c.to_raw()), 16),
        |dst, c| x86::fan_paar::mul_assign_ssse3_fp16(v2, dst, &FpTowerTables::new(c)),
    );
    check_mul_into(
        "fan_paar::mul_into_ssse3_fp16",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_into_ssse3_fp16(v2, dst, &FpTowerTables::new(c), src),
    );

    #[cfg(not(miri))]
    let fp32_coeffs: Vec<Elem<FanPaar32Field>> =
        [0u32, 1, 0x0001_0000, 0x0000_ffff, 0x1234_5678, u32::MAX]
            .iter()
            .map(|&c| Elem::<FanPaar32Field>::from_raw(c))
            .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp32_coeffs: Vec<Elem<FanPaar32Field>> = [0u32, 1, 0x1234_5678, u32::MAX]
        .iter()
        .map(|&c| Elem::<FanPaar32Field>::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_avx2_fp32",
        LENGTHS32,
        &fp32_coeffs,
        fp32_ref,
        |dst, c, src| x86::fan_paar::mul_add_avx2_fp32(v3, dst, c, src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_avx2_fp32",
        LENGTHS32,
        &fp32_coeffs,
        |dst, c| fp_buffer_mul_assign(dst, u64::from(c.to_raw()), 32),
        |dst, c| x86::fan_paar::mul_assign_avx2_fp32(v3, dst, c),
    );
    check_mul_into(
        "fan_paar::mul_into_avx2_fp32",
        LENGTHS32,
        &fp32_coeffs,
        fp32_ref,
        |dst, c, src| x86::fan_paar::mul_into_avx2_fp32(v3, dst, c, src),
    );

    #[cfg(not(miri))]
    let fp64_coeffs: Vec<Elem<FanPaar64Field>> = [
        0u64,
        1,
        0x0000_0001_0000_0000,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    ]
    .iter()
    .map(|&c| Elem::<FanPaar64Field>::from_raw(c))
    .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp64_coeffs: Vec<Elem<FanPaar64Field>> = [0u64, 1, 0x0123_4567_89ab_cdef, u64::MAX]
        .iter()
        .map(|&c| Elem::<FanPaar64Field>::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_avx2_fp64",
        LENGTHS64,
        &fp64_coeffs,
        fp64_ref,
        |dst, c, src| x86::fan_paar::mul_add_avx2_fp64(v3, dst, c, src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_avx2_fp64",
        LENGTHS64,
        &fp64_coeffs,
        |dst, c| fp_buffer_mul_assign(dst, c.to_raw(), 64),
        |dst, c| x86::fan_paar::mul_assign_avx2_fp64(v3, dst, c),
    );
    check_mul_into(
        "fan_paar::mul_into_avx2_fp64",
        LENGTHS64,
        &fp64_coeffs,
        fp64_ref,
        |dst, c, src| x86::fan_paar::mul_into_avx2_fp64(v3, dst, c, src),
    );
}

// ---------------------------------------------------------------------------
// Prime fields: canonical lanes only (kernel contract).
// ---------------------------------------------------------------------------

/// Noise with every 4-byte lane reduced to the canonical Mersenne31 range.
fn m31_lanes(n: usize, seed: u64) -> Vec<u8> {
    let mut bytes = noise(n * 4, seed);
    for chunk in bytes.as_chunks_mut::<4>().0 {
        let lane = u32::from_le_bytes(*chunk);
        chunk.copy_from_slice(&Elem::<Mersenne31>::from_raw(lane).to_bytes());
    }
    bytes
}

/// Noise with every 8-byte lane reduced to the canonical Goldilocks range.
fn gld_lanes(n: usize, seed: u64) -> Vec<u8> {
    let mut bytes = noise(n * 8, seed);
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let lane = u64::from_le_bytes(*chunk);
        chunk.copy_from_slice(&Elem::<Goldilocks>::from_raw(lane % goldilocks::MODULUS).to_bytes());
    }
    bytes
}

#[test]
fn proven_prime_kernels_match_scalar() {
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    let Some(v2) = X64V2Token::summon() else {
        eprintln!("skipping: no SSE4.2 on this host");
        return;
    };

    const LANES: &[usize] = &[0, 1, 3, 4, 7, 8, 9, 16, 75];
    let coeff = Elem::<Mersenne31>::from_raw(0x0532_1cde);
    type BinOp<'a> = Box<dyn Fn(&mut [u8], &[u8]) + 'a>;

    for &n in LANES {
        let src = m31_lanes(n, 0x91);
        let coeff_raw = coeff.to_raw();

        // add / sub against per-lane field arithmetic.
        for (name, kernel, sub) in [
            (
                "mersenne31::add_assign_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::mersenne31::add_assign_avx2(v3, dst, src)
                }) as BinOp,
                false,
            ),
            (
                "mersenne31::sub_assign_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::mersenne31::sub_assign_avx2(v3, dst, src)
                }),
                true,
            ),
            (
                "mersenne31::add_assign_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::mersenne31::add_assign_sse42(v2, dst, src)
                }),
                false,
            ),
            (
                "mersenne31::sub_assign_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::mersenne31::sub_assign_sse42(v2, dst, src)
                }),
                true,
            ),
        ] {
            let mut got = m31_lanes(n, 0x92);
            let mut want = got.clone();
            kernel(&mut got, &src);
            for (d, s) in want
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0)
            {
                let a = Elem::<Mersenne31>::from_bytes(*d);
                let b = Elem::<Mersenne31>::from_bytes(*s);
                let r = if sub { a.sub(b) } else { a.add(b) };
                d.copy_from_slice(&r.to_bytes());
            }
            assert_eq!(got, want, "{name}: lanes {n}");
        }

        // mul_add / mul_assign / mul_into against the scalar oracle.
        let mut got = m31_lanes(n, 0x93);
        let mut want = got.clone();
        x86::mersenne31::mul_add_avx2(v3, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mersenne31::mul_add_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x94);
        let mut want = got.clone();
        x86::mersenne31::mul_add_sse42(v2, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mersenne31::mul_add_sse42: lanes {n}");

        let mut got = m31_lanes(n, 0x95);
        let mut want = got.clone();
        x86::mersenne31::mul_assign_avx2(v3, &mut got, coeff_raw);
        scalar::mul_assign::<Mersenne31>(&mut want, coeff);
        assert_eq!(got, want, "mersenne31::mul_assign_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x96);
        let mut want = vec![0u8; n * 4];
        x86::mersenne31::mul_into_avx2(v3, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mersenne31::mul_into_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x97);
        let mut want = vec![0u8; n * 4];
        x86::mersenne31::mul_into_sse42(v2, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mersenne31::mul_into_sse42: lanes {n}");

        // Elementwise product of two canonical buffers.
        let a = m31_lanes(n, 0x98);
        let b = m31_lanes(n, 0x99);
        let mut got = m31_lanes(n, 0x9a);
        let mut want = got.clone();
        x86::mersenne31::mul_elementwise_avx2(v3, &mut got, &a, &b);
        scalar::mul_elementwise::<Mersenne31>(&mut want, &a, &b);
        assert_eq!(got, want, "mersenne31::mul_elementwise_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x9b);
        let mut want = got.clone();
        x86::mersenne31::mul_elementwise_sse42(v2, &mut got, &a, &b);
        scalar::mul_elementwise::<Mersenne31>(&mut want, &a, &b);
        assert_eq!(got, want, "mersenne31::mul_elementwise_sse42: lanes {n}");
    }

    // Goldilocks: the same sweep with 8-byte lanes.
    let gld_coeff = Elem::<Goldilocks>::from_raw(0x0123_4567_89ab_cdef % goldilocks::MODULUS);
    for &n in &[0usize, 1, 3, 8, 9, 32] {
        let src = gld_lanes(n, 0xa1);
        let coeff_raw = gld_coeff.to_raw();

        for (name, kernel, sub) in [
            (
                "goldilocks::add_assign_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::goldilocks::add_assign_avx2(v3, dst, src)
                }) as BinOp,
                false,
            ),
            (
                "goldilocks::sub_assign_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::goldilocks::sub_assign_avx2(v3, dst, src)
                }),
                true,
            ),
            (
                "goldilocks::add_assign_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::goldilocks::add_assign_sse42(v2, dst, src)
                }),
                false,
            ),
            (
                "goldilocks::sub_assign_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::goldilocks::sub_assign_sse42(v2, dst, src)
                }),
                true,
            ),
        ] {
            let mut got = gld_lanes(n, 0xa2);
            let mut want = got.clone();
            kernel(&mut got, &src);
            for (d, s) in want
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<8>().0)
            {
                let a = Elem::<Goldilocks>::from_bytes(*d);
                let b = Elem::<Goldilocks>::from_bytes(*s);
                let r = if sub { a.sub(b) } else { a.add(b) };
                d.copy_from_slice(&r.to_bytes());
            }
            assert_eq!(got, want, "{name}: lanes {n}");
        }

        let mut got = gld_lanes(n, 0xa3);
        let mut want = got.clone();
        x86::goldilocks::mul_add_avx2(v3, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "goldilocks::mul_add_avx2: lanes {n}");

        let mut got = gld_lanes(n, 0xa4);
        let mut want = got.clone();
        x86::goldilocks::mul_add_sse42(v2, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "goldilocks::mul_add_sse42: lanes {n}");

        let mut got = gld_lanes(n, 0xa5);
        let mut want = got.clone();
        x86::goldilocks::mul_assign_sse42(v2, &mut got, coeff_raw);
        scalar::mul_assign::<Goldilocks>(&mut want, gld_coeff);
        assert_eq!(got, want, "goldilocks::mul_assign_sse42: lanes {n}");

        let mut got = gld_lanes(n, 0xa6);
        let mut want = vec![0u8; n * 8];
        x86::goldilocks::mul_into_avx2(v3, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "goldilocks::mul_into_avx2: lanes {n}");

        let a = gld_lanes(n, 0xa7);
        let b = gld_lanes(n, 0xa8);
        let mut got = gld_lanes(n, 0xa9);
        let mut want = got.clone();
        x86::goldilocks::mul_elementwise_avx2(v3, &mut got, &a, &b);
        scalar::mul_elementwise::<Goldilocks>(&mut want, &a, &b);
        assert_eq!(got, want, "goldilocks::mul_elementwise_avx2: lanes {n}");

        let mut got = gld_lanes(n, 0xaa);
        let mut want = got.clone();
        x86::goldilocks::mul_elementwise_sse42(v2, &mut got, &a, &b);
        scalar::mul_elementwise::<Goldilocks>(&mut want, &a, &b);
        assert_eq!(got, want, "goldilocks::mul_elementwise_sse42: lanes {n}");
    }
}

#[derive(Clone, Copy)]
enum ResidueToken {
    Avx2(X64V3Token),
    #[cfg(feature = "simd512")]
    Avx512(archmage::X64V4Token),
}

fn for_residue_kernels(mut check: impl FnMut(ResidueToken)) {
    if let Some(token) = X64V3Token::summon() {
        eprintln!("residue differential: executing kernel=avx2 token=x64_v3");
        check(ResidueToken::Avx2(token));
    } else {
        eprintln!("residue differential: skipping unsupported kernel=avx2 token=x64_v3");
    }
    #[cfg(feature = "simd512")]
    if let Some(token) = archmage::X64V4Token::summon() {
        eprintln!("residue differential: executing kernel=avx512 token=x64_v4");
        check(ResidueToken::Avx512(token));
    } else {
        eprintln!("residue differential: skipping unsupported kernel=avx512 token=x64_v4");
    }
}

macro_rules! residue_call {
    ($token:expr, $avx2:ident, $avx512:ident $(, $arg:expr)* $(,)?) => {
        match $token {
            ResidueToken::Avx2(token) => x86::residue::$avx2(token $(, $arg)*),
            #[cfg(feature = "simd512")]
            ResidueToken::Avx512(token) => x86::residue::$avx512(token $(, $arg)*),
        }
    };
}

/// Four-byte generic residue lanes past dispatch: the direct AVX2 and AVX-512 entries
/// against the independent `u128` oracle, with raw lanes in and canonical
/// lanes out. Byte offsets 0 and 4 into a larger buffer exercise alignment.
#[test]
fn proven_residue_u32_elementwise_matches_oracle() {
    const PRIMES: [u64; 6] = [
        65_537,
        1_000_000_007,
        2_013_265_921,
        2_147_483_647,
        2_147_483_659,
        4_294_967_291,
    ];
    for_residue_kernels(|token| {
        let mut lens: Vec<usize> = (0..=33).collect();
        lens.push(517);
        for &p in &PRIMES {
            let params = x86::residue::U32Params::new(p as u32);
            for &lanes in &lens {
                for &offset in &[0usize, 4] {
                    let pad = offset + lanes * 4 + 4;
                    let mut backing_a = noise(pad, 0x1a2b + p + lanes as u64);
                    let mut backing_b = noise(pad, 0x3c4d + p + lanes as u64);
                    for (i, lane) in backing_a[offset..offset + lanes * 4]
                        .chunks_exact_mut(4)
                        .enumerate()
                    {
                        let edge = match i % 10 {
                            0 => 0u64,
                            1 => 1,
                            2 => p - 1,
                            3 => p,
                            4 => p + 1,
                            5 => 0x7FFF_FFFF,
                            6 => 0x8000_0000,
                            7 => 0xFFFF_FFFF,
                            _ => u64::from(u32::from_le_bytes(lane.try_into().unwrap())),
                        };
                        lane.copy_from_slice(&(edge as u32).to_le_bytes());
                    }
                    for (i, lane) in backing_b[offset..offset + lanes * 4]
                        .chunks_exact_mut(4)
                        .enumerate()
                    {
                        let edge = match (i + 4) % 10 {
                            0 => 0u64,
                            1 => 1,
                            2 => p - 1,
                            3 => p,
                            4 => p + 1,
                            5 => 0x7FFF_FFFF,
                            6 => 0x8000_0000,
                            7 => 0xFFFF_FFFF,
                            _ => u64::from(u32::from_le_bytes(lane.try_into().unwrap())),
                        };
                        lane.copy_from_slice(&(edge as u32).to_le_bytes());
                    }
                    let (a, b) = (
                        backing_a[offset..offset + lanes * 4].to_vec(),
                        backing_b[offset..offset + lanes * 4].to_vec(),
                    );
                    let mut want = vec![0u8; lanes * 4];
                    for ((w, x), y) in want
                        .chunks_exact_mut(4)
                        .zip(a.chunks_exact(4))
                        .zip(b.chunks_exact(4))
                    {
                        let x = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % p;
                        let y = u64::from(u32::from_le_bytes(y.try_into().unwrap())) % p;
                        w.copy_from_slice(
                            &((x as u128 * y as u128 % p as u128) as u32).to_le_bytes(),
                        );
                    }
                    // Offset windows of the retained backings, with guards.
                    let mut backing_d = noise(pad, 0x5e6f + p + lanes as u64);
                    let before_d = backing_d.clone();
                    {
                        let (dst, a_win, b_win) = (
                            &mut backing_d[offset..offset + lanes * 4],
                            &backing_a[offset..offset + lanes * 4],
                            &backing_b[offset..offset + lanes * 4],
                        );
                        residue_call!(
                            token,
                            mul_elementwise_u32_avx2,
                            mul_elementwise_u32_avx512,
                            dst,
                            a_win,
                            b_win,
                            params
                        );
                        assert_eq!(dst, &want, "residue mul p {p} lanes {lanes} off {offset}");
                    }
                    assert_eq!(&backing_d[..offset], &before_d[..offset], "head guard");
                    assert_eq!(
                        &backing_d[offset + lanes * 4..],
                        &before_d[offset + lanes * 4..],
                        "tail guard"
                    );
                    let mut backing_assign = backing_a.clone();
                    let before_assign = backing_assign.clone();
                    {
                        let (dst, src) = (
                            &mut backing_assign[offset..offset + lanes * 4],
                            &backing_b[offset..offset + lanes * 4],
                        );
                        residue_call!(
                            token,
                            mul_elementwise_assign_u32_avx2,
                            mul_elementwise_assign_u32_avx512,
                            dst,
                            src,
                            params
                        );
                        assert_eq!(
                            dst, &want,
                            "residue mul_assign p {p} lanes {lanes} off {offset}"
                        );
                    }
                    assert_eq!(
                        &backing_assign[..offset],
                        &before_assign[..offset],
                        "head guard"
                    );
                    assert_eq!(
                        &backing_assign[offset + lanes * 4..],
                        &before_assign[offset + lanes * 4..],
                        "tail guard"
                    );
                }
            }
        }
    });
}

/// Prepared Shoup fixed-coefficient kernels past dispatch: every four-byte
/// row class against the independent `u128` oracle
/// `((x mod P) * c) mod P`, with raw lanes in and canonical lanes out.
#[test]
fn proven_residue_u32_shoup_coeff_matches_oracle() {
    const PRIMES: [u64; 6] = [
        65_537,
        1_000_000_007,
        2_013_265_921,
        2_147_483_647,
        2_147_483_659,
        4_294_967_291,
    ];
    for_residue_kernels(|token| {
        let mut lens: Vec<usize> = (0..=33).collect();
        lens.push(517);
        for &p in &PRIMES {
            let p32 = p as u32;
            let mut coeffs = vec![2u64, p / 2, p - 1];
            let mut state = p.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                coeffs.push((state >> 11) % p);
            }
            for &c in &coeffs {
                let c32 = c as u32;
                let q = ((c as u128 * (1u128 << 32) / p as u128) as u64) as u32;
                for &lanes in &lens {
                    for &offset in &[0usize, 4] {
                        let pad = offset + lanes * 4 + 4;
                        let raw = noise(pad, 0x81 + p + c + lanes as u64);
                        let mut backing_s = raw.clone();
                        for (i, lane) in backing_s[offset..offset + lanes * 4]
                            .chunks_exact_mut(4)
                            .enumerate()
                        {
                            let edge = match i % 10 {
                                0 => 0u64,
                                1 => 1,
                                2 => p - 1,
                                3 => p,
                                4 => p + 1,
                                5 => 0x7FFF_FFFF,
                                6 => 0x8000_0000,
                                7 => 0xFFFF_FFFF,
                                9 => c,
                                _ => u64::from(u32::from_le_bytes(lane.try_into().unwrap())),
                            };
                            lane.copy_from_slice(&(edge as u32).to_le_bytes());
                        }
                        let src = backing_s[offset..offset + lanes * 4].to_vec();
                        let mut want = vec![0u8; lanes * 4];
                        for (w, x) in want.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                            let x = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % p;
                            w.copy_from_slice(
                                &((x as u128 * c as u128 % p as u128) as u32).to_le_bytes(),
                            );
                        }
                        // Overwrite into a retained destination window.
                        let mut backing_d = noise(pad, 0x83 + p + c + lanes as u64);
                        let before_d = backing_d.clone();
                        {
                            let (dst, s) = (
                                &mut backing_d[offset..offset + lanes * 4],
                                &backing_s[offset..offset + lanes * 4],
                            );
                            residue_call!(
                                token,
                                mul_into_u32_avx2,
                                mul_into_u32_avx512,
                                dst,
                                s,
                                c32,
                                q,
                                p32
                            );
                            assert_eq!(
                                dst, &want,
                                "shoup into p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_d[..offset], &before_d[..offset], "head guard");
                        assert_eq!(
                            &backing_d[offset + lanes * 4..],
                            &before_d[offset + lanes * 4..],
                            "tail guard"
                        );
                        // Accumulate into a retained destination window at
                        // the same offset as the source window.
                        let mut backing_a = noise(pad, 0x82 + p + c + lanes as u64);
                        let mut want_add = backing_a[offset..offset + lanes * 4].to_vec();
                        for (d, x) in want_add.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                            let a = u64::from(u32::from_le_bytes(d.try_into().unwrap())) % p;
                            let x = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % p;
                            d.copy_from_slice(
                                &(((a as u128 + x as u128 * c as u128 % p as u128) % p as u128)
                                    as u32)
                                    .to_le_bytes(),
                            );
                        }
                        let before_a = backing_a.clone();
                        {
                            let (dst, s) = (
                                &mut backing_a[offset..offset + lanes * 4],
                                &backing_s[offset..offset + lanes * 4],
                            );
                            residue_call!(
                                token,
                                mul_add_u32_avx2,
                                mul_add_u32_avx512,
                                dst,
                                s,
                                c32,
                                q,
                                p32
                            );
                            assert_eq!(
                                dst, &want_add,
                                "shoup add p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_a[..offset], &before_a[..offset], "head guard");
                        assert_eq!(
                            &backing_a[offset + lanes * 4..],
                            &before_a[offset + lanes * 4..],
                            "tail guard"
                        );
                        // In-place window of the retained source backing.
                        let before_s = backing_s.clone();
                        {
                            let dst = &mut backing_s[offset..offset + lanes * 4];
                            residue_call!(
                                token,
                                mul_assign_u32_avx2,
                                mul_assign_u32_avx512,
                                dst,
                                c32,
                                q,
                                p32
                            );
                            assert_eq!(
                                dst, &want,
                                "shoup assign p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_s[..offset], &before_s[..offset], "head guard");
                        assert_eq!(
                            &backing_s[offset + lanes * 4..],
                            &before_s[offset + lanes * 4..],
                            "tail guard"
                        );
                    }
                }
            }
        }
    });
}

/// Prepared doubleword-REDC fixed-coefficient kernels past dispatch:
/// eight-byte odd rows against the same oracle and geometry.
#[test]
fn proven_residue_u64_redc_coeff_matches_oracle() {
    const PRIMES: [u64; 3] = [
        4_294_967_311,
        1_000_000_000_000_000_003,
        18_446_744_073_709_551_557,
    ];
    for_residue_kernels(|token| {
        let mut lens: Vec<usize> = (0..=33).collect();
        lens.push(257);
        for &p in &PRIMES {
            let mut coeffs = vec![2u64 % p, p / 2, p - 1];
            let mut state = p.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                coeffs.push((state as u128 % p as u128) as u64);
            }
            for &c in &coeffs {
                let c_mont = (c as u128 * (1u128 << 64) % p as u128) as u64;
                for &lanes in &lens {
                    for &offset in &[0usize, 8] {
                        let pad = offset + lanes * 8 + 8;
                        let raw = noise(
                            pad,
                            0x91u64
                                .wrapping_add(p)
                                .wrapping_add(c)
                                .wrapping_add(lanes as u64),
                        );
                        let mut backing_s = raw.clone();
                        for (i, lane) in backing_s[offset..offset + lanes * 8]
                            .chunks_exact_mut(8)
                            .enumerate()
                        {
                            let edge: u128 = match i % 8 {
                                0 => 0,
                                1 => 1,
                                2 => (p - 1) as u128,
                                3 => p as u128,
                                4 => p as u128 + 1,
                                5 => 0xFFFF_FFFF_FFFF_FFFF,
                                7 => c as u128,
                                _ => u64::from_le_bytes(lane.try_into().unwrap()) as u128,
                            };
                            lane.copy_from_slice(&((edge % (1u128 << 64)) as u64).to_le_bytes());
                        }
                        let src = backing_s[offset..offset + lanes * 8].to_vec();
                        let mut want = vec![0u8; lanes * 8];
                        for (w, x) in want.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                            let x = (u64::from_le_bytes(x.try_into().unwrap()) as u128) % p as u128;
                            w.copy_from_slice(&((x * c as u128 % p as u128) as u64).to_le_bytes());
                        }
                        // Overwrite into a retained destination window.
                        let mut backing_d = noise(
                            pad,
                            0x93u64
                                .wrapping_add(p)
                                .wrapping_add(c)
                                .wrapping_add(lanes as u64),
                        );
                        let before_d = backing_d.clone();
                        {
                            let (dst, s) = (
                                &mut backing_d[offset..offset + lanes * 8],
                                &backing_s[offset..offset + lanes * 8],
                            );
                            residue_call!(
                                token,
                                mul_into_u64_avx2,
                                mul_into_u64_avx512,
                                dst,
                                s,
                                c_mont,
                                p
                            );
                            assert_eq!(
                                dst, &want,
                                "wide into p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_d[..offset], &before_d[..offset], "head guard");
                        assert_eq!(
                            &backing_d[offset + lanes * 8..],
                            &before_d[offset + lanes * 8..],
                            "tail guard"
                        );
                        // Accumulate into a retained destination window.
                        let mut backing_a_pad = noise(
                            pad,
                            0x94u64
                                .wrapping_add(p)
                                .wrapping_add(c)
                                .wrapping_add(lanes as u64),
                        );
                        let mut want_add = backing_a_pad[offset..offset + lanes * 8].to_vec();
                        for (d, x) in want_add.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                            let a = (u64::from_le_bytes(d.try_into().unwrap()) as u128) % p as u128;
                            let x = (u64::from_le_bytes(x.try_into().unwrap()) as u128) % p as u128;
                            d.copy_from_slice(
                                &(((a + x * c as u128 % p as u128) % p as u128) as u64)
                                    .to_le_bytes(),
                            );
                        }
                        let before_a = backing_a_pad.clone();
                        {
                            let (dst, s) = (
                                &mut backing_a_pad[offset..offset + lanes * 8],
                                &backing_s[offset..offset + lanes * 8],
                            );
                            residue_call!(
                                token,
                                mul_add_u64_avx2,
                                mul_add_u64_avx512,
                                dst,
                                s,
                                c_mont,
                                p
                            );
                            assert_eq!(
                                &dst[..],
                                &want_add[..],
                                "wide add p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_a_pad[..offset], &before_a[..offset], "head guard");
                        assert_eq!(
                            &backing_a_pad[offset + lanes * 8..],
                            &before_a[offset + lanes * 8..],
                            "tail guard"
                        );
                        // In-place window of the retained source backing.
                        let before_s = backing_s.clone();
                        {
                            let dst = &mut backing_s[offset..offset + lanes * 8];
                            residue_call!(
                                token,
                                mul_assign_u64_avx2,
                                mul_assign_u64_avx512,
                                dst,
                                c_mont,
                                p
                            );
                            assert_eq!(
                                dst, &want,
                                "wide assign p {p} c {c} lanes {lanes} off {offset}"
                            );
                        }
                        assert_eq!(&backing_s[..offset], &before_s[..offset], "head guard");
                        assert_eq!(
                            &backing_s[offset + lanes * 8..],
                            &before_s[offset + lanes * 8..],
                            "tail guard"
                        );
                    }
                }
            }
        }
    });
}

/// Byte nibble-lookup kernels, exhaustive: every odd prime `P <= 127`,
/// every coefficient, and every raw byte 0..=255, through all three
/// dispositions against the integer oracle, at lengths below, at, and
/// above both vector widths (tails included).
#[test]
fn proven_residue_u8_lookup_coeff_is_exhaustive() {
    const PRIMES: [u64; 30] = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127,
    ];
    const LENS: [usize; 8] = [0, 1, 31, 32, 33, 63, 64, 65];
    for_residue_kernels(|token| {
        for &p in &PRIMES {
            let p8 = p as u8;
            for c in 0..p {
                let c8 = c as u8;
                // Full raw-byte coverage in one 256-byte body plus tails at
                // every length class: the body lanes cycle all 256 values, so
                // vector lanes and scalar tails both meet raw bytes at and
                // above `P`.
                let full: Vec<u8> = (0..=255u16).map(|x| x as u8).collect();
                let mut want_full = vec![0u8; 256];
                for (w, &x) in want_full.iter_mut().zip(full.iter()) {
                    *w = ((x as u128 * c as u128 % p as u128) as u64) as u8;
                }
                let mut got = vec![0u8; 256];
                residue_call!(
                    token,
                    mul_into_u8_avx2,
                    mul_into_u8_avx512,
                    &mut got,
                    &full,
                    c8,
                    p8
                );
                assert_eq!(got, want_full, "lookup into p {p} c {c}");
                let mut acc = full.clone();
                let mut want_add = full.clone();
                for (d, &x) in want_add.iter_mut().zip(full.iter()) {
                    *d = (((*d as u128 % p as u128) + x as u128 * c as u128 % p as u128)
                        % p as u128) as u8;
                }
                residue_call!(
                    token,
                    mul_add_u8_avx2,
                    mul_add_u8_avx512,
                    &mut acc,
                    &full,
                    c8,
                    p8
                );
                assert_eq!(acc, want_add, "lookup add p {p} c {c}");
                let mut got_assign = full.clone();
                residue_call!(
                    token,
                    mul_assign_u8_avx2,
                    mul_assign_u8_avx512,
                    &mut got_assign,
                    c8,
                    p8
                );
                assert_eq!(got_assign, want_full, "lookup assign p {p} c {c}");
                for &len in &LENS {
                    // Retained backings with head/tail guards; the kernel sees
                    // the offset window, and source/destination offsets vary
                    // independently with the prime and coefficient.
                    let doff = (p as usize) % 3;
                    let soff = (c as usize) % 3;
                    let dpad = doff + len + 2;
                    let spad = soff + len + 2;
                    let src: Vec<u8> = (0..len)
                        .map(|i| ((i * 37 + c as usize) % 256) as u8)
                        .collect();
                    let mut want = vec![0u8; len];
                    for (w, &x) in want.iter_mut().zip(src.iter()) {
                        *w = ((x as u128 * c as u128 % p as u128) as u64) as u8;
                    }
                    let mut backing_s = noise(spad, 0xa1 + p + c);
                    backing_s[soff..soff + len].copy_from_slice(&src);
                    let before_s = backing_s.clone();
                    let mut backing_d = noise(dpad, 0xa2 + p + c);
                    let before_d = backing_d.clone();
                    {
                        let (dst, s) = (
                            &mut backing_d[doff..doff + len],
                            &backing_s[soff..soff + len],
                        );
                        residue_call!(token, mul_into_u8_avx2, mul_into_u8_avx512, dst, s, c8, p8);
                        assert_eq!(dst, &want, "lookup into tail p {p} c {c} len {len}");
                    }
                    assert_eq!(&backing_d[..doff], &before_d[..doff], "head guard");
                    assert_eq!(
                        &backing_d[doff + len..],
                        &before_d[doff + len..],
                        "tail guard"
                    );
                    assert_eq!(&backing_s[..], &before_s[..], "source untouched");
                    // Accumulate into a retained window over raw destination
                    // bytes (the destination doubles as the addend oracle).
                    let mut backing_a = noise(dpad, 0xa3 + p + c);
                    let mut want_add = backing_a[doff..doff + len].to_vec();
                    for (d, &x) in want_add.iter_mut().zip(src.iter()) {
                        *d = (((*d as u128 % p as u128) + x as u128 * c as u128 % p as u128)
                            % p as u128) as u8;
                    }
                    let before_a = backing_a.clone();
                    {
                        let (dst, s) = (
                            &mut backing_a[doff..doff + len],
                            &backing_s[soff..soff + len],
                        );
                        residue_call!(token, mul_add_u8_avx2, mul_add_u8_avx512, dst, s, c8, p8);
                        assert_eq!(dst, &want_add, "lookup add tail p {p} c {c} len {len}");
                    }
                    assert_eq!(&backing_a[..doff], &before_a[..doff], "head guard");
                    assert_eq!(
                        &backing_a[doff + len..],
                        &before_a[doff + len..],
                        "tail guard"
                    );
                    // In-place window of a retained backing.
                    let mut backing_m = noise(dpad, 0xa4 + p + c);
                    backing_m[doff..doff + len].copy_from_slice(&src);
                    let before_m = backing_m.clone();
                    {
                        let dst = &mut backing_m[doff..doff + len];
                        residue_call!(token, mul_assign_u8_avx2, mul_assign_u8_avx512, dst, c8, p8);
                        assert_eq!(dst, &want, "lookup assign tail p {p} c {c} len {len}");
                    }
                    assert_eq!(&backing_m[..doff], &before_m[..doff], "head guard");
                    assert_eq!(
                        &backing_m[doff + len..],
                        &before_m[doff + len..],
                        "tail guard"
                    );
                }
            }
        }
    });
}

/// Delayed-reduction prepared matrix kernels past dispatch: the direct
/// AVX2 and AVX-512 entries against the independent block-final `u128` oracle
/// `sum_i (x_i mod P) * c_i mod P`, then overwrite or modular add of the
/// raw destination. Sources carry raw lanes (noise plus all-ones words);
/// every stored lane is canonical. Rows sit at byte offsets 0 and 4 in a
/// larger buffer, whose guards must survive untouched.
#[test]
fn proven_residue_delayed_matrix_matches_oracle() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;

    const PRIMES: [u64; 8] = [
        65_537,
        1_000_000_007,
        1_073_741_827,
        1_280_000_003,
        1_431_655_751,
        2_013_265_921,
        2_147_483_647,
        4_294_967_291,
    ];
    const LONG_LANES: usize = 260;
    for_residue_kernels(|token| {
        for &p in &PRIMES {
            let k = ((1u128 << 32) / p as u128) as usize;
            if k < 2 {
                // The kernel itself rejects `k < 2`, and routing requires a
                // deeper delay, so the fallback covers this prime through
                // `ops`.
                continue;
            }
            let p32 = p as u32;
            let mut counts = vec![0usize, 1, k - 1, k, k + 1, 16, 17];
            counts.sort_unstable();
            counts.dedup();
            for &nterms in &counts {
                // Source counts past the small panel only run at two lane
                // counts; the oracle is linear in sources either way.
                let lane_counts: Vec<usize> = if nterms > 32 {
                    vec![1, 9]
                } else {
                    let mut lane_counts: Vec<usize> = (0..=17).collect();
                    lane_counts.push(LONG_LANES);
                    lane_counts
                };
                for &lanes in &lane_counts {
                    let row_len = lanes * 4;
                    for nrows in 1..=6usize {
                        for &offset in &[0usize, 4] {
                            // Coefficients cycle {0, 1, P-1} so every row
                            // and column covers all three.
                            let raw: Vec<u64> =
                                (0..nterms * nrows).map(|i| [0, 1, p - 1][i % 3]).collect();
                            let flat: Vec<PreparedResidue> = raw
                                .iter()
                                .map(|&c| {
                                    let q = (c as u128 * (1u128 << 32) / p as u128) as u64;
                                    PreparedResidue::from_word(c | (q << 32))
                                })
                                .collect();
                            // Two source fixtures: deterministic noise with
                            // edge lanes, and raw all-ones words.
                            for &ones in &[false, true] {
                                let sources: Vec<Vec<u8>> = (0..nterms)
                                    .map(|t| {
                                        let mut src = if ones {
                                            vec![0xFFu8; row_len]
                                        } else {
                                            noise(row_len, 0xd1 + p + t as u64 + lanes as u64)
                                        };
                                        for (i, lane) in src.chunks_exact_mut(4).enumerate().take(8)
                                        {
                                            let edge = match (i + t) % 10 {
                                                0 => 0u64,
                                                1 => 1,
                                                2 => p - 1,
                                                3 => p,
                                                4 => p + 1,
                                                5 => 0x7FFF_FFFF,
                                                6 => 0x8000_0000,
                                                7 => 0xFFFF_FFFF,
                                                _ => u64::from(u32::from_le_bytes(
                                                    lane.try_into().unwrap(),
                                                )),
                                            };
                                            lane.copy_from_slice(&(edge as u32).to_le_bytes());
                                        }
                                        src
                                    })
                                    .collect();
                                let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
                                let terms = FlatMatrix {
                                    coefficients: &flat,
                                    nrows,
                                    sources: &refs,
                                };
                                // Block-final oracle, one `u128` sum per lane.
                                let mut want = vec![0u8; nrows * row_len];
                                for j in 0..nrows {
                                    for (l, w) in want[j * row_len..(j + 1) * row_len]
                                        .chunks_exact_mut(4)
                                        .enumerate()
                                    {
                                        let mut acc: u128 = 0;
                                        for (t, src) in sources.iter().enumerate() {
                                            let x = u64::from(u32::from_le_bytes(
                                                src[l * 4..l * 4 + 4].try_into().unwrap(),
                                            )) % p;
                                            acc += x as u128 * raw[t * nrows + j] as u128;
                                        }
                                        w.copy_from_slice(
                                            &((acc % p as u128) as u32).to_le_bytes(),
                                        );
                                    }
                                }
                                // Overwrite past dispatch, with guards.
                                let pad = offset + nrows * row_len + offset;
                                let mut backing = noise(pad, 0xd2 + p + nterms as u64);
                                let before = backing.clone();
                                {
                                    let rows = &mut backing[offset..offset + nrows * row_len];
                                    // Nonzero destination catches
                                    // accumulation-into-destination bugs.
                                    for b in rows.iter_mut() {
                                        *b = b.wrapping_mul(3).wrapping_add(0x5a);
                                    }
                                    residue_call!(
                                        token,
                                        mul_into_matrix_u32_avx2_with,
                                        mul_into_matrix_u32_avx512_with,
                                        rows,
                                        row_len,
                                        nrows,
                                        &terms,
                                        p32,
                                    );
                                    assert_eq!(
                                        rows, &want,
                                        "delayed into p {p} terms {nterms} rows {nrows} lanes {lanes} ones {ones} off {offset}"
                                    );
                                }
                                assert_eq!(
                                    &backing[..offset],
                                    &before[..offset],
                                    "delayed into head guard p {p}"
                                );
                                assert_eq!(
                                    &backing[offset + nrows * row_len..],
                                    &before[offset + nrows * row_len..],
                                    "delayed into tail guard p {p}"
                                );
                                // Accumulation past dispatch, with guards. An
                                // empty sum is a no-op: the destination,
                                // canonical or not, survives byte-identical.
                                let mut backing = noise(pad, 0xd3 + p + nterms as u64);
                                let mut want_add =
                                    backing[offset..offset + nrows * row_len].to_vec();
                                if nterms > 0 {
                                    for j in 0..nrows {
                                        for (l, w) in want_add[j * row_len..(j + 1) * row_len]
                                            .chunks_exact_mut(4)
                                            .enumerate()
                                        {
                                            let d = u64::from(u32::from_le_bytes(
                                                backing[offset + j * row_len + l * 4
                                                    ..offset + j * row_len + l * 4 + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            )) % p;
                                            let r = u64::from(u32::from_le_bytes(
                                                want[j * row_len + l * 4..j * row_len + l * 4 + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            ));
                                            w.copy_from_slice(
                                                &(((d as u128 + r as u128) % p as u128) as u32)
                                                    .to_le_bytes(),
                                            );
                                        }
                                    }
                                }
                                let before = backing.clone();
                                {
                                    let rows = &mut backing[offset..offset + nrows * row_len];
                                    residue_call!(
                                        token,
                                        mul_add_matrix_u32_avx2_with,
                                        mul_add_matrix_u32_avx512_with,
                                        rows,
                                        row_len,
                                        nrows,
                                        &terms,
                                        p32,
                                    );
                                    assert_eq!(
                                        rows, &want_add,
                                        "delayed add p {p} terms {nterms} rows {nrows} lanes {lanes} ones {ones} off {offset}"
                                    );
                                }
                                assert_eq!(
                                    &backing[..offset],
                                    &before[..offset],
                                    "delayed add head guard p {p}"
                                );
                                assert_eq!(
                                    &backing[offset + nrows * row_len..],
                                    &before[offset + nrows * row_len..],
                                    "delayed add tail guard p {p}"
                                );
                            }
                        }
                    }
                }
            }
        }
    });
}

/// The delayed entries reject an even modulus before the first lane, with
/// the same message style as the doubleword siblings.
#[test]
fn proven_residue_delayed_matrix_rejects_even_modulus() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;
    for_residue_kernels(|token| {
        let coeff = PreparedResidue::from_word(1 | (1u64 << 32));
        let src = vec![0u8; 32];
        let terms = FlatMatrix {
            coefficients: &[coeff],
            nrows: 1,
            sources: &[src.as_slice()],
        };
        let rows_of = |entry: &str, into: bool| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rows = vec![0u8; 32];
                if into {
                    residue_call!(
                        token,
                        mul_into_matrix_u32_avx2_with,
                        mul_into_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        2
                    );
                } else {
                    residue_call!(
                        token,
                        mul_add_matrix_u32_avx2_with,
                        mul_add_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        2
                    );
                }
            }));
            let err = outcome.expect_err(&format!("{entry} must reject an even modulus"));
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.contains("modulus 2 is even"),
                "{entry}: panic must name the even modulus: {msg}"
            );
        };
        rows_of("residue::mul_into_matrix_u32_avx2_with", true);
        rows_of("residue::mul_add_matrix_u32_avx2_with", false);
    });
}
/// The delayed entries reject a non-positive modulus before the first lane,
/// with the same message style as the even-modulus guard.
#[test]
fn proven_residue_delayed_matrix_rejects_non_positive_modulus() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;
    for_residue_kernels(|token| {
        let coeff = PreparedResidue::from_word(0);
        let src = vec![0u8; 32];
        let terms = FlatMatrix {
            coefficients: &[coeff],
            nrows: 1,
            sources: &[src.as_slice()],
        };
        for (entry, into, modulus) in [
            ("residue::mul_into_matrix_u32_avx2_with", true, 0u32),
            ("residue::mul_add_matrix_u32_avx2_with", false, 0u32),
            ("residue::mul_into_matrix_u32_avx2_with", true, 1u32),
            ("residue::mul_add_matrix_u32_avx2_with", false, 1u32),
        ] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rows = vec![0u8; 32];
                if into {
                    residue_call!(
                        token,
                        mul_into_matrix_u32_avx2_with,
                        mul_into_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        modulus,
                    );
                } else {
                    residue_call!(
                        token,
                        mul_add_matrix_u32_avx2_with,
                        mul_add_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        modulus,
                    );
                }
            }));
            let err = outcome.expect_err(&format!("{entry} must reject modulus {modulus}"));
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.contains("not a delayed-reduction modulus"),
                "{entry}: panic must name the rejected modulus: {msg}"
            );
        }
    });
}

/// The delayed entries reject a modulus whose derived depth sits below two
/// (any odd `p` above `2^31`) before the first lane.
#[test]
fn proven_residue_delayed_matrix_rejects_shallow_depth() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;
    for_residue_kernels(|token| {
        let p = 4_294_967_291u32;
        let coeff = PreparedResidue::from_word(1 | (1u64 << 32));
        let src = vec![0u8; 32];
        let terms = FlatMatrix {
            coefficients: &[coeff],
            nrows: 1,
            sources: &[src.as_slice()],
        };
        for (entry, into) in [
            ("residue::mul_into_matrix_u32_avx2_with", true),
            ("residue::mul_add_matrix_u32_avx2_with", false),
        ] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rows = vec![0u8; 32];
                if into {
                    residue_call!(
                        token,
                        mul_into_matrix_u32_avx2_with,
                        mul_into_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        p
                    );
                } else {
                    residue_call!(
                        token,
                        mul_add_matrix_u32_avx2_with,
                        mul_add_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        1,
                        &terms,
                        p
                    );
                }
            }));
            let err = outcome.expect_err(&format!("{entry} must reject a depth-below-two modulus"));
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.contains("delay depth 1"),
                "{entry}: panic must name the shallow depth: {msg}"
            );
        }
    });
}

/// The delayed entries reject a noncanonical coefficient before the first
/// lane.
#[test]
fn proven_residue_delayed_matrix_rejects_noncanonical_coefficient() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;
    for_residue_kernels(|token| {
        let p = 1_000_000_007u32;
        let bad = PreparedResidue::from_word(u64::from(p) | (1u64 << 32));
        let src = vec![0u8; 32];
        let terms = FlatMatrix {
            coefficients: &[bad],
            nrows: 1,
            sources: &[src.as_slice()],
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rows = vec![0u8; 32];
            residue_call!(
                token,
                mul_into_matrix_u32_avx2_with,
                mul_into_matrix_u32_avx512_with,
                &mut rows,
                32,
                1,
                &terms,
                p
            );
        }));
        let err = outcome.expect_err("delayed overwrite must reject a noncanonical coefficient");
        let msg = err.downcast_ref::<String>().expect("string panic payload");
        assert!(
            msg.contains("reaches modulus"),
            "panic must name the noncanonical coefficient: {msg}"
        );
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rows = vec![0u8; 32];
            residue_call!(
                token,
                mul_add_matrix_u32_avx2_with,
                mul_add_matrix_u32_avx512_with,
                &mut rows,
                32,
                1,
                &terms,
                p
            );
        }));
        let err = outcome.expect_err("delayed accumulate must reject a noncanonical coefficient");
        let msg = err.downcast_ref::<String>().expect("string panic payload");
        assert!(
            msg.contains("reaches modulus"),
            "panic must name the noncanonical coefficient: {msg}"
        );
    });
}

/// The delayed entries reject a row geometry whose byte count overflows
/// before touching any lane.
#[test]
fn proven_residue_delayed_matrix_rejects_overflowing_geometry() {
    use fgf::internals::kernel::x86::residue::PreparedResidue;
    for_residue_kernels(|token| {
        let p = 1_000_000_007u32;
        let coeff = PreparedResidue::from_word(1 | (1u64 << 32));
        let src = vec![0u8; 32];
        let terms = FlatMatrix {
            coefficients: &[coeff],
            nrows: 1,
            sources: &[src.as_slice()],
        };
        for into in [true, false] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rows = vec![0u8; 32];
                if into {
                    residue_call!(
                        token,
                        mul_into_matrix_u32_avx2_with,
                        mul_into_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        usize::MAX,
                        &terms,
                        p,
                    );
                } else {
                    residue_call!(
                        token,
                        mul_add_matrix_u32_avx2_with,
                        mul_add_matrix_u32_avx512_with,
                        &mut rows,
                        32,
                        usize::MAX,
                        &terms,
                        p,
                    );
                }
            }));
            let err = outcome.expect_err("an overflowing row geometry must be rejected");
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.contains("row geometry overflows"),
                "panic must name the overflowing geometry: {msg}"
            );
        }
    });
}
