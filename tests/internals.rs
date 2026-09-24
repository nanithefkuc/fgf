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

#![cfg(all(
    feature = "simd",
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#![allow(clippy::cast_possible_truncation)]

use fgf::internals::field::wiedemann;
use fgf::internals::kernel::scalar;
use fgf::internals::kernel::tables::{
    FpTowerTables, ScaleTable, TowerCoeff, TowerTables, affine_8d, scale_table, scale_table_8d,
};
use fgf::internals::kernel::{
    FlatMatrix, SimdToken, X64V2Token, X64V3GfniCryptoToken, X64V3Token, X64V4Token, X64V4xToken,
    x86,
};
use fgf::{
    Gf8B, Gf8D, Gf16, Gf32, Gf64, Goldilocks, Mersenne31, fan_paar, gf8b, gf8d, gf16, gf32, gf64,
    goldilocks, mersenne31,
};

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
const ROW_LENS: &[usize] = &[2, 32, 34, 300];
/// Truncated under Miri: the long row only multiplies interpreted bytes.
#[cfg(miri)]
const ROW_LENS: &[usize] = &[2, 32, 34];
#[cfg(not(miri))]
const EVEN_ROW_LENS: &[usize] = &[2, 30, 32, 34];
/// Truncated under Miri on the same terms as `ROW_LENS`.
#[cfg(miri)]
const EVEN_ROW_LENS: &[usize] = &[2, 32, 34];
#[cfg(not(miri))]
const NTERMS: &[usize] = &[1, 2, 3, 9, 17];
/// Truncated under Miri to single, pair, and one count past the resolve
/// chunk, which keeps the scratch-staging seam.
#[cfg(miri)]
const NTERMS: &[usize] = &[1, 2, 9];

/// GF(2^8) coefficients: both short-circuits, the extremes, a spread.
fn gf8_coeffs() -> Vec<gf8b::Elem> {
    #[cfg(not(miri))]
    let raw = [0u8, 1, 2, 3, 7, 0x53, 0xd3, 0xff];
    // Short-circuits, one ordinary value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u8, 1, 0x53, 0xff];
    raw.iter().map(|&c| gf8b::Elem::from_raw(c)).collect()
}

fn gf8d_coeffs() -> Vec<gf8d::Elem> {
    #[cfg(not(miri))]
    let raw = [0u8, 1, 2, 3, 7, 0x53, 0xd3, 0xff];
    // Short-circuits, one ordinary value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u8, 1, 0x53, 0xff];
    raw.iter().map(|&c| gf8d::Elem::from_raw(c)).collect()
}

/// GF(2^16) coefficients: short-circuits, pure components, mixed, extremes.
fn gf16_coeffs() -> Vec<gf16::Elem> {
    #[cfg(not(miri))]
    let raw = [0u16, 1, 0x0100, 0x00ff, 0x1234, 0xbeef, 0x7411, 0xffff];
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u16, 1, 0x1234, 0xffff];
    raw.iter().map(|&c| gf16::Elem::from_raw(c)).collect()
}

fn gf32_coeffs() -> Vec<gf32::Elem> {
    #[cfg(not(miri))]
    let raw = [0u32, 1, 0x0001_0000, 0x0000_ffff, 0x1234_5678, u32::MAX];
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let raw = [0u32, 1, 0x1234_5678, u32::MAX];
    raw.iter().map(|&c| gf32::Elem::from_raw(c)).collect()
}

fn gf64_coeffs() -> Vec<gf64::Elem> {
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
    raw.iter().map(|&c| gf64::Elem::from_raw(c)).collect()
}

// ---------------------------------------------------------------------------
// Scalar oracles.
// ---------------------------------------------------------------------------

fn gf8b_ref(dst: &mut [u8], coeff: gf8b::Elem, src: &[u8]) {
    scalar::mul_add::<Gf8B>(dst, coeff, src);
}
fn gf8d_ref(dst: &mut [u8], coeff: gf8d::Elem, src: &[u8]) {
    scalar::mul_add::<Gf8D>(dst, coeff, src);
}
fn gf16_ref(dst: &mut [u8], coeff: gf16::Elem, src: &[u8]) {
    scalar::mul_add::<Gf16>(dst, coeff, src);
}
fn gf32_ref(dst: &mut [u8], coeff: gf32::Elem, src: &[u8]) {
    scalar::mul_add::<Gf32>(dst, coeff, src);
}
fn gf64_ref(dst: &mut [u8], coeff: gf64::Elem, src: &[u8]) {
    scalar::mul_add::<Gf64>(dst, coeff, src);
}
fn fp16_ref(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    wiedemann::mul_add(dst, u64::from(coeff.to_raw()), src, 16);
}
fn fp32_ref(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    wiedemann::mul_add(dst, u64::from(coeff.to_raw()), src, 32);
}
fn fp64_ref(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    wiedemann::mul_add(dst, coeff.to_raw(), src, 64);
}
fn m31_ref(dst: &mut [u8], coeff: mersenne31::Elem, src: &[u8]) {
    scalar::mul_add::<Mersenne31>(dst, coeff, src);
}
fn gld_ref(dst: &mut [u8], coeff: goldilocks::Elem, src: &[u8]) {
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

fn check_elementwise<F: fgf::Field>(
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

fn gf8b_coeff_at2(t: usize, j: usize) -> gf8b::Elem {
    gf8b::Elem::from_raw(((t * 31 + j * 29) % 256) as u8)
}
fn gf8d_coeff_at2(t: usize, j: usize) -> gf8d::Elem {
    gf8d::Elem::from_raw(((t * 31 + j * 29) % 256) as u8)
}
fn gf16_coeff_at2(t: usize, j: usize) -> gf16::Elem {
    gf16::Elem::from_raw(((t * 7919 + j * 613) % 65536) as u16)
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

    // Interleaved row XOR against per-row scalar XOR, across row grouping
    // boundaries, including zero rows.
    for &row_len in ROW_LENS {
        for &nrows in &[0usize, 1, 2, 3, 4, 5, 8, 9, 13] {
            let len = row_len * nrows;
            let src = noise(len, 0x44);
            let mut got = noise(len, 0x55);
            let mut want = got.clone();
            x86::bytes::xor_rows_avx2(v3, &mut got, &src, row_len);
            for (d, s) in want.chunks_exact_mut(row_len).zip(src.chunks(row_len)) {
                scalar::xor(d, s);
            }
            assert_eq!(got, want, "xor_rows_avx2: row_len {row_len}, nrows {nrows}");

            let mut got = noise(len, 0x66);
            let mut want = got.clone();
            x86::bytes::xor_rows_sse2(v2.v1(), &mut got, &src, row_len);
            for (d, s) in want.chunks_exact_mut(row_len).zip(src.chunks(row_len)) {
                scalar::xor(d, s);
            }
            assert_eq!(got, want, "xor_rows_sse2: row_len {row_len}, nrows {nrows}");
        }
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
    check_mul_add(
        "gf8::mul_add_gfni",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_add_gfni(v3gfni, dst, c, src),
    );
    check_mul_assign(
        "gf8::mul_assign_gfni",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<Gf8B>,
        |dst, c| x86::gf8::mul_assign_gfni(v3gfni, dst, c),
    );
    check_mul_into(
        "gf8::mul_into_gfni",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_into_gfni(v3gfni, dst, c, src),
    );
    check_mul_add(
        "gf8::mul_add_avx2",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_add_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8::mul_assign_avx2",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<Gf8B>,
        |dst, c| x86::gf8::mul_assign_avx2(v3, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8::mul_into_avx2",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_into_avx2(v3, dst, scale_table(c), src),
    );
    check_mul_add(
        "gf8::mul_add_ssse3",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_add_ssse3(v2, dst, scale_table(c), src),
    );
    check_mul_assign(
        "gf8::mul_assign_ssse3",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<Gf8B>,
        |dst, c| x86::gf8::mul_assign_ssse3(v2, dst, scale_table(c)),
    );
    check_mul_into(
        "gf8::mul_into_ssse3",
        LENGTHS8,
        &coeffs,
        gf8b_ref,
        |dst, c, src| x86::gf8::mul_into_ssse3(v2, dst, scale_table(c), src),
    );

    check_elementwise::<Gf8B>("gf8::mul_elementwise_gfni", LENGTHS8, |dst, a, b| {
        x86::gf8::mul_elementwise_gfni(v3gfni, dst, a, b)
    });
    check_elementwise::<Gf8B>("gf8::mul_elementwise_avx2", LENGTHS8, |dst, a, b| {
        x86::gf8::mul_elementwise_avx2::<0x1b>(v3, dst, a, b)
    });
    check_elementwise::<Gf8B>("gf8::mul_elementwise_ssse3", LENGTHS8, |dst, a, b| {
        x86::gf8::mul_elementwise_ssse3::<0x1b>(v2, dst, a, b)
    });
}

#[test]
fn proven_gf8d_affine_kernels_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let coeffs = gf8d_coeffs();
    check_mul_add(
        "gf8::mul_add_affine",
        LENGTHS8,
        &coeffs,
        gf8d_ref,
        |dst, c, src| x86::gf8::mul_add_affine(token, dst, affine_8d(c), scale_table_8d(c), src),
    );
    check_mul_assign(
        "gf8::mul_assign_affine",
        LENGTHS8,
        &coeffs,
        scalar::mul_assign::<Gf8D>,
        |dst, c| x86::gf8::mul_assign_affine(token, dst, affine_8d(c), scale_table_8d(c)),
    );
    check_mul_into(
        "gf8::mul_into_affine",
        LENGTHS8,
        &coeffs,
        gf8d_ref,
        |dst, c, src| x86::gf8::mul_into_affine(token, dst, affine_8d(c), scale_table_8d(c), src),
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
        gf8b_ref,
        |j| gf8b::Elem::from_raw((j as u8).wrapping_mul(29)),
        |rows, row_len, coeffs, src| {
            x86::gf8::mul_add_scatter_gfni(token, rows, row_len, coeffs, src)
        },
    );
    check_scatter(
        "gf8::mul_add_scatter_affine",
        ROW_LENS,
        gf8d_ref,
        |j| gf8d::Elem::from_raw((j as u8).wrapping_mul(29)),
        |rows, row_len, coeffs, src| {
            x86::gf8::mul_add_scatter_affine(token, rows, row_len, coeffs, src)
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
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(37).wrapping_add(3)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::mul_add_gather_gfni_axpy_tail",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(41).wrapping_add(5)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_axpy_tail(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::mul_add_gather_gfni_split",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(43).wrapping_add(7)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_split(token, dst, coeffs, srcs),
        );
        // The tiled gather is const-generic over its accumulator count; each
        // width is a distinct monomorphization with its own blocking.
        check_gather(
            "gf8::gather_gfni_tile1",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(47).wrapping_add(11)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_tile::<1>(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::gather_gfni_tile2",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(47).wrapping_add(11)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_tile::<2>(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::gather_gfni_tile3",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(47).wrapping_add(11)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_tile::<3>(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::gather_gfni_tile4",
            row_len,
            gf8b_ref,
            |i| gf8b::Elem::from_raw((i as u8).wrapping_mul(47).wrapping_add(11)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_tile::<4>(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf8::mul_add_gather_affine",
            row_len,
            gf8d_ref,
            |i| gf8d::Elem::from_raw((i as u8).wrapping_mul(53).wrapping_add(13)),
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_affine(token, dst, coeffs, srcs),
        );

        // Prepared affine factors: `0x11B` coefficients through the
        // experimental affine map, verified against the AES-field oracle.
        for &nterms in NTERMS {
            let sources: Vec<Vec<u8>> = (0..nterms)
                .map(|t| noise(row_len, 0x480 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<gf8b::Elem> = (0..nterms)
                .map(|i| gf8b::Elem::from_raw((i as u8).wrapping_mul(53).wrapping_add(13)))
                .collect();
            let factors: Vec<_> = coeffs
                .iter()
                .map(|&c| x86::gf8::prepare_affine_8b(c))
                .collect();
            let mut got = noise(row_len, 0xd1);
            let mut want = got.clone();
            x86::gf8::mul_add_gather_affine_8b(token, &mut got, &factors, &srcs);
            for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                gf8b_ref(&mut want, coeff, src);
            }
            assert_eq!(got, want, "gf8::mul_add_gather_affine_8b: terms {nterms}");
        }
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
        gf8b_coeff_at2,
        gf8b_ref,
        |rows, row_len, nrows, terms| {
            x86::gf8::mul_add_matrix_gfni(token, rows, row_len, nrows, terms)
        },
    );
    check_matrix_accumulate(
        "gf8::mul_add_matrix_affine",
        ROW_LENS,
        gf8d_coeff_at2,
        gf8d_ref,
        |rows, row_len, nrows, terms| {
            x86::gf8::mul_add_matrix_affine(token, rows, row_len, nrows, terms)
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni",
        ROW_LENS,
        gf8b_coeff_at2,
        gf8b_ref,
        |rows, row_len, nrows, terms| {
            x86::gf8::mul_into_matrix_gfni(token, rows, row_len, nrows, terms)
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_affine",
        ROW_LENS,
        gf8d_coeff_at2,
        gf8d_ref,
        |rows, row_len, nrows, terms| {
            x86::gf8::mul_into_matrix_affine(token, rows, row_len, nrows, terms)
        },
    );

    // The unsafe generic-provider twins, driven through the crate's own
    // stable FlatMatrix provider.
    check_matrix_accumulate(
        "gf8::mul_add_matrix_gfni_with",
        ROW_LENS,
        gf8b_coeff_at2,
        gf8b_ref,
        |rows, row_len, nrows, terms| {
            flat_gf8b(terms, nrows, |matrix| {
                x86::gf8::mul_add_matrix_gfni_with(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_gfni_with",
        ROW_LENS,
        gf8b_coeff_at2,
        gf8b_ref,
        |rows, row_len, nrows, terms| {
            flat_gf8b(terms, nrows, |matrix| {
                x86::gf8::mul_into_matrix_gfni_with(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_accumulate(
        "gf8::mul_add_matrix_affine_with",
        ROW_LENS,
        gf8d_coeff_at2,
        gf8d_ref,
        |rows, row_len, nrows, terms| {
            flat_gf8d(terms, nrows, |matrix| {
                x86::gf8::mul_add_matrix_affine_with(token, rows, row_len, nrows, matrix)
            });
        },
    );
    check_matrix_overwrite(
        "gf8::mul_into_matrix_affine_with",
        ROW_LENS,
        gf8d_coeff_at2,
        gf8d_ref,
        |rows, row_len, nrows, terms| {
            flat_gf8d(terms, nrows, |matrix| {
                x86::gf8::mul_into_matrix_affine_with(token, rows, row_len, nrows, matrix)
            });
        },
    );
}

/// Build a stable FlatMatrix over a term list and hand it to `body`.
fn flat_gf8b(
    terms: &[(&[gf8b::Elem], &[u8])],
    nrows: usize,
    body: impl FnOnce(&FlatMatrix<'_, gf8b::Elem>),
) {
    let mut flat = Vec::with_capacity(terms.len() * nrows);
    for &(coeffs, _) in terms {
        flat.extend_from_slice(coeffs);
    }
    let srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
    body(&FlatMatrix {
        coefficients: &flat,
        nrows,
        sources: &srcs,
    });
}

fn flat_gf8d(
    terms: &[(&[gf8d::Elem], &[u8])],
    nrows: usize,
    body: impl FnOnce(&FlatMatrix<'_, gf8d::Elem>),
) {
    let mut flat = Vec::with_capacity(terms.len() * nrows);
    for &(coeffs, _) in terms {
        flat.extend_from_slice(coeffs);
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
            scattered_case_gf8b(token, row_len, &starts[..nrows]);
            scattered_case_gf8d(token, row_len, &starts[..nrows]);
        }
    }
}

fn scattered_case_gf8b(token: X64V3GfniCryptoToken, row_len: usize, starts: &[usize]) {
    const TOTAL: usize = 512;
    let nterms = 3;
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x600 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<gf8b::Elem>> = (0..nterms)
        .map(|t| (0..starts.len()).map(|j| gf8b_coeff_at2(t, j)).collect())
        .collect();
    let terms: Vec<(&[gf8b::Elem], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut got = noise(TOTAL, 0xe1);
    let mut want = got.clone();
    x86::gf8::mul_add_matrix_at_gfni(token, &mut got, row_len, starts, &terms);
    for &(coeffs, src) in &terms {
        for (&start, &coeff) in starts.iter().zip(coeffs) {
            gf8b_ref(&mut want[start..start + row_len], coeff, src);
        }
    }
    assert_eq!(got, want, "mul_add_matrix_at_gfni: row_len {row_len}");
}

fn scattered_case_gf8d(token: X64V3GfniCryptoToken, row_len: usize, starts: &[usize]) {
    const TOTAL: usize = 512;
    let nterms = 3;
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x640 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..nterms)
        .map(|t| (0..starts.len()).map(|j| gf8d_coeff_at2(t, j)).collect())
        .collect();
    let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut got = noise(TOTAL, 0xe2);
    let mut want = got.clone();
    x86::gf8::mul_add_matrix_at_affine(token, &mut got, row_len, starts, &terms);
    for &(coeffs, src) in &terms {
        for (&start, &coeff) in starts.iter().zip(coeffs) {
            gf8d_ref(&mut want[start..start + row_len], coeff, src);
        }
    }
    assert_eq!(got, want, "mul_add_matrix_at_affine: row_len {row_len}");
}

// ---------------------------------------------------------------------------
// Degenerate boundaries: empty terms/sources and zero rows.
// ---------------------------------------------------------------------------

#[test]
fn proven_gf8_prepared_matrix_wrappers_accept_empty_geometry() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let mut rows = noise(32, 0xee);
    let want = rows.clone();
    let row_len = rows.len();

    x86::gf8::mul_add_matrix_affine_prepared_with(token, &mut rows, 0, 3, &[], &[]);
    x86::gf8::mul_into_matrix_affine_prepared_with(token, &mut rows, row_len, 0, &[], &[]);

    assert_eq!(rows, want);
}

#[test]
fn proven_overwrite_wrappers_zero_addressed_rows_on_empty_terms() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    let (row_len, nrows, surplus) = (96usize, 6, 17);

    // Empty term list: the overwrite sum is zero over the addressed rows;
    // the surplus stays untouched.
    let mut got = noise(nrows * row_len + surplus, 0xf1);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_gfni(token, &mut got, row_len, nrows, &[]);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_gfni: empty terms");

    let mut got = noise(nrows * row_len + surplus, 0xf2);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_affine(token, &mut got, row_len, nrows, &[]);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_affine: empty terms");

    // The FlatMatrix twins must agree with their slice twins.
    let srcs: Vec<&[u8]> = Vec::new();
    let flat: Vec<gf8b::Elem> = Vec::new();
    let matrix = FlatMatrix {
        coefficients: &flat,
        nrows,
        sources: &srcs,
    };
    let mut got = noise(nrows * row_len + surplus, 0xf3);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_gfni_with(token, &mut got, row_len, nrows, &matrix);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_gfni_with: empty terms");

    let flat: Vec<gf8d::Elem> = Vec::new();
    let matrix = FlatMatrix {
        coefficients: &flat,
        nrows,
        sources: &srcs,
    };
    let mut got = noise(nrows * row_len + surplus, 0xf4);
    let mut want = got.clone();
    x86::gf8::mul_into_matrix_affine_with(token, &mut got, row_len, nrows, &matrix);
    want[..nrows * row_len].fill(0);
    assert_eq!(got, want, "mul_into_matrix_affine_with: empty terms");

    // Accumulation with no terms is a no-op: nothing changes.
    let mut got = noise(nrows * row_len + surplus, 0xf5);
    let want = got.clone();
    x86::gf8::mul_add_matrix_gfni(token, &mut got, row_len, nrows, &[]);
    assert_eq!(got, want, "mul_add_matrix_gfni: empty terms is a no-op");

    // Zero rows: nothing is addressed, nothing changes (every term supplies
    // zero coefficients, matching the zero row count).
    let mut got = noise(row_len + surplus, 0xf6);
    let want = got.clone();
    let src = noise(row_len, 0xf7);
    let no_coeffs: [gf8b::Elem; 0] = [];
    let terms: Vec<(&[gf8b::Elem], &[u8])> = vec![(no_coeffs.as_slice(), src.as_slice())];
    x86::gf8::mul_add_matrix_gfni(token, &mut got, row_len, 0, &terms);
    x86::gf8::mul_into_matrix_gfni(token, &mut got, row_len, 0, &terms);
    assert_eq!(got, want, "zero rows address nothing");

    // Zero-length rows with a zero-length source are a documented no-op;
    // an empty gather with no sources must also leave the destination be.
    let mut got = noise(surplus, 0xf8);
    let want = got.clone();
    let coeffs = [gf8b::Elem::from_raw(7)];
    x86::gf8::mul_add_scatter_gfni(token, &mut got, 0, &coeffs, &[]);
    x86::gf8::mul_add_gather_gfni(token, &mut got, &[], &[]);
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
    x86::gf16::mul_add_matrix_gfni(token, &mut got, row_len, nrows, &[]);
    assert_eq!(
        got, want,
        "gf16::mul_add_matrix_gfni: empty terms is a no-op"
    );

    // Zero rows and empty gather sources.
    let mut got = noise(row_len + surplus, 0x172);
    let want = got.clone();
    let coeffs = [gf16::Elem::from_raw(0xbeef)];
    x86::gf16::mul_add_gather_gfni(token, &mut got, &[], &[]);
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
        x86::gf8::mul_add_gfni(token, &mut dst, gf8b::Elem::from_raw(3), &src);
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
            TowerCoeff::new(gf16::Elem::from_raw(7)),
            &src,
        );
    });
}

#[test]
fn proven_scatter_rejects_short_rows_buffer() {
    rejects_geometry("gf8 scatter", |token| {
        let mut rows = vec![0u8; 16];
        let src = vec![0u8; 16];
        let coeffs = [gf8b::Elem::from_raw(1), gf8b::Elem::from_raw(2)];
        x86::gf8::mul_add_scatter_gfni(token, &mut rows, 16, &coeffs, &src);
    });
}

#[test]
fn proven_matrix_rejects_term_coefficient_mismatch() {
    rejects_geometry("gf8 matrix", |token| {
        let mut rows = vec![0u8; 64];
        let src = vec![0u8; 32];
        let short = [gf8b::Elem::from_raw(1)];
        let terms: Vec<(&[gf8b::Elem], &[u8])> = vec![(short.as_slice(), src.as_slice())];
        x86::gf8::mul_add_matrix_gfni(token, &mut rows, 32, 2, &terms);
    });
}

#[test]
fn proven_matrix_scattered_rejects_overlapping_rows() {
    rejects_geometry("gf8 matrix_at overlap", |token| {
        let mut dst = vec![0u8; 128];
        let src = vec![0u8; 64];
        let coeffs = [gf8b::Elem::from_raw(1); 2];
        let terms: Vec<(&[gf8b::Elem], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
        let starts = [0usize, 32];
        x86::gf8::mul_add_matrix_at_gfni(token, &mut dst, 64, &starts, &terms);
    });
}

#[test]
fn proven_matrix_scattered_rejects_out_of_bounds_row() {
    rejects_geometry("gf8 matrix_at bounds", |token| {
        let mut dst = vec![0u8; 64];
        let src = vec![0u8; 64];
        let coeffs = [gf8b::Elem::from_raw(1)];
        let terms: Vec<(&[gf8b::Elem], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
        let starts = [16usize];
        x86::gf8::mul_add_matrix_at_gfni(token, &mut dst, 64, &starts, &terms);
    });
}

#[test]
fn proven_gather_rejects_count_mismatch() {
    rejects_geometry("gf8 gather", |token| {
        let mut dst = vec![0u8; 32];
        let src = vec![0u8; 32];
        let coeffs = [gf8b::Elem::from_raw(1), gf8b::Elem::from_raw(2)];
        let srcs: Vec<&[u8]> = vec![src.as_slice()];
        x86::gf8::mul_add_gather_gfni(token, &mut dst, &coeffs, &srcs);
    });
}

#[test]
fn proven_xor_rows_rejects_partial_row() {
    rejects_geometry("xor_rows avx2", |v3: X64V3Token| {
        let mut dst = vec![0u8; 33];
        let src = vec![0u8; 33];
        x86::bytes::xor_rows_avx2(v3, &mut dst, &src, 16);
    });
}

#[test]
fn proven_xor_rows_rejects_zero_row_len() {
    rejects_geometry("xor_rows sse2", |v2: X64V2Token| {
        let mut dst = vec![0u8; 16];
        let src = vec![0u8; 16];
        x86::bytes::xor_rows_sse2(v2.v1(), &mut dst, &src, 0);
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
        x86::prime::add_assign_m31_avx2(v3, &mut dst, &src);
    });
}

// ---------------------------------------------------------------------------
// Experimental overwrite shapes (perf candidates, `internals`-published).
// ---------------------------------------------------------------------------

/// A 32-byte-aligned window of deterministic noise; returns the backing and
/// the offset of the aligned window.
fn aligned_noise(len: usize, seed: u64) -> (Vec<u8>, usize) {
    let data = noise(len, seed);
    let mut backing = vec![0u8; len + 32];
    let off = (-(backing.as_ptr() as isize) & 31) as usize;
    backing[off..off + len].copy_from_slice(&data);
    (backing, off)
}

fn pack_table(table: &ScaleTable) -> [u8; 32] {
    let mut packed = [0u8; 32];
    packed[..16].copy_from_slice(&table.lo);
    packed[16..].copy_from_slice(&table.hi);
    packed
}

/// Build (oracle rows accumulated from zero, term list) for a fixed
/// two-row-shape experiment.
fn two_row_case<E: Copy>(
    row_len: usize,
    nterms: usize,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    body: impl FnOnce(Vec<u8>, Vec<(&[E], &[u8])>),
) {
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x440 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<E>> = (0..nterms)
        .map(|t| (0..2).map(|j| coeff_at(t, j)).collect())
        .collect();
    let terms: Vec<(&[E], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut want = vec![0u8; row_len * 2];
    for &(coeffs, src) in &terms {
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
            reference(row, coeff, src);
        }
    }
    body(want, terms);
}

#[test]
fn proven_experimental_two_row_overwrite_matches_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    #[cfg(not(miri))]
    let row_lens: &[usize] = &[32usize, 34, 96, 300];
    #[cfg(miri)]
    let row_lens: &[usize] = &[32usize, 34];
    #[cfg(not(miri))]
    let nterm_counts: &[usize] = &[1usize, 3, 17];
    // Single and wide term folds under Miri; every tile width keeps running.
    #[cfg(miri)]
    let nterm_counts: &[usize] = &[1usize, 17];
    for &row_len in row_lens {
        for &nterms in nterm_counts {
            // Temporal stores across every tile width.
            for &lanes in &[2usize, 3, 4] {
                two_row_case(row_len, nterms, gf8b_coeff_at2, gf8b_ref, |want, terms| {
                    let mut got = noise(row_len * 2, 0xd5);
                    x86::gf8::mul_into_matrix2_8b(token, &mut got, row_len, &terms, false, lanes);
                    assert_eq!(
                        got, want,
                        "two-row 8b: len {row_len} lanes {lanes} terms {nterms}"
                    );
                });
                two_row_case(row_len, nterms, gf8d_coeff_at2, gf8d_ref, |want, terms| {
                    let mut got = noise(row_len * 2, 0xd4);
                    x86::gf8::mul_into_matrix2_8d(token, &mut got, row_len, &terms, false, lanes);
                    assert_eq!(
                        got, want,
                        "two-row 8d: len {row_len} lanes {lanes} terms {nterms}"
                    );
                });
            }
            // Non-temporal stores: 32-byte-aligned buffer and 32-multiple
            // row length, the kernel's documented precondition. Skipped under
            // Miri, which cannot execute the fence intrinsic; the temporal
            // legs above already cover the offset-row residue.
            if row_len % 32 == 0 && !cfg!(miri) {
                two_row_case(row_len, nterms, gf8b_coeff_at2, gf8b_ref, |want, terms| {
                    let (mut backing, off) = aligned_noise(row_len * 2, 0xd6);
                    let rows = &mut backing[off..off + row_len * 2];
                    x86::gf8::mul_into_matrix2_8b(token, rows, row_len, &terms, true, 4);
                    assert_eq!(&*rows, &want[..], "two-row nt 8b: len {row_len}");
                });
                two_row_case(row_len, nterms, gf8d_coeff_at2, gf8d_ref, |want, terms| {
                    let (mut backing, off) = aligned_noise(row_len * 2, 0xd7);
                    let rows = &mut backing[off..off + row_len * 2];
                    x86::gf8::mul_into_matrix2_8d(token, rows, row_len, &terms, true, 4);
                    assert_eq!(&*rows, &want[..], "two-row nt 8d: len {row_len}");
                });
            }
        }
    }
}

#[test]
fn proven_experimental_two_row_overwrite_packed_matches_scalar() {
    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    #[cfg(not(miri))]
    let row_lens: &[usize] = &[32usize, 34, 128, 300];
    #[cfg(miri)]
    let row_lens: &[usize] = &[32usize, 34];
    #[cfg(not(miri))]
    let nterm_counts: &[usize] = &[1usize, 3, 17];
    // Single and wide term folds under Miri; both packed widths keep running.
    #[cfg(miri)]
    let nterm_counts: &[usize] = &[1usize, 17];
    for &row_len in row_lens {
        for &nterms in nterm_counts {
            for &lanes in &[1usize, 2] {
                two_row_case(row_len, nterms, gf8b_coeff_at2, gf8b_ref, |want, terms| {
                    let packed: Vec<[u8; 32]> = terms
                        .iter()
                        .flat_map(|&(coeffs, _)| coeffs.iter().map(|&c| pack_table(scale_table(c))))
                        .collect();
                    let srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
                    let mut got = noise(row_len * 2, 0xd6);
                    x86::gf8::mul_into_matrix2_shuffle_packed(
                        v3, &mut got, row_len, &packed, &srcs, lanes,
                    );
                    assert_eq!(
                        got, want,
                        "two-row packed 8b: len {row_len} lanes {lanes} terms {nterms}"
                    );
                });
            }
        }
    }
}

fn six_row_case<E: Copy>(
    row_len: usize,
    nterms: usize,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    body: impl FnOnce(Vec<u8>, Vec<(&[E], &[u8])>),
) {
    let sources: Vec<Vec<u8>> = (0..nterms)
        .map(|t| noise(row_len, 0x460 + t as u64))
        .collect();
    let coeff_sets: Vec<Vec<E>> = (0..nterms)
        .map(|t| (0..6).map(|j| coeff_at(t, j)).collect())
        .collect();
    let terms: Vec<(&[E], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut want = vec![0u8; row_len * 6];
    for &(coeffs, src) in &terms {
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
            reference(row, coeff, src);
        }
    }
    body(want, terms);
}

#[test]
fn proven_experimental_six_row_overwrite_matches_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    const NROWS: usize = 6;
    #[cfg(not(miri))]
    let row_lens: &[usize] = &[32usize, 34, 128, 300];
    #[cfg(miri)]
    let row_lens: &[usize] = &[32usize, 34];
    #[cfg(not(miri))]
    let nterm_counts: &[usize] = &[1usize, 2, 9, 17];
    // Single, pair, and one count past the resolve chunk under Miri.
    #[cfg(miri)]
    let nterm_counts: &[usize] = &[1usize, 9];
    for &row_len in row_lens {
        for &nterms in nterm_counts {
            six_row_case(row_len, nterms, gf8b_coeff_at2, gf8b_ref, |want, terms| {
                let mut got = noise(row_len * NROWS, 0xe6);
                x86::gf8::mul_into_matrix6_shuffle_8b(token, &mut got, row_len, &terms);
                assert_eq!(got, want, "six-row 8b: len {row_len} terms {nterms}");
            });
            six_row_case(row_len, nterms, gf8d_coeff_at2, gf8d_ref, |want, terms| {
                let mut got = noise(row_len * NROWS, 0xe7);
                x86::gf8::mul_into_matrix6_shuffle_8d(token, &mut got, row_len, &terms);
                assert_eq!(got, want, "six-row 8d: len {row_len} terms {nterms}");
            });
            six_row_case(row_len, nterms, gf8b_coeff_at2, gf8b_ref, |want, terms| {
                let packed: Vec<[u8; 32]> = terms
                    .iter()
                    .flat_map(|&(coeffs, _)| coeffs.iter().map(|&c| pack_table(scale_table(c))))
                    .collect();
                let srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
                let mut got = noise(row_len * NROWS, 0xe8);
                x86::gf8::mul_into_matrix6_shuffle_packed_8b(
                    token, &mut got, row_len, &packed, &srcs,
                );
                assert_eq!(got, want, "six-row packed 8b: len {row_len} terms {nterms}");
            });
            six_row_case(row_len, nterms, gf8d_coeff_at2, gf8d_ref, |want, terms| {
                let packed: Vec<[u8; 32]> = terms
                    .iter()
                    .flat_map(|&(coeffs, _)| coeffs.iter().map(|&c| pack_table(scale_table_8d(c))))
                    .collect();
                let srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
                let mut got = noise(row_len * NROWS, 0xe9);
                x86::gf8::mul_into_matrix6_shuffle_packed_8d(
                    token, &mut got, row_len, &packed, &srcs,
                );
                assert_eq!(got, want, "six-row packed 8d: len {row_len} terms {nterms}");
            });
        }
    }
}

#[test]
fn proven_experimental_one_row_overwrite_matches_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    // 384 is the first common multiple of both tile widths; the residues
    // exercise each body's cleanup loop.
    #[cfg(not(miri))]
    let row_lens: &[usize] = &[32usize, 96, 128, 384, 416];
    #[cfg(miri)]
    let row_lens: &[usize] = &[32usize, 128];
    #[cfg(not(miri))]
    let source_counts: &[usize] = &[1usize, 2, 5, 33];
    // Single and multi-source folds under Miri.
    #[cfg(miri)]
    let source_counts: &[usize] = &[1usize, 5];
    for &row_len in row_lens {
        for &sources in source_counts {
            one_row_case(row_len, sources, |want, _coeffs, srcs, maps, terms| {
                for &lanes in &[3usize, 4] {
                    let mut got = noise(row_len, 0x6b2);
                    x86::gf8::mul_into_matrix1_tile_8d(token, &mut got, row_len, &terms, lanes);
                    assert_eq!(got, want, "one-row tile: len {row_len} lanes {lanes}");

                    let mut got = noise(row_len, 0x7c2);
                    x86::gf8::mul_into_matrix1_fullinit_8d(token, &mut got, row_len, &terms, lanes);
                    assert_eq!(got, want, "one-row fullinit: len {row_len} lanes {lanes}");

                    for &chunk in &[32usize, 96] {
                        let mut got = noise(row_len, 0x7c3);
                        x86::gf8::mul_into_matrix1_chunk_8d(
                            token, &mut got, row_len, &terms, lanes, chunk,
                        );
                        assert_eq!(
                            got, want,
                            "one-row chunk {chunk}: len {row_len} lanes {lanes}"
                        );
                    }
                }
                if row_len % 32 == 0 {
                    for &lanes in &[3usize, 4] {
                        let mut got = noise(row_len, 0x6b3);
                        x86::gf8::mul_into_matrix1_external_8d(
                            token, &mut got, row_len, maps, srcs, lanes,
                        );
                        assert_eq!(got, want, "one-row external: len {row_len} lanes {lanes}");
                    }
                    for &chunk in &[0usize, 64] {
                        let mut got = noise(row_len, 0x7c4);
                        x86::gf8::mul_into_matrix1_external_chunk_8d(
                            token, &mut got, row_len, maps, srcs, 4, chunk,
                        );
                        assert_eq!(got, want, "one-row external chunk {chunk}: len {row_len}");
                    }
                }
            });
        }
    }
}

/// One-row experiment inputs: coefficients (one per term), sources, affine
/// maps, single-coefficient terms, and the zero-seeded oracle row.
fn one_row_case(
    row_len: usize,
    sources: usize,
    body: impl FnOnce(Vec<u8>, &[gf8d::Elem], &[&[u8]], &[u64], Vec<(&[gf8d::Elem], &[u8])>),
) {
    let buffers: Vec<Vec<u8>> = (0..sources)
        .map(|t| noise(row_len, 0x6a1 + t as u64 * 17 + row_len as u64))
        .collect();
    let srcs: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
    let coeffs: Vec<gf8d::Elem> = (0..sources)
        .map(|i| gf8d::Elem::from_raw((i as u8).wrapping_mul(29)))
        .collect();
    let maps: Vec<u64> = coeffs.iter().copied().map(affine_8d).collect();
    let terms: Vec<(&[gf8d::Elem], &[u8])> = coeffs
        .iter()
        .zip(&srcs)
        .map(|(c, s)| (core::slice::from_ref(c), *s))
        .collect();
    let mut want = vec![0u8; row_len];
    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
        gf8d_ref(&mut want, coeff, src);
    }
    body(want, &coeffs, &srcs, &maps, terms);
}

#[test]
fn proven_experimental_multi_row_chunk_and_grouped_match_scalar() {
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    #[cfg(not(miri))]
    const ROW_LEN: usize = 384;
    // Shorter rows under Miri: the residue is identical per tile.
    #[cfg(miri)]
    const ROW_LEN: usize = 128;
    #[cfg(not(miri))]
    let nrow_counts: &[usize] = &[1usize, 2, 3, 4, 6, 7];
    // Single, full group, and group-plus-pair under Miri.
    #[cfg(miri)]
    let nrow_counts: &[usize] = &[1usize, 4, 6];
    for &nrows in nrow_counts {
        for &sources in &[2usize, 33] {
            let buffers: Vec<Vec<u8>> = (0..sources)
                .map(|t| noise(ROW_LEN, 0x7d1 + t as u64 * 23 + nrows as u64))
                .collect();
            let srcs: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
            let columns: Vec<Vec<gf8d::Elem>> = (0..sources)
                .map(|t| {
                    (0..nrows)
                        .map(|r| gf8d::Elem::from_raw(((t * 7 + r) as u8).wrapping_mul(29)))
                        .collect()
                })
                .collect();
            let terms: Vec<(&[gf8d::Elem], &[u8])> = columns
                .iter()
                .zip(&srcs)
                .map(|(c, s)| (c.as_slice(), *s))
                .collect();

            let mut want = vec![0u8; nrows * ROW_LEN];
            for row in 0..nrows {
                for term in 0..sources {
                    let mut piece = vec![0u8; ROW_LEN];
                    gf8d_ref(&mut piece, columns[term][row], srcs[term]);
                    let target = &mut want[row * ROW_LEN..(row + 1) * ROW_LEN];
                    for (byte, &value) in target.iter_mut().zip(piece.iter()) {
                        *byte ^= value;
                    }
                }
            }

            for &chunk in &[32usize, 96] {
                let mut got = noise(nrows * ROW_LEN, 0x7d2);
                x86::gf8::mul_into_matrix_chunk_8d(token, &mut got, ROW_LEN, nrows, &terms, chunk);
                assert_eq!(got, want, "multi chunk {chunk}: nrows {nrows}");
            }

            // Group-major maps: every four-row group's words, then the
            // pair's, then the single row's, each term-major.
            let mut maps: Vec<u64> = Vec::with_capacity(nrows * sources);
            let mut g = 0;
            while g + 4 <= nrows {
                for column in &columns {
                    for coeff in &column[g..g + 4] {
                        maps.push(affine_8d(*coeff));
                    }
                }
                g += 4;
            }
            if g + 2 <= nrows {
                for column in &columns {
                    for coeff in &column[g..g + 2] {
                        maps.push(affine_8d(*coeff));
                    }
                }
                g += 2;
            }
            if g < nrows {
                for column in &columns {
                    maps.push(affine_8d(column[g]));
                }
            }
            for &chunk in &[0usize, 64] {
                let mut got = noise(nrows * ROW_LEN, 0x7d3);
                x86::gf8::mul_into_matrix_external_grouped_8d(
                    token, &mut got, ROW_LEN, nrows, &maps, &srcs, chunk,
                );
                assert_eq!(got, want, "grouped chunk {chunk}: nrows {nrows}");
            }
        }
    }
}

#[test]
fn proven_resolve_probe_sees_every_coefficient() {
    let coeffs = [gf8d::Elem::from_raw(7)];
    let src = vec![1u8; 8];
    let terms: Vec<(&[gf8d::Elem], &[u8])> = vec![(coeffs.as_slice(), src.as_slice())];
    let base = x86::gf8::resolve_probe_8d(&terms);
    let bumped = [gf8d::Elem::from_raw(7 ^ 0x5a)];
    let bumped_terms: Vec<(&[gf8d::Elem], &[u8])> = vec![(bumped.as_slice(), src.as_slice())];
    let other = x86::gf8::resolve_probe_8d(&bumped_terms);
    assert_ne!(base, other, "the probe must depend on each coefficient");
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

    check_elementwise::<Gf16>("gf16::mul_elementwise_gfni", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_gfni(v3gfni, dst, a, b)
    });
    check_elementwise::<Gf16>("gf16::mul_elementwise_avx2", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_avx2(v3, dst, a, b)
    });
    check_elementwise::<Gf16>("gf16::mul_elementwise_ssse3", LENGTHS16, |dst, a, b| {
        x86::gf16::mul_elementwise_ssse3(v2, dst, a, b)
    });

    // Scatter/gather/matrix across all three tiers.
    check_scatter(
        "gf16::mul_add_scatter_gfni",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| gf16::Elem::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_gfni(v3gfni, rows, row_len, coeffs, src)
        },
    );
    check_scatter(
        "gf16::mul_add_scatter_avx2",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| gf16::Elem::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_avx2(v3, rows, row_len, coeffs, src)
        },
    );
    check_scatter(
        "gf16::mul_add_scatter_ssse3",
        EVEN_ROW_LENS,
        gf16_ref,
        |j| gf16::Elem::from_raw((j as u16).wrapping_mul(7411)),
        |rows, row_len, coeffs, src| {
            x86::gf16::mul_add_scatter_ssse3(v2, rows, row_len, coeffs, src)
        },
    );
    for &row_len in &[32usize, 130] {
        check_gather(
            "gf16::mul_add_gather_gfni",
            row_len,
            gf16_ref,
            |i| gf16::Elem::from_raw((i as u16).wrapping_mul(613)),
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_gfni(v3gfni, dst, coeffs, srcs),
        );
        check_gather(
            "gf16::mul_add_gather_avx2",
            row_len,
            gf16_ref,
            |i| gf16::Elem::from_raw((i as u16).wrapping_mul(613)),
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_avx2(v3, dst, coeffs, srcs),
        );
        check_gather(
            "gf16::mul_add_gather_ssse3",
            row_len,
            gf16_ref,
            |i| gf16::Elem::from_raw((i as u16).wrapping_mul(613)),
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
        "gf16::mul_add_matrix_avx2",
        EVEN_ROW_LENS,
        gf16_coeff_at2,
        gf16_ref,
        |rows, row_len, nrows, terms| {
            x86::gf16::mul_add_matrix_avx2(v3, rows, row_len, nrows, terms)
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
    let fp16_coeffs: Vec<fan_paar::fp16::Elem> =
        [0u16, 1, 0x0100, 0x00ff, 0xbeef, 0x7411, u16::MAX]
            .iter()
            .map(|&c| fan_paar::fp16::Elem::from_raw(c))
            .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp16_coeffs: Vec<fan_paar::fp16::Elem> = [0u16, 1, 0xbeef, u16::MAX]
        .iter()
        .map(|&c| fan_paar::fp16::Elem::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_avx2",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_add_avx2(v3, dst, &FpTowerTables::new(c), src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_avx2",
        LENGTHS16,
        &fp16_coeffs,
        |dst, c| wiedemann::mul_assign(dst, u64::from(c.to_raw()), 16),
        |dst, c| x86::fan_paar::mul_assign_avx2(v3, dst, &FpTowerTables::new(c)),
    );
    check_mul_into(
        "fan_paar::mul_into_avx2",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_into_avx2(v3, dst, &FpTowerTables::new(c), src),
    );
    check_mul_add(
        "fan_paar::mul_add_ssse3",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_add_ssse3(v2, dst, &FpTowerTables::new(c), src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_ssse3",
        LENGTHS16,
        &fp16_coeffs,
        |dst, c| wiedemann::mul_assign(dst, u64::from(c.to_raw()), 16),
        |dst, c| x86::fan_paar::mul_assign_ssse3(v2, dst, &FpTowerTables::new(c)),
    );
    check_mul_into(
        "fan_paar::mul_into_ssse3",
        LENGTHS16,
        &fp16_coeffs,
        fp16_ref,
        |dst, c, src| x86::fan_paar::mul_into_ssse3(v2, dst, &FpTowerTables::new(c), src),
    );

    #[cfg(not(miri))]
    let fp32_coeffs: Vec<fan_paar::fp32::Elem> =
        [0u32, 1, 0x0001_0000, 0x0000_ffff, 0x1234_5678, u32::MAX]
            .iter()
            .map(|&c| fan_paar::fp32::Elem::from_raw(c))
            .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp32_coeffs: Vec<fan_paar::fp32::Elem> = [0u32, 1, 0x1234_5678, u32::MAX]
        .iter()
        .map(|&c| fan_paar::fp32::Elem::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_fp32_avx2",
        LENGTHS32,
        &fp32_coeffs,
        fp32_ref,
        |dst, c, src| x86::fan_paar::mul_add_fp32_avx2(v3, dst, c, src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_fp32_avx2",
        LENGTHS32,
        &fp32_coeffs,
        |dst, c| wiedemann::mul_assign(dst, u64::from(c.to_raw()), 32),
        |dst, c| x86::fan_paar::mul_assign_fp32_avx2(v3, dst, c),
    );
    check_mul_into(
        "fan_paar::mul_into_fp32_avx2",
        LENGTHS32,
        &fp32_coeffs,
        fp32_ref,
        |dst, c, src| x86::fan_paar::mul_into_fp32_avx2(v3, dst, c, src),
    );

    #[cfg(not(miri))]
    let fp64_coeffs: Vec<fan_paar::fp64::Elem> = [
        0u64,
        1,
        0x0000_0001_0000_0000,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    ]
    .iter()
    .map(|&c| fan_paar::fp64::Elem::from_raw(c))
    .collect();
    // Short-circuits, one mixed value, and the extreme under Miri.
    #[cfg(miri)]
    let fp64_coeffs: Vec<fan_paar::fp64::Elem> = [0u64, 1, 0x0123_4567_89ab_cdef, u64::MAX]
        .iter()
        .map(|&c| fan_paar::fp64::Elem::from_raw(c))
        .collect();
    check_mul_add(
        "fan_paar::mul_add_fp64_avx2",
        LENGTHS64,
        &fp64_coeffs,
        fp64_ref,
        |dst, c, src| x86::fan_paar::mul_add_fp64_avx2(v3, dst, c, src),
    );
    check_mul_assign(
        "fan_paar::mul_assign_fp64_avx2",
        LENGTHS64,
        &fp64_coeffs,
        |dst, c| wiedemann::mul_assign(dst, c.to_raw(), 64),
        |dst, c| x86::fan_paar::mul_assign_fp64_avx2(v3, dst, c),
    );
    check_mul_into(
        "fan_paar::mul_into_fp64_avx2",
        LENGTHS64,
        &fp64_coeffs,
        fp64_ref,
        |dst, c, src| x86::fan_paar::mul_into_fp64_avx2(v3, dst, c, src),
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
        chunk.copy_from_slice(&mersenne31::Elem::from_raw(lane).canonical().to_bytes());
    }
    bytes
}

/// Noise with every 8-byte lane reduced to the canonical Goldilocks range.
fn gld_lanes(n: usize, seed: u64) -> Vec<u8> {
    let mut bytes = noise(n * 8, seed);
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let lane = u64::from_le_bytes(*chunk);
        chunk.copy_from_slice(&goldilocks::Elem::from_raw(lane % goldilocks::MODULUS).to_bytes());
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
    let coeff = mersenne31::Elem::from_raw(0x0532_1cde).canonical();
    type BinOp<'a> = Box<dyn Fn(&mut [u8], &[u8]) + 'a>;

    for &n in LANES {
        let src = m31_lanes(n, 0x91);
        let coeff_raw = coeff.to_raw();

        // add / sub against per-lane field arithmetic.
        for (name, kernel, sub) in [
            (
                "add_assign_m31_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| x86::prime::add_assign_m31_avx2(v3, dst, src))
                    as BinOp,
                false,
            ),
            (
                "sub_assign_m31_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::sub_assign_m31_avx2(v3, dst, src)
                }),
                true,
            ),
            (
                "add_assign_m31_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::add_assign_m31_sse42(v2, dst, src)
                }),
                false,
            ),
            (
                "sub_assign_m31_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::sub_assign_m31_sse42(v2, dst, src)
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
                let a = mersenne31::Elem::from_bytes(*d);
                let b = mersenne31::Elem::from_bytes(*s);
                let r = if sub { a.sub(b) } else { a.add(b) };
                d.copy_from_slice(&r.to_bytes());
            }
            assert_eq!(got, want, "{name}: lanes {n}");
        }

        // mul_add / mul_assign / mul_into against the scalar oracle.
        let mut got = m31_lanes(n, 0x93);
        let mut want = got.clone();
        x86::prime::mul_add_m31_avx2(v3, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mul_add_m31_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x94);
        let mut want = got.clone();
        x86::prime::mul_add_m31_sse42(v2, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mul_add_m31_sse42: lanes {n}");

        let mut got = m31_lanes(n, 0x95);
        let mut want = got.clone();
        x86::prime::mul_assign_m31_avx2(v3, &mut got, coeff_raw);
        scalar::mul_assign::<Mersenne31>(&mut want, coeff);
        assert_eq!(got, want, "mul_assign_m31_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x96);
        let mut want = vec![0u8; n * 4];
        x86::prime::mul_into_m31_avx2(v3, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mul_into_m31_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x97);
        let mut want = vec![0u8; n * 4];
        x86::prime::mul_into_m31_sse42(v2, &mut got, coeff_raw, &src);
        m31_ref(&mut want, coeff, &src);
        assert_eq!(got, want, "mul_into_m31_sse42: lanes {n}");

        // Elementwise product of two canonical buffers.
        let a = m31_lanes(n, 0x98);
        let b = m31_lanes(n, 0x99);
        let mut got = m31_lanes(n, 0x9a);
        let mut want = got.clone();
        x86::prime::mul_elementwise_m31_avx2(v3, &mut got, &a, &b);
        scalar::mul_elementwise::<Mersenne31>(&mut want, &a, &b);
        assert_eq!(got, want, "mul_elementwise_m31_avx2: lanes {n}");

        let mut got = m31_lanes(n, 0x9b);
        let mut want = got.clone();
        x86::prime::mul_elementwise_m31_sse42(v2, &mut got, &a, &b);
        scalar::mul_elementwise::<Mersenne31>(&mut want, &a, &b);
        assert_eq!(got, want, "mul_elementwise_m31_sse42: lanes {n}");
    }

    // Goldilocks: the same sweep with 8-byte lanes.
    let gld_coeff = goldilocks::Elem::from_raw(0x0123_4567_89ab_cdef % goldilocks::MODULUS);
    for &n in &[0usize, 1, 3, 8, 9, 32] {
        let src = gld_lanes(n, 0xa1);
        let coeff_raw = gld_coeff.to_raw();

        for (name, kernel, sub) in [
            (
                "add_assign_gld_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| x86::prime::add_assign_gld_avx2(v3, dst, src))
                    as BinOp,
                false,
            ),
            (
                "sub_assign_gld_avx2",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::sub_assign_gld_avx2(v3, dst, src)
                }),
                true,
            ),
            (
                "add_assign_gld_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::add_assign_gld_sse42(v2, dst, src)
                }),
                false,
            ),
            (
                "sub_assign_gld_sse42",
                Box::new(|dst: &mut [u8], src: &[u8]| {
                    x86::prime::sub_assign_gld_sse42(v2, dst, src)
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
                let a = goldilocks::Elem::from_bytes(*d);
                let b = goldilocks::Elem::from_bytes(*s);
                let r = if sub { a.sub(b) } else { a.add(b) };
                d.copy_from_slice(&r.to_bytes());
            }
            assert_eq!(got, want, "{name}: lanes {n}");
        }

        let mut got = gld_lanes(n, 0xa3);
        let mut want = got.clone();
        x86::prime::mul_add_gld_avx2(v3, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "mul_add_gld_avx2: lanes {n}");

        let mut got = gld_lanes(n, 0xa4);
        let mut want = got.clone();
        x86::prime::mul_add_gld_sse42(v2, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "mul_add_gld_sse42: lanes {n}");

        let mut got = gld_lanes(n, 0xa5);
        let mut want = got.clone();
        x86::prime::mul_assign_gld_sse42(v2, &mut got, coeff_raw);
        scalar::mul_assign::<Goldilocks>(&mut want, gld_coeff);
        assert_eq!(got, want, "mul_assign_gld_sse42: lanes {n}");

        let mut got = gld_lanes(n, 0xa6);
        let mut want = vec![0u8; n * 8];
        x86::prime::mul_into_gld_avx2(v3, &mut got, coeff_raw, &src);
        gld_ref(&mut want, gld_coeff, &src);
        assert_eq!(got, want, "mul_into_gld_avx2: lanes {n}");

        let a = gld_lanes(n, 0xa7);
        let b = gld_lanes(n, 0xa8);
        let mut got = gld_lanes(n, 0xa9);
        let mut want = got.clone();
        x86::prime::mul_elementwise_gld_avx2(v3, &mut got, &a, &b);
        scalar::mul_elementwise::<Goldilocks>(&mut want, &a, &b);
        assert_eq!(got, want, "mul_elementwise_gld_avx2: lanes {n}");

        let mut got = gld_lanes(n, 0xaa);
        let mut want = got.clone();
        x86::prime::mul_elementwise_gld_sse42(v2, &mut got, &a, &b);
        scalar::mul_elementwise::<Goldilocks>(&mut want, &a, &b);
        assert_eq!(got, want, "mul_elementwise_gld_sse42: lanes {n}");
    }
}

// ---------------------------------------------------------------------------
// AVX-512 tier: type-checks everywhere, executes only where the token
// summons. The deferred tier is not on the dispatch ladder; on hosts
// without it these tests print a skip and return.
// ---------------------------------------------------------------------------

#[test]
fn proven_avx512_kernels_match_scalar_where_summonable() {
    let Some(v4) = X64V4Token::summon() else {
        eprintln!("skipping: AVX-512 not summonable on this host");
        return;
    };
    let Some(v4x) = X64V4xToken::summon() else {
        eprintln!("skipping: AVX-512+GFNI not summonable on this host");
        return;
    };

    for &len in LENGTHS8 {
        let src = noise(len, 0xb1);
        let mut got = noise(len, 0xb2);
        let mut want = got.clone();
        x86::avx512::proven::xor(v4, &mut got, &src);
        scalar::xor(&mut want, &src);
        assert_eq!(got, want, "avx512::xor: len {len}");

        let mut got = noise(len, 0xb3);
        let mut want = got.clone();
        x86::avx512::proven::gf8_mul_add(v4x, &mut got, gf8b::Elem::from_raw(0x53), &src);
        gf8b_ref(&mut want, gf8b::Elem::from_raw(0x53), &src);
        assert_eq!(got, want, "avx512::gf8_mul_add: len {len}");

        let mut got = noise(len, 0xb4);
        let mut want = vec![0u8; len];
        x86::avx512::proven::gf8_mul_into(v4x, &mut got, gf8b::Elem::from_raw(0xd3), &src);
        gf8b_ref(&mut want, gf8b::Elem::from_raw(0xd3), &src);
        assert_eq!(got, want, "avx512::gf8_mul_into: len {len}");
    }

    // Scatter, gather, and a small matrix over the 64-byte tier.
    let (row_len, nrows) = (96usize, 6);
    let src = noise(row_len, 0xb5);
    let coeffs: Vec<gf8b::Elem> = (0..nrows)
        .map(|j| gf8b::Elem::from_raw((j as u8).wrapping_mul(29)))
        .collect();
    let mut got = noise(row_len * nrows, 0xb6);
    let mut want = got.clone();
    x86::avx512::proven::gf8_mul_add_scatter(v4x, &mut got, row_len, &coeffs, &src);
    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
        gf8b_ref(row, coeff, &src);
    }
    assert_eq!(got, want, "avx512::gf8_mul_add_scatter");

    let sources: Vec<Vec<u8>> = (0..3).map(|t| noise(row_len, 0xb7 + t)).collect();
    let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let gcoeffs: Vec<gf8b::Elem> = (0..3)
        .map(|i| gf8b::Elem::from_raw((i as u8).wrapping_mul(37).wrapping_add(3)))
        .collect();
    let mut got = noise(row_len, 0xb8);
    let mut want = got.clone();
    x86::avx512::proven::gf8_mul_add_gather(v4x, &mut got, &gcoeffs, &srcs);
    for (&coeff, &src) in gcoeffs.iter().zip(&srcs) {
        gf8b_ref(&mut want, coeff, src);
    }
    assert_eq!(got, want, "avx512::gf8_mul_add_gather");

    let coeff_sets: Vec<Vec<gf8b::Elem>> = (0..3)
        .map(|t| (0..nrows).map(|j| gf8b_coeff_at2(t, j)).collect())
        .collect();
    let terms: Vec<(&[gf8b::Elem], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(c, s)| (c.as_slice(), s.as_slice()))
        .collect();
    let mut got = noise(row_len * nrows, 0xb9);
    let mut want = got.clone();
    x86::avx512::proven::gf8_mul_add_matrix(v4x, &mut got, row_len, nrows, &terms);
    fold_terms(&mut want, row_len, &terms, &gf8b_ref);
    assert_eq!(got, want, "avx512::gf8_mul_add_matrix");

    // GF(2^16) tier entries.
    let src = noise(128, 0xba);
    let mut got = noise(128, 0xbb);
    let mut want = got.clone();
    let coeff = gf16::Elem::from_raw(0xbeef);
    x86::avx512::proven::gf16_mul_add(v4x, &mut got, TowerCoeff::new(coeff), &src);
    gf16_ref(&mut want, coeff, &src);
    assert_eq!(got, want, "avx512::gf16_mul_add");
}
