//! Per-backend differential tests.
//!
//! [`crate::ops`] can only exercise the one backend the host selected. These
//! tests reach past dispatch and call every architecture kernel the CPU can
//! actually run, comparing each against the portable reference in
//! [`crate::kernel::scalar`]. On an AVX-512 GFNI machine that means AVX-512,
//! AVX2 GFNI, AVX2, SSSE3, and scalar are covered by one `cargo test`.
//!
//! Buffer lengths deliberately straddle every lane and unroll boundary. Most
//! SIMD bugs live in the tail, not the body.
// The tower and Fan–Paar differential helpers exist to compare a kernel
// against the portable reference. Only x86 has kernels for those fields
// today, so on any other target — or with `simd` off — the helpers have
// nothing to drive and are legitimately dead.
#![cfg_attr(
    not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))),
    allow(dead_code)
)]
// Seeds and geometry are deliberately reduced to field widths below.
#![allow(clippy::cast_possible_truncation)]

extern crate std;

use std::vec;
use std::vec::Vec;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::field::gf8d;
use crate::field::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf32, Gf64, Goldilocks, fan_paar, gf8b, gf16, gf32,
    gf64, quad_mersenne31,
};
use crate::kernel::FieldKernels;
use crate::kernel::scalar;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::FpTowerTables;
use crate::kernel::tables::{ScaleTable, TowerCoeff, TowerTables, scale_table};
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::{affine_8d, scale_table_8d};

/// Lengths covering: empty, sub-lane, exact lanes, lane+1, several unroll
/// tiles, and a large odd size. All even so GF(2^16) can use the same list.
const LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
];

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

/// GF(2^8) coefficients worth testing: the two short-circuits, the extremes,
/// and a spread through the field.
fn gf8_coeffs() -> Vec<gf8b::Elem> {
    let mut coeffs = vec![
        gf8b::Elem(0),
        gf8b::Elem(1),
        gf8b::Elem(2),
        gf8b::Elem(0xff),
    ];
    coeffs.extend((0..=u8::MAX).step_by(23).map(gf8b::Elem));
    coeffs
}

/// GF(2^16) coefficients: pure base-field, pure extension, and mixed. The
/// tower kernels have distinct code paths for the `same` and `cross` factors,
/// and a coefficient with a zero component silently masks one of them.
fn gf16_coeffs() -> Vec<gf16::Elem> {
    let mut coeffs = vec![
        gf16::Elem(0x0000),
        gf16::Elem(0x0001),
        gf16::Elem(0x0002),
        gf16::Elem(0x00ff),
        gf16::Elem(0x0100),
        gf16::Elem(0xff00),
        gf16::Elem(0xffff),
        gf16::Elem(0x0108),
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..24 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        coeffs.push(gf16::Elem((state >> 32) as u16));
    }
    coeffs
}

/// Row geometries for the multi-row kernels. Row counts straddle the
/// four-row and two-row grouping boundaries the blocked kernels use.
const ROW_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 13];
const ROW_LENS: &[usize] = &[2, 16, 32, 34, 64, 66, 128, 300];

/// Gather lengths exercise each side of the native 16/32/64/128-byte ladder,
/// compound remainders, and a body-plus-tail case.
const GATHER_LENGTHS: &[usize] = &[
    0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97, 111, 112, 113,
    127, 128, 129, 143, 144, 145, 255, 256, 257, 300,
];

/// Whether the host resolves to one of the tiers in `supported` — the shared
/// substitute for the crate's old `std::is_*_feature_detected!` test gates.
///
/// Declare the tier(s) a kernel needs and ask `simdispatch`'s selection,
/// keeping detection single-source. `SIMD_BACKEND` is honored, so
/// `SIMD_BACKEND=scalar` also skips the SIMD kernel tests.
fn host_supports(supported: &'static [crate::kernel::Backend]) -> bool {
    simdispatch::Selection::new("SIMD_BACKEND")
        .supports(supported)
        .resolve()
        != crate::kernel::Backend::Scalar
}

// ---------------------------------------------------------------------------
// Generic differential drivers, shared by every architecture below.
// ---------------------------------------------------------------------------

/// Compare a GF(2^8) `mul_add` kernel against the reference at every length
/// and coefficient.
fn check_gf8_mul_add(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for coeff in gf8_coeffs() {
            let mut got = noise(len, 0x62);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_assign(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable)) {
    for &len in LENGTHS {
        for coeff in gf8_coeffs() {
            let mut got = noise(len, 0x73);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff));
            scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_add_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x84);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x95);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_add::<gf16::Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_assign_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables)) {
    for &len in LENGTHS {
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xa6);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff));
            scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_into(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x136);
        for coeff in gf8_coeffs() {
            // Pre-fill the destination with noise: a fused kernel must
            // overwrite it, never accumulate into it.
            let mut got = noise(len, 0x147);
            let mut want = src.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_into_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x158);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x169);
            let mut want = src.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

/// Compare a scatter kernel against a per-row reference AXPY.
fn check_scatter<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, &[E], &[u8]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let src = noise(row_len, 0xb7);
            let coeffs: Vec<E> = (0..nrows).map(&coeff_at).collect();
            let mut got = noise(row_len * nrows, 0xc8);
            let mut want = got.clone();

            kernel(&mut got, row_len, &coeffs, &src);
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                reference(row, coeff, &src);
            }
            assert_eq!(got, want, "{name}: row_len {row_len}, nrows {nrows}");
        }
    }
}

/// Compare a matrix kernel against a per-term, per-row reference AXPY.
fn check_matrix<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            for nterms in [1usize, 2, 3, 7, 8, 9, 17] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x400 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<E>> = (0..nterms)
                    .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                    .collect();
                let terms: Vec<(&[E], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                let mut got = noise(row_len * nrows, 0xd9);
                let mut want = got.clone();

                kernel(&mut got, row_len, nrows, &terms);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "{name}: row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

/// Compare an overwrite matrix kernel against a per-term reference sum.
///
/// Unlike [`check_matrix`], the destination starts as nonzero noise the kernel
/// must ignore and the reference accumulates from zero, so the two agree only
/// if the kernel overwrites `rows[j] = sum_t coeffs[t][j] * src[t]`.
fn check_matrix_overwrite<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            for nterms in [1usize, 2, 3, 7, 8, 9, 17] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x400 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<E>> = (0..nterms)
                    .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                    .collect();
                let terms: Vec<(&[E], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                let mut got = noise(row_len * nrows, 0xd9);
                let mut want = vec![0u8; row_len * nrows];

                kernel(&mut got, row_len, nrows, &terms);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "{name}: row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

/// Fixed-six-row overwrite differential for ISA-L-shaped experimental bodies.
fn check_matrix_overwrite6<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, &[(&[E], &[u8])]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    const NROWS: usize = 6;
    for &row_len in ROW_LENS {
        for nterms in [1usize, 2, 3, 7, 8, 9, 17] {
            let sources: Vec<Vec<u8>> = (0..nterms)
                .map(|term| noise(row_len, 0x460 + term as u64))
                .collect();
            let coeff_sets: Vec<Vec<E>> = (0..nterms)
                .map(|term| (0..NROWS).map(|row| coeff_at(term, row)).collect())
                .collect();
            let terms: Vec<(&[E], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
                .collect();
            let mut got = noise(row_len * NROWS, 0xda);
            let mut want = vec![0u8; row_len * NROWS];

            kernel(&mut got, row_len, &terms);
            for &(coeffs, src) in &terms {
                for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                    reference(row, coeff, src);
                }
            }
            assert_eq!(got, want, "{name}: row_len {row_len}, terms {nterms}");
        }
    }
}

/// Fixed-six-row differential for prepacked shuffle-table candidates.
fn check_matrix_overwrite6_packed<E: Copy>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    pack: impl Fn(E) -> [u8; 32],
    kernel: impl Fn(&mut [u8], usize, &[[u8; 32]], &[&[u8]]),
) {
    const NROWS: usize = 6;
    for &row_len in ROW_LENS {
        for nterms in [1usize, 2, 3, 7, 8, 9, 17] {
            let sources: Vec<Vec<u8>> = (0..nterms)
                .map(|term| noise(row_len, 0x480 + term as u64))
                .collect();
            let source_refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeff_sets: Vec<Vec<E>> = (0..nterms)
                .map(|term| (0..NROWS).map(|row| coeff_at(term, row)).collect())
                .collect();
            let tables: Vec<[u8; 32]> = coeff_sets
                .iter()
                .flat_map(|coeffs| coeffs.iter().copied().map(&pack))
                .collect();
            let mut got = noise(row_len * NROWS, 0xdb);
            let mut want = vec![0u8; row_len * NROWS];

            kernel(&mut got, row_len, &tables, &source_refs);
            for (coeffs, src) in coeff_sets.iter().zip(&sources) {
                for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                    reference(row, coeff, src);
                }
            }
            assert_eq!(got, want, "{name}: row_len {row_len}, terms {nterms}");
        }
    }
}

/// Differential for the experimental two-row overwrite matrix (`p2`) over its
/// tunable tile width and store policy.
///
/// Temporal cases sweep arbitrary lengths (short bodies, 32-byte cleanup, and
/// sub-XMM tails); non-temporal cases use a 32-byte-aligned backing and
/// 32-multiple lengths, matching the store precondition. The destination
/// starts nonzero so only an overwrite (seed-from-zero) kernel agrees.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_matrix_overwrite2<E: Copy>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], usize, &[(&[E], &[u8])], bool, usize),
) {
    const NROWS: usize = 2;
    let temporal_lens: &[usize] = &[32, 34, 64, 66, 128, 130, 256, 300];
    let nt_lens: &[usize] = &[32, 64, 128, 256];
    for &nt in &[false, true] {
        for &lanes in &[2usize, 4] {
            let lens = if nt { nt_lens } else { temporal_lens };
            for &row_len in lens {
                for nterms in [1usize, 2, 3, 8, 17] {
                    let sources: Vec<Vec<u8>> = (0..nterms)
                        .map(|t| noise(row_len, 0x420 + t as u64))
                        .collect();
                    let coeff_sets: Vec<Vec<E>> = (0..nterms)
                        .map(|t| (0..NROWS).map(|j| coeff_at(t, j)).collect())
                        .collect();
                    let terms: Vec<(&[E], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();

                    let mut want = vec![0u8; row_len * NROWS];
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            reference(row, coeff, src);
                        }
                    }

                    if nt {
                        let mut backing = noise(row_len * NROWS + 32, 0xd7);
                        let off = backing.as_ptr().align_offset(32);
                        let got = &mut backing[off..off + row_len * NROWS];
                        got.fill(0xaa);
                        kernel(got, row_len, &terms, nt, lanes);
                        assert_eq!(
                            got,
                            want.as_slice(),
                            "{name}: nt lanes {lanes} row_len {row_len} terms {nterms}"
                        );
                    } else {
                        let mut got = noise(row_len * NROWS, 0xd8);
                        kernel(&mut got, row_len, &terms, nt, lanes);
                        assert_eq!(
                            got,
                            want.as_slice(),
                            "{name}: temporal lanes {lanes} row_len {row_len} terms {nterms}"
                        );
                    }
                }
            }
        }
    }
}

/// Differential for the experimental two-row shuffle overwrite matrix (`p2`)
/// over both spatial-unroll widths. `pack(coeff)` builds the `[lo;16, hi;16]`
/// packed record the kernel consumes.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_matrix_overwrite2_shuffle<E: Copy>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    pack: impl Fn(E) -> [u8; 32],
) {
    const NROWS: usize = 2;
    let lens: &[usize] = &[32, 34, 64, 66, 128, 130, 256, 300];
    for &lanes in &[1usize, 2] {
        for &row_len in lens {
            for nterms in [1usize, 2, 3, 8, 17] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x440 + t as u64))
                    .collect();
                let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
                let coeff_sets: Vec<Vec<E>> = (0..nterms)
                    .map(|t| (0..NROWS).map(|j| coeff_at(t, j)).collect())
                    .collect();
                let packed: Vec<[u8; 32]> = coeff_sets
                    .iter()
                    .flat_map(|cs| cs.iter().map(|&c| pack(c)))
                    .collect();

                let mut want = vec![0u8; row_len * NROWS];
                for (t, cs) in coeff_sets.iter().enumerate() {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(cs) {
                        reference(row, coeff, srcs[t]);
                    }
                }
                let mut got = noise(row_len * NROWS, 0xd6);
                crate::kernel::x86::gf8::matrix_overwrite2_shuffle_packed(
                    &mut got, row_len, &packed, &srcs, lanes,
                );
                assert_eq!(
                    got,
                    want.as_slice(),
                    "{name}: lanes {lanes} row_len {row_len} terms {nterms}"
                );
            }
        }
    }
}

/// Compare a gather kernel against repeated scalar AXPY calls.
fn check_gather<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], &[E], &[&[u8]]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &len in GATHER_LENGTHS {
        for nterms in [0usize, 1, 2, 3, 4, 7, 8, 9, 16, 17, 32] {
            let sources: Vec<Vec<u8>> = (0..nterms).map(|i| noise(len, 0x500 + i as u64)).collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<E> = (0..nterms).map(&coeff_at).collect();
            let mut got = noise(len, 0xe1);
            let mut want = got.clone();
            kernel(&mut got, &coeffs, &srcs);
            for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                reference(&mut want, coeff, src);
            }
            assert_eq!(got, want, "{name}: len {len}, terms {nterms}");
        }
    }
}

/// All 256 `0x11D` coefficients against noise sources at every lane-boundary
/// length, then against a source holding every byte value — so between the
/// two sweeps the affine kernels below see all 65 536 products outright.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn for_each_gf8d_affine_case(case: impl FnMut(gf8d::Elem, &[u8])) {
    let mut case = case;
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for c in 0..=u8::MAX {
            case(gf8d::Elem(c), &src);
        }
    }
    let every_byte: Vec<u8> = (0..=u8::MAX).collect();
    for c in 0..=u8::MAX {
        case(gf8d::Elem(c), &every_byte);
    }
}

/// Compare a `0x11D` affine `mul_add` kernel against the reference at every
/// length and coefficient.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8d_affine_mul_add(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable, &[u8])) {
    for_each_gf8d_affine_case(|coeff, src| {
        let mut got = noise(src.len(), 0x62);
        let mut want = got.clone();
        kernel(&mut got, affine_8d(coeff), scale_table_8d(coeff), src);
        scalar::mul_add::<gf8d::Gf8D>(&mut want, coeff, src);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8d_affine_mul_assign(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable)) {
    for_each_gf8d_affine_case(|coeff, src| {
        let mut got = noise(src.len(), 0x73);
        let mut want = got.clone();
        kernel(&mut got, affine_8d(coeff), scale_table_8d(coeff));
        scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8d_affine_mul_into(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable, &[u8])) {
    for_each_gf8d_affine_case(|coeff, src| {
        // Pre-fill the destination with noise: a fused kernel must overwrite
        // it, never accumulate into it.
        let mut got = noise(src.len(), 0x147);
        let mut want = src.to_vec();
        kernel(&mut got, affine_8d(coeff), scale_table_8d(coeff), src);
        scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

fn check_gf8_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0xf2);
        let b = noise(len, 0x103);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<gf8b::Gf8B>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Differential check for a `0x11D` elementwise kernel against the `Gf8D`
/// scalar oracle. `GF2P8MULB` is the AES field, so the `0x11D` field runs the
/// shift/reduce vector multiply on every x86 backend, including GFNI.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8d_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0xf2);
        let b = noise(len, 0x103);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<gf8d::Gf8D>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

fn check_gf16_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0x114);
        let b = noise(len, 0x125);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<gf16::Gf16>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Compare a tower-field `mul_add` kernel against the portable reference at
/// every length and coefficient.
fn check_tower_mul_add<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let src = noise(len, 0x90 ^ (ci as u64).wrapping_mul(7));
            let mut want = noise(len, 0x80 ^ (ci as u64).wrapping_mul(13));
            let mut got = want.clone();
            reference(&mut want, coeff, &src);
            kernel(&mut got, coeff, &src);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

/// Compare a tower-field `mul_assign` kernel against the portable reference.
fn check_tower_mul_assign<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E),
    kernel: impl Fn(&mut [u8], E),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let mut want = noise(len, 0x80 ^ (ci as u64).wrapping_mul(13));
            let mut got = want.clone();
            reference(&mut want, coeff);
            kernel(&mut got, coeff);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

/// Compare a tower-field `mul_into` kernel against copy-then-scale.
fn check_tower_mul_into<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let src = noise(len, 0x90 ^ (ci as u64).wrapping_mul(7));
            let mut want = vec![0; len];
            let mut got = vec![0; len];
            reference(&mut want, coeff, &src);
            kernel(&mut got, coeff, &src);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

fn gf8_coeff_at(j: usize) -> gf8b::Elem {
    // Includes 0 and 1 as j sweeps, which is what we want: the blocked
    // kernels must handle degenerate coefficients per row, not per call.
    gf8b::Elem((j as u8).wrapping_mul(29))
}

fn gf8_coeff_at2(t: usize, j: usize) -> gf8b::Elem {
    gf8b::Elem(((t * 31 + j * 29) % 256) as u8)
}

fn gf16_coeff_at(j: usize) -> gf16::Elem {
    gf16::Elem((j as u16).wrapping_mul(7411))
}

fn gf16_coeff_at2(t: usize, j: usize) -> gf16::Elem {
    gf16::Elem(((t * 7919 + j * 613) % 65536) as u16)
}

fn gf8_reference(dst: &mut [u8], coeff: gf8b::Elem, src: &[u8]) {
    scalar::mul_add::<gf8b::Gf8B>(dst, coeff, src);
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gf8d_coeff_at(j: usize) -> gf8d::Elem {
    gf8d::Elem((j as u8).wrapping_mul(29))
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gf8d_coeff_at2(t: usize, j: usize) -> gf8d::Elem {
    gf8d::Elem(((t * 31 + j * 29) % 256) as u8)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gf8d_reference(dst: &mut [u8], coeff: gf8d::Elem, src: &[u8]) {
    scalar::mul_add::<gf8d::Gf8D>(dst, coeff, src);
}

fn gf16_reference(dst: &mut [u8], coeff: gf16::Elem, src: &[u8]) {
    scalar::mul_add::<gf16::Gf16>(dst, coeff, src);
}

/// Fan–Paar GF(2^16) coefficients: short-circuits, extremes, the tower
/// generator and its components, and a deterministic spread.
fn fp16_coeffs() -> Vec<fan_paar::fp16::Elem> {
    use fan_paar::{fp8, fp16};
    let mut v = vec![
        fp16::Elem::ZERO,
        fp16::Elem::ONE,
        fp16::Elem(u16::MAX),
        // Pure base-field and pure extension.
        fp16::Elem::from_components(fp8::Elem::ONE, fp8::Elem::ZERO),
        fp16::Elem::from_components(fp8::Elem::ZERO, fp8::Elem::ONE),
        // The tower generator and its subfield tower generator.
        fp16::GENERATOR,
        fp16::ALPHA,
    ];
    let mut s = 0xf491u16;
    for _ in 0..16 {
        s = s.wrapping_mul(2057).wrapping_add(13849);
        v.push(fp16::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^16) lengths: multiples of 2 straddling 16/32-byte lanes.
const FP16_LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 126, 128, 130, 254, 256, 510, 512, 1022,
];

fn fp16_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar16>(dst, coeff, src);
}
fn fp16_assign_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem) {
    scalar::mul_assign::<FanPaar16>(dst, coeff);
}
fn fp16_into_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar16>(dst, coeff);
}

/// Fan–Paar GF(2^32) coefficients, the same shape as the polynomial towers.
fn fp32_coeffs() -> Vec<fan_paar::fp32::Elem> {
    use fan_paar::{fp16, fp32};
    let mut v = vec![
        fp32::Elem::ZERO,
        fp32::Elem::ONE,
        fp32::Elem(u32::MAX),
        fp32::Elem::from_components(fp16::Elem::ONE, fp16::Elem::ZERO),
        fp32::Elem::from_components(fp16::Elem::ZERO, fp16::Elem::ONE),
        fp32::ALPHA,
        fp32::Elem::from_components(fp16::Elem::ZERO, fp16::ALPHA),
        fp32::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(fp32::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^64) coefficients.
fn fp64_coeffs() -> Vec<fan_paar::fp64::Elem> {
    use fan_paar::{fp32, fp64};
    let mut v = vec![
        fp64::Elem::ZERO,
        fp64::Elem::ONE,
        fp64::Elem(u64::MAX),
        fp64::Elem::from_components(fp32::Elem::ONE, fp32::Elem::ZERO),
        fp64::ALPHA,
        fp64::Elem::from_components(fp32::Elem::ZERO, fp32::ALPHA),
        fp64::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(fp64::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^32) lengths: multiples of 4 straddling 32-byte lanes.
const FP32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// Fan–Paar GF(2^64) lengths: multiples of 8.
const FP64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

fn fp32_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar32>(dst, coeff, src);
}
fn fp32_assign_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem) {
    scalar::mul_assign::<FanPaar32>(dst, coeff);
}
fn fp32_into_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar32>(dst, coeff);
}
fn fp64_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar64>(dst, coeff, src);
}
fn fp64_assign_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem) {
    scalar::mul_assign::<FanPaar64>(dst, coeff);
}
fn fp64_into_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar64>(dst, coeff);
}

/// GF(2^32) coefficients: the short-circuits, the extremes, pure tower
/// components, the tower constant, the generator, and a deterministic spread.
fn gf32_coeffs() -> Vec<gf32::Elem> {
    let mut v = vec![
        gf32::Elem::ZERO,
        gf32::Elem::ONE,
        gf32::Elem(u32::MAX),
        gf32::Elem(0x0000_ffff),
        gf32::Elem(0xffff_0000),
        // Pure base field and pure extension.
        gf32::Elem::from_components(gf16::Elem::ONE, gf16::Elem::ZERO),
        gf32::Elem::from_components(gf16::Elem::ZERO, gf16::Elem::ONE),
        // The tower constant in each component.
        gf32::Elem::from_components(gf32::DELTA, gf16::Elem::ZERO),
        gf32::Elem::from_components(gf16::Elem::ZERO, gf32::DELTA),
        gf32::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(gf32::Elem(s));
    }
    v
}

/// GF(2^64) coefficients, the same shape one level up.
fn gf64_coeffs() -> Vec<gf64::Elem> {
    let mut v = vec![
        gf64::Elem::ZERO,
        gf64::Elem::ONE,
        gf64::Elem(u64::MAX),
        gf64::Elem(0x0000_0000_ffff_ffff),
        gf64::Elem(0xffff_ffff_0000_0000),
        gf64::Elem::from_components(gf32::Elem::ONE, gf32::Elem::ZERO),
        gf64::Elem::from_components(gf32::Elem::ZERO, gf32::Elem::ONE),
        gf64::Elem::from_components(gf64::DELTA, gf32::Elem::ZERO),
        gf64::Elem::from_components(gf32::Elem::ZERO, gf64::DELTA),
        gf64::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(gf64::Elem(s));
    }
    v
}

/// GF(2^32) lengths: multiples of 4 straddling every 32-byte lane boundary.
const GF32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// GF(2^64) lengths: multiples of 8.
const GF64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

fn gf32_reference(dst: &mut [u8], coeff: gf32::Elem, src: &[u8]) {
    scalar::mul_add::<Gf32>(dst, coeff, src);
}
fn gf32_assign_reference(dst: &mut [u8], coeff: gf32::Elem) {
    scalar::mul_assign::<Gf32>(dst, coeff);
}
fn gf32_into_reference(dst: &mut [u8], coeff: gf32::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf32>(dst, coeff);
}
fn gf64_reference(dst: &mut [u8], coeff: gf64::Elem, src: &[u8]) {
    scalar::mul_add::<Gf64>(dst, coeff, src);
}
fn gf64_assign_reference(dst: &mut [u8], coeff: gf64::Elem) {
    scalar::mul_assign::<Gf64>(dst, coeff);
}
fn gf64_into_reference(dst: &mut [u8], coeff: gf64::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf64>(dst, coeff);
}

// ---------------------------------------------------------------------------
// Backend-independent
// ---------------------------------------------------------------------------

#[test]
fn scalar_nibble_paths_match_the_generic_reference() {
    check_gf8_mul_add("gf8 nibble", |dst, table, src| {
        super::gf8::mul_add_nibble(dst, table, src);
    });
    check_gf8_mul_assign("gf8 nibble", super::gf8::mul_assign_nibble);
    for &len in LENGTHS {
        let src = noise(len, 0xea);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xfb);
            let mut want = got.clone();
            super::gf16::mul_add_scalar(&mut got, coeff, &src);
            scalar::mul_add::<gf16::Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "gf16 scalar: len {len}, coeff {coeff:?}");
        }
    }
}

#[test]
fn tower_coefficient_derivation_is_self_consistent() {
    for coeff in gf16_coeffs() {
        let compact = TowerCoeff::new(coeff);
        let tables = TowerTables::new(coeff);
        for (factor, table) in compact.factors().iter().zip(&tables.factors) {
            assert_eq!(*factor, table.coeff, "table/factor mismatch for {coeff:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// x86
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    use super::*;
    use crate::kernel::{Backend, x86};

    // The 64-byte AVX-512 kernels are the deferred V4x tier (not in the
    // ladder) and have no `Backend` to resolve, so this hardware gate stays a
    // direct `is_x86_feature_detected!` — the one sanctioned exception. The
    // kernels are experimental (`internals`) until a validated 512-bit GFNI
    // kernel exists and V4x ships.
    #[test]
    #[cfg(feature = "internals")]
    fn avx512_kernels_match_reference() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("gfni"))
        {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        }

        check_gf8_mul_add("gf8 avx512", |dst, table, src| {
            x86::avx512::gf8_mul_add(dst, table.coeff, src);
        });
        check_gf8_mul_assign("gf8 avx512", |dst, table| {
            x86::avx512::gf8_mul_assign(dst, table.coeff);
        });
        check_gf16_mul_add_tables("gf16 avx512", |dst, tables, src| {
            x86::avx512::gf16_mul_add(dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf16 avx512", |dst, tables| {
            x86::avx512::gf16_mul_assign(dst, TowerCoeff::new(tables.coeff));
        });
        check_scatter(
            "gf8 avx512 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::avx512::gf8_scatter,
        );
        check_scatter(
            "gf16 avx512 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::avx512::gf16_scatter,
        );
        check_gather(
            "gf8 avx512 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::avx512::gf8_gather,
        );
        check_gather(
            "gf16 avx512 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::avx512::gf16_gather,
        );
        check_matrix(
            "gf8 avx512 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::avx512::gf8_matrix,
        );
        check_matrix(
            "gf16 avx512 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::avx512::gf16_matrix,
        );
        // The overwrite kernels: junk destination, must equal scaling from
        // a zeroed one.
        for &len in LENGTHS {
            let src = noise(len, 0x3a);
            let coeff = gf8_coeff_at(5);
            let mut base = vec![0u8; len];
            scalar::mul_add::<gf8b::Gf8B>(&mut base, coeff, &src);
            let mut got = noise(len, 0x3b);
            x86::avx512::gf8_mul_into(&mut got, coeff, &src);
            assert_eq!(got, base, "gf8 avx512 mul_into at len {len}");

            let coeff = TowerCoeff::new(gf16_coeff_at(7));
            let mut base = vec![0u8; len];
            scalar::mul_add::<gf16::Gf16>(&mut base, coeff.coeff, &src);
            let mut got = noise(len, 0x3c);
            x86::avx512::gf16_mul_into(&mut got, coeff, &src);
            assert_eq!(got, base, "gf16 avx512 mul_into at len {len}");
        }
        check_gf8_elementwise("gf8 avx512 elementwise", x86::avx512::gf8_elementwise);
        check_gf16_elementwise("gf16 avx512 elementwise", x86::avx512::gf16_elementwise);
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut simd = want.clone();
            scalar::xor(&mut want, &src);
            x86::avx512::xor(&mut simd, &src);
            assert_eq!(simd, want, "avx512 xor: len {len}");
        }
    }

    fn check_gf8_six_row_shuffle_candidates() {
        check_matrix_overwrite6(
            "gf8 six-row shuffle overwrite",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_overwrite6_shuffle_8b,
        );
        check_matrix_overwrite6_packed(
            "gf8 six-row packed shuffle overwrite",
            gf8_coeff_at2,
            gf8_reference,
            |coefficient| {
                let table = scale_table(coefficient);
                let mut packed = [0u8; 32];
                packed[..16].copy_from_slice(&table.lo);
                packed[16..].copy_from_slice(&table.hi);
                packed
            },
            x86::gf8::matrix_overwrite6_shuffle_packed_8b,
        );
    }

    fn check_gf8d_six_row_shuffle_candidates() {
        check_matrix_overwrite6(
            "gf8d six-row shuffle overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            x86::gf8::matrix_overwrite6_shuffle_8d,
        );
        check_matrix_overwrite6_packed(
            "gf8d six-row packed shuffle overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            |coefficient| {
                let table = scale_table_8d(coefficient);
                let mut packed = [0u8; 32];
                packed[..16].copy_from_slice(&table.lo);
                packed[16..].copy_from_slice(&table.hi);
                packed
            },
            x86::gf8::matrix_overwrite6_shuffle_packed_8d,
        );
    }

    fn check_gf8_two_row_candidates() {
        check_matrix_overwrite2(
            "gf8 two-row overwrite",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_overwrite2_8b,
        );
        check_matrix_overwrite2_shuffle(
            "gf8 two-row shuffle overwrite",
            gf8_coeff_at2,
            gf8_reference,
            |coefficient| {
                let table = scale_table(coefficient);
                let mut packed = [0u8; 32];
                packed[..16].copy_from_slice(&table.lo);
                packed[16..].copy_from_slice(&table.hi);
                packed
            },
        );
    }

    fn check_gf8d_two_row_candidates() {
        check_matrix_overwrite2(
            "gf8d two-row overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            x86::gf8::matrix_overwrite2_8d,
        );
        check_matrix_overwrite2_shuffle(
            "gf8d two-row shuffle overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            |coefficient| {
                let table = scale_table_8d(coefficient);
                let mut packed = [0u8; 32];
                packed[..16].copy_from_slice(&table.lo);
                packed[16..].copy_from_slice(&table.hi);
                packed
            },
        );
    }

    #[test]
    fn gfni_kernels_match_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        check_gf8_mul_add("gf8 gfni", |dst, table, src| {
            x86::gf8::mul_add_gfni(dst, table.coeff, src);
        });
        check_gf8_mul_assign("gf8 gfni", |dst, table| {
            x86::gf8::mul_assign_gfni(dst, table.coeff);
        });
        check_gf16_mul_add_tables("gf16 gfni", |dst, tables, src| {
            x86::gf16::mul_add_gfni(dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf16 gfni", |dst, tables| {
            x86::gf16::mul_assign_gfni(dst, TowerCoeff::new(tables.coeff));
        });

        check_scatter(
            "gf8 gfni scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_gfni,
        );
        check_scatter(
            "gf16 gfni scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_gfni,
        );
        check_matrix(
            "gf8 gfni matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_gfni,
        );
        check_matrix_overwrite(
            "gf8 gfni matrix overwrite",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_overwrite_gfni,
        );
        check_gf8_six_row_shuffle_candidates();
        check_gf8_two_row_candidates();
        check_matrix(
            "gf16 gfni matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_gfni,
        );
        check_gather(
            "gf8 gfni gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni,
        );
        check_gather(
            "gf8 gfni 32-byte tile",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni_tile::<1>,
        );
        check_gather(
            "gf8 gfni 64-byte tile",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni_tile::<2>,
        );
        check_gather(
            "gf8 gfni 96-byte tile",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni_tile::<3>,
        );
        check_gather(
            "gf8 gfni split accumulators",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni_split,
        );
        check_gather(
            "gf16 gfni gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_gfni,
        );
        check_gf8_elementwise("gf8 gfni elementwise", x86::gf8::elementwise_gfni);
        check_gf16_elementwise("gf16 gfni elementwise", x86::gf16::elementwise_gfni);
        // Level-2/3 tower kernels: the period-2 lane multiply one and two
        // levels up from the GF(2^16) kernel.
        check_tower_gfni_kernels();
    }

    /// Differential-check the GFNI GF(2^32) and GF(2^64) kernels — the level-2
    /// and level-3 tower multiplies — against the portable scalar oracle.
    fn check_tower_gfni_kernels() {
        check_tower_mul_add(
            "gf32 gfni mul_add",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_reference,
            |dst, coeff, src| {
                x86::gf32::mul_add_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf32 gfni mul_assign",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_assign_reference,
            |dst, coeff| {
                x86::gf32::mul_assign_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf32 gfni mul_into",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_into_reference,
            |dst, coeff, src| {
                x86::gf32::mul_into_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        // Level-3: the same identity over GF(2^32) lanes.
        check_tower_mul_add(
            "gf64 gfni mul_add",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_reference,
            |dst, coeff, src| {
                x86::gf64::mul_add_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf64 gfni mul_assign",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_assign_reference,
            |dst, coeff| {
                x86::gf64::mul_assign_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf64 gfni mul_into",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_into_reference,
            |dst, coeff, src| {
                x86::gf64::mul_into_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
    }

    /// Exhaustive silicon check for the experimental `Gf8B` affine map and
    /// differential coverage across the gather tile/remainder ladder.
    #[test]
    fn gf8b_affine_gather_matches_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        let every_byte: Vec<u8> = (0..=u8::MAX).collect();
        // Remainder lengths, so the scalar tail past the last vector also
        // runs through the blocked coefficient accessors.
        for tail_len in [1usize, 27, 33] {
            let tail = &every_byte[..tail_len];
            let tail_srcs = [tail];
            for c in [0u8, 1, 0x53] {
                let coeff = gf8b::Elem(c);
                let factors = [x86::gf8::prepare_affine_8b(coeff)];
                let mut got = vec![0u8; tail_len];
                x86::gf8::gather_affine_8b(&mut got, &factors, &tail_srcs);
                let mut want = vec![0u8; tail_len];
                scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, tail);
                assert_eq!(
                    got, want,
                    "affine gather tail len {tail_len} coeff {c:#04x}"
                );
            }
        }
        let srcs = [every_byte.as_slice()];
        for c in 0..=u8::MAX {
            let coeff = gf8b::Elem(c);
            let factors = [x86::gf8::prepare_affine_8b(coeff)];
            let mut got = vec![0u8; every_byte.len()];
            x86::gf8::gather_affine_8b(&mut got, &factors, &srcs);
            let mut want = vec![0u8; every_byte.len()];
            scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, &every_byte);
            assert_eq!(got, want, "gf8b affine coeff {c:#04x}");
        }

        check_gather(
            "gf8b affine gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| {
                let factors: Vec<_> = coeffs
                    .iter()
                    .copied()
                    .map(x86::gf8::prepare_affine_8b)
                    .collect();
                x86::gf8::gather_affine_8b(dst, &factors, srcs);
            },
        );
    }

    /// The `0x11D` field's GFNI path: `VGF2P8AFFINEQB` with the const-derived
    /// affine bank. The sweep is exhaustive over coefficients, and the
    /// all-byte-values source makes it exhaustive over products — this is the
    /// silicon-level proof of the affine map's bit-order convention, and the
    /// loud failure if `GF2P8MULB` (the AES field) were ever substituted.
    #[test]
    fn gf8d_affine_kernels_match_reference() {
        // The non-temporal `mul_into` body sits far above the shared lengths;
        // destination offsets 0 and 1 cover both sides of the alignment peel.
        const NT_LEN: usize = (2 << 20) + 130;

        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        check_gf8d_affine_mul_add("gf8d affine", x86::gf8::mul_add_affine);
        check_gf8d_affine_mul_assign("gf8d affine", x86::gf8::mul_assign_affine);
        check_gf8d_affine_mul_into("gf8d affine", x86::gf8::mul_into_affine);
        let source = noise(NT_LEN + 2, 0x2b8);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut buffer = vec![0u8; NT_LEN + 1];
            for coeff in [0u8, 1, 0x53, 0xff].map(gf8d::Elem) {
                let mut want = src.to_vec();
                scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
                let got = &mut buffer[offset..offset + NT_LEN];
                x86::gf8::mul_into_affine(got, affine_8d(coeff), scale_table_8d(coeff), src);
                assert_eq!(got, want.as_slice(), "gf8d affine nt mul_into: {coeff:?}");
            }
        }

        // The register-blocked affine multi-row shapes against per-row AXPY,
        // over the same row-group and tile boundaries as the `Gf8B` kernels.
        check_scatter(
            "gf8d affine scatter",
            gf8d_coeff_at,
            gf8d_reference,
            x86::gf8::scatter_affine,
        );
        check_matrix(
            "gf8d affine matrix",
            gf8d_coeff_at2,
            gf8d_reference,
            x86::gf8::matrix_affine,
        );
        check_matrix_overwrite(
            "gf8d affine matrix overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            x86::gf8::matrix_overwrite_affine,
        );
        check_gf8d_six_row_shuffle_candidates();
        check_gf8d_two_row_candidates();
        check_gather(
            "gf8d affine gather",
            gf8d_coeff_at,
            gf8d_reference,
            x86::gf8::gather_affine,
        );
        // Prepared coefficients broadcast the stored affine map straight into
        // the gather, hoisting the per-tile lookup; output must be identical.
        check_gather(
            "gf8d affine gather prepared",
            |j| {
                let c = gf8d_coeff_at(j);
                crate::kernel::gf8::Prepared8D {
                    table: scale_table_8d(c),
                    affine: affine_8d(c),
                }
            },
            |dst, prep: crate::kernel::gf8::Prepared8D, src| {
                gf8d_reference(dst, gf8d::Elem(prep.table.coeff.0), src);
            },
            x86::gf8::gather_affine_prepared,
        );
        // Scattered rows: disjoint offsets, blocked affine against per-term AXPY.
        for &row_len in ROW_LENS {
            for &nrows in ROW_COUNTS {
                let starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
                let span = starts.last().map_or(0, |&s| s + row_len);
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0x900 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8d_coeff_at2(t, j)).collect())
                    .collect();
                let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(span, 0xda);
                let mut want = got.clone();
                x86::gf8::matrix_scattered_affine(&mut got, row_len, &starts, &terms);
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        scalar::mul_add::<gf8d::Gf8D>(
                            &mut want[starts[j]..starts[j] + row_len],
                            coeff,
                            src,
                        );
                    }
                }
                assert_eq!(
                    got, want,
                    "gf8d affine matrix_scattered: row_len {row_len}, nrows {nrows}"
                );
            }
        }
    }

    #[test]
    fn avx2_kernels_match_reference() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        check_gf8_mul_add("gf8 avx2", x86::gf8::mul_add_avx2);
        check_gf8_mul_assign("gf8 avx2", x86::gf8::mul_assign_avx2);
        check_gf16_mul_add_tables("gf16 avx2", x86::gf16::mul_add_avx2);
        check_gf16_mul_assign_tables("gf16 avx2", x86::gf16::mul_assign_avx2);
        check_gf8_mul_into("gf8 avx2 mul_into", x86::gf8::mul_into_avx2);
        check_gf16_mul_into_tables("gf16 avx2 mul_into", x86::gf16::mul_into_avx2);
        check_scatter(
            "gf8 avx2 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_avx2,
        );
        check_scatter(
            "gf16 avx2 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_avx2,
        );
        check_gather(
            "gf8 avx2 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_avx2,
        );
        check_gather(
            "gf16 avx2 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_avx2,
        );
        check_matrix(
            "gf8 avx2 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_avx2,
        );
        check_matrix(
            "gf16 avx2 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_avx2,
        );
        check_gf8_elementwise("gf8 avx2 elementwise", x86::gf8::elementwise_avx2::<0x1b>);
        check_gf8d_elementwise("gf8d avx2 elementwise", x86::gf8::elementwise_avx2::<0x1d>);
        check_gf16_elementwise("gf16 avx2 elementwise", x86::gf16::elementwise_avx2);
        // Fan–Paar tower (GF(2^16)/32/64): the fp8 nibble tower and its
        // period-2 lane-mul extensions.
        check_fan_paar_avx2_kernels();
    }

    /// Differential-check the Fan–Paar GF(2^16)/32/64 AVX2 kernels — the fp8
    /// nibble tower and its period-2 lane-mul extensions — against the scalar
    /// oracle.
    fn check_fan_paar_avx2_kernels() {
        check_tower_mul_add(
            "fp16 avx2 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_avx2(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 avx2 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_avx2(dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 avx2 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_avx2(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_add(
            "fp32 avx2 mul_add",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_fp32_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp32 avx2 mul_assign",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_fp32_avx2(dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp32 avx2 mul_into",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_fp32_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_add(
            "fp64 avx2 mul_add",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_fp64_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp64 avx2 mul_assign",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_fp64_avx2(dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp64 avx2 mul_into",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_fp64_avx2(dst, coeff, src);
            },
        );
    }

    #[test]
    fn ssse3_kernels_match_reference() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        check_gf8_mul_add("gf8 ssse3", x86::gf8::mul_add_ssse3);
        check_gf8_mul_assign("gf8 ssse3", x86::gf8::mul_assign_ssse3);
        check_gf16_mul_add_tables("gf16 ssse3", x86::gf16::mul_add_ssse3);
        check_gf16_mul_assign_tables("gf16 ssse3", x86::gf16::mul_assign_ssse3);
        check_gf8_mul_into("gf8 ssse3 mul_into", x86::gf8::mul_into_ssse3);
        check_gf16_mul_into_tables("gf16 ssse3 mul_into", x86::gf16::mul_into_ssse3);
        check_scatter(
            "gf8 ssse3 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_ssse3,
        );
        check_scatter(
            "gf16 ssse3 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_ssse3,
        );
        check_gather(
            "gf8 ssse3 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_ssse3,
        );
        check_gather(
            "gf16 ssse3 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_ssse3,
        );
        check_matrix(
            "gf8 ssse3 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_ssse3,
        );
        check_matrix(
            "gf16 ssse3 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_ssse3,
        );
        check_gf8_elementwise("gf8 ssse3 elementwise", x86::gf8::elementwise_ssse3::<0x1b>);
        check_gf8d_elementwise(
            "gf8d ssse3 elementwise",
            x86::gf8::elementwise_ssse3::<0x1d>,
        );
        check_gf16_elementwise("gf16 ssse3 elementwise", x86::gf16::elementwise_ssse3);
        check_tower_mul_add(
            "fp16 ssse3 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_ssse3(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 ssse3 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_ssse3(dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 ssse3 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_ssse3(dst, &FpTowerTables::new(coeff), src);
            },
        );
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut avx2 = want.clone();
            let mut sse2 = want.clone();
            scalar::xor(&mut want, &src);
            if host_supports(&[Backend::V3]) {
                x86::xor_avx2(&mut avx2, &src);
                assert_eq!(avx2, want, "avx2 xor: len {len}");
            }
            x86::xor_sse2(&mut sse2, &src);
            assert_eq!(sse2, want, "sse2 xor: len {len}");
        }
    }

    /// `mul_into` over a buffer past the non-temporal store threshold.
    ///
    /// The shared `LENGTHS` are all far below it, so nothing else reaches the
    /// `vmovntdq` body. Destination offsets 0 and 1 cover both sides of the
    /// alignment peel — offset 1 leaves the GF(2^16) peel an odd number of
    /// bytes, which must fall back to ordinary stores rather than split a
    /// buffer mid-element.
    #[test]
    fn non_temporal_mul_into_matches_reference() {
        const NT_LEN: usize = (2 << 20) + 130;
        // The non-temporal split is a store-side choice, independent of the
        // coefficient, so a zero, a one and a mixed value are enough.
        const GF8_COEFFS: [gf8b::Elem; 3] = [gf8b::Elem(0), gf8b::Elem(1), gf8b::Elem(0x53)];
        const GF16_COEFFS: [gf16::Elem; 3] = [gf16::Elem(0), gf16::Elem(1), gf16::Elem(0x53a7)];

        let source = noise(NT_LEN + 2, 0x1a7);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];

            for coeff in GF8_COEFFS {
                let table = scale_table(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
                if host_supports(&[Backend::V3GfniCrypto]) {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_gfni(got, coeff, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if host_supports(&[Backend::V3]) {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_avx2(got, table, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + NT_LEN];
                x86::gf8::mul_into_ssse3(got, table, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }

            // GF(2^16) needs an even length and an even peel.
            let src = &src[..NT_LEN - 1];
            for coeff in GF16_COEFFS {
                let tables = TowerTables::new(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                if host_supports(&[Backend::V3GfniCrypto]) {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_gfni(got, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if host_supports(&[Backend::V3]) {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_avx2(got, &tables, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + src.len()];
                x86::gf16::mul_into_ssse3(got, &tables, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    }

    /// Multi-row scatter over rows long enough to take the alignment peel.
    ///
    /// `ROW_LENS` tops out at 300 bytes, below the peel's row-length floor, so
    /// no other test reaches the peeled path. The destination is offset by
    /// every residue that matters: 0 and 16 pick different peel lengths, and 1
    /// makes a 32-byte boundary unreachable in whole GF(2^16) elements, which
    /// must skip the peel rather than split an element.
    #[test]
    fn aligned_scatter_matches_reference() {
        const ROW_LEN: usize = 4096;

        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        for nrows in [1usize, 4, 6, 9] {
            for skew in [0usize, 1, 16] {
                let src = noise(ROW_LEN, 0x1b8);
                let mut backing = noise(ROW_LEN * nrows + skew, 0x1c9);
                let rows = &mut backing[skew..];

                let coeffs8: Vec<_> = (0..nrows).map(gf8_coeff_at).collect();
                let mut want = rows.to_vec();
                x86::gf8::scatter_gfni(rows, ROW_LEN, &coeffs8, &src);
                for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs8) {
                    gf8_reference(row, coeff, &src);
                }
                assert_eq!(rows, want.as_slice(), "gf8 peeled scatter: {nrows}/{skew}");

                let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                let mut want = rows.to_vec();
                x86::gf16::scatter_gfni(rows, ROW_LEN, &coeffs16, &src);
                for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs16) {
                    gf16_reference(row, coeff, &src);
                }
                assert_eq!(rows, want.as_slice(), "gf16 peeled scatter: {nrows}/{skew}");

                for (name, kernel) in [
                    (
                        "avx2",
                        x86::gf16::scatter_avx2::<gf16::Elem>
                            as fn(&mut [u8], usize, &[gf16::Elem], &[u8]),
                    ),
                    ("ssse3", x86::gf16::scatter_ssse3::<gf16::Elem>),
                ] {
                    let mut want = rows.to_vec();
                    kernel(rows, ROW_LEN, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 {name} peeled scatter: {nrows}/{skew}"
                    );
                }
            }
        }
    }

    // Prime-field integer-SIMD kernels versus the portable prime reference.
    use crate::field::goldilocks::{self, Goldilocks};
    use crate::field::mersenne31::{self, Mersenne31};
    use crate::kernel::prime;

    // Multiples of 4 (Mersenne31) / 8 (Goldilocks) straddling the 16-byte SSE
    // and 32-byte AVX2 lanes, with odd element tails.
    const M31_LENS: &[usize] = &[0, 4, 8, 16, 20, 32, 36, 64, 68, 128, 260, 1020];
    const GLD_LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
    const M31_COEFFS: [u32; 8] = [
        0,
        1,
        2,
        7,
        0x5555_5555,
        0x7FFF_FFFE,
        0x7FFF_FFFF, // non-canonical zero
        0xFFFF_FFFF, // non-canonical one
    ];
    const GLD_COEFFS: [u64; 8] = [
        0,
        1,
        2,
        7,
        0x5555_5555_5555_5555,
        goldilocks::MODULUS - 1,
        goldilocks::MODULUS, // non-canonical zero
        u64::MAX,
    ];

    #[allow(clippy::too_many_arguments)]
    fn drive_prime<F: crate::field::Field, C: Copy + core::fmt::Debug>(
        lens: &[usize],
        coeffs: &[C],
        to_elem: impl Fn(C) -> F::Elem,
        add: fn(&mut [u8], &[u8]),
        sub: fn(&mut [u8], &[u8]),
        muladd: fn(&mut [u8], C, &[u8]),
        mulinto: fn(&mut [u8], C, &[u8]),
        mulassign: fn(&mut [u8], C),
        elemwise: fn(&mut [u8], &[u8], &[u8]),
        label: &str,
    ) {
        // Canonicalize a buffer through the portable field so both sides start
        // from the canonical bytes the kernels are contracted to produce.
        let canon = |buf: &mut Vec<u8>| {
            let z = vec![0u8; buf.len()];
            prime::add_assign::<F>(buf, &z);
        };
        for &len in lens {
            let mut src = noise(len, 0xa1);
            canon(&mut src);
            let mut base = noise(len, 0xb2);
            canon(&mut base);

            let mut got = base.clone();
            let mut want = base.clone();
            add(&mut got, &src);
            prime::add_assign::<F>(&mut want, &src);
            assert_eq!(got, want, "{label} add_assign len {len}");

            let mut got = base.clone();
            let mut want = base.clone();
            sub(&mut got, &src);
            prime::sub_assign::<F>(&mut want, &src);
            assert_eq!(got, want, "{label} sub_assign len {len}");

            let mut b2 = noise(len, 0xc3);
            canon(&mut b2);
            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            elemwise(&mut got, &src, &b2);
            prime::mul_elementwise::<F>(&mut want, &src, &b2);
            assert_eq!(got, want, "{label} mul_elementwise len {len}");

            for &c in coeffs {
                let mut got = base.clone();
                let mut want = base.clone();
                muladd(&mut got, c, &src);
                prime::mul_add::<F>(&mut want, to_elem(c), &src);
                assert_eq!(got, want, "{label} mul_add len {len} coeff {c:?}");

                let mut got = base.clone();
                let mut want = src.clone();
                mulinto(&mut got, c, &src);
                prime::mul_assign::<F>(&mut want, to_elem(c));
                assert_eq!(got, want, "{label} mul_into len {len} coeff {c:?}");

                let mut got = base.clone();
                let mut want = base.clone();
                mulassign(&mut got, c);
                prime::mul_assign::<F>(&mut want, to_elem(c));
                assert_eq!(got, want, "{label} mul_assign len {len} coeff {c:?}");
            }
        }
    }

    #[test]
    fn prime_avx2_kernels_match_reference() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        drive_prime::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            mersenne31::Elem,
            x86::prime::add_assign_m31_avx2,
            x86::prime::sub_assign_m31_avx2,
            x86::prime::mul_add_m31_avx2,
            x86::prime::mul_into_m31_avx2,
            x86::prime::mul_assign_m31_avx2,
            x86::prime::mul_elementwise_m31_avx2,
            "m31 avx2",
        );
        drive_prime::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            goldilocks::Elem,
            x86::prime::add_assign_gld_avx2,
            x86::prime::sub_assign_gld_avx2,
            x86::prime::mul_add_gld_avx2,
            x86::prime::mul_into_gld_avx2,
            x86::prime::mul_assign_gld_avx2,
            x86::prime::mul_elementwise_gld_avx2,
            "gld avx2",
        );
    }

    #[test]
    fn prime_sse42_kernels_match_reference() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSE4.2 on this host");
            return;
        }
        drive_prime::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            mersenne31::Elem,
            x86::prime::add_assign_m31_sse41,
            x86::prime::sub_assign_m31_sse41,
            x86::prime::mul_add_m31_sse41,
            x86::prime::mul_into_m31_sse41,
            x86::prime::mul_assign_m31_sse41,
            x86::prime::mul_elementwise_m31_sse41,
            "m31 sse4.2",
        );
        drive_prime::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            goldilocks::Elem,
            x86::prime::add_assign_gld_sse41,
            x86::prime::sub_assign_gld_sse41,
            x86::prime::mul_add_gld_sse41,
            x86::prime::mul_into_gld_sse41,
            x86::prime::mul_assign_gld_sse41,
            x86::prime::mul_elementwise_gld_sse41,
            "gld sse4.2",
        );
    }
}

// ---------------------------------------------------------------------------
// AArch64
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64 {
    use super::*;
    use crate::kernel::{Backend, aarch64};

    #[test]
    fn neon_kernels_match_reference() {
        if !host_supports(&[Backend::Neon]) {
            eprintln!("skipping: no NEON on this host");
            return;
        }
        check_gf8_mul_add("gf8 neon", aarch64::gf8::mul_add_neon);
        check_gf8_mul_assign("gf8 neon", aarch64::gf8::mul_assign_neon);
        check_gf16_mul_add_tables("gf16 neon", aarch64::gf16::mul_add_neon);
        check_gf16_mul_assign_tables("gf16 neon", aarch64::gf16::mul_assign_neon);
        check_gf8_mul_into("gf8 neon mul_into", aarch64::gf8::mul_into_neon);
        check_gf16_mul_into_tables("gf16 neon mul_into", aarch64::gf16::mul_into_neon);

        check_scatter(
            "gf8 neon scatter",
            gf8_coeff_at,
            gf8_reference,
            aarch64::gf8::scatter_neon,
        );
        check_scatter(
            "gf16 neon scatter",
            gf16_coeff_at,
            gf16_reference,
            aarch64::gf16::scatter_neon,
        );
        check_matrix(
            "gf8 neon matrix",
            gf8_coeff_at2,
            gf8_reference,
            aarch64::gf8::matrix_neon,
        );
        check_matrix(
            "gf16 neon matrix",
            gf16_coeff_at2,
            gf16_reference,
            aarch64::gf16::matrix_neon,
        );
        check_gather(
            "gf8 neon gather",
            gf8_coeff_at,
            gf8_reference,
            aarch64::gf8::gather_neon,
        );
        check_gather(
            "gf16 neon gather",
            gf16_coeff_at,
            gf16_reference,
            aarch64::gf16::gather_neon,
        );
        check_gf8_elementwise("gf8 neon elementwise", aarch64::gf8::elementwise_neon);
        check_gf16_elementwise("gf16 neon elementwise", aarch64::gf16::elementwise_neon);
    }

    #[test]
    fn pmull_kernels_match_reference() {
        if !host_supports(&[Backend::NeonAes]) {
            eprintln!("skipping: no AArch64 PMULL extension on this host");
            return;
        }
        check_gf8_elementwise("gf8 pmull elementwise", aarch64::gf8::elementwise_pmull);
        // The tower elementwise and every fixed-coefficient PMULL kernel were
        // measured against the nibble/bit-serial paths and lost; GF(2^8)
        // elementwise is the shape that won and the only one dispatch selects.
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut neon = want.clone();
            scalar::xor(&mut want, &src);
            aarch64::xor_neon(&mut neon, &src);
            assert_eq!(neon, want, "neon xor: len {len}");
        }
    }
}

// ---------------------------------------------------------------------------
// WebAssembly
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32 {
    use super::*;
    use crate::kernel::wasm32;

    #[test]
    fn simd128_kernels_match_reference() {
        check_gf8_mul_add("gf8 simd128", wasm32::gf8::mul_add_simd128);
        check_gf8_mul_assign("gf8 simd128", wasm32::gf8::mul_assign_simd128);
        check_gf16_mul_add_tables("gf16 simd128", wasm32::gf16::mul_add_simd128);
        check_gf16_mul_assign_tables("gf16 simd128", wasm32::gf16::mul_assign_simd128);
        check_gf8_mul_into("gf8 simd128 mul_into", wasm32::gf8::mul_into_simd128);
        check_gf16_mul_into_tables("gf16 simd128 mul_into", wasm32::gf16::mul_into_simd128);
        check_scatter(
            "gf8 simd128 scatter",
            gf8_coeff_at,
            gf8_reference,
            wasm32::gf8::scatter_simd128,
        );
        check_scatter(
            "gf16 simd128 scatter",
            gf16_coeff_at,
            gf16_reference,
            wasm32::gf16::scatter_simd128,
        );
        check_gather(
            "gf8 simd128 gather",
            gf8_coeff_at,
            gf8_reference,
            wasm32::gf8::gather_simd128,
        );
        check_gather(
            "gf16 simd128 gather",
            gf16_coeff_at,
            gf16_reference,
            wasm32::gf16::gather_simd128,
        );
        check_matrix(
            "gf8 simd128 matrix",
            gf8_coeff_at2,
            gf8_reference,
            wasm32::gf8::matrix_simd128,
        );
        check_matrix(
            "gf16 simd128 matrix",
            gf16_coeff_at2,
            gf16_reference,
            wasm32::gf16::matrix_simd128,
        );
        check_gf8_elementwise("gf8 simd128 elementwise", wasm32::gf8::elementwise_simd128);
        check_gf16_elementwise(
            "gf16 simd128 elementwise",
            wasm32::gf16::elementwise_simd128,
        );
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut simd = want.clone();
            scalar::xor(&mut want, &src);
            wasm32::xor_simd128(&mut simd, &src);
            assert_eq!(simd, want, "simd128 xor: len {len}");
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage of dispatch-level contracts the differential drivers skip
// ---------------------------------------------------------------------------

#[test]
fn default_kernels_handle_empty_terms() {
    // The defaulted `FieldKernels::dot_product` (used by the prime and
    // Fan-Paar fields) zeroes the destination when there are no terms.
    let mut dst = noise(16, 0x5a);
    <Goldilocks as FieldKernels>::dot_product(&mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
    let mut dst = noise(16, 0x5b);
    <FanPaar32 as FieldKernels>::dot_product(&mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
#[test]
fn internals_only_gfni_controls_match_reference() {
    use crate::kernel::x86;

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    }

    // The retained pre-fusion gather control and the full-width tile
    // delegation must agree with the production gather.
    check_gather(
        "gf8 gfni axpy-tail control",
        gf8_coeff_at,
        gf8_reference,
        x86::gf8::gather_gfni_axpy_tail,
    );
    check_gather(
        "gf8 gfni 128-byte tile",
        gf8_coeff_at,
        gf8_reference,
        x86::gf8::gather_gfni_tile::<4>,
    );

    // `TableCoefficient::with_tables` must fall back to building tables for
    // the `Plain` prepared form, which the dispatch layer never produces on
    // a shuffle host but internals callers may hold.
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("skipping: no AVX2 on this host");
        return;
    }
    for &len in LENGTHS {
        let src = noise(len, 0x61);
        let mut got = noise(2 * len, 0x62);
        let mut want = got.clone();
        let coeffs = [
            crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(1)),
            crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(2)),
        ];
        x86::gf16::scatter_avx2(&mut got, len, &coeffs, &src);
        scalar::mul_add::<gf16::Gf16>(&mut want[..len], gf16_coeff_at(1), &src);
        scalar::mul_add::<gf16::Gf16>(&mut want[len..], gf16_coeff_at(2), &src);
        assert_eq!(got, want, "plain-prepared scatter at len {len}");
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
#[test]
fn x86_geometry_guards_accept_zero_length_rows() {
    use crate::kernel::x86;

    // `row_len == 0` must fall through the vector prologues untouched.
    if !std::is_x86_feature_detected!("ssse3") {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    }

    let coeffs = [gf8b::Elem(3)];
    x86::gf8::scatter_ssse3(&mut [], 0, &coeffs, &[]);
    let gf16_coeffs = [crate::kernel::gf16::Prepared::Plain(gf16::Elem(5))];
    x86::gf16::scatter_ssse3(&mut [], 0, &gf16_coeffs, &[]);

    if std::is_x86_feature_detected!("avx2") {
        x86::gf8::scatter_avx2(&mut [], 0, &coeffs, &[]);
        x86::gf16::scatter_avx2(&mut [], 0, &gf16_coeffs, &[]);
        x86::gf8::matrix_avx2(&mut [], 0, 1, &[]);
        x86::gf16::matrix_avx2(&mut [], 0, 1, &[]);
    }
    x86::gf8::matrix_ssse3(&mut [], 0, 1, &[]);
    x86::gf16::matrix_ssse3(&mut [], 0, 1, &[]);

    // Empty term lists must return before any store, in the checked
    // scattered wrappers as well.
    let mut dst = [0u8; 8];
    x86::gf8::matrix_scattered_gfni(&mut dst, 8, &[0], &[]);
    assert_eq!(dst, [0; 8]);
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn scatter_gfni_rejects_short_rows_buffer() {
    use crate::kernel::x86;

    // `#[should_panic]` cannot express "skip on hosts that cannot select the
    // tier", so the panic is caught explicitly after the capability guard.
    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    }
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::scatter_gfni(&mut [0u8; 4], 4, &[gf8b::Elem(1), gf8b::Elem(2)], &[0u8; 4]);
    })
    .is_err();
    assert!(panicked, "scatter_gfni must reject a short rows buffer");
}

#[test]
fn macro_kernels_matrix_with_matches_per_row_application() {
    // `impl_field_kernels!`'s prepared-terms matrix override (FanPaar8):
    // apply the same prepared coefficients row by row and compare.
    let row_len = 8;
    let srcs: Vec<Vec<u8>> = vec![noise(row_len, 0x71), noise(row_len, 0x72)];
    let coeffs = [fan_paar::fp8::Elem(1), fan_paar::fp8::Elem(0x8d)];
    let prepared: Vec<_> = coeffs.iter().copied().map(FanPaar8::prepare).collect();

    let mut got = noise(2 * row_len, 0x73);
    let mut want = got.clone();
    <FanPaar8 as FieldKernels>::mul_add_matrix_with(
        &mut got,
        row_len,
        2,
        &[
            (&[prepared[0], prepared[1]], &srcs[0]),
            (&[prepared[1], prepared[0]], &srcs[1]),
        ],
    );
    for (term, src) in srcs.iter().enumerate() {
        let row_coeffs = if term == 0 {
            [&prepared[0], &prepared[1]]
        } else {
            [&prepared[1], &prepared[0]]
        };
        for (row, coeff) in want.chunks_exact_mut(row_len).zip(row_coeffs) {
            <FanPaar8 as FieldKernels>::mul_add(row, coeff, src);
        }
    }
    assert_eq!(got, want, "prepared matrix");
}

#[test]
fn default_prepared_kernels_handle_empty_inputs() {
    // `FieldKernels::dot_product_with`'s defaulted empty arm (the scalar and
    // Fan-Paar fields): no terms zeroes the destination.
    let mut dst = noise(16, 0x66);
    <FanPaar8 as FieldKernels>::dot_product_with(&mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
}

#[test]
fn quad_mersenne31_kernel_handles_zero_and_one_coefficients() {
    // The kernel-level zero/one shortcuts sit below the `ops` wrappers,
    // which short-circuit first; drive them directly.
    use crate::field::QuadMersenne31;

    let mut zeroed = noise(16, 0x67);
    <QuadMersenne31 as FieldKernels>::mul_assign(
        &mut zeroed,
        &QuadMersenne31::prepare(quad_zero()),
    );
    assert!(
        zeroed.iter().all(|&b| b == 0),
        "zero coefficient fills zero"
    );

    let original = noise(16, 0x68);
    let mut scaled = original.clone();
    <QuadMersenne31 as FieldKernels>::mul_assign(&mut scaled, &QuadMersenne31::prepare(quad_one()));
    assert_eq!(scaled, original, "one coefficient is the identity");
}

fn quad_zero() -> quad_mersenne31::Elem {
    quad_mersenne31::Elem(0, 0)
}

fn quad_one() -> quad_mersenne31::Elem {
    quad_mersenne31::Elem(1, 0)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
#[test]
fn gfni_matrix_wrappers_accept_empty_terms() {
    use crate::kernel::FlatMatrix;
    use crate::kernel::x86;

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    }

    // Zero-row and empty-source matrices return before any store, through
    // both the element and prepared coefficient entry points.
    let empty: FlatMatrix<'_, gf8b::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    let mut rows = [0u8; 8];
    x86::gf8::matrix_gfni_with(&mut rows, 8, 0, &empty);
    let empty_8d: FlatMatrix<'_, gf8d::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf8::matrix_affine_with(&mut rows, 8, 0, &empty_8d);
    let gf16_empty: FlatMatrix<'_, gf16::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::matrix_gfni_with(&mut rows, 8, 0, &gf16_empty);
    assert_eq!(rows, [0; 8]);
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
#[test]
fn scatter_affine_rejects_short_rows_buffer() {
    use crate::kernel::x86;

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    }
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::scatter_affine(&mut [0u8; 4], 4, &[gf8d::Elem(1), gf8d::Elem(2)], &[0u8; 4]);
    })
    .is_err();
    assert!(panicked, "scatter_affine must reject a short rows buffer");
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
#[test]
fn gf16_gfni_wrappers_tolerate_degenerate_geometry() {
    use crate::kernel::x86;

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    }

    // The gf16 wrappers clamp rather than panic: zero-length rows, zero
    // rows, and room for fewer rows than coefficients are all no-ops.
    x86::gf16::scatter_gfni(&mut [], 0, &[gf16::Elem(1)], &[]);
    let mut rows = [0u8; 8];
    x86::gf16::scatter_gfni(&mut rows, 0, &[gf16::Elem(1)], &[]);
    assert_eq!(rows, [0; 8]);

    let empty: crate::kernel::FlatMatrix<'_, gf16::Elem> = crate::kernel::FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::matrix_gfni_with(&mut rows, 8, 2, &empty);
    x86::gf16::matrix_gfni_with(&mut rows, 0, 2, &empty);
    x86::gf16::matrix_gfni_with(&mut rows, 8, 0, &empty);
    assert_eq!(rows, [0; 8]);
}

// ---------------------------------------------------------------------------
// GF(2) bit-packed kernels
// ---------------------------------------------------------------------------

/// The GF(2) kernels are portable-only today; the differential partner is the
/// per-bit scalar oracle over `gf2::Elem` — a genuinely different path from
/// the word loops (no packing, no masks). Geometry sweeps the same tail and
/// boundary cases the public suite drives, minus the wrapper checks, so a
/// wrapper bug cannot mask a kernel bug.
mod gf2_bits {
    use super::Vec;
    use crate::field::gf2;
    use crate::kernel::gf2 as k;

    fn bit(buf: &[u8], i: usize) -> bool {
        buf[i / 8] >> (i % 8) & 1 == 1
    }

    fn put(buf: &mut [u8], i: usize, value: bool) {
        let mask = 1u8 << (i % 8);
        if value {
            buf[i / 8] |= mask;
        } else {
            buf[i / 8] &= !mask;
        }
    }

    fn zeros(len: usize) -> Vec<u8> {
        (0..len).map(|_| 0u8).collect()
    }

    fn noise_bits(bits: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut buf = zeros(bits.div_ceil(8));
        for byte in &mut buf {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        for i in bits..8 * buf.len() {
            put(&mut buf, i, false);
        }
        buf
    }

    fn assert_padding_zero(buf: &[u8], bits: usize) {
        for i in bits..8 * buf.len() {
            assert!(!bit(buf, i), "padding bit {i} set");
        }
    }

    /// Element counts across sub-byte, byte, word, and word+tail boundaries.
    const BITS: &[usize] = &[0, 1, 3, 7, 8, 9, 13, 63, 64, 65, 100, 128, 200, 257];

    #[test]
    fn whole_buffer_kernels_match_oracle() {
        for &bits_len in BITS {
            let a = noise_bits(bits_len, 0x10 + bits_len as u64);
            let b = noise_bits(bits_len, 0x20 + bits_len as u64);

            let mut dst = a.clone();
            k::xor(&mut dst, &b);
            for i in 0..bits_len {
                assert_eq!(bit(&dst, i), bit(&a, i) ^ bit(&b, i), "xor {bits_len}/{i}");
            }

            let mut dst = noise_bits(8 * a.len(), 0);
            k::and_into(&mut dst, &a, &b);
            for i in 0..8 * dst.len() {
                assert_eq!(
                    bit(&dst, i),
                    bit(&a, i) && bit(&b, i),
                    "and_into {bits_len}/{i}"
                );
            }

            let mut dst = a.clone();
            k::and_assign(&mut dst, &b);
            for i in 0..bits_len {
                assert_eq!(
                    bit(&dst, i),
                    bit(&a, i) && bit(&b, i),
                    "and_assign {bits_len}/{i}"
                );
            }

            let mut dst = a.clone();
            k::andnot_assign(&mut dst, &b);
            for i in 0..bits_len {
                assert_eq!(
                    bit(&dst, i),
                    bit(&a, i) && !bit(&b, i),
                    "andnot {bits_len}/{i}"
                );
            }
            assert_padding_zero(&dst, bits_len);
        }
    }

    #[test]
    fn range_kernels_match_oracle() {
        for &bits_len in BITS {
            for from in [0, 1, 5, 8, 63, 64] {
                for span in [1, 3, 64, 65] {
                    let to = from + span;
                    if to > bits_len {
                        continue;
                    }
                    let src = noise_bits(bits_len, 0x30 + bits_len as u64);

                    let mut dst = noise_bits(bits_len, 0x40 + bits_len as u64);
                    let mut oracle = dst.clone();
                    k::xor_range(&mut dst, &src, from, to);
                    for i in from..to {
                        if bit(&src, i) {
                            let flipped = !bit(&oracle, i);
                            put(&mut oracle, i, flipped);
                        }
                    }
                    assert_eq!(dst, oracle, "xor_range {from}..{to} of {bits_len}");

                    let mut dst = noise_bits(bits_len, 0x50 + bits_len as u64);
                    let original = dst.clone();
                    k::clear_range(&mut dst, from, to);
                    for i in 0..bits_len {
                        let expected = !(from..to).contains(&i) && bit(&original, i);
                        assert_eq!(bit(&dst, i), expected, "clear {from}..{to}/{i}");
                    }

                    let mut dst = zeros(bits_len.div_ceil(8));
                    k::set_range(&mut dst, from, to);
                    for i in 0..bits_len {
                        assert_eq!(
                            bit(&dst, i),
                            (from..to).contains(&i),
                            "set {from}..{to}/{i}"
                        );
                    }
                    assert_padding_zero(&dst, bits_len);
                }
            }
        }
    }

    #[test]
    fn weight_parity_and_gather_match_oracle() {
        for &bits_len in BITS {
            let a = noise_bits(bits_len, 0x60 + bits_len as u64);
            let b = noise_bits(bits_len, 0x70 + bits_len as u64);

            let expected_weight: u32 = (0..bits_len).map(|i| u32::from(bit(&a, i))).sum();
            assert_eq!(
                k::weight(&a, bits_len),
                expected_weight,
                "weight {bits_len}"
            );

            let mut acc = gf2::Elem::ZERO;
            for i in 0..bits_len {
                acc = acc.add(gf2::Elem::from_raw(u8::from(bit(&a, i) && bit(&b, i))));
            }
            assert_eq!(
                gf2::Elem::from_raw(k::parity(&a, &b, bits_len) as u8),
                acc,
                "parity {bits_len}"
            );

            let srcs: Vec<Vec<u8>> = (0..4)
                .map(|index| noise_bits(bits_len, 0x80 + index + bits_len as u64))
                .collect();
            let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
            for selector in [0u64, 0b1, 0b1111, 0b1010] {
                let mut dst = noise_bits(bits_len, 0x90 + selector + bits_len as u64);
                let mut oracle = dst.clone();
                k::xor_gather(&mut dst, &refs, selector);
                for (index, src) in srcs.iter().enumerate() {
                    if selector >> index & 1 == 1 {
                        for i in 0..bits_len {
                            if bit(src, i) {
                                let flipped = !bit(&oracle, i);
                                put(&mut oracle, i, flipped);
                            }
                        }
                    }
                }
                assert_eq!(dst, oracle, "xor_gather {selector:#b}/{bits_len}");
                assert_padding_zero(&dst, bits_len);
            }
        }
    }

    /// Empty ranges are no-ops through the kernel layer too.
    #[test]
    fn empty_ranges_are_no_ops() {
        let mut dst = [0b1010_1010u8];
        let src = [0b0101_0101u8];
        k::xor_range(&mut dst, &src, 3, 3);
        k::clear_range(&mut dst, 4, 4);
        k::set_range(&mut dst, 5, 5);
        assert_eq!(dst, [0b1010_1010]);
        k::xor_gather(&mut dst, &[], 0);
        assert_eq!(dst, [0b1010_1010]);
    }
}
