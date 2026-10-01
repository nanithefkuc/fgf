//! Per-backend differential tests.
//!
//! [`crate::ops`] can only exercise the one backend the host selected. These
//! tests reach past dispatch and call every architecture kernel the CPU can
//! actually run, comparing each against an independent scalar reference. The
//! canonical Fan–Paar kernels use [`crate::field::wiedemann`]; the other
//! families use [`crate::kernel::scalar`].
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
    FanPaar8, FanPaar32, Gf32, Gf64, Goldilocks, fan_paar, gf8b, gf16, gf32, gf64, quad_mersenne31,
    wiedemann,
};
use crate::kernel::scalar;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::FpTowerTables;
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::affine_8b;
use crate::kernel::tables::{ScaleTable, TowerCoeff, TowerTables, scale_table};
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::{affine_8d, scale_table_8d};
use crate::kernel::{KernelDispatch, RawDispatch};

/// Lengths covering: empty, sub-lane, exact lanes, lane+1, several unroll
/// tiles, and a large odd size. All even so GF(2^16) can use the same list.
const LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
];
/// Byte-XOR lengths around every 16- and 32-byte lane boundary, including
/// compound AVX2 tails and body-plus-tail cases.
const XOR_LENGTHS: &[usize] = &[
    0, 1, 2, 7, 8, 15, 16, 17, 30, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129,
    255, 256, 257, 511, 512, 513, 1023,
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

/// Compare a gather kernel against repeated scalar AXPY calls.
fn check_gather<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], &[E], &[&[u8]]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    check_gather_aligned(name, 1, coeff_at, reference, kernel);
}

/// Compare a gather kernel whose elements span `element_bytes`.
fn check_gather_aligned<E: Copy, F>(
    name: &str,
    element_bytes: usize,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], &[E], &[&[u8]]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &len in GATHER_LENGTHS
        .iter()
        .filter(|&&len| len.is_multiple_of(element_bytes))
    {
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

/// Compare a `0x11B` affine `mul_add` kernel against the reference at every
/// length and coefficient: the 512-bit bodies are field-agnostic, so `Gf8B`
/// runs the same entry with its own bank.
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_affine_mul_add(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable, &[u8])) {
    for_each_gf8_affine_case(|coeff, src| {
        let mut got = noise(src.len(), 0x62);
        let mut want = got.clone();
        kernel(&mut got, affine_8b(coeff), scale_table(coeff), src);
        scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, src);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Compare a `0x11B` affine `mul_assign` kernel against the reference.
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_affine_mul_assign(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable)) {
    for_each_gf8_affine_case(|coeff, src| {
        let mut got = src.to_vec();
        let mut want = src.to_vec();
        kernel(&mut got, affine_8b(coeff), scale_table(coeff));
        scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Compare a `0x11B` affine `mul_into` kernel against the reference.
#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_affine_mul_into(name: &str, kernel: impl Fn(&mut [u8], u64, &ScaleTable, &[u8])) {
    for_each_gf8_affine_case(|coeff, src| {
        // Pre-fill the destination with noise: a fused kernel must overwrite
        // it, never accumulate into it.
        let mut got = noise(src.len(), 0x147);
        let mut want = src.to_vec();
        kernel(&mut got, affine_8b(coeff), scale_table(coeff), src);
        scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

#[cfg(all(feature = "simd512", any(target_arch = "x86", target_arch = "x86_64")))]
fn for_each_gf8_affine_case(case: impl FnMut(gf8b::Elem, &[u8])) {
    let mut case = case;
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for c in 0..=u8::MAX {
            case(gf8b::Elem(c), &src);
        }
    }
    let every_byte: Vec<u8> = (0..=u8::MAX).collect();
    for c in 0..=u8::MAX {
        case(gf8b::Elem(c), &every_byte);
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

/// Differential check for an in-place elementwise kernel: the destination
/// starts as one operand, the kernel multiplies it by the other in place.
fn check_elementwise_assign<F: crate::field::Field>(name: &str, kernel: impl Fn(&mut [u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0xf2);
        let b = noise(len, 0x103);
        let mut got = a.clone();
        kernel(&mut got, &b);
        let mut want = a.clone();
        scalar::mul_elementwise_assign::<F>(&mut want, &b);
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

/// Compare a tower-field `mul_add` kernel against its independent scalar
/// reference at every length and coefficient.
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
    wiedemann::mul_add(dst, u64::from(coeff.to_raw()), src, 16);
}
fn fp16_assign_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem) {
    wiedemann::mul_assign(dst, u64::from(coeff.to_raw()), 16);
}
fn fp16_into_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    wiedemann::mul_assign(dst, u64::from(coeff.to_raw()), 16);
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
    wiedemann::mul_add(dst, u64::from(coeff.to_raw()), src, 32);
}
fn fp32_assign_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem) {
    wiedemann::mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
fn fp32_into_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    wiedemann::mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
fn fp64_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    wiedemann::mul_add(dst, coeff.to_raw(), src, 64);
}
fn fp64_assign_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem) {
    wiedemann::mul_assign(dst, coeff.to_raw(), 64);
}
fn fp64_into_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    wiedemann::mul_assign(dst, coeff.to_raw(), 64);
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
    // Past-dispatch kernels are driven under their own `host_supports` tier
    // gate, which an SIMD_BACKEND downgrade can leave selecting a different
    // tier than the process backend — so each test summons the token its gate
    // proved, rather than reading the process-wide proof.
    #[cfg(feature = "simd512")]
    use archmage::X64V4xToken;
    use archmage::{SimdToken, X64V1Token, X64V2Token, X64V3GfniCryptoToken, X64V3Token};

    /// Checks the token-proven tower lanes directly against scalar arithmetic.
    #[cfg(feature = "simd512")]
    #[test]
    fn gf16_wide_kernels_match_reference() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        check_gf16_mul_add_tables("gf16 wide", |dst, tables, src| {
            x86::gf16::mul_add_avx512(token, dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf16 wide", |dst, tables| {
            x86::gf16::mul_assign_avx512(token, dst, TowerCoeff::new(tables.coeff));
        });
        for &len in LENGTHS {
            let src = noise(len, 0xb51);
            for coeff in gf16_coeffs() {
                let mut expected = vec![0u8; len];
                scalar::mul_add::<gf16::Gf16>(&mut expected, coeff, &src);
                let mut got = noise(len, 0xb52);
                x86::gf16::mul_into_avx512(token, &mut got, TowerCoeff::new(coeff), &src);
                assert_eq!(
                    got, expected,
                    "gf16 wide mul_into len {len} coeff {coeff:?}"
                );
            }
        }
        check_gf16_elementwise("gf16 wide elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_avx512(token, dst, a, b);
        });
        check_elementwise_assign::<gf16::Gf16>("gf16 wide elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_avx512(token, dst, src);
        });
    }

    /// Misaligned destinations on both sides of the `mul_add` peel floor,
    /// with even (peelable) and odd (unpeelable) heads.
    #[cfg(feature = "simd512")]
    #[test]
    fn gf16_wide_mul_add_peels_misaligned_rows() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        let floor = x86::gf16::MUL_ADD_PEEL_MIN;
        for len in [floor - 2, floor, floor + 2, floor + 130, 4096 + 66] {
            for offset in [0usize, 1, 2, 16, 34, 62] {
                let src_storage = noise(len + 64, 0xb61);
                let mut dst_storage = noise(len + 128, 0xb62);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<gf16::Gf16>(&mut want, coeff, src);
                    x86::gf16::mul_add_avx512(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_add_avx512: len {len}, offset {offset}"
                    );
                    dst.copy_from_slice(&before);
                }
            }
        }
    }

    /// Misaligned destinations across the shared 64-byte peel floor for the
    /// wide overwrite body, beside the `mul_add` coverage above.
    #[cfg(feature = "simd512")]
    #[test]
    fn gf16_wide_mul_into_peels_misaligned_rows() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        let floor = x86::gf16::MUL_ADD_PEEL_MIN;
        for len in [floor - 2, floor, floor + 2, floor + 130, 4096 + 66] {
            for offset in [0usize, 1, 2, 16, 34, 62] {
                let src_storage = noise(len + 64, 0xb65);
                let mut dst_storage = noise(len + 128, 0xb66);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = src.to_vec();
                    scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                    x86::gf16::mul_into_avx512(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_into_avx512: len {len}, offset {offset}"
                    );
                }
            }
        }
    }

    /// The 64-byte byte kernels: XOR over every lane boundary and mismatched
    /// source/destination offsets, and `VPOPCNTQ` counts against `count_ones`.
    #[cfg(feature = "simd512")]
    #[test]
    fn bytes512_kernels_match_scalar() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        let peel = x86::bytes::XOR_PEEL_MIN;
        let edges = [peel - 1, peel, peel + 1, peel + 63, 1024 + 255, 4096 + 63];
        for &len in XOR_LENGTHS.iter().chain(&edges) {
            for &(dst_prefix, src_prefix) in &[(0usize, 0usize), (1, 3), (16, 16), (31, 7), (63, 0)]
            {
                let mut dst_storage = noise(dst_prefix + len, 0x2e);
                let src_storage = noise(src_prefix + len, 0x1d);
                let dst = &mut dst_storage[dst_prefix..];
                let src = &src_storage[src_prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                x86::bytes::xor_avx512(token.v4(), dst, src);
                assert_eq!(
                    &*dst, want,
                    "xor_avx512: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                );
            }
            // The dispatched entry routes on the short-buffer threshold.
            for prefix in [0usize, 16] {
                let mut dst_storage = noise(prefix + len, 0x4e);
                let src_storage = noise(prefix + len, 0x4d);
                let dst = &mut dst_storage[prefix..];
                let src = &src_storage[prefix..];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                crate::kernel::xor(dst, src);
                assert_eq!(&*dst, want, "dispatched xor: len {len}, prefix {prefix}");
            }
            for prefix in [0usize, 5, 17] {
                let storage = noise(prefix + len, 0x3f);
                let buf = &storage[prefix..];
                let want: u64 = buf.iter().map(|b| u64::from(b.count_ones())).sum();
                assert_eq!(
                    x86::bytes::count_ones_avx512(token, buf),
                    want,
                    "count_ones_avx512: len {len}, prefix {prefix}",
                );
            }
        }
        assert_eq!(x86::bytes::count_ones_avx512(token, &[0xff; 1000]), 8000);
    }

    /// Compares wide matrix tiles and their remainders with independent row sums.
    #[cfg(feature = "simd512")]
    #[test]
    fn gf16_wide_matrix_matches_reference() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        check_matrix(
            "gf16 wide matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, len, nrows, terms| {
                x86::gf16::mul_add_matrix_avx512(token, rows, len, nrows, terms);
            },
        );
        check_matrix(
            "gf16 wide matrix provider",
            gf16_coeff_at2,
            gf16_reference,
            |rows, len, nrows, terms| {
                x86::gf16::mul_add_matrix_avx512_with(token, rows, len, nrows, terms);
            },
        );
    }

    /// Checks the deferred affine kernels against the scalar oracle.
    ///
    /// The token summon permits direct kernel testing independently of
    /// dispatch. The sweep covers coefficients through the affine helpers,
    /// whose all-byte-values source covers the products.
    #[cfg(feature = "simd512")]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn gf8_avx512_kernels_match_reference() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        };

        check_gf8d_affine_mul_add("gf8d avx512", |dst, map, table, src| {
            x86::gf8::mul_add_avx512(token, dst, map, table, src);
        });
        check_gf8d_affine_mul_assign("gf8d avx512", |dst, map, table| {
            x86::gf8::mul_assign_avx512(token, dst, map, table);
        });
        check_gf8d_affine_mul_into("gf8d avx512", |dst, map, table, src| {
            x86::gf8::mul_into_avx512(token, dst, map, table, src);
        });
        // `Gf8B` through the same field-agnostic 512-bit bodies: the affine
        // bank differs (`0x11B`), the instruction does not.
        check_gf8_affine_mul_add("gf8b avx512", |dst, map, table, src| {
            x86::gf8::mul_add_avx512(token, dst, map, table, src);
        });
        check_gf8_affine_mul_assign("gf8b avx512", |dst, map, table| {
            x86::gf8::mul_assign_avx512(token, dst, map, table);
        });
        check_gf8_affine_mul_into("gf8b avx512", |dst, map, table, src| {
            x86::gf8::mul_into_avx512(token, dst, map, table, src);
        });
        // Prefetch coverage: at and above `PREFETCH_MIN` the into body names
        // lines 512 and 640 bytes ahead, so the length must clear the guard
        // plus tiles, lanes, and a sub-lane tail.
        for len in [24 * 1024, 24 * 1024 + 359, 32 * 1024 + 71] {
            let src = noise(len, 0xb20 + len as u64);
            for &coeff in &[
                gf8d::Elem(0),
                gf8d::Elem(1),
                gf8d::Elem(0x53),
                gf8d::Elem(0xff),
            ] {
                let mut got = noise(len, 0xb21);
                let mut want = src.clone();
                x86::gf8::mul_into_avx512(
                    token,
                    &mut got,
                    affine_8d(coeff),
                    scale_table_8d(coeff),
                    &src,
                );
                scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "gf8d avx512 into prefetch: len {len}, coeff {coeff:?}"
                );
            }
        }
        check_scatter(
            "gf8d avx512 scatter",
            gf8d_coeff_at,
            gf8d_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx512(token, rows, row_len, coeffs, src);
            },
        );
        check_gather(
            "gf8d avx512 gather",
            gf8d_coeff_at,
            gf8d_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_avx512(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8d avx512 matrix",
            gf8d_coeff_at2,
            gf8d_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_avx512(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix_overwrite(
            "gf8d avx512 matrix overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_avx512(token, rows, row_len, nrows, terms);
            },
        );
        // `Gf8B` through the generic 512-bit multi-row bodies.
        check_scatter(
            "gf8b avx512 scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx512(token, rows, row_len, coeffs, src);
            },
        );
        check_gather(
            "gf8b avx512 gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_avx512(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8b avx512 matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_avx512(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix_overwrite(
            "gf8b avx512 matrix overwrite",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_avx512(token, rows, row_len, nrows, terms);
            },
        );
        // 512-bit tile transitions: sub-tile, exact tiles, tile-plus-lanes,
        // and lane-plus-AVX2-ladder remainders, with few and many terms.
        for &row_len in &[64usize, 192, 255, 256, 257, 320, 448, 511, 512, 513, 640] {
            for &nrows in &[1usize, 3, 4, 5, 8, 9] {
                for nterms in [1usize, 3, 9] {
                    let sources: Vec<Vec<u8>> = (0..nterms)
                        .map(|t| noise(row_len, 0xb00 + t as u64))
                        .collect();
                    let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..nterms)
                        .map(|t| (0..nrows).map(|j| gf8d_coeff_at2(t, j)).collect())
                        .collect();
                    let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();
                    let mut got = noise(row_len * nrows, 0xb10);
                    let mut want = got.clone();
                    x86::gf8::mul_add_matrix_avx512(token, &mut got, row_len, nrows, &terms);
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8d_reference(row, coeff, src);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "gf8d avx512 tiles: {row_len}B x {nrows} x {nterms}t"
                    );
                    // Empty terms: accumulate leaves rows unchanged, overwrite zeroes them.
                    let mut got = noise(row_len * nrows, 0xb11);
                    let want = got.clone();
                    let empty: &[(&[gf8d::Elem], &[u8])] = &[];
                    x86::gf8::mul_add_matrix_avx512(token, &mut got, row_len, nrows, empty);
                    assert_eq!(
                        got, want,
                        "gf8d avx512 empty accumulate: {row_len}B x {nrows}"
                    );
                    let mut got = noise(row_len * nrows, 0xb12);
                    x86::gf8::mul_into_matrix_avx512(token, &mut got, row_len, nrows, empty);
                    assert_eq!(
                        got,
                        vec![0u8; row_len * nrows],
                        "gf8d avx512 empty overwrite: {row_len}B x {nrows}"
                    );
                }
            }
        }
        // Alignment peel: rows long enough to peel, placed off a 64-byte
        // boundary, so the head runs through the tail helper with a source
        // window shorter than the row. One-, two-, and four-row groups each
        // peel; accumulate and overwrite share the tail. Lengths track the
        // floor so the peel stays exercised if the floor moves. The exact
        // floor length with a single row leaves a one-byte final remainder,
        // pinning the covered-prefix boundary a restructure must honor.
        let floor = x86::gf8::MATRIX_PEEL_MIN;
        for &offset in &[1usize, 16, 48] {
            for &row_len in &[floor - 64, floor, floor + 64] {
                for &nrows in &[1usize, 2, 4, 7] {
                    let sources: Vec<Vec<u8>> = (0..3usize)
                        .map(|t| noise(row_len, 0xb30 + t as u64))
                        .collect();
                    let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..3usize)
                        .map(|t| (0..nrows).map(|j| gf8d_coeff_at2(t, j)).collect())
                        .collect();
                    let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();
                    let span = row_len * nrows;
                    let mut backing = noise(span + 128, 0xb31);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + span].to_vec();
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8d_reference(row, coeff, src);
                        }
                    }
                    let rows = &mut backing[start..start + span];
                    x86::gf8::mul_add_matrix_avx512(token, rows, row_len, nrows, &terms);
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8d avx512 peeled accumulate: +{offset} {row_len}B x {nrows}"
                    );
                    let mut want = vec![0u8; span];
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8d_reference(row, coeff, src);
                        }
                    }
                    x86::gf8::mul_into_matrix_avx512(token, rows, row_len, nrows, &terms);
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8d avx512 peeled overwrite: +{offset} {row_len}B x {nrows}"
                    );
                }
            }
        }
        // Scatter rows straddle the floor and have both whole-line and
        // sub-line pitches; gather sources exercise the peeled head as well.
        for &offset in &[1usize, 16, 48] {
            for len in [
                x86::gf8::SCATTER_PEEL_MIN,
                x86::gf8::SCATTER_PEEL_MIN + 16,
                1152,
                1168,
                1184,
                1200,
            ] {
                for &nrows in &[1usize, 2, 3, 4, 6] {
                    let src_backing = noise(len + 128, 0xb40);
                    let s0 = src_backing.as_ptr().align_offset(64) + offset;
                    let src = &src_backing[s0..s0 + len];
                    let coeffs: Vec<gf8d::Elem> = (0..nrows).map(gf8d_coeff_at).collect();
                    let mut backing = noise(len * nrows + 128, 0xb41);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len * nrows].to_vec();
                    for (row, &coeff) in want.chunks_exact_mut(len).zip(&coeffs) {
                        gf8d_reference(row, coeff, src);
                    }
                    let rows = &mut backing[start..start + len * nrows];
                    x86::gf8::mul_add_scatter_avx512(token, rows, len, &coeffs, src);
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8d avx512 peeled scatter: +{offset} {len}B x {nrows}"
                    );
                }
            }
            for len in [
                x86::gf8::GATHER_PEEL_MIN - 1,
                x86::gf8::GATHER_PEEL_MIN,
                x86::gf8::GATHER_PEEL_MIN + 64,
            ] {
                for &nsrcs in &[1usize, 3, 16] {
                    let backings: Vec<Vec<u8>> = (0..nsrcs)
                        .map(|t| noise(len + 128, 0xb50 + t as u64))
                        .collect();
                    let srcs: Vec<&[u8]> = backings
                        .iter()
                        .map(|b| {
                            let s = b.as_ptr().align_offset(64) + offset;
                            &b[s..s + len]
                        })
                        .collect();
                    let coeffs: Vec<gf8d::Elem> = (0..nsrcs).map(gf8d_coeff_at).collect();
                    let mut backing = noise(len + 128, 0xb51);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len].to_vec();
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        gf8d_reference(&mut want, coeff, src);
                    }
                    let dst = &mut backing[start..start + len];
                    x86::gf8::mul_add_gather_avx512(token, dst, &coeffs, &srcs);
                    assert_eq!(
                        dst,
                        want.as_slice(),
                        "gf8d avx512 peeled gather: +{offset} {len}B x {nsrcs}"
                    );
                }
            }
        }
        // Scattered rows: disjoint offsets, blocked affine against per-term AXPY.
        // Even iterations use ascending starts, odd iterations a reversed
        // (non-monotonic) order over the same windows.
        for &row_len in ROW_LENS {
            for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
                let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
                if ri % 2 == 1 {
                    starts.reverse();
                }
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
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
                x86::gf8::mul_add_matrix_at_avx512(token, &mut got, row_len, &starts, &terms);
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        scalar::mul_add::<gf8d::Gf8D>(
                            &mut want[starts[j]..starts[j] + row_len],
                            coeff,
                            src,
                        );
                    }
                }
                assert_eq!(got, want, "gf8d avx512 scattered: {row_len}B x {nrows}");
                // Empty terms leave scattered rows unchanged; zero rows are a no-op.
                let mut got = noise(span, 0xdb);
                let want = got.clone();
                let empty: &[(&[gf8d::Elem], &[u8])] = &[];
                x86::gf8::mul_add_matrix_at_avx512(token, &mut got, row_len, &starts, empty);
                assert_eq!(
                    got, want,
                    "gf8d avx512 scattered empty: {row_len}B x {nrows}"
                );
            }
        }
        {
            let mut rows = noise(64, 0xdc);
            let want = rows.clone();
            let empty: &[(&[gf8d::Elem], &[u8])] = &[];
            x86::gf8::mul_add_matrix_at_avx512(token, &mut rows, 16, &[], empty);
            assert_eq!(rows, want, "gf8d avx512 scattered zero rows");
        }
        // `Gf8B` through the generic scattered body: same windows, AES field.
        for &row_len in ROW_LENS {
            for &nrows in ROW_COUNTS {
                let starts: Vec<usize> = (0..nrows).rev().map(|j| j * (row_len + 7)).collect();
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0x940 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<gf8b::Elem>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2(t, j)).collect())
                    .collect();
                let terms: Vec<(&[gf8b::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(span, 0xdd);
                let mut want = got.clone();
                x86::gf8::mul_add_matrix_at_avx512(token, &mut got, row_len, &starts, &terms);
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        gf8_reference(&mut want[starts[j]..starts[j] + row_len], coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8b avx512 scattered: {row_len}B x {nrows}");
            }
        }
        // 64-byte elementwise products for both byte fields: full lanes, the
        // 32/16-byte ladder, and the scalar tail.
        check_gf8_elementwise("gf8 avx512 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx512(token, dst, a, b);
        });
        check_elementwise_assign::<gf8b::Gf8B>("gf8 avx512 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_avx512(token, dst, src);
        });
        check_gf8d_elementwise("gf8d avx512_8d elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx512_8d(token, dst, a, b);
        });
        check_elementwise_assign::<gf8d::Gf8D>("gf8d avx512_8d elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_avx512_8d(token, dst, src);
        });
        // The `_with` provider path over the flat test-slice terms.
        for &row_len in ROW_LENS {
            for &nrows in ROW_COUNTS {
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0xa00 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8d_coeff_at2(t, j)).collect())
                    .collect();
                let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let flat: Vec<gf8d::Elem> = terms
                    .iter()
                    .flat_map(|&(coeffs, _)| coeffs.iter().copied())
                    .collect();
                let flat_srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &flat,
                    nrows,
                    sources: &flat_srcs,
                };
                let mut got = noise(row_len * nrows, 0xdb);
                let mut want = got.clone();
                x86::gf8::mul_add_matrix_avx512_with(token, &mut got, row_len, nrows, &matrix);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        scalar::mul_add::<gf8d::Gf8D>(row, coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8d avx512 with: {row_len}B x {nrows}");
                let mut got = noise(row_len * nrows, 0xdc);
                x86::gf8::mul_into_matrix_avx512_with(token, &mut got, row_len, nrows, &matrix);
                let mut want = vec![0u8; row_len * nrows];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        scalar::mul_add::<gf8d::Gf8D>(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "gf8d avx512 overwrite with: {row_len}B x {nrows}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn gfni_kernels_match_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        check_gf8_mul_add("gf8 gfni", |dst, table, src| {
            x86::gf8::mul_add_gfni(token, dst, table.coeff, src);
        });
        check_gf8_mul_assign("gf8 gfni", |dst, table| {
            x86::gf8::mul_assign_gfni(token, dst, table.coeff);
        });
        check_gf16_mul_add_tables("gf16 gfni", |dst, tables, src| {
            x86::gf16::mul_add_gfni(token, dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf8 gfni", |dst, tables| {
            x86::gf16::mul_assign_gfni(token, dst, TowerCoeff::new(tables.coeff));
        });

        check_scatter(
            "gf8 gfni scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_gfni(token, rows, row_len, coeffs, src);
            },
        );
        check_scatter(
            "gf16 gfni scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                x86::gf16::mul_add_scatter_gfni(token, rows, row_len, coeffs, src);
            },
        );
        check_matrix(
            "gf8 gfni matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_gfni(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix_overwrite(
            "gf8 gfni matrix overwrite",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_gfni(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix(
            "gf16 gfni matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms| {
                x86::gf16::mul_add_matrix_gfni(token, rows, row_len, nrows, terms);
            },
        );
        check_gather(
            "gf8 gfni gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni(token, dst, coeffs, srcs),
        );
        check_gather_aligned(
            "gf16 gfni gather",
            2,
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_gfni(token, dst, coeffs, srcs),
        );
        check_gf8_elementwise("gf8 gfni elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_gfni(token, dst, a, b);
        });
        check_gf16_elementwise("gf16 gfni elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_gfni(token, dst, a, b);
        });
        let v3 = X64V3Token::summon().expect("GFNI implies AVX2");
        check_elementwise_assign::<gf8b::Gf8B>("gf8 gfni elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_gfni(token, dst, src);
        });
        check_elementwise_assign::<gf8d::Gf8D>("gf8d gfni elementwise assign", |dst, src| {
            // `GF2P8MULB` is the AES field; the assign twin runs the same
            // `0x1d` shift/reduce form the three-slice kernel does.
            x86::gf8::mul_elementwise_assign_avx2::<0x1d>(v3, dst, src);
        });
        check_elementwise_assign::<gf8d::Gf8D>("gf8d gfni_8d elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_gfni_8d(token, dst, src);
        });
        check_elementwise_assign::<gf16::Gf16>("gf16 gfni elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_gfni(token, dst, src);
        });
        // Level-2/3 tower kernels: the period-2 lane multiply one and two
        // levels up from the GF(2^16) kernel.
        check_tower_gfni_kernels(token);
    }

    /// Differential-check the GFNI GF(2^32) and GF(2^64) kernels — the level-2
    /// and level-3 tower multiplies — against the portable scalar oracle.
    fn check_tower_gfni_kernels(token: X64V3GfniCryptoToken) {
        check_tower_mul_add(
            "gf32 gfni mul_add",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_reference,
            |dst, coeff, src| {
                x86::gf32::mul_add_gfni(token, dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf32 gfni mul_assign",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_assign_reference,
            |dst, coeff| {
                x86::gf32::mul_assign_gfni(token, dst, coeff, x86::gf32::gf32_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf32 gfni mul_into",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_into_reference,
            |dst, coeff, src| {
                x86::gf32::mul_into_gfni(token, dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        // Level-3: the same identity over GF(2^32) lanes.
        check_tower_mul_add(
            "gf64 gfni mul_add",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_reference,
            |dst, coeff, src| {
                x86::gf64::mul_add_gfni(token, dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf64 gfni mul_assign",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_assign_reference,
            |dst, coeff| {
                x86::gf64::mul_assign_gfni(token, dst, coeff, x86::gf64::gf64_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf64 gfni mul_into",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_into_reference,
            |dst, coeff, src| {
                x86::gf64::mul_into_gfni(token, dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
    }

    /// Scattered-rows differential for the blocked affine matrix.
    fn check_gf8d_scattered_gfni_rows(token: X64V3GfniCryptoToken) {
        // Scattered rows: disjoint offsets, blocked affine against per-term AXPY.
        // Even iterations use ascending starts, odd iterations a reversed
        // (non-monotonic) order over the same windows.
        for &row_len in ROW_LENS {
            for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
                let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
                if ri % 2 == 1 {
                    starts.reverse();
                }
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
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
                x86::gf8::mul_add_matrix_at_gfni_8d(token, &mut got, row_len, &starts, &terms);
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
                    "gf8d gfni_8d matrix_scattered: row_len {row_len}, nrows {nrows}"
                );
            }
        }
    }

    /// The `0x11D` field's GFNI path: `VGF2P8AFFINEQB` with the const-derived
    /// affine bank. The sweep is exhaustive over coefficients, and the
    /// all-byte-values source makes it exhaustive over products — this is the
    /// silicon-level proof of the affine map's bit-order convention, and the
    /// loud failure if `GF2P8MULB` (the AES field) were ever substituted.
    #[test]
    fn gf8d_gfni_kernels_match_reference() {
        // The non-temporal `mul_into` body sits far above the shared lengths;
        // destination offsets 0 and 1 cover both sides of the alignment peel.
        const NT_LEN: usize = (2 << 20) + 130;

        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");

        check_gf8d_affine_mul_add("gf8d gfni_8d", |dst, map, table, src| {
            x86::gf8::mul_add_gfni_8d(token, dst, map, table, src);
        });
        check_gf8d_affine_mul_assign("gf8d gfni_8d", |dst, map, table| {
            x86::gf8::mul_assign_gfni_8d(token, dst, map, table);
        });
        check_gf8d_affine_mul_into("gf8d gfni_8d", |dst, map, table, src| {
            x86::gf8::mul_into_gfni_8d(token, dst, map, table, src);
        });
        let source = noise(NT_LEN + 2, 0x2b8);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            // Noise, not zeros: a streaming body that accumulated into the
            // destination instead of overwriting it must fail this check.
            let mut buffer = noise(NT_LEN + 1, 0x2b9);
            for coeff in [0u8, 1, 0x53, 0xff].map(gf8d::Elem) {
                let mut want = src.to_vec();
                scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
                let got = &mut buffer[offset..offset + NT_LEN];
                x86::gf8::mul_into_gfni_8d(
                    token,
                    got,
                    affine_8d(coeff),
                    scale_table_8d(coeff),
                    src,
                );
                assert_eq!(&*got, want, "gf8d gfni_8d NT mul_into: offset {offset}");
            }
        }

        // The register-blocked affine multi-row shapes against per-row AXPY,
        // over the same row-group and tile boundaries as the `Gf8B` kernels.
        check_scatter(
            "gf8d gfni_8d scatter",
            gf8d_coeff_at,
            gf8d_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_gfni_8d(token, rows, row_len, coeffs, src);
            },
        );
        check_matrix(
            "gf8d gfni_8d matrix",
            gf8d_coeff_at2,
            gf8d_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_gfni_8d(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix_overwrite(
            "gf8d gfni_8d matrix overwrite",
            gf8d_coeff_at2,
            gf8d_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_gfni_8d(token, rows, row_len, nrows, terms);
            },
        );
        check_gather(
            "gf8d gfni_8d gather",
            gf8d_coeff_at,
            gf8d_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_gfni_8d(token, dst, coeffs, srcs),
        );
        check_gf8d_scattered_gfni_rows(token);
    }

    /// Field-independent XOR across the half-lane peel floor: the 16-mod-32
    /// residues that peel, their odd neighbours that decline, and every
    /// destination offset against the byte-wise reference.
    #[test]
    fn bytes_avx2_peels_half_lane() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        let floor = x86::HALF_LANE_PEEL_MIN;
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c1);
                let mut dst_storage = noise(len + 128, 0x2c2);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                let mut want = dst.to_vec();
                scalar::xor(&mut want, src);
                x86::bytes::xor_avx2(token, dst, src);
                assert_eq!(&*dst, want, "xor_avx2: len {len}, offset {offset}");
            }
        }
    }

    /// The `GF2P8MULB` byte kernels across the half-lane peel floor, against
    /// the scalar reference for all three single-row forms.
    #[test]
    fn gf8_gfni_peels_half_lane() {
        if !host_supports(&[Backend::V3GfniCrypto]) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let floor = x86::HALF_LANE_PEEL_MIN;
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c3);
                let mut dst_storage = noise(len + 128, 0x2c4);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in [0u8, 1, 0x53, 0xff].map(gf8b::Elem) {
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, src);
                    x86::gf8::mul_add_gfni(token, dst, coeff, src);
                    assert_eq!(&*dst, want, "mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
                    x86::gf8::mul_assign_gfni(token, dst, coeff);
                    assert_eq!(&*dst, want, "mul_assign_gfni: len {len}, offset {offset}");
                    let mut want = src.to_vec();
                    scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
                    x86::gf8::mul_into_gfni(token, dst, coeff, src);
                    assert_eq!(&*dst, want, "mul_into_gfni: len {len}, offset {offset}");
                }
            }
        }
    }

    /// The `VGF2P8AFFINEQB` byte kernels across the half-lane peel floor,
    /// against the scalar reference for all three single-row forms.
    #[test]
    fn gf8d_gfni_peels_half_lane() {
        if !host_supports(&[Backend::V3GfniCrypto]) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let floor = x86::HALF_LANE_PEEL_MIN;
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c5);
                let mut dst_storage = noise(len + 128, 0x2c6);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in [0u8, 1, 0x53, 0xff].map(gf8d::Elem) {
                    let map = affine_8d(coeff);
                    let table = scale_table_8d(coeff);
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<gf8d::Gf8D>(&mut want, coeff, src);
                    x86::gf8::mul_add_gfni_8d(token, dst, map, table, src);
                    assert_eq!(&*dst, want, "mul_add_gfni_8d: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
                    x86::gf8::mul_assign_gfni_8d(token, dst, map, table);
                    assert_eq!(
                        &*dst, want,
                        "mul_assign_gfni_8d: len {len}, offset {offset}"
                    );
                    let mut want = src.to_vec();
                    scalar::mul_assign::<gf8d::Gf8D>(&mut want, coeff);
                    x86::gf8::mul_into_gfni_8d(token, dst, map, table, src);
                    assert_eq!(&*dst, want, "mul_into_gfni_8d: len {len}, offset {offset}");
                }
            }
        }
    }

    /// The GF(2^16) GFNI kernels across the half-lane peel floor: even
    /// lengths over every destination residue, so declined odd heads stay
    /// covered beside the peeled 16-mod-32 ones.
    #[test]
    fn gf16_gfni_peels_half_lane() {
        if !host_supports(&[Backend::V3GfniCrypto]) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let floor = x86::HALF_LANE_PEEL_MIN;
        for len in [floor - 2, floor, floor + 2, floor + 130, 1152, 4096 + 66] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c7);
                let mut dst_storage = noise(len + 128, 0x2c8);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let prepared = TowerCoeff::new(coeff);
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<gf16::Gf16>(&mut want, coeff, src);
                    x86::gf16::mul_add_gfni(token, dst, prepared, src);
                    assert_eq!(&*dst, want, "gf16 mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                    x86::gf16::mul_assign_gfni(token, dst, prepared);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_assign_gfni: len {len}, offset {offset}"
                    );
                    let mut want = src.to_vec();
                    scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                    x86::gf16::mul_into_gfni(token, dst, prepared, src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_into_gfni: len {len}, offset {offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_kernels_match_reference() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        check_gf8_mul_add("gf8 avx2", |dst, table, src| {
            x86::gf8::mul_add_avx2(token, dst, table, src);
        });
        check_gf8_mul_assign("gf8 avx2", |dst, table| {
            x86::gf8::mul_assign_avx2(token, dst, table);
        });
        check_gf16_mul_add_tables("gf16 avx2", |dst, tables, src| {
            x86::gf16::mul_add_avx2(token, dst, tables, src);
        });
        check_gf16_mul_assign_tables("gf16 avx2", |dst, tables| {
            x86::gf16::mul_assign_avx2(token, dst, tables);
        });
        check_gf8_mul_into("gf8 avx2 mul_into", |dst, table, src| {
            x86::gf8::mul_into_avx2(token, dst, table, src);
        });
        check_gf16_mul_into_tables("gf16 avx2 mul_into", |dst, tables, src| {
            x86::gf16::mul_into_avx2(token, dst, tables, src);
        });
        check_scatter(
            "gf8 avx2 scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx2(token, rows, row_len, coeffs, src);
            },
        );
        check_scatter(
            "gf16 avx2 scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                x86::gf16::mul_add_scatter_avx2(token, rows, row_len, coeffs, src);
            },
        );
        check_gather(
            "gf8 avx2 gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_avx2(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8 avx2 matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_avx2(token, rows, row_len, nrows, terms);
            },
        );
        check_gf8_elementwise("gf8 avx2 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx2::<0x1b>(token, dst, a, b);
        });
        check_gf8d_elementwise("gf8d avx2 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx2::<0x1d>(token, dst, a, b);
        });
        if let Some(gfni) = X64V3GfniCryptoToken::summon() {
            check_gf8d_elementwise("gf8d gfni_8d elementwise", |dst, a, b| {
                x86::gf8::mul_elementwise_gfni_8d(gfni, dst, a, b);
            });
        } else {
            eprintln!("skipping: gf8d iso elementwise needs GFNI");
        }
        check_gf16_elementwise("gf16 avx2 elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_avx2(token, dst, a, b);
        });
        // Fan–Paar tower (GF(2^16)/32/64): the fp8 nibble tower and its
        // period-2 lane-mul extensions.
        check_elementwise_assign::<gf8b::Gf8B>("gf8 avx2 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_avx2::<0x1b>(token, dst, src);
        });
        check_elementwise_assign::<gf8d::Gf8D>("gf8d avx2 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_avx2::<0x1d>(token, dst, src);
        });
        check_elementwise_assign::<gf16::Gf16>("gf16 avx2 elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_avx2(token, dst, src);
        });
        check_fan_paar_avx2_kernels(token);
    }

    /// Differential-check the Fan–Paar GF(2^16)/32/64 AVX2 kernels — the fp8
    /// nibble tower and its period-2 lane-mul extensions — against the
    /// Wiedemann scalar oracle.
    fn check_fan_paar_avx2_kernels(token: X64V3Token) {
        check_tower_mul_add(
            "fp16 avx2 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_avx2_fp16(token, dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 avx2 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_avx2_fp16(token, dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 avx2 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_avx2_fp16(token, dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_add(
            "fp32 avx2 mul_add",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_avx2_fp32(token, dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp32 avx2 mul_assign",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_avx2_fp32(token, dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp32 avx2 mul_into",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_avx2_fp32(token, dst, coeff, src);
            },
        );
        check_tower_mul_add(
            "fp64 avx2 mul_add",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_avx2_fp64(token, dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp64 avx2 mul_assign",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_avx2_fp64(token, dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp64 avx2 mul_into",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_avx2_fp64(token, dst, coeff, src);
            },
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn ssse3_kernels_match_reference() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        let token = X64V2Token::summon().expect("guard passed: SSSE3 summons here");
        check_gf8_mul_add("gf8 ssse3", |dst, table, src| {
            x86::gf8::mul_add_ssse3(token, dst, table, src);
        });
        check_gf8_mul_assign("gf8 ssse3", |dst, table| {
            x86::gf8::mul_assign_ssse3(token, dst, table);
        });
        check_gf16_mul_add_tables("gf16 ssse3", |dst, tables, src| {
            x86::gf16::mul_add_ssse3(token, dst, tables, src);
        });
        check_gf16_mul_assign_tables("gf16 ssse3", |dst, tables| {
            x86::gf16::mul_assign_ssse3(token, dst, tables);
        });
        check_gf8_mul_into("gf8 ssse3 mul_into", |dst, table, src| {
            x86::gf8::mul_into_ssse3(token, dst, table, src);
        });
        check_gf16_mul_into_tables("gf16 ssse3 mul_into", |dst, tables, src| {
            x86::gf16::mul_into_ssse3(token, dst, tables, src);
        });
        check_scatter(
            "gf8 ssse3 scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_ssse3(token, rows, row_len, coeffs, src);
            },
        );
        check_scatter(
            "gf16 ssse3 scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                x86::gf16::mul_add_scatter_ssse3(token, rows, row_len, coeffs, src);
            },
        );
        check_gather(
            "gf8 ssse3 gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| x86::gf8::mul_add_gather_ssse3(token, dst, coeffs, srcs),
        );
        check_gather_aligned(
            "gf16 ssse3 gather",
            2,
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_ssse3(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8 ssse3 matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_ssse3(token, rows, row_len, nrows, terms);
            },
        );
        check_matrix(
            "gf16 ssse3 matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms| {
                x86::gf16::mul_add_matrix_ssse3(token, rows, row_len, nrows, terms);
            },
        );
        check_gf8_elementwise("gf8 ssse3 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_ssse3::<0x1b>(token, dst, a, b);
        });
        check_gf8d_elementwise("gf8d ssse3 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_ssse3::<0x1d>(token, dst, a, b);
        });
        check_gf16_elementwise("gf16 ssse3 elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_ssse3(token, dst, a, b);
        });
        check_elementwise_assign::<gf8b::Gf8B>("gf8 ssse3 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_ssse3::<0x1b>(token, dst, src);
        });
        check_elementwise_assign::<gf8d::Gf8D>("gf8d ssse3 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_ssse3::<0x1d>(token, dst, src);
        });
        check_elementwise_assign::<gf16::Gf16>("gf16 ssse3 elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_ssse3(token, dst, src);
        });
        check_tower_mul_add(
            "fp16 ssse3 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_ssse3_fp16(token, dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 ssse3 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_ssse3_fp16(token, dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 ssse3 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_ssse3_fp16(token, dst, &FpTowerTables::new(coeff), src);
            },
        );
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        let v1 = X64V1Token::summon().expect("SSE2 is available on this x86 host");
        let v3 = X64V3Token::summon();
        for &len in XOR_LENGTHS {
            for &(dst_prefix, src_prefix) in &[(0usize, 0usize), (1, 3), (15, 17), (31, 7)] {
                let mut dst_storage = noise(dst_prefix + len, 0x2d);
                let src_storage = noise(src_prefix + len, 0x1c);
                let dst = &mut dst_storage[dst_prefix..];
                let src = &src_storage[src_prefix..];
                let mut want = dst.to_vec();
                let mut avx2 = want.clone();
                let mut sse2 = want.clone();
                scalar::xor(&mut want, src);
                if let Some(token) = v3 {
                    x86::xor_avx2(token, &mut avx2, src);
                    assert_eq!(
                        avx2, want,
                        "avx2 xor: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                    );
                }
                x86::xor_sse2(v1, &mut sse2, src);
                assert_eq!(
                    sse2, want,
                    "sse2 xor: len {len}, dst prefix {dst_prefix}, src prefix {src_prefix}",
                );
            }
        }
    }

    /// The broadcast XOR kernels directly — every value width the dispatch
    /// can produce (1-byte GF(2^8), 2-byte GF(2^16), 4-byte GF(2^32),
    /// 8-byte GF(2^64), and a 16-byte width), plus the zero value and a
    /// width that divides neither lane, against the portable byte-cyclic
    /// reference.
    #[test]
    fn xor_broadcast_matches_scalar_reference() {
        const VALUES: [&[u8]; 8] = [
            &[0x00],
            &[0x53],
            &[0xa7, 0x53],
            &[0x01, 0x00],
            &[0x1a, 0x2b, 0x3c, 0x4d],
            &[0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78],
            &[
                0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
            // A width that divides neither lane, exercising the portable
            // fallback in both kernels.
            &[0x11, 0x22, 0x33],
        ];
        let v1 = X64V1Token::summon().expect("SSE2 is baseline on this x86 host");
        let v3 = X64V3Token::summon();
        for &len in XOR_LENGTHS {
            for value in VALUES {
                let dst = &mut noise(len, 0x2e);
                let mut want = dst.clone();
                let mut avx2 = want.clone();
                let mut sse2 = want.clone();
                for (w, &b) in want.iter_mut().zip(value.iter().cycle()) {
                    *w ^= b;
                }
                if let Some(token) = v3 {
                    x86::bytes::xor_broadcast_avx2(token, &mut avx2, value);
                    assert_eq!(avx2, want, "avx2 xor_broadcast: len {len}, value {value:?}");
                }
                x86::bytes::xor_broadcast_sse2(v1, &mut sse2, value);
                assert_eq!(sse2, want, "sse2 xor_broadcast: len {len}, value {value:?}");
            }
        }
    }

    /// The documented value-width contract of the direct broadcast entries:
    /// an empty value and a 17-byte value both panic, and the panic message
    /// names the operation.
    #[test]
    fn xor_broadcast_entries_reject_invalid_value_widths() {
        let mut dst = [0u8; 32];
        let sse = X64V1Token::summon().expect("SSE2 is baseline on this x86 host");
        for bad in [&[][..], &[0u8; 17][..]] {
            if let Some(token) = X64V3Token::summon() {
                let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    x86::bytes::xor_broadcast_avx2(token, &mut dst, bad);
                }))
                .expect_err("xor_broadcast_avx2 must reject an invalid value width");
                let msg = err.downcast_ref::<String>().expect("string panic payload");
                assert!(
                    msg.starts_with("xor_broadcast_avx2: value is"),
                    "panic must name the operation: {msg}"
                );
            }
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                x86::bytes::xor_broadcast_sse2(sse, &mut dst, bad);
            }))
            .expect_err("xor_broadcast_sse2 must reject an invalid value width");
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.starts_with("xor_broadcast_sse2: value is"),
                "panic must name the operation: {msg}"
            );
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
        let gfni = X64V3GfniCryptoToken::summon();
        let v3 = X64V3Token::summon();
        let v2 = X64V2Token::summon().expect("SSSE3 summons on this x86_64 host");
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF8_COEFFS {
                let table = scale_table(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
                if host_supports(&[Backend::V3GfniCrypto]) {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_gfni(
                        gfni.expect("guard passed: GFNI summons here"),
                        got,
                        coeff,
                        src,
                    );
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if host_supports(&[Backend::V3]) {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_avx2(
                        v3.expect("guard passed: AVX2 summons here"),
                        got,
                        table,
                        src,
                    );
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + NT_LEN];
                x86::gf8::mul_into_ssse3(v2, got, table, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }

            // GF(2^16) needs an even length and an even peel.
            let src = &src[..NT_LEN];
            for coeff in GF16_COEFFS {
                let tables = TowerTables::new(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                if host_supports(&[Backend::V3GfniCrypto]) {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_gfni(
                        gfni.expect("guard passed: GFNI summons here"),
                        got,
                        TowerCoeff::new(coeff),
                        src,
                    );
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if host_supports(&[Backend::V3]) {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_avx2(
                        v3.expect("guard passed: AVX2 summons here"),
                        got,
                        &tables,
                        src,
                    );
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + src.len()];
                x86::gf16::mul_into_ssse3(v2, got, &tables, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    }

    #[test]
    fn half_lane_payloads_match_reference() {
        if !host_supports(&[Backend::V3GfniCrypto]) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let gfni = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let avx2 = X64V3Token::summon().expect("GFNI implies AVX2");
        let coeff = gf8b::Elem::from_raw(0x53);
        for len in [511usize, 512, 1152, 1168, 1200, 2048] {
            for off in [0usize, 1, 16, 32, 48] {
                let src_backing = noise(len + 128, 0x522);
                let src_start = src_backing.as_ptr().align_offset(64) + (off + 16) % 64;
                let src = &src_backing[src_start..src_start + len];
                let mut backing = noise(len + 128, 0x533);
                let start = backing.as_ptr().align_offset(64) + off;
                let dst = &mut backing[start..start + len];
                let mut want = dst.to_vec();

                x86::bytes::xor_avx2(avx2, dst, src);
                scalar::xor(&mut want, src);
                assert_eq!(dst, want, "xor {len} +{off}");

                x86::gf8::mul_add_gfni(gfni, dst, coeff, src);
                scalar::mul_add::<gf8b::Gf8B>(&mut want, coeff, src);
                assert_eq!(dst, want, "mul_add {len} +{off}");

                x86::gf8::mul_assign_gfni(gfni, dst, coeff);
                scalar::mul_assign::<gf8b::Gf8B>(&mut want, coeff);
                assert_eq!(dst, want, "mul_assign {len} +{off}");
            }
        }
    }

    /// Multi-row scatter across short and long rows, including pitches that
    /// alternate the destination alignment within a row group.
    ///
    /// The source and destination start at independent offsets; reference
    /// multiplication checks the peeled head and the vector/tail seam.
    #[test]
    fn aligned_scatter_matches_reference() {
        const ROW_LENS: &[usize] = &[512, 1152, 1168, 1184, 1200, 2048, 4096];
        type ScatterKernel<'a> = dyn Fn(&mut [u8], usize, &[gf16::Elem], &[u8]) + 'a;

        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let v3 = X64V3Token::summon().expect("GFNI implies AVX2");
        let v2 = X64V2Token::summon().expect("GFNI implies SSSE3");
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let coeffs8: Vec<_> = (0..nrows).map(gf8_coeff_at).collect();
                    let mut want = rows.to_vec();
                    x86::gf8::mul_add_scatter_gfni(token, rows, row_len, &coeffs8, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs8) {
                        gf8_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 scatter: {row_len}/{nrows}/{skew}"
                    );

                    let affine_coeffs: Vec<_> = (0..nrows).map(gf8d_coeff_at).collect();
                    let mut want = rows.to_vec();
                    x86::gf8::mul_add_scatter_gfni_8d(token, rows, row_len, &affine_coeffs, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&affine_coeffs) {
                        gf8d_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8d scatter: {row_len}/{nrows}/{skew}"
                    );

                    let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                    let mut want = rows.to_vec();
                    x86::gf16::mul_add_scatter_gfni(token, rows, row_len, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 scatter: {row_len}/{nrows}/{skew}"
                    );

                    let kernels: [(&str, &ScatterKernel<'_>); 2] = [
                        ("avx2", &|rows: &mut [u8],
                                   row_len: usize,
                                   coeffs: &[gf16::Elem],
                                   src: &[u8]| {
                            x86::gf16::mul_add_scatter_avx2(v3, rows, row_len, coeffs, src);
                        }),
                        ("ssse3", &|rows: &mut [u8],
                                    row_len: usize,
                                    coeffs: &[gf16::Elem],
                                    src: &[u8]| {
                            x86::gf16::mul_add_scatter_ssse3(v2, rows, row_len, coeffs, src);
                        }),
                    ];
                    for (name, kernel) in kernels {
                        let mut want = rows.to_vec();
                        kernel(rows, row_len, &coeffs16, &src);
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs16) {
                            gf16_reference(row, coeff, &src);
                        }
                        assert_eq!(
                            rows,
                            want.as_slice(),
                            "gf16 {name} scatter: {row_len}/{nrows}/{skew}"
                        );
                    }
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
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], C, &[u8]),
        mulinto: impl Fn(&mut [u8], C, &[u8]),
        mulassign: impl Fn(&mut [u8], C),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], C),
        subscalar: impl Fn(&mut [u8], C),
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

            let mut got = base.clone();
            elemwise_assign(&mut got, &src);
            let mut want = base.clone();
            prime::mul_elementwise::<F>(&mut want, &base, &src);
            assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");

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

                let mut got = base.clone();
                addscalar(&mut got, c);
                let mut want = base.clone();
                prime::add_assign_scalar::<F>(&mut want, to_elem(c));
                assert_eq!(got, want, "{label} add_assign_scalar len {len} coeff {c:?}");

                let mut got = base.clone();
                subscalar(&mut got, c);
                let mut want = base.clone();
                prime::sub_assign_scalar::<F>(&mut want, to_elem(c));
                assert_eq!(got, want, "{label} sub_assign_scalar len {len} coeff {c:?}");
            }
        }
    }

    #[test]
    fn prime_avx2_kernels_match_reference() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        drive_prime::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            mersenne31::Elem,
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, coeff, src| x86::mersenne31::mul_add_avx2(token, dst, coeff, src),
            |dst, coeff, src| x86::mersenne31::mul_into_avx2(token, dst, coeff, src),
            |dst, coeff| x86::mersenne31::mul_assign_avx2(token, dst, coeff),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
            "m31 avx2",
        );
        drive_prime::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            goldilocks::Elem,
            |dst, src| x86::goldilocks::add_assign_avx2(token, dst, src),
            |dst, src| x86::goldilocks::sub_assign_avx2(token, dst, src),
            |dst, coeff, src| x86::goldilocks::mul_add_avx2(token, dst, coeff, src),
            |dst, coeff, src| x86::goldilocks::mul_into_avx2(token, dst, coeff, src),
            |dst, coeff| x86::goldilocks::mul_assign_avx2(token, dst, coeff),
            |dst, a, b| x86::goldilocks::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::goldilocks::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::goldilocks::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::goldilocks::sub_assign_scalar_avx2(token, dst, c),
            "gld avx2",
        );
    }

    #[cfg(feature = "simd512")]
    #[test]
    fn prime_avx512_kernels_match_reference() {
        // Lane-straddling lengths past the 64-byte vector, each ending in an
        // AVX2 and a scalar remainder.
        const M31_WIDE: &[usize] = &[0, 4, 60, 64, 68, 96, 128, 196, 260, 1020, 1028];
        const GLD_WIDE: &[usize] = &[0, 8, 56, 64, 72, 96, 128, 200, 264, 1024, 1032];
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        let token = token.v4();
        drive_prime::<Mersenne31, u32>(
            M31_WIDE,
            &M31_COEFFS,
            mersenne31::Elem,
            |dst, src| x86::mersenne31::add_assign_avx512(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx512(token, dst, src),
            |dst, coeff, src| x86::mersenne31::mul_add_avx512(token, dst, coeff, src),
            |dst, coeff, src| x86::mersenne31::mul_into_avx512(token, dst, coeff, src),
            |dst, coeff| x86::mersenne31::mul_assign_avx512(token, dst, coeff),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx512(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token, dst, c),
            "m31 avx512",
        );
        drive_prime::<Goldilocks, u64>(
            GLD_WIDE,
            &GLD_COEFFS,
            goldilocks::Elem,
            |dst, src| x86::goldilocks::add_assign_avx512(token, dst, src),
            |dst, src| x86::goldilocks::sub_assign_avx512(token, dst, src),
            |dst, coeff, src| x86::goldilocks::mul_add_avx512(token, dst, coeff, src),
            |dst, coeff, src| x86::goldilocks::mul_into_avx512(token, dst, coeff, src),
            |dst, coeff| x86::goldilocks::mul_assign_avx512(token, dst, coeff),
            |dst, a, b| x86::goldilocks::mul_elementwise_avx512(token, dst, a, b),
            |dst, s| x86::goldilocks::mul_elementwise_assign_avx512(token, dst, s),
            |dst, c| x86::goldilocks::add_assign_scalar_avx512(token, dst, c),
            |dst, c| x86::goldilocks::sub_assign_scalar_avx512(token, dst, c),
            "gld avx512",
        );
    }

    /// Goldilocks lane edges noise buffers rarely hit — non-canonical
    /// inputs, operands at `p − 1`, and products whose fold both borrows and
    /// wraps — against the `u128 % p` model.
    #[cfg(feature = "simd512")]
    #[test]
    fn gld_avx512_lane_edges_match_u128_model() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no V4x token on this host");
            return;
        };
        let modulus = goldilocks::MODULUS;
        let edges: [u64; 12] = [
            0,
            1,
            2,
            0xFFFF_FFFF,
            0x1_0000_0000,
            modulus - 2,
            modulus - 1,
            modulus,
            modulus + 1,
            u64::MAX - 1,
            u64::MAX,
            0xFFFF_FFFE_FFFF_FFFF,
        ];
        let (lhs, rhs): (Vec<u64>, Vec<u64>) = edges
            .iter()
            .flat_map(|&x| edges.iter().map(move |&y| (x, y)))
            .unzip();
        let bytes = |v: &[u64]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        let (lhs_bytes, rhs_bytes) = (bytes(&lhs), bytes(&rhs));
        let word = |buf: &[u8], lane: usize| {
            u64::from_le_bytes(buf[8 * lane..8 * lane + 8].try_into().unwrap())
        };
        let wide = u128::from(modulus);

        let mut product = vec![0u8; lhs_bytes.len()];
        x86::goldilocks::mul_elementwise_avx512(token.v4(), &mut product, &lhs_bytes, &rhs_bytes);
        let mut sum = lhs_bytes.clone();
        x86::goldilocks::add_assign_avx512(token.v4(), &mut sum, &rhs_bytes);
        let mut diff = lhs_bytes.clone();
        x86::goldilocks::sub_assign_avx512(token.v4(), &mut diff, &rhs_bytes);
        for (lane, (&left, &right)) in lhs.iter().zip(&rhs).enumerate() {
            let (x, y) = (u128::from(left) % wide, u128::from(right) % wide);
            assert_eq!(
                u128::from(word(&product, lane)),
                x * y % wide,
                "gld mul {left:#x} * {right:#x}"
            );
            assert_eq!(
                u128::from(word(&sum, lane)),
                (x + y) % wide,
                "gld add {left:#x} {right:#x}"
            );
            assert_eq!(
                u128::from(word(&diff, lane)),
                (x + wide - y) % wide,
                "gld sub {left:#x} {right:#x}"
            );
        }
    }

    #[test]
    fn prime_sse42_kernels_match_reference() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSE4.2 on this host");
            return;
        }
        let token = X64V2Token::summon().expect("guard passed: SSE4.2 summons here");
        drive_prime::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            mersenne31::Elem,
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, coeff, src| x86::mersenne31::mul_add_sse42(token, dst, coeff, src),
            |dst, coeff, src| x86::mersenne31::mul_into_sse42(token, dst, coeff, src),
            |dst, coeff| x86::mersenne31::mul_assign_sse42(token, dst, coeff),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
            "m31 sse4.2",
        );
        drive_prime::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            goldilocks::Elem,
            |dst, src| x86::goldilocks::add_assign_sse42(token, dst, src),
            |dst, src| x86::goldilocks::sub_assign_sse42(token, dst, src),
            |dst, coeff, src| x86::goldilocks::mul_add_sse42(token, dst, coeff, src),
            |dst, coeff, src| x86::goldilocks::mul_into_sse42(token, dst, coeff, src),
            |dst, coeff| x86::goldilocks::mul_assign_sse42(token, dst, coeff),
            |dst, a, b| x86::goldilocks::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::goldilocks::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::goldilocks::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::goldilocks::sub_assign_scalar_sse42(token, dst, c),
            "gld sse4.2",
        );
    }
    #[test]
    fn qm31_avx2_kernels_match_reference() {
        // Multiples of 8 (one complex element) straddling the 32-byte AVX2
        // lane, with odd element tails.
        const QM31_LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
        const P: u32 = crate::field::mersenne31::MODULUS;
        const QM31_VALUES: [crate::field::quad_mersenne31::Elem; 6] = [
            crate::field::quad_mersenne31::Elem(0, 0),
            crate::field::quad_mersenne31::Elem(1, 0),
            crate::field::quad_mersenne31::Elem(0, 1),
            crate::field::quad_mersenne31::Elem(7, 2),
            crate::field::quad_mersenne31::Elem(P - 1, 3),
            crate::field::quad_mersenne31::Elem(P - 1, P - 1),
        ];
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        drive_prime::<
            crate::field::quad_mersenne31::QuadMersenne31,
            crate::field::quad_mersenne31::Elem,
        >(
            QM31_LENS,
            &QM31_VALUES,
            |e| e,
            |dst, src| x86::quad_mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::quad_mersenne31::sub_assign_avx2(token, dst, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_add_avx2(token, dst, coeff, src),
            |dst, coeff, src| x86::quad_mersenne31::mul_into_avx2(token, dst, coeff, src),
            |dst, coeff| x86::quad_mersenne31::mul_assign_avx2(token, dst, coeff),
            |dst, a, b| x86::quad_mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::quad_mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::quad_mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::quad_mersenne31::sub_assign_scalar_avx2(token, dst, c),
            "qm31 avx2",
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
    use archmage::SimdToken as _;

    #[test]
    fn neon_kernels_match_reference() {
        if !host_supports(&[Backend::Neon]) {
            eprintln!("skipping: no NEON on this host");
            return;
        }
        let token = archmage::NeonToken::summon().expect("host_supports guard: NEON summons here");
        check_gf8_mul_add("gf8 neon", |dst, table, src| {
            aarch64::gf8::mul_add_neon(token, dst, table, src)
        });
        check_gf8_mul_assign("gf8 neon", |dst, table| {
            aarch64::gf8::mul_assign_neon(token, dst, table)
        });
        check_gf16_mul_add_tables("gf16 neon", |dst, tables, src| {
            aarch64::gf16::mul_add_neon(token, dst, tables, src)
        });
        check_gf16_mul_assign_tables("gf16 neon", |dst, tables| {
            aarch64::gf16::mul_assign_neon(token, dst, tables)
        });
        check_gf8_mul_into("gf8 neon mul_into", |dst, table, src| {
            aarch64::gf8::mul_into_neon(token, dst, table, src)
        });
        check_gf16_mul_into_tables("gf16 neon mul_into", |dst, tables, src| {
            aarch64::gf16::mul_into_neon(token, dst, tables, src)
        });

        check_scatter(
            "gf8 neon scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                aarch64::gf8::mul_add_scatter_neon(token, rows, row_len, coeffs, src)
            },
        );
        check_scatter(
            "gf16 neon scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                aarch64::gf16::mul_add_scatter_neon(token, rows, row_len, coeffs, src)
            },
        );
        check_matrix(
            "gf8 neon matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                aarch64::gf8::mul_add_matrix_neon(token, rows, row_len, nrows, terms)
            },
        );
        check_matrix(
            "gf16 neon matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms| {
                aarch64::gf16::mul_add_matrix_neon(token, rows, row_len, nrows, terms)
            },
        );
        check_gather(
            "gf8 neon gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| aarch64::gf8::mul_add_gather_neon(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf16 neon gather",
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| aarch64::gf16::mul_add_gather_neon(token, dst, coeffs, srcs),
        );
        check_gf8_elementwise("gf8 neon elementwise", |dst, a, b| {
            aarch64::gf8::mul_elementwise_neon(token, dst, a, b)
        });
        check_gf16_elementwise("gf16 neon elementwise", |dst, a, b| {
            aarch64::gf16::mul_elementwise_neon(token, dst, a, b)
        });
    }

    #[test]
    fn pmull_kernels_match_reference() {
        if !host_supports(&[Backend::NeonAes]) {
            eprintln!("skipping: no AArch64 PMULL extension on this host");
            return;
        }
        let token =
            archmage::NeonAesToken::summon().expect("host_supports guard: NEON-AES summons here");
        check_gf8_elementwise("gf8 pmull elementwise", |dst, a, b| {
            aarch64::gf8::mul_elementwise_neon_aes(token, dst, a, b)
        });
        // The tower elementwise and every fixed-coefficient PMULL kernel were
        // measured against the nibble/bit-serial paths and lost; GF(2^8)
        // elementwise is the shape that won and the only one dispatch selects.
    }
    #[test]
    fn vector_xor_matches_scalar_xor() {
        let token = archmage::NeonToken::summon().expect("NEON is baseline on AArch64");
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut neon = want.clone();
            scalar::xor(&mut want, &src);
            aarch64::xor_neon(token, &mut neon, &src);
            assert_eq!(neon, want, "neon xor: len {len}");
        }
    }

    /// The broadcast XOR kernel directly — every value width the dispatch
    /// can produce (1-byte GF(2^8), 2-byte GF(2^16), 4-byte GF(2^32),
    /// 8-byte GF(2^64), and a 16-byte width), plus the zero value and a
    /// width that divides neither lane, against the portable byte-cyclic
    /// reference.
    #[test]
    fn xor_broadcast_matches_scalar_reference() {
        const VALUES: [&[u8]; 8] = [
            &[0x00],
            &[0x53],
            &[0xa7, 0x53],
            &[0x01, 0x00],
            &[0x1a, 0x2b, 0x3c, 0x4d],
            &[0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78],
            &[
                0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
            // A width that divides neither lane, exercising the portable
            // fallback in the kernel.
            &[0x11, 0x22, 0x33],
        ];
        let token = archmage::NeonToken::summon().expect("NEON is baseline on AArch64");
        for &len in XOR_LENGTHS {
            for value in VALUES {
                let dst = &mut noise(len, 0x2e);
                let mut want = dst.clone();
                let mut neon = want.clone();
                for (w, &b) in want.iter_mut().zip(value.iter().cycle()) {
                    *w ^= b;
                }
                aarch64::bytes::xor_broadcast_neon(token, &mut neon, value);
                assert_eq!(neon, want, "neon xor_broadcast: len {len}, value {value:?}");
            }
        }
    }

    /// The documented value-width contract of the direct broadcast entry:
    /// an empty value and a 17-byte value both panic, and the panic message
    /// names the operation.
    #[test]
    fn xor_broadcast_neon_rejects_invalid_value_widths() {
        let token = archmage::NeonToken::summon().expect("NEON is baseline on AArch64");
        let mut dst = [0u8; 32];
        for bad in [&[][..], &[0u8; 17][..]] {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                aarch64::bytes::xor_broadcast_neon(token, &mut dst, bad);
            }))
            .expect_err("xor_broadcast_neon must reject an invalid value width");
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.starts_with("xor_broadcast_neon: value is"),
                "panic must name the operation: {msg}"
            );
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
    use archmage::SimdToken as _;

    #[test]
    fn simd128_kernels_match_reference() {
        let token =
            archmage::Wasm128Token::summon().expect("host_supports guard: simd128 summons here");
        check_gf8_mul_add("gf8 simd128", |dst, table, src| {
            wasm32::gf8::mul_add_simd128(token, dst, table, src)
        });
        check_gf8_mul_assign("gf8 simd128", |dst, table| {
            wasm32::gf8::mul_assign_simd128(token, dst, table)
        });
        check_gf16_mul_add_tables("gf16 simd128", |dst, tables, src| {
            wasm32::gf16::mul_add_simd128(token, dst, tables, src)
        });
        check_gf16_mul_assign_tables("gf16 simd128", |dst, tables| {
            wasm32::gf16::mul_assign_simd128(token, dst, tables)
        });
        check_gf8_mul_into("gf8 simd128 mul_into", |dst, table, src| {
            wasm32::gf8::mul_into_simd128(token, dst, table, src)
        });
        check_gf16_mul_into_tables("gf16 simd128 mul_into", |dst, tables, src| {
            wasm32::gf16::mul_into_simd128(token, dst, tables, src)
        });
        check_scatter(
            "gf8 simd128 scatter",
            gf8_coeff_at,
            gf8_reference,
            |rows, row_len, coeffs, src| {
                wasm32::gf8::mul_add_scatter_simd128(token, rows, row_len, coeffs, src)
            },
        );
        check_scatter(
            "gf16 simd128 scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                wasm32::gf16::mul_add_scatter_simd128(token, rows, row_len, coeffs, src)
            },
        );
        check_gather(
            "gf8 simd128 gather",
            gf8_coeff_at,
            gf8_reference,
            |dst, coeffs, srcs| wasm32::gf8::mul_add_gather_simd128(token, dst, coeffs, srcs),
        );
        check_gather(
            "gf16 simd128 gather",
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| wasm32::gf16::mul_add_gather_simd128(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8 simd128 matrix",
            gf8_coeff_at2,
            gf8_reference,
            |rows, row_len, nrows, terms| {
                wasm32::gf8::mul_add_matrix_simd128(token, rows, row_len, nrows, terms)
            },
        );
        check_matrix(
            "gf16 simd128 matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms| {
                wasm32::gf16::mul_add_matrix_simd128(token, rows, row_len, nrows, terms)
            },
        );
        check_gf8_elementwise("gf8 simd128 elementwise", |dst, a, b| {
            wasm32::gf8::mul_elementwise_simd128(token, dst, a, b)
        });
        check_gf16_elementwise("gf16 simd128 elementwise", |dst, a, b| {
            wasm32::gf16::mul_elementwise_simd128(token, dst, a, b)
        });
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        let token =
            archmage::Wasm128Token::summon().expect("host_supports guard: simd128 summons here");
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut simd = want.clone();
            scalar::xor(&mut want, &src);
            wasm32::xor_simd128(token, &mut simd, &src);
            assert_eq!(simd, want, "simd128 xor: len {len}");
        }
    }

    /// The broadcast XOR kernel directly — every value width the dispatch
    /// can produce (1-byte GF(2^8), 2-byte GF(2^16), 4-byte GF(2^32),
    /// 8-byte GF(2^64), and a 16-byte width), plus the zero value and a
    /// width that divides neither lane, against the portable byte-cyclic
    /// reference.
    #[test]
    fn xor_broadcast_matches_scalar_reference() {
        const VALUES: [&[u8]; 8] = [
            &[0x00],
            &[0x53],
            &[0xa7, 0x53],
            &[0x01, 0x00],
            &[0x1a, 0x2b, 0x3c, 0x4d],
            &[0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78],
            &[
                0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
            // A width that divides neither lane, exercising the portable
            // fallback in the kernel.
            &[0x11, 0x22, 0x33],
        ];
        let token =
            archmage::Wasm128Token::summon().expect("host_supports guard: simd128 summons here");
        for &len in XOR_LENGTHS {
            for value in VALUES {
                let dst = &mut noise(len, 0x2e);
                let mut want = dst.clone();
                let mut simd = want.clone();
                for (w, &b) in want.iter_mut().zip(value.iter().cycle()) {
                    *w ^= b;
                }
                wasm32::bytes::xor_broadcast_simd128(token, &mut simd, value);
                assert_eq!(
                    simd, want,
                    "simd128 xor_broadcast: len {len}, value {value:?}"
                );
            }
        }
    }

    /// The documented value-width contract of the direct broadcast entry:
    /// an empty value and a 17-byte value both panic, and the panic message
    /// names the operation.
    #[test]
    fn xor_broadcast_simd128_rejects_invalid_value_widths() {
        let token =
            archmage::Wasm128Token::summon().expect("host_supports guard: simd128 summons here");
        let mut dst = [0u8; 32];
        for bad in [&[][..], &[0u8; 17][..]] {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                wasm32::bytes::xor_broadcast_simd128(token, &mut dst, bad);
            }))
            .expect_err("xor_broadcast_simd128 must reject an invalid value width");
            let msg = err.downcast_ref::<String>().expect("string panic payload");
            assert!(
                msg.starts_with("xor_broadcast_simd128: value is"),
                "panic must name the operation: {msg}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage of dispatch-level contracts the differential drivers skip
// ---------------------------------------------------------------------------

#[test]
fn default_kernels_handle_empty_terms() {
    // The defaulted `KernelDispatch::mul_into_gather` (used by the prime and
    // Fan-Paar fields) zeroes the destination when there are no terms.
    let mut dst = noise(16, 0x5a);
    <Goldilocks as KernelDispatch>::mul_into_gather(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
    let mut dst = noise(16, 0x5b);
    <FanPaar32 as KernelDispatch>::mul_into_gather(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
}

/// The dispatch-level binary broadcast for every characteristic-two field —
/// including the wider fields that have no hand-written broadcast overrides
/// and reach the field-independent primitive through the trait default —
/// against the portable byte-cyclic reference, at element counts that
/// straddle the 16- and 32-byte lane boundaries.
fn check_binary_broadcast_dispatch<F: KernelDispatch>(values: &[F::Elem]) {
    const ELEMS: [usize; 4] = [0, 17, 33, 49];
    for &n in &ELEMS {
        let len = n * F::BYTES;
        let base = noise(len, 0x2f00 + u64::from(F::BITS));
        for &v in values {
            let mut encoded = [0u8; 8];
            F::encode(&mut encoded[..F::BYTES], v);
            let mut want = base.clone();
            for (w, &b) in want.iter_mut().zip(encoded[..F::BYTES].iter().cycle()) {
                *w ^= b;
            }
            let prepared = F::prepare(RawDispatch, v);
            let mut got = base.clone();
            <F as KernelDispatch>::add_assign_scalar(RawDispatch, &mut got, &prepared);
            assert_eq!(got, want, "{} add_assign_scalar broadcast", F::NAME);
            let mut got = base.clone();
            <F as KernelDispatch>::sub_assign_scalar(RawDispatch, &mut got, &prepared);
            assert_eq!(got, want, "{} sub_assign_scalar broadcast", F::NAME);
        }
    }
}

/// The scalar fallback and the vector broadcast arms produce byte-identical
/// output, so no differential can catch a dropped dispatch arm; the run
/// reports the resolved backend instead, which is the observability
/// `just test-tiers` relies on (inspect the reported backend, never a green
/// exit alone).
#[test]
fn binary_broadcast_dispatch_matches_reference_and_reports_backend() {
    std::eprintln!(
        "binary broadcast dispatch backend: {:?}",
        crate::kernel::backend()
    );
    check_binary_broadcast_dispatch::<gf8b::Gf8B>(&[
        gf8b::Elem(0),
        gf8b::Elem(1),
        gf8b::Elem(0x53),
    ]);
    check_binary_broadcast_dispatch::<crate::field::gf8d::Gf8D>(&[
        crate::field::gf8d::Elem(0),
        crate::field::gf8d::Elem(1),
        crate::field::gf8d::Elem(0x53),
    ]);
    check_binary_broadcast_dispatch::<gf16::Gf16>(&[
        gf16::Elem(0),
        gf16::Elem(1),
        gf16::Elem(0x53a7),
    ]);
    check_binary_broadcast_dispatch::<gf32::Gf32>(&[
        gf32::Elem(0),
        gf32::Elem(1),
        gf32::Elem(0xdead_beef),
    ]);
    check_binary_broadcast_dispatch::<gf64::Gf64>(&[
        gf64::Elem(0),
        gf64::Elem(1),
        gf64::Elem(0x0123_4567_89ab_cdef),
    ]);
    check_binary_broadcast_dispatch::<FanPaar8>(&[
        fan_paar::fp8::Elem(0),
        fan_paar::fp8::Elem(1),
        fan_paar::fp8::Elem(0xa5),
    ]);
    check_binary_broadcast_dispatch::<fan_paar::FanPaar16>(&[
        fan_paar::fp16::Elem(0),
        fan_paar::fp16::Elem(1),
        fan_paar::fp16::Elem(0xa55a),
    ]);
    check_binary_broadcast_dispatch::<FanPaar32>(&[
        fan_paar::fp32::Elem(0),
        fan_paar::fp32::Elem(1),
        fan_paar::fp32::Elem(0xa55a_1234),
    ]);
    check_binary_broadcast_dispatch::<fan_paar::FanPaar64>(&[
        fan_paar::fp64::Elem(0),
        fan_paar::fp64::Elem(1),
        fan_paar::fp64::Elem(0xa55a_1234_dead_beef),
    ]);
}

/// `TableCoefficient::with_tables` builds tables for the `Plain` prepared
/// form, which dispatch never produces on a shuffle host but direct callers
/// may hold.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn gf16_scatter_accepts_plain_prepared_coefficients() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3Token};

    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipping: no AVX2 on this host");
        return;
    };
    for &len in LENGTHS {
        let src = noise(len, 0x61);
        let mut got = noise(2 * len, 0x62);
        let mut want = got.clone();
        let coeffs = [
            crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(1)),
            crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(2)),
        ];
        x86::gf16::mul_add_scatter_avx2(v3, &mut got, len, &coeffs, &src);
        scalar::mul_add::<gf16::Gf16>(&mut want[..len], gf16_coeff_at(1), &src);
        scalar::mul_add::<gf16::Gf16>(&mut want[len..], gf16_coeff_at(2), &src);
        assert_eq!(got, want, "plain-prepared scatter at len {len}");
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn x86_geometry_guards_accept_zero_length_rows() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V2Token, X64V3GfniCryptoToken, X64V3Token};

    // `row_len == 0` must fall through the vector prologues untouched.
    if !std::is_x86_feature_detected!("ssse3") {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    }

    let v2 = X64V2Token::summon().expect("SSSE3 detected above");
    let coeffs = [gf8b::Elem(3)];
    x86::gf8::mul_add_scatter_ssse3(v2, &mut [], 0, &coeffs, &[]);
    let gf16_coeffs = [crate::kernel::gf16::Prepared::Plain(gf16::Elem(5))];
    x86::gf16::mul_add_scatter_ssse3(v2, &mut [], 0, &gf16_coeffs, &[]);

    if let Some(v3) = X64V3Token::summon() {
        x86::gf8::mul_add_scatter_avx2(v3, &mut [], 0, &coeffs, &[]);
        x86::gf16::mul_add_scatter_avx2(v3, &mut [], 0, &gf16_coeffs, &[]);
        x86::gf8::mul_add_matrix_avx2(v3, &mut [], 0, 1, &[]);
    }
    x86::gf8::mul_add_matrix_ssse3(v2, &mut [], 0, 1, &[]);
    x86::gf16::mul_add_matrix_ssse3(v2, &mut [], 0, 1, &[]);

    // Empty term lists must return before any store, in the checked
    // scattered wrappers as well. The GFNI wrapper runs only where its
    // token summons; the guarded empty-terms differential below covers
    // GFNI hosts.
    if let Some(token) = X64V3GfniCryptoToken::summon() {
        let mut dst = [0u8; 8];
        x86::gf8::mul_add_matrix_at_gfni(token, &mut dst, 8, &[0], &[]);
        assert_eq!(dst, [0; 8]);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn scatter_gfni_rejects_short_rows_buffer() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3GfniCryptoToken};

    // `#[should_panic]` cannot express "skip on hosts that cannot select the
    // tier", so the panic is caught explicitly after the capability guard.
    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    }
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::mul_add_scatter_gfni(
            X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here"),
            &mut [0u8; 4],
            4,
            &[gf8b::Elem(1), gf8b::Elem(2)],
            &[0u8; 4],
        );
    })
    .is_err();
    assert!(
        panicked,
        "mul_add_scatter_gfni must reject a short rows buffer"
    );
}

#[test]
fn macro_kernels_matrix_with_matches_per_row_application() {
    // `impl_field_kernels!`'s prepared-terms matrix override (FanPaar8):
    // apply the same prepared coefficients row by row and compare.
    let row_len = 8;
    let srcs: Vec<Vec<u8>> = vec![noise(row_len, 0x71), noise(row_len, 0x72)];
    let coeffs = [fan_paar::fp8::Elem(1), fan_paar::fp8::Elem(0x8d)];
    let prepared: Vec<_> = coeffs
        .iter()
        .copied()
        .map(|coeff| FanPaar8::prepare(RawDispatch, coeff))
        .collect();

    let mut got = noise(2 * row_len, 0x73);
    let mut want = got.clone();
    <FanPaar8 as KernelDispatch>::mul_add_matrix_with(
        RawDispatch,
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
            <FanPaar8 as KernelDispatch>::mul_add(RawDispatch, row, coeff, src);
        }
    }
    assert_eq!(got, want, "prepared matrix");
}

#[test]
fn default_prepared_kernels_handle_empty_inputs() {
    // `KernelDispatch::mul_into_gather_with`'s defaulted empty arm (the scalar and
    // Fan-Paar fields): no terms zeroes the destination.
    let mut dst = noise(16, 0x66);
    <FanPaar8 as KernelDispatch>::mul_into_gather_with(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
}

#[test]
fn quad_mersenne31_kernel_handles_zero_and_one_coefficients() {
    // The kernel-level zero/one shortcuts sit below the `ops` wrappers,
    // which short-circuit first; drive them directly.
    use crate::field::QuadMersenne31;

    let mut zeroed = noise(16, 0x67);
    <QuadMersenne31 as KernelDispatch>::mul_assign(
        RawDispatch,
        &mut zeroed,
        &QuadMersenne31::prepare(RawDispatch, quad_zero()),
    );
    assert!(
        zeroed.iter().all(|&b| b == 0),
        "zero coefficient fills zero"
    );

    let original = noise(16, 0x68);
    let mut scaled = original.clone();
    <QuadMersenne31 as KernelDispatch>::mul_assign(
        RawDispatch,
        &mut scaled,
        &QuadMersenne31::prepare(RawDispatch, quad_one()),
    );
    assert_eq!(scaled, original, "one coefficient is the identity");
}

fn quad_zero() -> quad_mersenne31::Elem {
    quad_mersenne31::Elem(0, 0)
}

fn quad_one() -> quad_mersenne31::Elem {
    quad_mersenne31::Elem(1, 0)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn gfni_matrix_wrappers_accept_empty_terms() {
    use crate::kernel::FlatMatrix;
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3GfniCryptoToken};

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    }

    let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
    // Zero-row and empty-source matrices return before any store, through
    // both the element and prepared coefficient entry points.
    let empty: FlatMatrix<'_, gf8b::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    let mut rows = [0u8; 8];
    x86::gf8::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &empty);
    let empty_8d: FlatMatrix<'_, gf8d::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf8::mul_add_matrix_gfni_8d_with(token, &mut rows, 8, 0, &empty_8d);
    let gf16_empty: FlatMatrix<'_, gf16::Elem> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &gf16_empty);
    assert_eq!(rows, [0; 8]);
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn scatter_gfni_8d_rejects_short_rows_buffer() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3GfniCryptoToken};

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    }
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::mul_add_scatter_gfni_8d(
            X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here"),
            &mut [0u8; 4],
            4,
            &[gf8d::Elem(1), gf8d::Elem(2)],
            &[0u8; 4],
        );
    })
    .is_err();
    assert!(
        panicked,
        "mul_add_scatter_gfni_8d must reject a short rows buffer"
    );
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn gf16_gfni_wrappers_tolerate_degenerate_geometry() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3GfniCryptoToken};

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    }

    let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
    // The gf16 wrappers clamp rather than panic: zero-length rows, zero
    // rows, and room for fewer rows than coefficients are all no-ops.
    x86::gf16::mul_add_scatter_gfni(token, &mut [], 0, &[gf16::Elem(1)], &[]);
    let mut rows = [0u8; 8];
    x86::gf16::mul_add_scatter_gfni(token, &mut rows, 0, &[gf16::Elem(1)], &[]);
    assert_eq!(rows, [0; 8]);

    let empty: crate::kernel::FlatMatrix<'_, gf16::Elem> = crate::kernel::FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 2, &empty);
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 0, 2, &empty);
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &empty);
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
                    k::xor_window(&mut dst, &src, &k::window(from, to));
                    for i in from..to {
                        if bit(&src, i) {
                            let flipped = !bit(&oracle, i);
                            put(&mut oracle, i, flipped);
                        }
                    }
                    assert_eq!(dst, oracle, "xor_window {from}..{to} of {bits_len}");

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
        k::xor_window(&mut dst, &src, &k::window(3, 3));
        k::clear_range(&mut dst, 4, 4);
        k::set_range(&mut dst, 5, 5);
        assert_eq!(dst, [0b1010_1010]);
        k::xor_gather(&mut dst, &[], 0);
        assert_eq!(dst, [0b1010_1010]);
    }
}
