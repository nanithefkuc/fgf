//! Per-backend differential tests.
//!
//! [`crate::ops`] can only exercise the one backend the host selected. These
//! tests reach past dispatch and call every architecture kernel the CPU can
//! actually run, comparing each against an independent scalar reference. The
//! canonical Fan–Paar kernels use the tower recurrence; the other
//! families use [`crate::kernel::scalar`].
//!
//! Buffer lengths deliberately straddle every lane and unroll boundary. Most
//! SIMD bugs live in the tail, not the body.
// Seeds and geometry are deliberately reduced to field widths below.
#![allow(clippy::cast_possible_truncation)]

extern crate std;

use std::vec;
use std::vec::Vec;

use crate::field::binary::{AES, REED_SOLOMON};
use crate::field::{Elem, Gf, Gf8, Poly};
use crate::field::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf16, Gf32, Gf64, Goldilocks, quad_mersenne31,
};
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
use crate::kernel::gf8::Prepared;
use crate::kernel::scalar;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::tables::{FpTowerTables, TowerCoeff};
use crate::kernel::tables::{ScaleTable, TowerTables, scale_table};
use crate::kernel::{KernelDispatch, RawDispatch};

/// Row lengths whose GFNI scatter/gather/matrix tail is 17-30 bytes past
/// the last 32-byte window: the regression geometry for the staged
/// `mul_add_tail_gfni` chunking.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
const GFNI_TAIL_LENGTHS: &[usize] = &[50, 62, 94, 126, 158, 190, 222, 254, 350, 510, 1086];

/// Lengths covering: empty, sub-lane, exact lanes, lane+1, several unroll
/// tiles, and a large odd size. All even so GF(2^16) can use the same list.
const LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
];
/// Byte-XOR lengths around every 16- and 32-byte lane boundary, including
/// compound AVX2 tails and body-plus-tail cases.
#[allow(dead_code)]
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
fn gf8_aes_coeffs() -> Vec<Elem<Gf<8, Poly<AES>>>> {
    gf8_coeffs_of::<AES>()
}

/// The Reed–Solomon field coefficients, same byte spread under `0x11D`.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gf8_rs_coeffs() -> Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>> {
    gf8_coeffs_of::<REED_SOLOMON>()
}

fn gf8_coeffs_of<const POLY: u128>() -> Vec<Elem<Gf<8, Poly<POLY>>>> {
    let mut coeffs = vec![
        Elem::<Gf<8, Poly<POLY>>>::ZERO,
        Elem::<Gf<8, Poly<POLY>>>::ONE,
        Elem::<Gf<8, Poly<POLY>>>::from_raw(2),
        Elem::<Gf<8, Poly<POLY>>>::from_raw(0xff),
    ];
    coeffs.extend(
        (0..=u8::MAX)
            .step_by(23)
            .map(Elem::<Gf<8, Poly<POLY>>>::from_raw),
    );
    coeffs
}

/// The Rijndael tower's relation constant and base reduction byte: the raw
/// facts the elementwise kernels receive as plain values.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
const DELTA_BYTE: u8 = 0x20;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn delta_table() -> &'static ScaleTable {
    scale_table(Elem::<Gf<8, Poly<AES>>>::from_raw(0x20))
}
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
const AES_REDUCTION: u8 = Poly::<AES>::REDUCTION_LOW;

/// A custom degree-16 tower over the Reed-Solomon base: the relation
/// `t^2 + 2*t + 0x80` (the absolute trace of `0x80 / 2^2` is one). It is
/// not a pinned presentation, so every bulk operation routes through the
/// typed scalar fallback and `backend_for` reports `Scalar`.
#[derive(Clone, Copy)]
pub(crate) struct RsTower;

impl crate::field::binary::tower::TowerSpec for RsTower {
    type Base = Gf<8, Poly<REED_SOLOMON>>;
    const A: u64 = 2;
    const B: u64 = 0x80;
    const NAME: &'static str = "GF(2^16)/0x11D";
}

pub(crate) type RsField = Gf<16, crate::field::binary::tower::Tower<RsTower>>;
#[allow(dead_code)]
pub(crate) type RsElem = Elem<RsField>;

/// A custom degree-16 tower over the AES base with a non-unit linear
/// coefficient: `t^2 + 2*t + 0x01`, with generator `0x0104`.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct AesTower;

impl crate::field::binary::tower::TowerSpec for AesTower {
    type Base = Gf<8, Poly<AES>>;
    const A: u64 = 2;
    const B: u64 = 0x01;
    const NAME: &'static str = "GF(2^16)/0x11B/A2";
}

impl crate::field::binary::tower::TowerGeneratorSpec for AesTower {
    const GENERATOR: u64 = 0x0104;
}

#[allow(dead_code)]
pub(crate) type AesField = Gf<16, crate::field::binary::tower::Tower<AesTower>>;
#[allow(dead_code)]
pub(crate) type AesElem = Elem<AesField>;

/// GF(2^16) coefficients: pure base-field, pure extension, and mixed. The
/// tower kernels have distinct code paths for the `same` and `cross` factors,
/// and a coefficient with a zero component silently masks one of them.
fn gf16_coeffs() -> Vec<Elem<Gf16>> {
    let mut coeffs = vec![
        Elem::<Gf16>::from_raw(0x0000),
        Elem::<Gf16>::from_raw(0x0001),
        Elem::<Gf16>::from_raw(0x0002),
        Elem::<Gf16>::from_raw(0x00ff),
        Elem::<Gf16>::from_raw(0x0100),
        Elem::<Gf16>::from_raw(0xff00),
        Elem::<Gf16>::from_raw(0xffff),
        Elem::<Gf16>::from_raw(0x0108),
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..24 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        coeffs.push(Elem::<Gf16>::from_raw((state >> 32) as u16));
    }
    coeffs
}

/// Row geometries for the multi-row kernels. Row counts straddle the
/// four-row and two-row grouping boundaries the blocked kernels use.
#[allow(dead_code)]
const ROW_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 13];
#[allow(dead_code)]
const ROW_LENS: &[usize] = &[2, 16, 32, 34, 64, 66, 128, 300];

/// Gather lengths exercise each side of the native 16/32/64/128-byte ladder,
/// compound remainders, and a body-plus-tail case.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
fn check_gf8_mul_add<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<Gf<8, Poly<POLY>>>],
    kernel: impl Fn(&mut [u8], &ScaleTable, &[u8]),
) {
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for &coeff in coeffs {
            let mut got = noise(len, 0x62);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_add::<Gf8<Poly<POLY>>>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_assign<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<Gf<8, Poly<POLY>>>],
    kernel: impl Fn(&mut [u8], &ScaleTable),
) {
    for &len in LENGTHS {
        for &coeff in coeffs {
            let mut got = noise(len, 0x73);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff));
            scalar::mul_assign::<Gf8<Poly<POLY>>>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn check_gf16_mul_add_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x84);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x95);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_add::<Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn check_gf16_mul_assign_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables)) {
    for &len in LENGTHS {
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xa6);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff));
            scalar::mul_assign::<Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn check_gf8_mul_into<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<Gf<8, Poly<POLY>>>],
    kernel: impl Fn(&mut [u8], &ScaleTable, &[u8]),
) {
    for &len in LENGTHS {
        let src = noise(len, 0x136);
        for &coeff in coeffs {
            // Pre-fill the destination with noise: a fused kernel must
            // overwrite it, never accumulate into it.
            let mut got = noise(len, 0x147);
            let mut want = src.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_assign::<Gf8<Poly<POLY>>>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn check_gf16_mul_into_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x158);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x169);
            let mut want = src.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_assign::<Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

/// Compare a scatter kernel against a per-row reference AXPY.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// All 256 coefficients of `POLY` against noise sources at every
/// lane-boundary length, then against a source holding every byte value — so
/// between the two sweeps the blocked kernels below see all 65 536 products
/// outright.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn for_each_gf8_coeff<const POLY: u128>(case: impl FnMut(Elem<Gf<8, Poly<POLY>>>, &[u8])) {
    let mut case = case;
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for c in 0..=u8::MAX {
            case(Elem::<Gf<8, Poly<POLY>>>::from_raw(c), &src);
        }
    }
    let every_byte: Vec<u8> = (0..=u8::MAX).collect();
    for c in 0..=u8::MAX {
        case(Elem::<Gf<8, Poly<POLY>>>::from_raw(c), &every_byte);
    }
}

/// Exhaustive `mul_add` sweep through `prepare`'s coefficient form — the raw
/// element for the blocked single-buffer entries, [`Prepared`](crate::kernel::gf8::Prepared)
/// for the form dispatch hands them.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_exhaustive_mul_add<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<Gf<8, Poly<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C, &[u8]),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x62);
        let mut want = got.clone();
        kernel(&mut got, prepare(coeff), src);
        scalar::mul_add::<Gf8<Poly<POLY>>>(&mut want, coeff, src);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Exhaustive `mul_assign` sweep through `prepare`'s coefficient form.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_exhaustive_mul_assign<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<Gf<8, Poly<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x73);
        let mut want = got.clone();
        kernel(&mut got, prepare(coeff));
        scalar::mul_assign::<Gf8<Poly<POLY>>>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Exhaustive `mul_into` sweep through `prepare`'s coefficient form: the
/// destination carries noise the fused kernel must overwrite.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn check_gf8_exhaustive_mul_into<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<Gf<8, Poly<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C, &[u8]),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x147);
        let mut want = src.to_vec();
        kernel(&mut got, prepare(coeff), src);
        scalar::mul_assign::<Gf8<Poly<POLY>>>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}
#[allow(dead_code)]
fn check_gf8_elementwise<const POLY: u128>(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0xf2);
        let b = noise(len, 0x103);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<Gf8<Poly<POLY>>>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Differential check for an in-place elementwise kernel: the destination
/// starts as one operand, the kernel multiplies it by the other in place.
#[allow(dead_code)]
fn check_elementwise_assign<F: crate::field::FieldBuffer>(
    name: &str,
    kernel: impl Fn(&mut [u8], &[u8]),
) {
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

#[allow(dead_code)]
fn check_gf16_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0x114);
        let b = noise(len, 0x125);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<Gf16>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Compare a tower-field `mul_add` kernel against its independent scalar
/// reference at every length and coefficient.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

#[allow(dead_code)]
fn gf8_coeff_at<const POLY: u128>(j: usize) -> Elem<Gf<8, Poly<POLY>>> {
    // Includes 0 and 1 as j sweeps, which is what we want: the blocked
    // kernels must handle degenerate coefficients per row, not per call.
    Elem::<Gf<8, Poly<POLY>>>::from_raw((j as u8).wrapping_mul(29))
}

#[allow(dead_code)]
fn gf8_coeff_at2<const POLY: u128>(t: usize, j: usize) -> Elem<Gf<8, Poly<POLY>>> {
    Elem::<Gf<8, Poly<POLY>>>::from_raw(((t * 31 + j * 29) % 256) as u8)
}

#[allow(dead_code)]
fn gf16_coeff_at(j: usize) -> Elem<Gf16> {
    Elem::<Gf16>::from_raw((j as u16).wrapping_mul(7411))
}

#[allow(dead_code)]
fn gf16_coeff_at2(t: usize, j: usize) -> Elem<Gf16> {
    Elem::<Gf16>::from_raw(((t * 7919 + j * 613) % 65536) as u16)
}

#[allow(dead_code)]
fn gf8_reference<const POLY: u128>(dst: &mut [u8], coeff: Elem<Gf<8, Poly<POLY>>>, src: &[u8]) {
    scalar::mul_add::<Gf8<Poly<POLY>>>(dst, coeff, src);
}

#[allow(dead_code)]
fn gf16_reference(dst: &mut [u8], coeff: Elem<Gf16>, src: &[u8]) {
    scalar::mul_add::<Gf16>(dst, coeff, src);
}

/// Fan–Paar GF(2^16) coefficients: short-circuits, extremes, the tower
/// generator and its components, and a deterministic spread.
#[allow(dead_code)]
fn fp16_coeffs() -> Vec<Elem<FanPaar16>> {
    let mut v = vec![
        Elem::<FanPaar16>::ZERO,
        Elem::<FanPaar16>::ONE,
        Elem::<FanPaar16>::from_raw(u16::MAX),
        // Pure base-field and pure extension.
        Elem::<FanPaar16>::from_components(Elem::<FanPaar8>::ONE, Elem::<FanPaar8>::ZERO),
        Elem::<FanPaar16>::from_components(Elem::<FanPaar8>::ZERO, Elem::<FanPaar8>::ONE),
        // The tower generator and its subfield tower generator.
        Elem::<FanPaar16>::GENERATOR,
        Elem::<FanPaar16>::from_raw(0x0100),
    ];
    let mut s = 0xf491u16;
    for _ in 0..16 {
        s = s.wrapping_mul(2057).wrapping_add(13849);
        v.push(Elem::<FanPaar16>::from_raw(s));
    }
    v
}

/// Fan–Paar GF(2^16) lengths: multiples of 2 straddling 16/32-byte lanes.
#[allow(dead_code)]
const FP16_LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 126, 128, 130, 254, 256, 510, 512, 1022,
];

/// `dst ^= coeff * src` through the independent Fan-Paar recurrence.
#[allow(dead_code)]
fn fp_mul_add(dst: &mut [u8], coeff: u64, src: &[u8], bits: u32) {
    let width = bits as usize / 8;
    for (d, s) in dst.chunks_exact_mut(width).zip(src.chunks_exact(width)) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(s);
        let product =
            crate::field::binary::tower::fp_multiply(u64::from_le_bytes(word), coeff, bits);
        let mut acc = [0u8; 8];
        acc[..width].copy_from_slice(d);
        let summed = u64::from_le_bytes(acc) ^ product;
        d.copy_from_slice(&summed.to_le_bytes()[..width]);
    }
}

/// `dst *= coeff` through the independent Fan-Paar recurrence.
#[allow(dead_code)]
fn fp_mul_assign(dst: &mut [u8], coeff: u64, bits: u32) {
    let width = bits as usize / 8;
    for d in dst.chunks_exact_mut(width) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(d);
        let scaled =
            crate::field::binary::tower::fp_multiply(u64::from_le_bytes(word), coeff, bits);
        d.copy_from_slice(&scaled.to_le_bytes()[..width]);
    }
}

#[allow(dead_code)]
fn fp16_reference(dst: &mut [u8], coeff: Elem<FanPaar16>, src: &[u8]) {
    fp_mul_add(dst, u64::from(coeff.to_raw()), src, 16);
}
#[allow(dead_code)]
fn fp16_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar16>) {
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 16);
}
#[allow(dead_code)]
fn fp16_into_reference(dst: &mut [u8], coeff: Elem<FanPaar16>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 16);
}

/// Fan–Paar GF(2^32) coefficients, the same shape as the polynomial towers.
#[allow(dead_code)]
fn fp32_coeffs() -> Vec<Elem<FanPaar32>> {
    let mut v = vec![
        Elem::<FanPaar32>::ZERO,
        Elem::<FanPaar32>::ONE,
        Elem::<FanPaar32>::from_raw(u32::MAX),
        Elem::<FanPaar32>::from_components(Elem::<FanPaar16>::ONE, Elem::<FanPaar16>::ZERO),
        Elem::<FanPaar32>::from_components(Elem::<FanPaar16>::ZERO, Elem::<FanPaar16>::ONE),
        Elem::<FanPaar32>::from_raw(0x0001_0000),
        Elem::<FanPaar32>::from_components(
            Elem::<FanPaar16>::ZERO,
            Elem::<FanPaar16>::from_raw(0x0100),
        ),
        Elem::<FanPaar32>::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(Elem::<FanPaar32>::from_raw(s));
    }
    v
}

/// Fan–Paar GF(2^64) coefficients.
#[allow(dead_code)]
fn fp64_coeffs() -> Vec<Elem<FanPaar64>> {
    let mut v = vec![
        Elem::<FanPaar64>::ZERO,
        Elem::<FanPaar64>::ONE,
        Elem::<FanPaar64>::from_raw(u64::MAX),
        Elem::<FanPaar64>::from_components(Elem::<FanPaar32>::ONE, Elem::<FanPaar32>::ZERO),
        Elem::<FanPaar64>::from_raw(0x0000_0001_0000_0000),
        Elem::<FanPaar64>::from_components(
            Elem::<FanPaar32>::ZERO,
            Elem::<FanPaar32>::from_raw(0x0001_0000),
        ),
        Elem::<FanPaar64>::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(Elem::<FanPaar64>::from_raw(s));
    }
    v
}

/// Fan–Paar GF(2^32) lengths: multiples of 4 straddling 32-byte lanes.
#[allow(dead_code)]
const FP32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// Fan–Paar GF(2^64) lengths: multiples of 8.
#[allow(dead_code)]
const FP64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

#[allow(dead_code)]
fn fp32_reference(dst: &mut [u8], coeff: Elem<FanPaar32>, src: &[u8]) {
    fp_mul_add(dst, u64::from(coeff.to_raw()), src, 32);
}
#[allow(dead_code)]
fn fp32_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar32>) {
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
#[allow(dead_code)]
fn fp32_into_reference(dst: &mut [u8], coeff: Elem<FanPaar32>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
#[allow(dead_code)]
fn fp64_reference(dst: &mut [u8], coeff: Elem<FanPaar64>, src: &[u8]) {
    fp_mul_add(dst, coeff.to_raw(), src, 64);
}
#[allow(dead_code)]
fn fp64_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar64>) {
    fp_mul_assign(dst, coeff.to_raw(), 64);
}
#[allow(dead_code)]
fn fp64_into_reference(dst: &mut [u8], coeff: Elem<FanPaar64>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, coeff.to_raw(), 64);
}

/// GF(2^32) coefficients: the short-circuits, the extremes, pure tower
/// components, the tower constant, the generator, and a deterministic spread.
#[allow(dead_code)]
fn gf32_coeffs() -> Vec<Elem<Gf32>> {
    let mut v = vec![
        Elem::<Gf32>::ZERO,
        Elem::<Gf32>::ONE,
        Elem::<Gf32>::from_raw(u32::MAX),
        Elem::<Gf32>::from_raw(0x0000_ffff),
        Elem::<Gf32>::from_raw(0xffff_0000),
        // Pure base field and pure extension.
        Elem::<Gf32>::from_components(Elem::<Gf16>::ONE, Elem::<Gf16>::ZERO),
        Elem::<Gf32>::from_components(Elem::<Gf16>::ZERO, Elem::<Gf16>::ONE),
        // The tower constant in each component.
        Elem::<Gf32>::from_components(Elem::<Gf16>::from_raw(0x2000), Elem::<Gf16>::ZERO),
        Elem::<Gf32>::from_components(Elem::<Gf16>::ZERO, Elem::<Gf16>::from_raw(0x2000)),
        Elem::<Gf32>::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(Elem::<Gf32>::from_raw(s));
    }
    v
}

/// GF(2^64) coefficients, the same shape one level up.
#[allow(dead_code)]
fn gf64_coeffs() -> Vec<Elem<Gf64>> {
    let mut v = vec![
        Elem::<Gf64>::ZERO,
        Elem::<Gf64>::ONE,
        Elem::<Gf64>::from_raw(u64::MAX),
        Elem::<Gf64>::from_raw(0x0000_0000_ffff_ffff),
        Elem::<Gf64>::from_raw(0xffff_ffff_0000_0000),
        Elem::<Gf64>::from_components(Elem::<Gf32>::ONE, Elem::<Gf32>::ZERO),
        Elem::<Gf64>::from_components(Elem::<Gf32>::ZERO, Elem::<Gf32>::ONE),
        Elem::<Gf64>::from_components(Elem::<Gf32>::from_raw(0x2000_0000), Elem::<Gf32>::ZERO),
        Elem::<Gf64>::from_components(Elem::<Gf32>::ZERO, Elem::<Gf32>::from_raw(0x2000_0000)),
        Elem::<Gf64>::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(Elem::<Gf64>::from_raw(s));
    }
    v
}

/// GF(2^32) lengths: multiples of 4 straddling every 32-byte lane boundary.
#[allow(dead_code)]
const GF32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// GF(2^64) lengths: multiples of 8.
#[allow(dead_code)]
const GF64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

#[allow(dead_code)]
fn gf32_reference(dst: &mut [u8], coeff: Elem<Gf32>, src: &[u8]) {
    scalar::mul_add::<Gf32>(dst, coeff, src);
}
#[allow(dead_code)]
fn gf32_assign_reference(dst: &mut [u8], coeff: Elem<Gf32>) {
    scalar::mul_assign::<Gf32>(dst, coeff);
}
#[allow(dead_code)]
fn gf32_into_reference(dst: &mut [u8], coeff: Elem<Gf32>, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf32>(dst, coeff);
}
#[allow(dead_code)]
fn gf64_reference(dst: &mut [u8], coeff: Elem<Gf64>, src: &[u8]) {
    scalar::mul_add::<Gf64>(dst, coeff, src);
}
#[allow(dead_code)]
fn gf64_assign_reference(dst: &mut [u8], coeff: Elem<Gf64>) {
    scalar::mul_assign::<Gf64>(dst, coeff);
}
#[allow(dead_code)]
fn gf64_into_reference(dst: &mut [u8], coeff: Elem<Gf64>, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf64>(dst, coeff);
}

// ---------------------------------------------------------------------------
// Backend-independent
// ---------------------------------------------------------------------------

/// A custom tower is not a pinned presentation: every tier routes it to
/// the typed scalar fallback, `backend_for` reports `Scalar`, and
/// elementwise stays on the scalar path.
#[test]
fn rs_tower_routing_reports_serving_tiers() {
    use crate::kernel::FieldKernels;

    assert_eq!(
        <RsField as FieldKernels>::backend(),
        crate::kernel::Backend::Scalar,
        "a custom tower routes to the typed scalar fallback"
    );
    assert!(
        !<RsField as FieldKernels>::has_vector_elementwise(),
        "a custom tower keeps elementwise scalar"
    );
    assert!(
        <Gf16 as FieldKernels>::has_vector_elementwise()
            || matches!(crate::kernel::backend(), crate::kernel::Backend::Scalar),
        "the Rijndael tower keeps its vector elementwise"
    );
}

/// A prepared GF(2^16) coefficient for a custom tower carries the raw word
/// on every host: the scalar fallback is the only route.
#[test]
fn rs_tower_prepared_form_matches_serving_route() {
    use crate::kernel::tower::gf16::Prepared;

    let prepared =
        <RsField as KernelDispatch>::prepare(RawDispatch, Elem::<RsField>::from_raw(0x0100));
    assert!(
        matches!(prepared, Prepared::Plain(_)),
        "a custom tower prepares the raw coefficient word"
    );
}

#[test]
fn scalar_nibble_paths_match_the_generic_reference() {
    check_gf8_mul_add("gf8 nibble", &gf8_aes_coeffs(), |dst, table, src| {
        super::gf8::mul_add_nibble(dst, table, src);
    });
    check_gf8_mul_assign(
        "gf8 nibble",
        &gf8_aes_coeffs(),
        super::gf8::mul_assign_nibble,
    );
    for &len in LENGTHS {
        let src = noise(len, 0xea);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xfb);
            let mut want = got.clone();
            super::gf16::mul_add_scalar(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_add::<Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "gf16 scalar: len {len}, coeff {coeff:?}");
        }
    }
}

/// Resolve raw GF(2^8) coefficients into the prepared form the blocked
/// entries take.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
#[allow(dead_code)]
fn prepared_coeffs<const POLY: u128>(coeffs: &[Elem<Gf<8, Poly<POLY>>>]) -> Vec<Prepared> {
    coeffs.iter().map(|&c| Prepared::new(c)).collect()
}

/// A [`Matrix`](crate::kernel::Matrix) of prepared coefficients over raw
/// element terms, resolving each coefficient on construction: the provider
/// shape the dispatch layer hands the blocked matrix entries.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
#[allow(dead_code)]
struct PreparedMatrix<'a, const POLY: u128> {
    sets: Vec<Vec<Prepared>>,
    srcs: Vec<&'a [u8]>,
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
#[allow(dead_code)]
impl<const POLY: u128> crate::kernel::Matrix<Prepared> for PreparedMatrix<'_, POLY> {
    fn len(&self) -> usize {
        self.srcs.len()
    }

    fn coefficient(&self, term: usize, row: usize) -> Prepared {
        self.sets[term][row]
    }

    fn source(&self, term: usize) -> &[u8] {
        self.srcs[term]
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
#[allow(dead_code)]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn prepared_matrix<'a, const POLY: u128>(
    terms: &[(&'a [Elem<Gf<8, Poly<POLY>>>], &'a [u8])],
) -> PreparedMatrix<'a, POLY> {
    PreparedMatrix {
        sets: terms
            .iter()
            .map(|(coeffs, _)| prepared_coeffs(coeffs))
            .collect(),
        srcs: terms.iter().map(|&(_, src)| src).collect(),
    }
}

// ---------------------------------------------------------------------------
// x86
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    use super::*;
    use crate::field::Normal;
    use crate::kernel::tables::isomorphism_to_aes;
    use crate::kernel::{Backend, x86};

    /// Transport a polynomial-coordinate word of the test normal base
    /// `Normal<0x11B, 0x20>` into its normal-basis coordinates: the `B`
    /// constant of the custom normal-base tower spec.
    #[allow(clippy::cast_possible_truncation)]
    const fn normal_transport(word: u64) -> u64 {
        let to_base = crate::field::binary::normal::to_base::<0x11B, 0x20, 8>();
        let from = crate::field::binary::normal::inverse_columns(to_base, 8);
        let mut image = 0u64;
        let mut i = 0;
        while i < 8 {
            if (word >> i) & 1 == 1 {
                image ^= from[i] as u64;
            }
            i += 1;
        }
        image
    }
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
            x86::gf16::mul_add_avx512(token, dst, tables.compact(), src);
        });
        check_gf16_mul_assign_tables("gf16 wide", |dst, tables| {
            x86::gf16::mul_assign_avx512(token, dst, tables.compact());
        });
        for &len in LENGTHS {
            let src = noise(len, 0xb51);
            for coeff in gf16_coeffs() {
                let mut expected = vec![0u8; len];
                scalar::mul_add::<Gf16>(&mut expected, coeff, &src);
                let mut got = noise(len, 0xb52);
                x86::gf16::mul_into_avx512(token, &mut got, TowerCoeff::new(coeff), &src);
                assert_eq!(
                    got, expected,
                    "gf16 wide mul_into len {len} coeff {coeff:?}"
                );
            }
        }
        check_gf16_elementwise("gf16 wide elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_avx512(token, dst, a, b, DELTA_BYTE, AES_REDUCTION);
        });
        check_elementwise_assign::<Gf16>("gf16 wide elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_avx512(token, dst, src, DELTA_BYTE, AES_REDUCTION);
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
                    scalar::mul_add::<Gf16>(&mut want, coeff, src);
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
                    scalar::mul_assign::<Gf16>(&mut want, coeff);
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

    /// Checks the 512-bit kernels against the scalar oracle.
    ///
    /// The token summon permits direct kernel testing independently of
    /// dispatch. The sweep is exhaustive over coefficients in both
    /// coefficient forms, and the all-byte-values source covers the products.
    #[cfg(feature = "simd512")]
    #[test]
    // Term geometry nests the unified element spelling; the slices stay slices.
    #[allow(clippy::type_complexity)]
    // Exhaustive coefficient forms with all-byte-values sources.
    #[allow(clippy::too_many_lines)]
    fn gf8_avx512_kernels_match_reference() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        };

        // The blocked entries take the coefficient form directly: raw here,
        // and the dispatch-shaped `Prepared` twin in the resolve test below.
        check_gf8_exhaustive_mul_add(
            "gf8 rs avx512",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_add_avx512(token, dst, c, src);
            },
        );
        check_gf8_exhaustive_mul_assign(
            "gf8 rs avx512",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c| {
                x86::gf8::mul_assign_avx512(token, dst, c);
            },
        );
        check_gf8_exhaustive_mul_into(
            "gf8 rs avx512",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_into_avx512(token, dst, c, src);
            },
        );
        // The AES field through the same 512-bit bodies: every coefficient
        // folds in through its affine map, `GF2P8MULB` never enters.
        check_gf8_exhaustive_mul_add(
            "gf8 aes avx512",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_add_avx512(token, dst, c, src);
            },
        );
        check_gf8_exhaustive_mul_assign(
            "gf8 aes avx512",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c| {
                x86::gf8::mul_assign_avx512(token, dst, c);
            },
        );
        check_gf8_exhaustive_mul_into(
            "gf8 aes avx512",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_into_avx512(token, dst, c, src);
            },
        );
        // Prefetch coverage: at and above `PREFETCH_MIN` the into body names
        // lines 512 and 640 bytes ahead, so the length must clear the guard
        // plus tiles, lanes, and a sub-lane tail.
        for len in [24 * 1024, 24 * 1024 + 359, 32 * 1024 + 71] {
            let src = noise(len, 0xb20 + len as u64);
            for &coeff in &[
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::ZERO,
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::ONE,
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0x53),
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0xff),
            ] {
                let mut got = noise(len, 0xb21);
                let mut want = src.clone();
                x86::gf8::mul_into_avx512(token, &mut got, Prepared::new(coeff), &src);
                scalar::mul_assign::<Gf8<Poly<REED_SOLOMON>>>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 into prefetch: len {len}, coeff {coeff:?}"
                );
            }
        }
        check_scatter(
            "gf8 rs avx512 scatter",
            gf8_coeff_at::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx512(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs)[..],
                    src,
                );
            },
        );
        check_gather(
            "gf8 rs avx512 gather",
            gf8_coeff_at::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_avx512(token, dst, &prepared_coeffs(coeffs)[..], srcs);
            },
        );
        check_matrix(
            "gf8 rs avx512 matrix",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_avx512_with(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                );
            },
        );
        check_matrix_overwrite(
            "gf8 rs avx512 matrix overwrite",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_avx512_with(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                );
            },
        );
        // The AES field through the generic 512-bit multi-row bodies.
        check_scatter(
            "gf8 aes avx512 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx512(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs)[..],
                    src,
                );
            },
        );
        check_gather(
            "gf8 aes avx512 gather",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_avx512(token, dst, &prepared_coeffs(coeffs)[..], srcs);
            },
        );
        check_matrix(
            "gf8 aes avx512 matrix",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_add_matrix_avx512_with(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                );
            },
        );
        check_matrix_overwrite(
            "gf8 aes avx512 matrix overwrite",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                x86::gf8::mul_into_matrix_avx512_with(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                );
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
                    let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>>> = (0..nterms)
                        .map(|t| {
                            (0..nrows)
                                .map(|j| gf8_coeff_at2::<REED_SOLOMON>(t, j))
                                .collect()
                        })
                        .collect();
                    let terms: Vec<(&[Elem<Gf<8, Poly<REED_SOLOMON>>>], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();
                    let mut got = noise(row_len * nrows, 0xb10);
                    let mut want = got.clone();
                    x86::gf8::mul_add_matrix_avx512_with(
                        token,
                        &mut got,
                        row_len,
                        nrows,
                        &prepared_matrix(&terms),
                    );
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8_reference::<REED_SOLOMON>(row, coeff, src);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "gf8 rs avx512 tiles: {row_len}B x {nrows} x {nterms}t"
                    );
                    // Empty terms: accumulate leaves rows unchanged, overwrite zeroes them.
                    let mut got = noise(row_len * nrows, 0xb11);
                    let want = got.clone();
                    let empty: &[(&[Prepared], &[u8])] = &[];
                    x86::gf8::mul_add_matrix_avx512_with(token, &mut got, row_len, nrows, empty);
                    assert_eq!(
                        got, want,
                        "gf8 rs avx512 empty accumulate: {row_len}B x {nrows}"
                    );
                    let mut got = noise(row_len * nrows, 0xb12);
                    x86::gf8::mul_into_matrix_avx512_with(token, &mut got, row_len, nrows, empty);
                    assert_eq!(
                        got,
                        vec![0u8; row_len * nrows],
                        "gf8 rs avx512 empty overwrite: {row_len}B x {nrows}"
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
                    let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>>> = (0..3usize)
                        .map(|t| {
                            (0..nrows)
                                .map(|j| gf8_coeff_at2::<REED_SOLOMON>(t, j))
                                .collect()
                        })
                        .collect();
                    let terms: Vec<(&[Elem<Gf<8, Poly<REED_SOLOMON>>>], &[u8])> = coeff_sets
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
                            gf8_reference::<REED_SOLOMON>(row, coeff, src);
                        }
                    }
                    let rows = &mut backing[start..start + span];
                    x86::gf8::mul_add_matrix_avx512_with(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &prepared_matrix(&terms),
                    );
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled accumulate: +{offset} {row_len}B x {nrows}"
                    );
                    let mut want = vec![0u8; span];
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8_reference::<REED_SOLOMON>(row, coeff, src);
                        }
                    }
                    x86::gf8::mul_into_matrix_avx512_with(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &prepared_matrix(&terms),
                    );
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled overwrite: +{offset} {row_len}B x {nrows}"
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
                    let coeffs: Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>> =
                        (0..nrows).map(gf8_coeff_at::<REED_SOLOMON>).collect();
                    let mut backing = noise(len * nrows + 128, 0xb41);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len * nrows].to_vec();
                    for (row, &coeff) in want.chunks_exact_mut(len).zip(&coeffs) {
                        gf8_reference::<REED_SOLOMON>(row, coeff, src);
                    }
                    let rows = &mut backing[start..start + len * nrows];
                    x86::gf8::mul_add_scatter_avx512(
                        token,
                        rows,
                        len,
                        &prepared_coeffs(&coeffs),
                        src,
                    );
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled scatter: +{offset} {len}B x {nrows}"
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
                    let coeffs: Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>> =
                        (0..nsrcs).map(gf8_coeff_at::<REED_SOLOMON>).collect();
                    let mut backing = noise(len + 128, 0xb51);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len].to_vec();
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        gf8_reference::<REED_SOLOMON>(&mut want, coeff, src);
                    }
                    let dst = &mut backing[start..start + len];
                    x86::gf8::mul_add_gather_avx512(token, dst, &prepared_coeffs(&coeffs), &srcs);
                    assert_eq!(
                        dst,
                        want.as_slice(),
                        "gf8 rs avx512 peeled gather: +{offset} {len}B x {nsrcs}"
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
                let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>>> = (0..3usize)
                    .map(|t| {
                        (0..nrows)
                            .map(|j| gf8_coeff_at2::<REED_SOLOMON>(t, j))
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[Elem<Gf<8, Poly<REED_SOLOMON>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(span, 0xda);
                let mut want = got.clone();
                x86::gf8::mul_add_matrix_at_avx512_with(
                    token,
                    &mut got,
                    row_len,
                    &starts,
                    &prepared_matrix(&terms),
                );
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        scalar::mul_add::<Gf8<Poly<REED_SOLOMON>>>(
                            &mut want[starts[j]..starts[j] + row_len],
                            coeff,
                            src,
                        );
                    }
                }
                assert_eq!(got, want, "gf8 rs avx512 scattered: {row_len}B x {nrows}");
                // Empty terms leave scattered rows unchanged; zero rows are a no-op.
                let mut got = noise(span, 0xdb);
                let want = got.clone();
                let empty: &[(&[Prepared], &[u8])] = &[];
                x86::gf8::mul_add_matrix_at_avx512_with(token, &mut got, row_len, &starts, empty);
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 scattered empty: {row_len}B x {nrows}"
                );
            }
        }
        {
            let mut rows = noise(64, 0xdc);
            let want = rows.clone();
            let empty: &[(&[Prepared], &[u8])] = &[];
            x86::gf8::mul_add_matrix_at_avx512_with(token, &mut rows, 16, &[], empty);
            assert_eq!(rows, want, "gf8 rs avx512 scattered zero rows");
        }
        // The AES field through the generic scattered body: same windows.
        for &row_len in ROW_LENS {
            for &nrows in ROW_COUNTS {
                let starts: Vec<usize> = (0..nrows).rev().map(|j| j * (row_len + 7)).collect();
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0x940 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<AES>>>>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<AES>(t, j)).collect())
                    .collect();
                let terms: Vec<(&[Elem<Gf<8, Poly<AES>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(span, 0xdd);
                let mut want = got.clone();
                x86::gf8::mul_add_matrix_at_avx512_with(
                    token,
                    &mut got,
                    row_len,
                    &starts,
                    &prepared_matrix(&terms),
                );
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        gf8_reference::<AES>(&mut want[starts[j]..starts[j] + row_len], coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8 aes avx512 scattered: {row_len}B x {nrows}");
            }
        }
        // 64-byte elementwise products for both byte fields: full lanes, the
        // 32/16-byte ladder, and the scalar tail.
        check_gf8_elementwise::<AES>("gf8 aes avx512 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx512::<true>(
                token,
                isomorphism_to_aes::<AES>(),
                Poly::<AES>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
        check_elementwise_assign::<Gf8<Poly<AES>>>(
            "gf8 aes avx512 elementwise assign",
            |dst, src| {
                x86::gf8::mul_elementwise_assign_avx512::<true>(
                    token,
                    isomorphism_to_aes::<AES>(),
                    Poly::<AES>::REDUCTION_LOW,
                    dst,
                    src,
                );
            },
        );
        check_gf8_elementwise::<REED_SOLOMON>("gf8 rs avx512 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx512::<false>(
                token,
                isomorphism_to_aes::<REED_SOLOMON>(),
                Poly::<REED_SOLOMON>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
        check_elementwise_assign::<Gf8<Poly<REED_SOLOMON>>>(
            "gf8 rs avx512 elementwise assign",
            |dst, src| {
                x86::gf8::mul_elementwise_assign_avx512::<false>(
                    token,
                    isomorphism_to_aes::<REED_SOLOMON>(),
                    Poly::<REED_SOLOMON>::REDUCTION_LOW,
                    dst,
                    src,
                );
            },
        );
        // The `_with` provider path over the flat test-slice terms.
        for &row_len in ROW_LENS {
            for &nrows in ROW_COUNTS {
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0xa00 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>>> = (0..3usize)
                    .map(|t| {
                        (0..nrows)
                            .map(|j| gf8_coeff_at2::<REED_SOLOMON>(t, j))
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[Elem<Gf<8, Poly<REED_SOLOMON>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let flat: Vec<Prepared> = terms
                    .iter()
                    .flat_map(|&(coeffs, _)| prepared_coeffs(coeffs))
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
                        scalar::mul_add::<Gf8<Poly<REED_SOLOMON>>>(row, coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8 rs avx512 with: {row_len}B x {nrows}");
                let mut got = noise(row_len * nrows, 0xdc);
                x86::gf8::mul_into_matrix_avx512_with(token, &mut got, row_len, nrows, &matrix);
                let mut want = vec![0u8; row_len * nrows];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        scalar::mul_add::<Gf8<Poly<REED_SOLOMON>>>(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 overwrite with: {row_len}B x {nrows}"
                );
            }
        }
    }

    /// One AVX-512 matrix entry under test: label, whether it overwrites, and
    /// the call over contiguous rows.
    #[cfg(feature = "simd512")]
    type MatrixEntry<'a, E> = (
        &'a str,
        bool,
        &'a dyn Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
    );

    /// Flat row-major coefficients and per-term sources, the layout the
    /// provider entries read.
    #[cfg(feature = "simd512")]
    fn flatten<'a, E: Copy>(terms: &[(&[E], &'a [u8])]) -> (Vec<E>, Vec<&'a [u8]>) {
        let coeffs = terms
            .iter()
            .flat_map(|&(coeffs, _)| coeffs.iter().copied())
            .collect();
        (coeffs, terms.iter().map(|&(_, src)| src).collect())
    }

    /// Resolve raw GF(2^8) terms into the prepared coefficient pairs the
    /// blocked matrix entries take; the sets stay alive across `body`.
    macro_rules! prepared_pairs {
        ($terms:expr, |$pairs:ident| $body:expr) => {{
            let sets: Vec<Vec<Prepared>> = $terms
                .iter()
                .map(|(coeffs, _)| prepared_coeffs(coeffs))
                .collect();
            let $pairs: Vec<(&[Prepared], &[u8])> = sets
                .iter()
                .zip($terms.iter().map(|&(_, src)| src))
                .map(|(set, src)| (set.as_slice(), src))
                .collect();
            $body
        }};
    }

    /// Compare AVX-512 matrix entries against a per-term reference sum at term
    /// counts straddling the coefficient resolve chunk.
    ///
    /// Row counts cover every mix of four-, two-, and one-row groups. Rows sit
    /// 16 bytes past a 64-byte boundary, so the whole-lane floor length runs
    /// the head peel; the long lengths keep to few row counts to bound
    /// debug-build runtime.
    #[cfg(feature = "simd512")]
    fn check_matrix_resolve_chunks<E: Copy>(
        field: &str,
        coeff_at: impl Fn(usize, usize) -> E,
        reference: impl Fn(&mut [u8], E, &[u8]),
        entries: &[MatrixEntry<'_, E>],
    ) {
        let chunk = x86::gf8::RESOLVE_CHUNK;
        let floor = x86::gf8::MATRIX_PEEL_MIN;
        let every: Vec<usize> = (1..=9).collect();
        let few: &[usize] = &[2, 7];
        // Sub-lane, one lane, tile plus lanes plus sub-lane tail, and the
        // peel floor on both sides of a whole lane.
        let shapes: [(usize, &[usize]); 8] = [
            (40, &every),
            (64, &every),
            (256 + 2 * 64 + 17, &every),
            (512 - 64, &every),
            (512, &every),
            (512 + 64 + 17, &every),
            (floor, few),
            (floor + 1, few),
        ];
        for nterms in [chunk - 1, chunk, chunk + 1, 2 * chunk + 1] {
            for &(row_len, row_counts) in &shapes {
                for &nrows in row_counts {
                    let sources: Vec<Vec<u8>> = (0..nterms)
                        .map(|t| noise(row_len, 0xc00 + t as u64))
                        .collect();
                    let coeff_sets: Vec<Vec<E>> = (0..nterms)
                        .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                        .collect();
                    let terms: Vec<(&[E], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();
                    let span = row_len * nrows;
                    let mut sum = vec![0u8; span];
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in sum.chunks_exact_mut(row_len).zip(coeffs) {
                            reference(row, coeff, src);
                        }
                    }
                    let mut backing = noise(span + 128, 0xc01);
                    let start = backing.as_ptr().align_offset(64) + 16;
                    let initial = backing[start..start + span].to_vec();
                    // Field addition is XOR, so accumulating the reference
                    // sum into the initial rows gives the accumulate result.
                    let accumulated: Vec<u8> =
                        initial.iter().zip(&sum).map(|(a, b)| a ^ b).collect();
                    for &(label, overwrites, kernel) in entries {
                        let rows = &mut backing[start..start + span];
                        rows.copy_from_slice(&initial);
                        kernel(rows, row_len, nrows, &terms);
                        let want = if overwrites { &sum } else { &accumulated };
                        assert_eq!(
                            rows,
                            want.as_slice(),
                            "{field} {label}: {row_len}B x {nrows} x {nterms}t"
                        );
                    }
                }
            }
        }
    }

    /// Every AVX-512 matrix entry for both byte fields, at term counts that
    /// split the row groups' coefficients across several resolve passes.
    #[allow(clippy::too_many_lines)]
    // Term geometry nests the unified element spelling; the slices stay slices.
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "simd512")]
    #[test]
    fn gf8_avx512_matrix_straddles_resolve_chunk() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        };
        let entries: [MatrixEntry<'_, Elem<Gf<8, Poly<REED_SOLOMON>>>>; 7] = [
            ("mul_add", false, &|rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            }),
            ("mul_into", true, &|rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            }),
            ("mul_add with", false, &|rows, row_len, nrows, terms| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_into with", true, &|rows, row_len, nrows, terms| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_add prepared", false, &|rows,
                                          row_len,
                                          nrows,
                                          terms: &[(
                &[Elem<Gf<8, Poly<REED_SOLOMON>>>],
                &[u8],
            )]| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_into prepared", true, &|rows,
                                          row_len,
                                          nrows,
                                          terms: &[(
                &[Elem<Gf<8, Poly<REED_SOLOMON>>>],
                &[u8],
            )]| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_add at", false, &|rows, row_len, nrows, terms| {
                let starts: Vec<usize> = (0..nrows).map(|j| j * row_len).collect();
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_at_avx512_with(
                        token,
                        rows,
                        row_len,
                        &starts,
                        &pairs[..],
                    );
                });
            }),
        ];
        check_matrix_resolve_chunks(
            "gf8 rs",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            &entries,
        );
        let entries: [MatrixEntry<'_, Elem<Gf<8, Poly<AES>>>>; 7] = [
            ("mul_add", false, &|rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            }),
            ("mul_into", true, &|rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            }),
            ("mul_add with", false, &|rows, row_len, nrows, terms| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_into with", true, &|rows, row_len, nrows, terms| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_add prepared", false, &|rows,
                                          row_len,
                                          nrows,
                                          terms: &[(
                &[Elem<Gf<8, Poly<AES>>>],
                &[u8],
            )]| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_into prepared", true, &|rows,
                                          row_len,
                                          nrows,
                                          terms: &[(
                &[Elem<Gf<8, Poly<AES>>>],
                &[u8],
            )]| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &matrix);
            }),
            ("mul_add at", false, &|rows, row_len, nrows, terms| {
                let starts: Vec<usize> = (0..nrows).map(|j| j * row_len).collect();
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_at_avx512_with(
                        token,
                        rows,
                        row_len,
                        &starts,
                        &pairs[..],
                    );
                });
            }),
        ];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }

    /// The AVX-512 matrix walks over rows long enough to split into column
    /// blocks. A destination off a 64-byte boundary gives the walk a
    /// non-empty leading block before the aligned ones; an aligned
    /// destination gives it none.
    #[cfg(feature = "simd512")]
    #[test]
    fn gf8_avx512_matrix_column_blocks_match_reference() {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        };

        for offset in [16usize, 0] {
            check_matrix_column_blocks(
                "gf8 rs avx512 matrix blocks",
                false,
                offset,
                gf8_coeff_at2::<REED_SOLOMON>,
                gf8_reference::<REED_SOLOMON>,
                |rows, row_len, nrows, terms| {
                    prepared_pairs!(terms, |pairs| {
                        x86::gf8::mul_add_matrix_avx512_with(
                            token,
                            rows,
                            row_len,
                            nrows,
                            &pairs[..],
                        );
                    });
                },
            );
        }
        check_matrix_column_blocks(
            "gf8 rs avx512 matrix overwrite blocks",
            true,
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 aes avx512 matrix blocks",
            false,
            16,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, &pairs[..]);
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 rs avx512 matrix with blocks",
            false,
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                with_flat::<REED_SOLOMON>(terms, nrows, |matrix| {
                    x86::gf8::mul_add_matrix_avx512_with(token, rows, row_len, nrows, matrix);
                });
            },
        );
        check_matrix_at_column_blocks(
            "gf8 rs avx512 matrix_at blocks",
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |dst, row_len, starts, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_at_avx512_with(
                        token,
                        dst,
                        row_len,
                        starts,
                        &pairs[..],
                    );
                });
            },
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn gfni_kernels_match_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        // The single-buffer entries take the prepared coefficient: the AES
        // field runs the native multiply, the affine form follows.
        check_gf8_exhaustive_mul_add(
            "gf8 aes gfni",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_add_gfni::<true>(token, dst, c, src);
            },
        );
        check_gf8_exhaustive_mul_assign(
            "gf8 aes gfni",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c| {
                x86::gf8::mul_assign_gfni::<true>(token, dst, c);
            },
        );
        check_gf8_exhaustive_mul_into(
            "gf8 aes gfni",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_into_gfni::<true>(token, dst, c, src);
            },
        );
        // The affine multiply on the same field: the dispatch shape every
        // non-AES representation takes, exercised on AES data.
        check_gf8_exhaustive_mul_add(
            "gf8 aes gfni affine",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_add_gfni::<false>(token, dst, c, src);
            },
        );
        check_gf8_exhaustive_mul_assign(
            "gf8 aes gfni affine",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c| {
                x86::gf8::mul_assign_gfni::<false>(token, dst, c);
            },
        );
        check_gf8_exhaustive_mul_into(
            "gf8 aes gfni affine",
            |c: Elem<Gf<8, Poly<AES>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_into_gfni::<false>(token, dst, c, src);
            },
        );
        check_gf16_mul_add_tables("gf16 gfni", |dst, tables, src| {
            x86::gf16::mul_add_gfni(token, dst, tables.compact(), src);
        });
        check_gf16_mul_assign_tables("gf16 gfni", |dst, tables| {
            x86::gf16::mul_assign_gfni(token, dst, tables.compact());
        });

        check_scatter(
            "gf8 gfni scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_gfni::<true>(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs),
                    src,
                );
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
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_overwrite(
            "gf8 gfni matrix overwrite",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_gfni_matrix_prefetch(
            "gf8 gfni matrix prefetch",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
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
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_gfni::<true>(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
        check_gather_aligned(
            "gf16 gfni gather",
            2,
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| x86::gf16::mul_add_gather_gfni(token, dst, coeffs, srcs),
        );
        check_gf8_elementwise::<AES>("gf8 gfni elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_gfni::<true>(
                token,
                isomorphism_to_aes::<AES>(),
                Poly::<AES>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
        check_gf8_elementwise::<REED_SOLOMON>("gf8 rs gfni elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_gfni::<false>(
                token,
                isomorphism_to_aes::<REED_SOLOMON>(),
                Poly::<REED_SOLOMON>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
        check_gf16_elementwise("gf16 gfni elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_gfni(token, dst, a, b, DELTA_BYTE, AES_REDUCTION);
        });
        check_elementwise_assign::<Gf8<Poly<AES>>>("gf8 gfni elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_gfni::<true>(
                token,
                isomorphism_to_aes::<AES>(),
                Poly::<AES>::REDUCTION_LOW,
                dst,
                src,
            );
        });
        check_elementwise_assign::<Gf8<Poly<REED_SOLOMON>>>(
            "gf8 rs gfni elementwise assign",
            |dst, src| {
                x86::gf8::mul_elementwise_assign_gfni::<false>(
                    token,
                    isomorphism_to_aes::<REED_SOLOMON>(),
                    Poly::<REED_SOLOMON>::REDUCTION_LOW,
                    dst,
                    src,
                );
            },
        );
        check_elementwise_assign::<Gf16>("gf16 gfni elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_gfni(token, dst, src, DELTA_BYTE, AES_REDUCTION);
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

    /// Scattered-rows differential for the blocked matrix: disjoint offsets,
    /// blocked against per-term AXPY. Even iterations use ascending starts,
    /// odd iterations a reversed (non-monotonic) order over the same windows.
    // Term geometry nests the unified element spelling; the slices stay slices.
    #[allow(clippy::type_complexity)]
    fn check_gf8_scattered_gfni_rows<const POLY: u128>(name: &str, token: X64V3GfniCryptoToken) {
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
                let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<POLY>>>>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<POLY>(t, j)).collect())
                    .collect();
                let terms: Vec<(&[Elem<Gf<8, Poly<POLY>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(span, 0xda);
                let mut want = got.clone();
                if POLY == AES {
                    x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                        token,
                        &mut got,
                        row_len,
                        &starts,
                        &prepared_matrix(&terms),
                    );
                } else {
                    x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                        token,
                        &mut got,
                        row_len,
                        &starts,
                        &prepared_matrix(&terms),
                    );
                }
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        scalar::mul_add::<Gf8<Poly<POLY>>>(
                            &mut want[starts[j]..starts[j] + row_len],
                            coeff,
                            src,
                        );
                    }
                }
                assert_eq!(
                    got, want,
                    "{name} matrix_scattered: row_len {row_len}, nrows {nrows}"
                );
            }
        }
    }

    /// Column-blocked matrix geometry as `(row_len, nrows, nterms)`.
    ///
    /// Row lengths sit one byte and one lane past two blocks, on a ragged span
    /// with a sub-lane remainder, and on an exact multiple of the block; row
    /// counts run several row groups. One single-group case at a long row
    /// pins the unblocked walk.
    fn column_block_cases() -> Vec<(usize, usize, usize)> {
        let block = x86::gf8::MATRIX_COLUMN_BLOCK;
        let mut cases = Vec::new();
        for row_len in [2 * block + 1, 2 * block + 64, 3 * block + 31, 5 * block] {
            for nrows in [3, 5, 6, 8, 9] {
                for nterms in [1, 3, 10] {
                    cases.push((row_len, nrows, nterms));
                }
            }
        }
        cases.push((3 * block + 31, 4, 3));
        // Term counts past one resolve chunk over a lane-multiple blocked
        // row, so later chunks accumulate inside every block, including the
        // leading unaligned one.
        let chunk = x86::gf8::RESOLVE_CHUNK;
        for nterms in [chunk + 1, 2 * chunk + 1] {
            for nrows in [3, 5, 7] {
                cases.push((3 * block + 64, nrows, nterms));
            }
        }
        cases
    }

    /// Deterministic matrix terms: per-term noise sources of `row_len` bytes
    /// and `coeff_at(term, row)` coefficients.
    fn column_block_terms<E: Copy>(
        row_len: usize,
        nrows: usize,
        nterms: usize,
        coeff_at: impl Fn(usize, usize) -> E,
    ) -> (Vec<Vec<u8>>, Vec<Vec<E>>) {
        let sources = (0..nterms)
            .map(|t| noise(row_len, 0xc00 + t as u64))
            .collect();
        let coeff_sets = (0..nterms)
            .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
            .collect();
        (sources, coeff_sets)
    }

    /// Contiguous matrix differential over column-blocked geometry, with the
    /// destination `offset` bytes past a 64-byte boundary. Accumulate starts
    /// from noise; overwrite starts from noise the kernel must ignore.
    fn check_matrix_column_blocks<E: Copy>(
        name: &str,
        overwrite: bool,
        offset: usize,
        coeff_at: impl Fn(usize, usize) -> E,
        reference: impl Fn(&mut [u8], E, &[u8]),
        kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
    ) {
        for (row_len, nrows, nterms) in column_block_cases() {
            let (sources, coeff_sets) = column_block_terms(row_len, nrows, nterms, &coeff_at);
            let terms: Vec<(&[E], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let span = row_len * nrows;
            let mut backing = noise(span + 128, 0xc10);
            let start = backing.as_ptr().align_offset(64) + offset;
            let mut want = if overwrite {
                vec![0u8; span]
            } else {
                backing[start..start + span].to_vec()
            };
            for &(coeffs, src) in &terms {
                for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                    reference(row, coeff, src);
                }
            }
            let rows = &mut backing[start..start + span];
            kernel(rows, row_len, nrows, &terms);
            assert!(
                *rows == *want,
                "{name}: +{offset} {row_len}B x {nrows} x {nterms}t"
            );
        }
    }

    /// Scattered-row matrix differential over column-blocked geometry: rows
    /// separated by gaps, in reversed order for odd row counts, with the
    /// destination `offset` bytes past a 64-byte boundary.
    fn check_matrix_at_column_blocks<E: Copy>(
        name: &str,
        offset: usize,
        coeff_at: impl Fn(usize, usize) -> E,
        reference: impl Fn(&mut [u8], E, &[u8]),
        kernel: impl Fn(&mut [u8], usize, &[usize], &[(&[E], &[u8])]),
    ) {
        for (row_len, nrows, nterms) in column_block_cases() {
            let (sources, coeff_sets) = column_block_terms(row_len, nrows, nterms, &coeff_at);
            let terms: Vec<(&[E], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
            if nrows % 2 == 1 {
                starts.reverse();
            }
            let span = (nrows - 1) * (row_len + 7) + row_len;
            let mut backing = noise(span + 128, 0xc20);
            let start = backing.as_ptr().align_offset(64) + offset;
            let mut want = backing[start..start + span].to_vec();
            for &(coeffs, src) in &terms {
                for (&row_start, &coeff) in starts.iter().zip(coeffs) {
                    reference(&mut want[row_start..row_start + row_len], coeff, src);
                }
            }
            let dst = &mut backing[start..start + span];
            kernel(dst, row_len, &starts, &terms);
            assert!(
                *dst == *want,
                "{name}: +{offset} {row_len}B x {nrows} x {nterms}t"
            );
        }
    }

    /// Run `f` over a [`FlatMatrix`](crate::kernel::FlatMatrix) holding the
    /// same terms as `terms`, so the `_with` forms see a provider other than
    /// the term slice.
    // Term geometry nests the unified element spelling; the slices stay slices.
    #[allow(clippy::type_complexity)]
    fn with_flat<const POLY: u128>(
        terms: &[(&[Elem<Gf<8, Poly<POLY>>>], &[u8])],
        nrows: usize,
        f: impl FnOnce(&crate::kernel::FlatMatrix<'_, Prepared>),
    ) {
        let coefficients: Vec<Prepared> = terms
            .iter()
            .flat_map(|&(coeffs, _)| prepared_coeffs(coeffs))
            .collect();
        let sources: Vec<&[u8]> = terms.iter().map(|&(_, src)| src).collect();
        f(&crate::kernel::FlatMatrix {
            coefficients: &coefficients,
            nrows,
            sources: &sources,
        });
    }

    /// The AES GFNI matrix walks over rows long enough to split into
    /// column blocks, through the slice, `_with`, and scattered-row entries.
    #[test]
    fn gfni_matrix_column_blocks_match_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");

        check_matrix_column_blocks(
            "gf8 gfni matrix blocks",
            false,
            0,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 gfni matrix overwrite blocks",
            true,
            0,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<true, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 gfni matrix with blocks",
            false,
            0,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                with_flat::<AES>(terms, nrows, |matrix| {
                    x86::gf8::mul_add_matrix_gfni_with::<true, _>(
                        token, rows, row_len, nrows, matrix,
                    );
                });
            },
        );
        check_matrix_at_column_blocks(
            "gf8 gfni matrix_at blocks",
            0,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |dst, row_len, starts, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(
                        token,
                        dst,
                        row_len,
                        starts,
                        &pairs[..],
                    );
                });
            },
        );
    }

    /// The Reed–Solomon GFNI matrix walks over rows long enough to split into
    /// column blocks, through the slice, `_with`, and scattered-row entries.
    #[test]
    fn gfni_rs_matrix_column_blocks_match_reference() {
        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");

        check_matrix_column_blocks(
            "gf8 rs gfni matrix blocks",
            false,
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 rs gfni matrix overwrite blocks",
            true,
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_column_blocks(
            "gf8 rs gfni matrix overwrite with blocks",
            true,
            0,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                with_flat::<REED_SOLOMON>(terms, nrows, |matrix| {
                    x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                        token, rows, row_len, nrows, matrix,
                    );
                });
            },
        );
        check_matrix_at_column_blocks(
            "gf8 rs gfni matrix_at blocks",
            16,
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |dst, row_len, starts, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_at_gfni_with::<false, _>(
                        token,
                        dst,
                        row_len,
                        starts,
                        &pairs[..],
                    );
                });
            },
        );
    }

    /// Prefetch-boundary differential for the GFNI matrix row-group bodies.
    ///
    /// Row lengths sit just below, at, and just above
    /// [`MATRIX_PREFETCH_MIN`](crate::kernel::x86::gf8::MATRIX_PREFETCH_MIN),
    /// so every case runs both the prefetching and the non-prefetching body
    /// across each row-group shape and term count. The accumulate and
    /// overwrite forms share each case against per-term scalar AXPY.
    fn check_gfni_matrix_prefetch<E: Copy, F>(
        name: &str,
        coeff_at: impl Fn(usize, usize) -> E,
        reference: F,
        kernel_add: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
        kernel_overwrite: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
    ) where
        F: Fn(&mut [u8], E, &[u8]),
    {
        let floor = x86::gf8::MATRIX_PREFETCH_MIN;
        for &row_len in &[floor - 32, floor, floor + 33] {
            for &nrows in &[1usize, 2, 3, 4, 6] {
                for nterms in [1usize, 4, 10] {
                    let sources: Vec<Vec<u8>> = (0..nterms)
                        .map(|t| noise(row_len, 0xc10 + t as u64))
                        .collect();
                    let coeff_sets: Vec<Vec<E>> = (0..nterms)
                        .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                        .collect();
                    let terms: Vec<(&[E], &[u8])> = coeff_sets
                        .iter()
                        .zip(&sources)
                        .map(|(c, s)| (c.as_slice(), s.as_slice()))
                        .collect();

                    let mut got = noise(row_len * nrows, 0xc11);
                    let mut want = got.clone();
                    kernel_add(&mut got, row_len, nrows, &terms);
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            reference(row, coeff, src);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "{name}: accumulate row_len {row_len}, nrows {nrows}, terms {nterms}"
                    );

                    let mut got = noise(row_len * nrows, 0xc12);
                    kernel_overwrite(&mut got, row_len, nrows, &terms);
                    let mut want = vec![0u8; row_len * nrows];
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            reference(row, coeff, src);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "{name}: overwrite row_len {row_len}, nrows {nrows}, terms {nterms}"
                    );
                }
            }
        }
    }
    /// The Reed–Solomon field's GFNI path: `VGF2P8AFFINEQB` with the
    /// const-derived affine bank. The sweep is exhaustive over coefficients,
    /// and the all-byte-values source makes it exhaustive over products —
    /// this is the silicon-level proof of the affine map's bit-order
    /// convention, and the loud failure if `GF2P8MULB` (the AES field) were
    /// ever substituted.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn gf8_rs_gfni_kernels_match_reference() {
        // The non-temporal `mul_into` body sits far above the shared lengths;
        // destination offsets 0 and 1 cover both sides of the alignment peel.
        const NT_LEN: usize = (2 << 20) + 130;

        if !(host_supports(&[Backend::V3GfniCrypto])) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");

        check_gf8_exhaustive_mul_add(
            "gf8 rs gfni",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_add_gfni::<false>(token, dst, c, src);
            },
        );
        check_gf8_exhaustive_mul_assign(
            "gf8 rs gfni",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c| {
                x86::gf8::mul_assign_gfni::<false>(token, dst, c);
            },
        );
        check_gf8_exhaustive_mul_into(
            "gf8 rs gfni",
            |c: Elem<Gf<8, Poly<REED_SOLOMON>>>| Prepared::new(c),
            |dst, c, src| {
                x86::gf8::mul_into_gfni::<false>(token, dst, c, src);
            },
        );
        let source = noise(NT_LEN + 2, 0x2b8);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            // Noise, not zeros: a streaming body that accumulated into the
            // destination instead of overwriting it must fail this check.
            let mut buffer = noise(NT_LEN + 1, 0x2b9);
            for &coeff in &[
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::ZERO,
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::ONE,
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0x53),
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0xff),
            ] {
                let mut want = src.to_vec();
                scalar::mul_assign::<Gf8<Poly<REED_SOLOMON>>>(&mut want, coeff);
                let got = &mut buffer[offset..offset + NT_LEN];
                x86::gf8::mul_into_gfni::<false>(token, got, Prepared::new(coeff), src);
                assert_eq!(&*got, want, "gf8 rs gfni NT mul_into: offset {offset}");
            }
        }

        // The register-blocked affine multi-row shapes against per-row AXPY,
        // over the same row-group and tile boundaries as the AES kernels.
        check_scatter(
            "gf8 rs gfni scatter",
            gf8_coeff_at::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_gfni::<false>(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs),
                    src,
                );
            },
        );
        check_matrix(
            "gf8 rs gfni matrix",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_matrix_overwrite(
            "gf8 rs gfni matrix overwrite",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_gfni_matrix_prefetch(
            "gf8 rs gfni matrix prefetch",
            gf8_coeff_at2::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_into_matrix_gfni_with::<false, _>(
                        token,
                        rows,
                        row_len,
                        nrows,
                        &pairs[..],
                    );
                });
            },
        );
        check_gather(
            "gf8 rs gfni gather",
            gf8_coeff_at::<REED_SOLOMON>,
            gf8_reference::<REED_SOLOMON>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_gfni::<false>(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
        check_gf8_scattered_gfni_rows::<REED_SOLOMON>("gf8 rs gfni", token);
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
                for coeff in [0u8, 1, 0x53, 0xff].map(Elem::<Gf<8, Poly<AES>>>::from_raw) {
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<Gf8<Poly<AES>>>(&mut want, coeff, src);
                    x86::gf8::mul_add_gfni::<true>(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<Gf8<Poly<AES>>>(&mut want, coeff);
                    x86::gf8::mul_assign_gfni::<true>(token, dst, Prepared::new(coeff));
                    assert_eq!(&*dst, want, "mul_assign_gfni: len {len}, offset {offset}");
                    let mut want = src.to_vec();
                    scalar::mul_assign::<Gf8<Poly<AES>>>(&mut want, coeff);
                    x86::gf8::mul_into_gfni::<true>(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "mul_into_gfni: len {len}, offset {offset}");
                }
            }
        }
    }

    /// The `VGF2P8AFFINEQB` byte kernels across the half-lane peel floor,
    /// against the scalar reference for all three single-row forms.
    #[test]
    fn gf8_rs_gfni_peels_half_lane() {
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
                for coeff in [0u8, 1, 0x53, 0xff].map(Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw) {
                    let before = dst.to_vec();
                    let mut want = before.clone();
                    scalar::mul_add::<Gf8<Poly<REED_SOLOMON>>>(&mut want, coeff, src);
                    x86::gf8::mul_add_gfni::<false>(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "rs mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<Gf8<Poly<REED_SOLOMON>>>(&mut want, coeff);
                    x86::gf8::mul_assign_gfni::<false>(token, dst, Prepared::new(coeff));
                    assert_eq!(
                        &*dst, want,
                        "rs mul_assign_gfni: len {len}, offset {offset}"
                    );
                    let mut want = src.to_vec();
                    scalar::mul_assign::<Gf8<Poly<REED_SOLOMON>>>(&mut want, coeff);
                    x86::gf8::mul_into_gfni::<false>(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "rs mul_into_gfni: len {len}, offset {offset}");
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
                    scalar::mul_add::<Gf16>(&mut want, coeff, src);
                    x86::gf16::mul_add_gfni(token, dst, prepared, src);
                    assert_eq!(&*dst, want, "gf16 mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(&before);
                    let mut want = before.clone();
                    scalar::mul_assign::<Gf16>(&mut want, coeff);
                    x86::gf16::mul_assign_gfni(token, dst, prepared);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_assign_gfni: len {len}, offset {offset}"
                    );
                    let mut want = src.to_vec();
                    scalar::mul_assign::<Gf16>(&mut want, coeff);
                    x86::gf16::mul_into_gfni(token, dst, prepared, src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_into_gfni: len {len}, offset {offset}"
                    );
                }
            }
        }
    }

    /// The staged GFNI tail past whole 32-byte windows: 17-30-byte
    /// remainders chunked at 16 bytes, across aligned and misaligned
    /// destinations. The single 16-byte staging of `mul_add_tail_gfni`
    /// overflowed on exactly this geometry; the blocked GFNI routes hand
    /// it remainders up to 30 bytes.
    #[test]
    fn gf16_gfni_blocked_tails_match_scalar() {
        if !host_supports(&[Backend::V3GfniCrypto]) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
        let coeffs: Vec<Elem<Gf16>> = [0x0000u16, 0x0001, 0x0100, 0x53a7, 0xbeef, 0xffff]
            .iter()
            .copied()
            .map(Elem::<Gf16>::from_raw)
            .collect();
        for &row_len in GFNI_TAIL_LENGTHS {
            for nrows in [1usize, 2, 3, 5] {
                for skew in [0usize, 1] {
                    let src = noise(row_len, 0x311);
                    let row_coeffs: Vec<Elem<Gf16>> =
                        (0..nrows).map(|j| coeffs[j % coeffs.len()]).collect();
                    let mut backing = noise(row_len * nrows + skew, 0x322);
                    let rows = &mut backing[skew..skew + row_len * nrows];
                    let mut want = rows.to_vec();
                    x86::gf16::mul_add_scatter_gfni(token, rows, row_len, &row_coeffs, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&row_coeffs) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 gfni scatter tail: row_len {row_len}, nrows {nrows}, skew {skew}"
                    );
                }
            }
            let sources: Vec<Vec<u8>> = (0..5usize)
                .map(|t| noise(row_len, 0x333 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let gather_coeffs: Vec<Elem<Gf16>> = (0..5).map(|t| coeffs[t % coeffs.len()]).collect();
            let mut got = noise(row_len, 0x344);
            let mut want = got.clone();
            x86::gf16::mul_add_gather_gfni(token, &mut got, &gather_coeffs, &srcs);
            for (&coeff, &src) in gather_coeffs.iter().zip(&srcs) {
                gf16_reference(&mut want, coeff, src);
            }
            assert_eq!(got, want, "gf16 gfni gather tail: row_len {row_len}");
            let nrows = 5;
            let coeff_sets: Vec<Vec<Elem<Gf16>>> = (0..3)
                .map(|t| (0..nrows).map(|j| gf16_coeff_at2(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<Gf16>], &[u8])> = coeff_sets
                .iter()
                .zip(&srcs)
                .map(|(c, s)| (c.as_slice(), *s))
                .collect();
            let mut got = noise(row_len * nrows, 0x355);
            let mut want = got.clone();
            x86::gf16::mul_add_matrix_gfni(token, &mut got, row_len, nrows, &terms);
            for &(coeffs, src) in &terms {
                for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                    gf16_reference(row, coeff, src);
                }
            }
            assert_eq!(got, want, "gf16 gfni matrix tail: row_len {row_len}");
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn avx2_kernels_match_reference() {
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        check_gf8_mul_add("gf8 aes avx2", &gf8_aes_coeffs(), |dst, table, src| {
            x86::gf8::mul_add_avx2(token, dst, table, src);
        });
        check_gf8_mul_assign("gf8 aes avx2", &gf8_aes_coeffs(), |dst, table| {
            x86::gf8::mul_assign_avx2(token, dst, table);
        });
        // The shuffle kernels are polynomial-agnostic: the coefficient's own
        // bank serves any polynomial, RS included.
        check_gf8_mul_add::<REED_SOLOMON>("gf8 rs avx2", &gf8_rs_coeffs(), |dst, table, src| {
            x86::gf8::mul_add_avx2(token, dst, table, src);
        });
        check_gf8_mul_assign::<REED_SOLOMON>("gf8 rs avx2", &gf8_rs_coeffs(), |dst, table| {
            x86::gf8::mul_assign_avx2(token, dst, table);
        });
        check_gf16_mul_add_tables("gf16 avx2", |dst, tables, src| {
            x86::gf16::mul_add_avx2(token, dst, tables, src);
        });
        check_gf16_mul_assign_tables("gf16 avx2", |dst, tables| {
            x86::gf16::mul_assign_avx2(token, dst, tables);
        });
        check_gf8_mul_into("gf8 avx2 mul_into", &gf8_aes_coeffs(), |dst, table, src| {
            x86::gf8::mul_into_avx2(token, dst, table, src);
        });
        check_gf8_mul_into::<REED_SOLOMON>(
            "gf8 rs avx2 mul_into",
            &gf8_rs_coeffs(),
            |dst, table, src| {
                x86::gf8::mul_into_avx2(token, dst, table, src);
            },
        );
        check_gf16_mul_into_tables("gf16 avx2 mul_into", |dst, tables, src| {
            x86::gf16::mul_into_avx2(token, dst, tables, src);
        });
        check_scatter(
            "gf8 avx2 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_avx2(token, rows, row_len, &prepared_coeffs(coeffs), src);
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
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_avx2(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
        check_matrix(
            "gf8 avx2 matrix",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_avx2_with(token, rows, row_len, nrows, &pairs[..]);
                });
            },
        );
        check_gf8_elementwise::<AES>("gf8 avx2 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx2(token, Poly::<AES>::REDUCTION_LOW, dst, a, b);
        });
        check_gf8_elementwise::<REED_SOLOMON>("gf8 rs avx2 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_avx2(token, Poly::<REED_SOLOMON>::REDUCTION_LOW, dst, a, b);
        });
        if let Some(gfni) = X64V3GfniCryptoToken::summon() {
            check_gf8_elementwise::<REED_SOLOMON>("gf8 rs gfni elementwise", |dst, a, b| {
                x86::gf8::mul_elementwise_gfni::<false>(
                    gfni,
                    isomorphism_to_aes::<REED_SOLOMON>(),
                    Poly::<REED_SOLOMON>::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            });
        } else {
            eprintln!("skipping: rs iso elementwise needs GFNI");
        }
        check_gf16_elementwise("gf16 avx2 elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_avx2(token, dst, a, b, delta_table(), AES_REDUCTION);
        });
        // Fan–Paar tower (GF(2^16)/32/64): the fp8 nibble tower and its
        // period-2 lane-mul extensions.
        check_elementwise_assign::<Gf8<Poly<AES>>>("gf8 avx2 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_avx2(token, Poly::<AES>::REDUCTION_LOW, dst, src);
        });
        check_elementwise_assign::<Gf8<Poly<REED_SOLOMON>>>(
            "gf8 rs avx2 elementwise assign",
            |dst, src| {
                x86::gf8::mul_elementwise_assign_avx2(
                    token,
                    Poly::<REED_SOLOMON>::REDUCTION_LOW,
                    dst,
                    src,
                );
            },
        );
        check_elementwise_assign::<Gf16>("gf16 avx2 elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_avx2(token, dst, src, delta_table(), AES_REDUCTION);
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
        check_gf8_mul_add("gf8 aes ssse3", &gf8_aes_coeffs(), |dst, table, src| {
            x86::gf8::mul_add_ssse3(token, dst, table, src);
        });
        check_gf8_mul_assign("gf8 aes ssse3", &gf8_aes_coeffs(), |dst, table| {
            x86::gf8::mul_assign_ssse3(token, dst, table);
        });
        check_gf8_mul_add::<REED_SOLOMON>("gf8 rs ssse3", &gf8_rs_coeffs(), |dst, table, src| {
            x86::gf8::mul_add_ssse3(token, dst, table, src);
        });
        check_gf8_mul_assign::<REED_SOLOMON>("gf8 rs ssse3", &gf8_rs_coeffs(), |dst, table| {
            x86::gf8::mul_assign_ssse3(token, dst, table);
        });
        check_gf16_mul_add_tables("gf16 ssse3", |dst, tables, src| {
            x86::gf16::mul_add_ssse3(token, dst, tables, src);
        });
        check_gf16_mul_assign_tables("gf16 ssse3", |dst, tables| {
            x86::gf16::mul_assign_ssse3(token, dst, tables);
        });
        check_gf8_mul_into(
            "gf8 ssse3 mul_into",
            &gf8_aes_coeffs(),
            |dst, table, src| {
                x86::gf8::mul_into_ssse3(token, dst, table, src);
            },
        );
        check_gf8_mul_into::<REED_SOLOMON>(
            "gf8 rs ssse3 mul_into",
            &gf8_rs_coeffs(),
            |dst, table, src| {
                x86::gf8::mul_into_ssse3(token, dst, table, src);
            },
        );
        check_gf16_mul_into_tables("gf16 ssse3 mul_into", |dst, tables, src| {
            x86::gf16::mul_into_ssse3(token, dst, tables, src);
        });
        check_scatter(
            "gf8 ssse3 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                x86::gf8::mul_add_scatter_ssse3(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs),
                    src,
                );
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
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                x86::gf8::mul_add_gather_ssse3(token, dst, &prepared_coeffs(coeffs), srcs);
            },
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
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                prepared_pairs!(terms, |pairs| {
                    x86::gf8::mul_add_matrix_ssse3_with(token, rows, row_len, nrows, &pairs[..]);
                });
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
        check_gf8_elementwise::<AES>("gf8 ssse3 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_ssse3(token, Poly::<AES>::REDUCTION_LOW, dst, a, b);
        });
        check_gf8_elementwise::<REED_SOLOMON>("gf8 rs ssse3 elementwise", |dst, a, b| {
            x86::gf8::mul_elementwise_ssse3(token, Poly::<REED_SOLOMON>::REDUCTION_LOW, dst, a, b);
        });
        check_gf16_elementwise("gf16 ssse3 elementwise", |dst, a, b| {
            x86::gf16::mul_elementwise_ssse3(token, dst, a, b, delta_table(), AES_REDUCTION);
        });
        check_elementwise_assign::<Gf8<Poly<AES>>>("gf8 ssse3 elementwise assign", |dst, src| {
            x86::gf8::mul_elementwise_assign_ssse3(token, Poly::<AES>::REDUCTION_LOW, dst, src);
        });
        check_elementwise_assign::<Gf8<Poly<REED_SOLOMON>>>(
            "gf8 rs ssse3 elementwise assign",
            |dst, src| {
                x86::gf8::mul_elementwise_assign_ssse3(
                    token,
                    Poly::<REED_SOLOMON>::REDUCTION_LOW,
                    dst,
                    src,
                );
            },
        );
        check_elementwise_assign::<Gf16>("gf16 ssse3 elementwise assign", |dst, src| {
            x86::gf16::mul_elementwise_assign_ssse3(token, dst, src, delta_table(), AES_REDUCTION);
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
        const GF8_COEFFS: [Elem<Gf<8, Poly<AES>>>; 3] = [
            Elem::<Gf<8, Poly<AES>>>::ZERO,
            Elem::<Gf<8, Poly<AES>>>::ONE,
            Elem::<Gf<8, Poly<AES>>>::from_raw(0x53),
        ];
        const GF16_COEFFS: [Elem<Gf16>; 3] = [
            Elem::<Gf16>::from_raw(0),
            Elem::<Gf16>::from_raw(1),
            Elem::<Gf16>::from_raw(0x53a7),
        ];

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
                scalar::mul_assign::<Gf8<Poly<AES>>>(&mut want, coeff);
                if host_supports(&[Backend::V3GfniCrypto]) {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_gfni::<true>(
                        gfni.expect("guard passed: GFNI summons here"),
                        got,
                        Prepared::new(coeff),
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
                scalar::mul_assign::<Gf16>(&mut want, coeff);
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
        let coeff = Elem::<Gf<8, Poly<AES>>>::from_raw(0x53);
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

                x86::gf8::mul_add_gfni::<true>(gfni, dst, Prepared::new(coeff), src);
                scalar::mul_add::<Gf8<Poly<AES>>>(&mut want, coeff, src);
                assert_eq!(dst, want, "mul_add {len} +{off}");

                x86::gf8::mul_assign_gfni::<true>(gfni, dst, Prepared::new(coeff));
                scalar::mul_assign::<Gf8<Poly<AES>>>(&mut want, coeff);
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
        type ScatterKernel<'a> = dyn Fn(&mut [u8], usize, &[Elem<Gf16>], &[u8]) + 'a;

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

                    let coeffs8: Vec<_> = (0..nrows).map(gf8_coeff_at::<AES>).collect();
                    let mut want = rows.to_vec();
                    x86::gf8::mul_add_scatter_gfni::<true>(
                        token,
                        rows,
                        row_len,
                        &prepared_coeffs(&coeffs8),
                        &src,
                    );
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs8) {
                        gf8_reference::<AES>(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 scatter: {row_len}/{nrows}/{skew}"
                    );

                    let rs_coeffs: Vec<_> = (0..nrows).map(gf8_coeff_at::<REED_SOLOMON>).collect();
                    let mut want = rows.to_vec();
                    x86::gf8::mul_add_scatter_gfni::<false>(
                        token,
                        rows,
                        row_len,
                        &prepared_coeffs(&rs_coeffs),
                        &src,
                    );
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&rs_coeffs) {
                        gf8_reference::<REED_SOLOMON>(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs scatter: {row_len}/{nrows}/{skew}"
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
                                   coeffs: &[Elem<Gf16>],
                                   src: &[u8]| {
                            x86::gf16::mul_add_scatter_avx2(v3, rows, row_len, coeffs, src);
                        }),
                        ("ssse3", &|rows: &mut [u8],
                                    row_len: usize,
                                    coeffs: &[Elem<Gf16>],
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

    /// Direct differentials for polynomials beyond the two named constants:
    /// Data Matrix `0x12D` and the CCSDS Reed–Solomon field `0x187`, exactly
    /// the shapes a caller spells as `Gf8<Poly<0x12D>>`. Each tier exercises the
    /// single-buffer and elementwise kernels at lane/tail boundaries — raw
    /// coefficients everywhere, plus the dispatch-shaped `Prepared` form on
    /// the blocked backends — and skips through its token convention when
    /// the host cannot prove the instructions.
    #[test]
    fn gf8_additional_polynomials_match_reference() {
        check_gf8_polynomial_kernels::<0x12D>("0x12d");
        check_gf8_polynomial_kernels::<0x187>("0x187");
    }

    #[allow(clippy::too_many_lines)]
    fn check_gf8_polynomial_kernels<const POLY: u128>(name: &str) {
        let coeffs = gf8_coeffs_of::<POLY>();

        // GFNI: the affine multiply folds any polynomial's coefficient in.
        // The dispatch shape selects the native body only for the AES
        // polynomial, which none of these sweeps name.
        if host_supports(&[Backend::V3GfniCrypto]) {
            let token = X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here");
            check_tower_mul_add(
                &format!("{name} gfni mul_add"),
                LENGTHS,
                &coeffs,
                gf8_reference::<POLY>,
                |dst, c, src| x86::gf8::mul_add_gfni::<false>(token, dst, Prepared::new(c), src),
            );
            check_tower_mul_assign(
                &format!("{name} gfni mul_assign"),
                LENGTHS,
                &coeffs,
                scalar::mul_assign::<Gf8<Poly<POLY>>>,
                |dst, c| x86::gf8::mul_assign_gfni::<false>(token, dst, Prepared::new(c)),
            );
            check_tower_mul_into(
                &format!("{name} gfni mul_into"),
                LENGTHS,
                &coeffs,
                gf8_reference::<POLY>,
                |dst, c, src| x86::gf8::mul_into_gfni::<false>(token, dst, Prepared::new(c), src),
            );
            check_gf8_elementwise::<POLY>(&format!("{name} gfni elementwise"), |dst, a, b| {
                if POLY == AES {
                    x86::gf8::mul_elementwise_gfni::<true>(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                } else {
                    x86::gf8::mul_elementwise_gfni::<false>(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        a,
                        b,
                    );
                }
            });
            check_elementwise_assign::<Gf8<Poly<POLY>>>(
                &format!("{name} gfni elementwise assign"),
                |dst, src| {
                    if POLY == AES {
                        x86::gf8::mul_elementwise_assign_gfni::<true>(
                            token,
                            isomorphism_to_aes::<POLY>(),
                            Poly::<POLY>::REDUCTION_LOW,
                            dst,
                            src,
                        );
                    } else {
                        x86::gf8::mul_elementwise_assign_gfni::<false>(
                            token,
                            isomorphism_to_aes::<POLY>(),
                            Poly::<POLY>::REDUCTION_LOW,
                            dst,
                            src,
                        );
                    }
                },
            );
        } else {
            eprintln!("skipping {name}: no AVX2+GFNI on this host");
        }

        // AVX2 and SSSE3: nibble tables from the polynomial's own bank.
        if host_supports(&[Backend::V3]) {
            let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
            check_gf8_mul_add::<POLY>(&format!("{name} avx2"), &coeffs, |dst, table, src| {
                x86::gf8::mul_add_avx2(token, dst, table, src);
            });
            check_gf8_mul_assign::<POLY>(&format!("{name} avx2"), &coeffs, |dst, table| {
                x86::gf8::mul_assign_avx2(token, dst, table);
            });
            check_gf8_elementwise::<POLY>(&format!("{name} avx2 elementwise"), |dst, a, b| {
                x86::gf8::mul_elementwise_avx2(token, Poly::<POLY>::REDUCTION_LOW, dst, a, b);
            });
            check_elementwise_assign::<Gf8<Poly<POLY>>>(
                &format!("{name} avx2 elementwise assign"),
                |dst, src| {
                    x86::gf8::mul_elementwise_assign_avx2(
                        token,
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        } else {
            eprintln!("skipping {name}: no AVX2 on this host");
        }
        if host_supports(&[Backend::V2]) {
            let token = X64V2Token::summon().expect("guard passed: SSSE3 summons here");
            check_gf8_mul_add::<POLY>(&format!("{name} ssse3"), &coeffs, |dst, table, src| {
                x86::gf8::mul_add_ssse3(token, dst, table, src);
            });
            check_gf8_mul_assign::<POLY>(&format!("{name} ssse3"), &coeffs, |dst, table| {
                x86::gf8::mul_assign_ssse3(token, dst, table);
            });
            check_gf8_elementwise::<POLY>(&format!("{name} ssse3 elementwise"), |dst, a, b| {
                x86::gf8::mul_elementwise_ssse3(token, Poly::<POLY>::REDUCTION_LOW, dst, a, b);
            });
            check_elementwise_assign::<Gf8<Poly<POLY>>>(
                &format!("{name} ssse3 elementwise assign"),
                |dst, src| {
                    x86::gf8::mul_elementwise_assign_ssse3(
                        token,
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        } else {
            eprintln!("skipping {name}: no SSSE3 on this host");
        }

        // AVX-512: the affine multiply at 64-byte lanes over the prepared
        // coefficient.
        #[cfg(feature = "simd512")]
        check_gf8_polynomial_avx512::<POLY>(name, &coeffs);
        #[cfg(not(feature = "simd512"))]
        let _ = name;
    }

    /// The 512-bit tier of [`check_gf8_polynomial_kernels`], compiled only
    /// where the dispatch tier exists.
    #[cfg(feature = "simd512")]
    fn check_gf8_polynomial_avx512<const POLY: u128>(
        name: &str,
        coeffs: &[Elem<Gf<8, Poly<POLY>>>],
    ) {
        let Some(token) = X64V4xToken::summon() else {
            eprintln!("skipping {name}: no V4x token on this host");
            return;
        };
        check_tower_mul_add(
            &format!("{name} avx512 mul_add"),
            LENGTHS,
            coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| x86::gf8::mul_add_avx512(token, dst, Prepared::new(c), src),
        );
        check_tower_mul_assign(
            &format!("{name} avx512 mul_assign"),
            LENGTHS,
            coeffs,
            scalar::mul_assign::<Gf8<Poly<POLY>>>,
            |dst, c| x86::gf8::mul_assign_avx512(token, dst, Prepared::new(c)),
        );
        check_tower_mul_into(
            &format!("{name} avx512 mul_into"),
            LENGTHS,
            coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| x86::gf8::mul_into_avx512(token, dst, Prepared::new(c), src),
        );
        check_gf8_elementwise::<POLY>(&format!("{name} avx512 elementwise"), |dst, a, b| {
            if POLY == AES {
                x86::gf8::mul_elementwise_avx512::<true>(
                    token,
                    isomorphism_to_aes::<POLY>(),
                    Poly::<POLY>::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            } else {
                x86::gf8::mul_elementwise_avx512::<false>(
                    token,
                    isomorphism_to_aes::<POLY>(),
                    Poly::<POLY>::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            }
        });
        check_elementwise_assign::<Gf8<Poly<POLY>>>(
            &format!("{name} avx512 elementwise assign"),
            |dst, src| {
                if POLY == AES {
                    x86::gf8::mul_elementwise_assign_avx512::<true>(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                } else {
                    x86::gf8::mul_elementwise_assign_avx512::<false>(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Poly::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                }
            },
        );
    }

    // Prime-field integer-SIMD kernels versus the portable prime reference.
    use crate::field::goldilocks::{self, Goldilocks};
    use crate::field::mersenne31::Mersenne31;
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
    fn drive_prime<F: crate::field::FieldBuffer, C: Copy + core::fmt::Debug>(
        lens: &[usize],
        coeffs: &[C],
        to_elem: impl Fn(C) -> Elem<F>,
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
            Elem::<Mersenne31>::from_raw,
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
            Elem::<Goldilocks>::from_raw,
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
            Elem::<Mersenne31>::from_raw,
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
            Elem::<Goldilocks>::from_raw,
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
            Elem::<Mersenne31>::from_raw,
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
            Elem::<Goldilocks>::from_raw,
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
    /// Non-canonical lanes through every prime entry: buffers whose lanes
    /// hold `p`, wider aliases, and all-ones words must match the portable
    /// prime reference lane for lane, and every stored lane must be
    /// canonical — accumulate, in-place, overwrite, elementwise, broadcast,
    /// and tails included, at lane-straddling lengths.
    #[test]
    // Nine entries times three tiers at a dozen lengths each: a table by nature.
    #[allow(clippy::too_many_lines)]
    fn prime_entries_canonicalize_noncanonical_lanes() {
        // Cycle lane words across a buffer, little-endian lanes.
        fn fill(len: usize, words: &[u64], lane_bytes: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            let mut i = 0;
            while out.len() < len {
                out.extend_from_slice(&words[i % words.len()].to_le_bytes()[..lane_bytes]);
                i += 1;
            }
            out.truncate(len);
            out
        }
        // Every lane-sized chunk holds a value below the modulus.
        fn assert_canonical(label: &str, buf: &[u8], lane_bytes: usize, modulus: u64) {
            for (i, lane) in buf.chunks_exact(lane_bytes).enumerate() {
                let mut word = 0u64;
                for (j, &b) in lane.iter().enumerate() {
                    word |= u64::from(b) << (8 * j);
                }
                assert!(
                    word < modulus,
                    "{label}: lane {i} holds {word:#x}, want < {modulus:#x}"
                );
            }
        }

        const P31: u64 = crate::field::mersenne31::MODULUS as u64;
        const PGLD: u64 = crate::field::goldilocks::MODULUS;
        const L32: &[u64] = &[
            0,
            1,
            P31 - 1,
            P31,
            P31 + 1,
            0x8000_0000,
            0xFFFF_FFFE,
            0xFFFF_FFFF,
        ];
        const C32: &[u64] = &[0, 1, 2, 0x53, P31, 0xFFFF_FFFF];
        const L64: &[u64] = &[
            0,
            1,
            PGLD - 1,
            PGLD,
            PGLD + 1,
            0x1_0000_0000,
            u64::MAX - 1,
            u64::MAX,
        ];
        const C64: &[u64] = &[0, 1, 2, 0x53, PGLD, u64::MAX];

        #[allow(clippy::too_many_arguments)]
        fn drive_m31(
            label: &str,
            add: impl Fn(&mut [u8], &[u8]),
            sub: impl Fn(&mut [u8], &[u8]),
            muladd: impl Fn(&mut [u8], u32, &[u8]),
            mulinto: impl Fn(&mut [u8], u32, &[u8]),
            mulassign: impl Fn(&mut [u8], u32),
            elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
            elemwise_assign: impl Fn(&mut [u8], &[u8]),
            addscalar: impl Fn(&mut [u8], u32),
            subscalar: impl Fn(&mut [u8], u32),
        ) {
            use crate::field::FieldElem as _;
            use crate::field::mersenne31::Mersenne31;
            use crate::kernel::prime;
            for &len in M31_LENS {
                let src = fill(len, L32, 4);
                let base = fill(len, &L32[3..], 4);
                let other = fill(len, &L32[5..], 4);

                let mut got = base.clone();
                let mut want = base.clone();
                add(&mut got, &src);
                prime::add_assign::<Mersenne31>(&mut want, &src);
                assert_eq!(got, want, "{label} add_assign len {len}");
                assert_canonical(label, &got, 4, P31);

                let mut got = base.clone();
                let mut want = base.clone();
                sub(&mut got, &src);
                prime::sub_assign::<Mersenne31>(&mut want, &src);
                assert_eq!(got, want, "{label} sub_assign len {len}");
                assert_canonical(label, &got, 4, P31);

                // A small-minuend, aliased-subtrahend pair the rotation
                // above never aligns: `0 - 0xFFFF_FFFF` must be `p - 1`.
                let sub_base = fill(len, &L32[1..], 4);
                let mut got = sub_base.clone();
                let mut want = sub_base.clone();
                sub(&mut got, &src);
                prime::sub_assign::<Mersenne31>(&mut want, &src);
                assert_eq!(got, want, "{label} sub_assign shifted len {len}");
                assert_canonical(label, &got, 4, P31);

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                elemwise(&mut got, &src, &other);
                prime::mul_elementwise::<Mersenne31>(&mut want, &src, &other);
                assert_eq!(got, want, "{label} mul_elementwise len {len}");
                assert_canonical(label, &got, 4, P31);

                let mut got = base.clone();
                let mut want = base.clone();
                elemwise_assign(&mut got, &src);
                prime::mul_elementwise::<Mersenne31>(&mut want, &base, &src);
                assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
                assert_canonical(label, &got, 4, P31);

                for &c in C32 {
                    let c32 = c as u32;
                    let e = Elem::<Mersenne31>::from_raw(c32);
                    let mut got = base.clone();
                    let mut want = base.clone();
                    muladd(&mut got, c32, &src);
                    prime::mul_add::<Mersenne31>(&mut want, e, &src);
                    assert_eq!(got, want, "{label} mul_add len {len} coeff {c:#x}");
                    // A zero coefficient is a defined no-op at every level
                    // (ops fast path, scalar reference, kernel entry), so
                    // untouched input lanes are preserved, not canonicalized.
                    if !e.is_zero() {
                        assert_canonical(label, &got, 4, P31);
                    }

                    let mut got = vec![0u8; len];
                    let mut want = vec![0u8; len];
                    mulinto(&mut got, c32, &src);
                    prime::mul_add::<Mersenne31>(&mut want, e, &src);
                    // Overwrite of a zero buffer is the product itself.
                    assert_eq!(got, want, "{label} mul_into len {len} coeff {c:#x}");
                    assert_canonical(label, &got, 4, P31);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    mulassign(&mut got, c32);
                    prime::mul_assign::<Mersenne31>(&mut want, e);
                    assert_eq!(got, want, "{label} mul_assign len {len} coeff {c:#x}");
                    // A unit coefficient is a defined no-op, preserving
                    // untouched input lanes.
                    if !e.is_one() {
                        assert_canonical(label, &got, 4, P31);
                    }

                    let mut got = base.clone();
                    let mut want = base.clone();
                    addscalar(&mut got, c32);
                    prime::add_assign_scalar::<Mersenne31>(&mut want, e);
                    assert_eq!(
                        got, want,
                        "{label} add_assign_scalar len {len} coeff {c:#x}"
                    );
                    assert_canonical(label, &got, 4, P31);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    subscalar(&mut got, c32);
                    prime::sub_assign_scalar::<Mersenne31>(&mut want, e);
                    assert_eq!(
                        got, want,
                        "{label} sub_assign_scalar len {len} coeff {c:#x}"
                    );
                    assert_canonical(label, &got, 4, P31);
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn drive_gld(
            label: &str,
            add: impl Fn(&mut [u8], &[u8]),
            sub: impl Fn(&mut [u8], &[u8]),
            muladd: impl Fn(&mut [u8], u64, &[u8]),
            mulinto: impl Fn(&mut [u8], u64, &[u8]),
            mulassign: impl Fn(&mut [u8], u64),
            elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
            elemwise_assign: impl Fn(&mut [u8], &[u8]),
            addscalar: impl Fn(&mut [u8], u64),
            subscalar: impl Fn(&mut [u8], u64),
        ) {
            use crate::field::FieldElem as _;
            use crate::field::goldilocks::Goldilocks;
            use crate::kernel::prime;
            for &len in GLD_LENS {
                let src = fill(len, L64, 8);
                let base = fill(len, &L64[3..], 8);
                let other = fill(len, &L64[5..], 8);

                let mut got = base.clone();
                let mut want = base.clone();
                add(&mut got, &src);
                prime::add_assign::<Goldilocks>(&mut want, &src);
                assert_eq!(got, want, "{label} add_assign len {len}");
                assert_canonical(label, &got, 8, PGLD);

                let mut got = base.clone();
                let mut want = base.clone();
                sub(&mut got, &src);
                prime::sub_assign::<Goldilocks>(&mut want, &src);
                assert_eq!(got, want, "{label} sub_assign len {len}");
                assert_canonical(label, &got, 8, PGLD);

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                elemwise(&mut got, &src, &other);
                prime::mul_elementwise::<Goldilocks>(&mut want, &src, &other);
                assert_eq!(got, want, "{label} mul_elementwise len {len}");
                assert_canonical(label, &got, 8, PGLD);

                let mut got = base.clone();
                let mut want = base.clone();
                elemwise_assign(&mut got, &src);
                prime::mul_elementwise::<Goldilocks>(&mut want, &base, &src);
                assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
                assert_canonical(label, &got, 8, PGLD);

                for &c in C64 {
                    let e = Elem::<Goldilocks>::from_raw(c);
                    let mut got = base.clone();
                    let mut want = base.clone();
                    muladd(&mut got, c, &src);
                    prime::mul_add::<Goldilocks>(&mut want, e, &src);
                    assert_eq!(got, want, "{label} mul_add len {len} coeff {c:#x}");
                    // A zero coefficient is a defined no-op at every level
                    // (ops fast path, scalar reference, kernel entry), so
                    // untouched input lanes are preserved, not canonicalized.
                    if !e.is_zero() {
                        assert_canonical(label, &got, 8, PGLD);
                    }

                    let mut got = vec![0u8; len];
                    let mut want = vec![0u8; len];
                    mulinto(&mut got, c, &src);
                    prime::mul_add::<Goldilocks>(&mut want, e, &src);
                    assert_eq!(got, want, "{label} mul_into len {len} coeff {c:#x}");
                    assert_canonical(label, &got, 8, PGLD);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    mulassign(&mut got, c);
                    prime::mul_assign::<Goldilocks>(&mut want, e);
                    assert_eq!(got, want, "{label} mul_assign len {len} coeff {c:#x}");
                    // A unit coefficient is a defined no-op, preserving
                    // untouched input lanes.
                    if !e.is_one() {
                        assert_canonical(label, &got, 8, PGLD);
                    }

                    let mut got = base.clone();
                    let mut want = base.clone();
                    addscalar(&mut got, c);
                    prime::add_assign_scalar::<Goldilocks>(&mut want, e);
                    assert_eq!(
                        got, want,
                        "{label} add_assign_scalar len {len} coeff {c:#x}"
                    );
                    assert_canonical(label, &got, 8, PGLD);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    subscalar(&mut got, c);
                    prime::sub_assign_scalar::<Goldilocks>(&mut want, e);
                    assert_eq!(
                        got, want,
                        "{label} sub_assign_scalar len {len} coeff {c:#x}"
                    );
                    assert_canonical(label, &got, 8, PGLD);
                }
            }
        }

        if host_supports(&[Backend::V3]) {
            let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
            drive_m31(
                "m31 avx2 noncanonical",
                |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
                |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
                |dst, coeff, src| x86::mersenne31::mul_add_avx2(token, dst, coeff, src),
                |dst, coeff, src| x86::mersenne31::mul_into_avx2(token, dst, coeff, src),
                |dst, coeff| x86::mersenne31::mul_assign_avx2(token, dst, coeff),
                |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
                |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
                |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
                |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
            );
            drive_gld(
                "gld avx2 noncanonical",
                |dst, src| x86::goldilocks::add_assign_avx2(token, dst, src),
                |dst, src| x86::goldilocks::sub_assign_avx2(token, dst, src),
                |dst, coeff, src| x86::goldilocks::mul_add_avx2(token, dst, coeff, src),
                |dst, coeff, src| x86::goldilocks::mul_into_avx2(token, dst, coeff, src),
                |dst, coeff| x86::goldilocks::mul_assign_avx2(token, dst, coeff),
                |dst, a, b| x86::goldilocks::mul_elementwise_avx2(token, dst, a, b),
                |dst, s| x86::goldilocks::mul_elementwise_assign_avx2(token, dst, s),
                |dst, c| x86::goldilocks::add_assign_scalar_avx2(token, dst, c),
                |dst, c| x86::goldilocks::sub_assign_scalar_avx2(token, dst, c),
            );
        } else {
            eprintln!("skipping: no AVX2 on this host");
        }

        if host_supports(&[Backend::V2]) {
            let token = X64V2Token::summon().expect("guard passed: SSE4.2 summons here");
            drive_m31(
                "m31 sse4.2 noncanonical",
                |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
                |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
                |dst, coeff, src| x86::mersenne31::mul_add_sse42(token, dst, coeff, src),
                |dst, coeff, src| x86::mersenne31::mul_into_sse42(token, dst, coeff, src),
                |dst, coeff| x86::mersenne31::mul_assign_sse42(token, dst, coeff),
                |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
                |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
                |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
                |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
            );
            drive_gld(
                "gld sse4.2 noncanonical",
                |dst, src| x86::goldilocks::add_assign_sse42(token, dst, src),
                |dst, src| x86::goldilocks::sub_assign_sse42(token, dst, src),
                |dst, coeff, src| x86::goldilocks::mul_add_sse42(token, dst, coeff, src),
                |dst, coeff, src| x86::goldilocks::mul_into_sse42(token, dst, coeff, src),
                |dst, coeff| x86::goldilocks::mul_assign_sse42(token, dst, coeff),
                |dst, a, b| x86::goldilocks::mul_elementwise_sse42(token, dst, a, b),
                |dst, s| x86::goldilocks::mul_elementwise_assign_sse42(token, dst, s),
                |dst, c| x86::goldilocks::add_assign_scalar_sse42(token, dst, c),
                |dst, c| x86::goldilocks::sub_assign_scalar_sse42(token, dst, c),
            );
        } else {
            eprintln!("skipping: no SSE4.2 on this host");
        }

        #[cfg(feature = "simd512")]
        {
            let Some(v4x) = X64V4xToken::summon() else {
                eprintln!("skipping: no V4x token on this host");
                return;
            };
            let token = v4x.v4();
            drive_m31(
                "m31 avx512 noncanonical",
                |dst, src| x86::mersenne31::add_assign_avx512(token, dst, src),
                |dst, src| x86::mersenne31::sub_assign_avx512(token, dst, src),
                |dst, coeff, src| x86::mersenne31::mul_add_avx512(token, dst, coeff, src),
                |dst, coeff, src| x86::mersenne31::mul_into_avx512(token, dst, coeff, src),
                |dst, coeff| x86::mersenne31::mul_assign_avx512(token, dst, coeff),
                |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token, dst, a, b),
                |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token, dst, s),
                |dst, c| x86::mersenne31::add_assign_scalar_avx512(token, dst, c),
                |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token, dst, c),
            );
            drive_gld(
                "gld avx512 noncanonical",
                |dst, src| x86::goldilocks::add_assign_avx512(token, dst, src),
                |dst, src| x86::goldilocks::sub_assign_avx512(token, dst, src),
                |dst, coeff, src| x86::goldilocks::mul_add_avx512(token, dst, coeff, src),
                |dst, coeff, src| x86::goldilocks::mul_into_avx512(token, dst, coeff, src),
                |dst, coeff| x86::goldilocks::mul_assign_avx512(token, dst, coeff),
                |dst, a, b| x86::goldilocks::mul_elementwise_avx512(token, dst, a, b),
                |dst, s| x86::goldilocks::mul_elementwise_assign_avx512(token, dst, s),
                |dst, c| x86::goldilocks::add_assign_scalar_avx512(token, dst, c),
                |dst, c| x86::goldilocks::sub_assign_scalar_avx512(token, dst, c),
            );
        }
    }
    #[test]
    // Nine entries at a dozen lengths: a table by nature.
    #[allow(clippy::too_many_lines)]
    fn qm31_entries_canonicalize_noncanonical_lanes() {
        use crate::field::quad_mersenne31::QuadMersenne31;
        use crate::kernel::prime;

        type Qm = Elem<QuadMersenne31>;
        const P: u32 = crate::field::mersenne31::MODULUS;
        // Limb pairs cycling both canonical and alias limbs through each
        // element position.
        const LIMBS: &[u32] = &[0, 1, P - 1, P, P + 1, 0x8000_0000, 0xFFFF_FFFE, u32::MAX];
        const LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
        const COEFFS: &[(u32, u32)] = &[(0, 0), (1, 0), (0, 1), (3, 5), (P, 7), (u32::MAX, P)];

        fn fill(len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            let mut i = 0;
            while out.len() < len {
                let (re, im) = (LIMBS[i % LIMBS.len()], LIMBS[(i + 3) % LIMBS.len()]);
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
                i += 1;
            }
            out.truncate(len);
            out
        }
        fn assert_canonical(label: &str, buf: &[u8]) {
            for (i, limb) in buf.chunks_exact(4).enumerate() {
                let lane = u32::from_le_bytes(limb.try_into().unwrap());
                assert!(lane < P, "{label}: limb {i} holds {lane:#x}, want < {P:#x}");
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn drive(
            label: &str,
            add: impl Fn(&mut [u8], &[u8]),
            sub: impl Fn(&mut [u8], &[u8]),
            muladd: impl Fn(&mut [u8], Qm, &[u8]),
            mulinto: impl Fn(&mut [u8], Qm, &[u8]),
            mulassign: impl Fn(&mut [u8], Qm),
            elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
            elemwise_assign: impl Fn(&mut [u8], &[u8]),
            addscalar: impl Fn(&mut [u8], Qm),
            subscalar: impl Fn(&mut [u8], Qm),
        ) {
            for &len in LENS {
                let src = fill(len);
                let base = {
                    let mut shifted = fill(len + 8);
                    shifted.drain(..8);
                    shifted.truncate(len);
                    shifted
                };
                let other = {
                    let mut shifted = fill(len + 16);
                    shifted.drain(..16);
                    shifted.truncate(len);
                    shifted
                };

                let mut got = base.clone();
                let mut want = base.clone();
                add(&mut got, &src);
                prime::add_assign::<QuadMersenne31>(&mut want, &src);
                assert_eq!(got, want, "{label} add_assign len {len}");
                assert_canonical(label, &got);

                let mut got = base.clone();
                let mut want = base.clone();
                sub(&mut got, &src);
                prime::sub_assign::<QuadMersenne31>(&mut want, &src);
                assert_eq!(got, want, "{label} sub_assign len {len}");
                assert_canonical(label, &got);

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                elemwise(&mut got, &src, &other);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &other);
                assert_eq!(got, want, "{label} mul_elementwise len {len}");
                assert_canonical(label, &got);

                let mut got = base.clone();
                let mut want = base.clone();
                elemwise_assign(&mut got, &src);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &base, &src);
                assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
                assert_canonical(label, &got);

                for &(re, im) in COEFFS {
                    let coeff = Qm::from_raw(re, im);
                    let mut got = base.clone();
                    let mut want = base.clone();
                    muladd(&mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(got, want, "{label} mul_add len {len} coeff {re:#x},{im:#x}");
                    // A zero coefficient is a defined no-op at every level,
                    // so untouched input lanes are preserved.
                    if coeff != Qm::ZERO {
                        assert_canonical(label, &got);
                    }

                    let mut got = vec![0u8; len];
                    let mut want = vec![0u8; len];
                    mulinto(&mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(
                        got, want,
                        "{label} mul_into len {len} coeff {re:#x},{im:#x}"
                    );
                    assert_canonical(label, &got);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    mulassign(&mut got, coeff);
                    prime::mul_assign::<QuadMersenne31>(&mut want, coeff);
                    assert_eq!(
                        got, want,
                        "{label} mul_assign len {len} coeff {re:#x},{im:#x}"
                    );
                    // A unit coefficient is a defined no-op, preserving
                    // untouched input lanes.
                    if coeff != Qm::ONE {
                        assert_canonical(label, &got);
                    }

                    let mut got = base.clone();
                    let mut want = base.clone();
                    addscalar(&mut got, coeff);
                    prime::add_assign_scalar::<QuadMersenne31>(&mut want, coeff);
                    assert_eq!(
                        got, want,
                        "{label} add_assign_scalar len {len} coeff {re:#x},{im:#x}"
                    );
                    assert_canonical(label, &got);

                    let mut got = base.clone();
                    let mut want = base.clone();
                    subscalar(&mut got, coeff);
                    prime::sub_assign_scalar::<QuadMersenne31>(&mut want, coeff);
                    assert_eq!(
                        got, want,
                        "{label} sub_assign_scalar len {len} coeff {re:#x},{im:#x}"
                    );
                    assert_canonical(label, &got);
                }
            }
        }

        if host_supports(&[Backend::V3]) {
            let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
            drive(
                "qm31 avx2 noncanonical",
                |dst, src| x86::quad_mersenne31::add_assign_avx2(token, dst, src),
                |dst, src| x86::quad_mersenne31::sub_assign_avx2(token, dst, src),
                |dst, coeff, src| x86::quad_mersenne31::mul_add_avx2(token, dst, coeff, src),
                |dst, coeff, src| x86::quad_mersenne31::mul_into_avx2(token, dst, coeff, src),
                |dst, coeff| x86::quad_mersenne31::mul_assign_avx2(token, dst, coeff),
                |dst, a, b| x86::quad_mersenne31::mul_elementwise_avx2(token, dst, a, b),
                |dst, s| x86::quad_mersenne31::mul_elementwise_assign_avx2(token, dst, s),
                |dst, c| x86::quad_mersenne31::add_assign_scalar_avx2(token, dst, c),
                |dst, c| x86::quad_mersenne31::sub_assign_scalar_avx2(token, dst, c),
            );
        } else {
            eprintln!("skipping: no AVX2 on this host");
        }

        #[cfg(feature = "simd512")]
        {
            let Some(v4x) = X64V4xToken::summon() else {
                eprintln!("skipping: no V4x token on this host");
                return;
            };
            let token = v4x.v4();
            // The AVX-512 QM31 family covers the multiply shapes; the
            // remaining operations run the AVX2 entries above.
            for &len in LENS {
                let src = fill(len);
                let base = fill(len);
                for &(re, im) in COEFFS {
                    let coeff = Qm::from_raw(re, im);
                    let mut got = base.clone();
                    let mut want = base.clone();
                    x86::quad_mersenne31::mul_add_avx512(token, &mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(got, want, "qm31 avx512 mul_add len {len}");
                    // A zero coefficient is a defined no-op, preserving
                    // untouched input lanes.
                    if coeff != Qm::ZERO {
                        assert_canonical("qm31 avx512 mul_add", &got);
                    }

                    let mut got = vec![0u8; len];
                    let mut want = vec![0u8; len];
                    x86::quad_mersenne31::mul_into_avx512(token, &mut got, coeff, &src);
                    prime::mul_add::<QuadMersenne31>(&mut want, coeff, &src);
                    assert_eq!(got, want, "qm31 avx512 mul_into len {len}");
                    assert_canonical("qm31 avx512 mul_into", &got);
                }
                let other = fill(len);
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                x86::quad_mersenne31::mul_elementwise_avx512(token, &mut got, &src, &other);
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &src, &other);
                assert_eq!(got, want, "qm31 avx512 mul_elementwise len {len}");
                assert_canonical("qm31 avx512 mul_elementwise", &got);
            }
        }
    }
    #[test]
    fn qm31_avx2_kernels_match_reference() {
        // Multiples of 8 (one complex element) straddling the 32-byte AVX2
        // lane, with odd element tails.
        const QM31_LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
        const P: u32 = crate::field::mersenne31::MODULUS;
        const QM31_VALUES: [crate::field::Elem<crate::field::quad_mersenne31::QuadMersenne31>; 6] = [
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(0, 0),
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(1, 0),
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(0, 1),
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(7, 2),
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(P - 1, 3),
            crate::field::Elem::<crate::field::quad_mersenne31::QuadMersenne31>::from_raw(
                P - 1,
                P - 1,
            ),
        ];
        if !host_supports(&[Backend::V3]) {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: AVX2 summons here");
        drive_prime::<
            crate::field::quad_mersenne31::QuadMersenne31,
            crate::field::Elem<crate::field::quad_mersenne31::QuadMersenne31>,
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

    /// Differential coverage for the Reed-Solomon-rooted custom tower: a
    /// description without a pinned strategy routes every dispatched
    /// single-row operation through the typed scalar fallback, at lengths
    /// straddling the 32-byte lane boundaries the vector kernels would use.
    #[test]
    fn rs_tower_kernels_match_scalar() {
        let coeffs: Vec<RsElem> = [0u16, 1, 2, 0x00ff, 0x0100, 0xff00, 0xffff, 0xbeef]
            .iter()
            .copied()
            .map(RsElem::from_raw)
            .collect();

        check_tower_mul_add(
            "rs16 fallback mul_add",
            LENGTHS,
            &coeffs,
            scalar::mul_add::<RsField>,
            |dst, c, src| {
                let prepared = <RsField as KernelDispatch>::prepare(RawDispatch, c);
                <RsField as KernelDispatch>::mul_add(RawDispatch, dst, &prepared, src);
            },
        );
        check_tower_mul_assign(
            "rs16 fallback mul_assign",
            LENGTHS,
            &coeffs,
            scalar::mul_assign::<RsField>,
            |dst, c| {
                let prepared = <RsField as KernelDispatch>::prepare(RawDispatch, c);
                <RsField as KernelDispatch>::mul_assign(RawDispatch, dst, &prepared);
            },
        );
        check_tower_mul_into(
            "rs16 fallback mul_into",
            LENGTHS,
            &coeffs,
            scalar::mul_add::<RsField>,
            |dst, c, src| {
                let prepared = <RsField as KernelDispatch>::prepare(RawDispatch, c);
                <RsField as KernelDispatch>::mul_into(RawDispatch, dst, &prepared, src);
            },
        );

        // The vector elementwise bodies fold the relation's linear term into
        // the Karatsuba cross sum, which holds only for a unit linear
        // coefficient — so the RS tower's elementwise route is the scalar
        // path, exercised in the dispatch test below.
    }

    /// The dispatched operations for the Reed–Solomon-rooted tower: the
    /// one-shot raw-element forms resolve against the RS bank on access.
    #[test]
    fn rs_tower_dispatch_matches_scalar() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        for &len in LENGTHS {
            let src = noise(len, 0x238);
            for raw in [0u16, 1, 0x0100, 0xbeef] {
                let coeff = RsElem::from_raw(raw);
                let prepared = <RsField as KernelDispatch>::prepare(RawDispatch, coeff);

                let mut got = noise(len, 0x249);
                let mut want = got.clone();
                <RsField as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
                scalar::mul_add::<RsField>(&mut want, coeff, &src);
                assert_eq!(
                    got, want,
                    "rs16 dispatch mul_add: len {len} coeff {raw:04x}"
                );

                let mut got = noise(len, 0x25a);
                let mut want = got.clone();
                <RsField as KernelDispatch>::mul_assign(RawDispatch, &mut got, &prepared);
                scalar::mul_assign::<RsField>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "rs16 dispatch mul_assign: len {len} coeff {raw:04x}"
                );

                let mut got = vec![0xaa; len];
                let mut want = vec![0u8; len];
                <RsField as KernelDispatch>::mul_into(RawDispatch, &mut got, &prepared, &src);
                scalar::mul_add::<RsField>(&mut want, coeff, &src);
                assert_eq!(
                    got, want,
                    "rs16 dispatch mul_into: len {len} coeff {raw:04x}"
                );
            }

            // Elementwise dispatch keeps the scalar path for the non-unit
            // linear coefficient.
            let a = noise(len, 0x26b);
            let b = noise(len, 0x27c);
            let mut got = vec![0; len];
            let mut want = vec![0; len];
            <RsField as KernelDispatch>::mul_elementwise(RawDispatch, &mut got, &a, &b);
            scalar::mul_elementwise::<RsField>(&mut want, &a, &b);
            assert_eq!(got, want, "rs16 dispatch elementwise: len {len}");
        }

        // Multi-row shapes over raw elements through dispatch.
        let row_len = 34;
        let nrows = 5;
        let src = noise(row_len, 0x28d);
        let coeffs: Vec<RsElem> = (0..nrows as u16)
            .map(|j| RsElem::from_raw(j * 8191))
            .collect();
        let mut got = noise(row_len * nrows, 0x29e);
        let mut want = got.clone();
        <RsField as KernelDispatch>::mul_add_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
            scalar::mul_add::<RsField>(row, coeff, &src);
        }
        assert_eq!(got, want, "rs16 dispatch scatter");

        let srcs: Vec<Vec<u8>> = (0u16..3)
            .map(|t| noise(row_len, 0x2af + u64::from(t)))
            .collect();
        let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        let gather_coeffs: Vec<RsElem> = (0u16..3)
            .map(|t| RsElem::from_raw(t.wrapping_mul(40_961).wrapping_add(7)))
            .collect();
        let mut got = noise(row_len, 0x2c0);
        let mut want = got.clone();
        <RsField as KernelDispatch>::mul_add_gather(
            RawDispatch,
            &mut got,
            &gather_coeffs,
            &src_refs,
        );
        for (&coeff, &s) in gather_coeffs.iter().zip(&src_refs) {
            scalar::mul_add::<RsField>(&mut want, coeff, s);
        }
        assert_eq!(got, want, "rs16 dispatch gather");

        let matrix_coeffs: Vec<Vec<RsElem>> = (0u16..3)
            .map(|t| {
                (0..nrows)
                    .map(|j| {
                        RsElem::from_raw(
                            t.wrapping_mul(7919)
                                .wrapping_add((j as u16).wrapping_mul(613)),
                        )
                    })
                    .collect()
            })
            .collect();
        let terms: Vec<(&[RsElem], &[u8])> = (0..3)
            .map(|t| (matrix_coeffs[t].as_slice(), src_refs[t]))
            .collect();
        let mut got = noise(row_len * nrows, 0x2d1);
        let mut want = got.clone();
        <RsField as KernelDispatch>::mul_add_matrix(RawDispatch, &mut got, row_len, nrows, &terms);
        for &(coeffs, s) in &terms {
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                scalar::mul_add::<RsField>(row, coeff, s);
            }
        }
        assert_eq!(got, want, "rs16 dispatch matrix");
    }

    /// The AES-base tower with a non-unit linear coefficient: the only
    /// spec that drives the GFNI `Compact` route with a general relation.
    /// Every dispatched entry runs, against the portable scalar reference.
    #[test]
    fn aes_tower_dispatch_matches_scalar() {
        use crate::kernel::tower::gf16::Prepared;

        // A custom tower takes the scalar fallback on every tier: the
        // backend reports `Scalar` and the prepared form is the raw word.
        assert_eq!(
            <AesField as crate::kernel::FieldKernels>::backend(),
            crate::kernel::Backend::Scalar
        );
        assert!(matches!(
            <AesField as KernelDispatch>::prepare(RawDispatch, AesElem::GENERATOR),
            Prepared::Plain(_)
        ));
        assert!(!<AesField as crate::kernel::FieldKernels>::has_vector_elementwise());

        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        for &len in LENGTHS {
            let src = noise(len, 0x238);
            for raw in [0u16, 1, 0x0104, 0xbeef] {
                let coeff = AesElem::from_raw(raw);
                let prepared = <AesField as KernelDispatch>::prepare(RawDispatch, coeff);

                let mut got = noise(len, 0x249);
                let mut want = got.clone();
                <AesField as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
                scalar::mul_add::<AesField>(&mut want, coeff, &src);
                assert_eq!(
                    got, want,
                    "aes16 dispatch mul_add: len {len} coeff {raw:04x}"
                );

                let mut got = noise(len, 0x25a);
                let mut want = got.clone();
                <AesField as KernelDispatch>::mul_assign(RawDispatch, &mut got, &prepared);
                scalar::mul_assign::<AesField>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "aes16 dispatch mul_assign: len {len} coeff {raw:04x}"
                );

                let mut got = vec![0xaa; len];
                let mut want = vec![0u8; len];
                <AesField as KernelDispatch>::mul_into(RawDispatch, &mut got, &prepared, &src);
                scalar::mul_add::<AesField>(&mut want, coeff, &src);
                assert_eq!(
                    got, want,
                    "aes16 dispatch mul_into: len {len} coeff {raw:04x}"
                );
            }

            // A non-unit linear coefficient keeps elementwise scalar.
            let a = noise(len, 0x26b);
            let b = noise(len, 0x27c);
            let mut got = vec![0; len];
            let mut want = vec![0; len];
            <AesField as KernelDispatch>::mul_elementwise(RawDispatch, &mut got, &a, &b);
            scalar::mul_elementwise::<AesField>(&mut want, &a, &b);
            assert_eq!(got, want, "aes16 dispatch elementwise: len {len}");
            let mut got = a.clone();
            let mut want = a.clone();
            <AesField as KernelDispatch>::mul_elementwise_assign(RawDispatch, &mut got, &b);
            scalar::mul_elementwise_assign::<AesField>(&mut want, &b);
            assert_eq!(got, want, "aes16 dispatch elementwise assign: len {len}");
        }

        aes_tower_multi_row_matches_scalar();
    }

    /// A custom tower over a normal-basis degree-8 base: bulk operations
    /// run the typed scalar fallback on every tier and `backend_for`
    /// reports `Scalar`.
    #[test]
    fn custom_normal_base_tower_bulk_matches_scalar() {
        type NormalBase = Gf<8, Normal<0x11B, 0x20>>;

        #[derive(Clone, Copy)]
        struct OverNormal;

        impl crate::field::binary::tower::TowerSpec for OverNormal {
            type Base = NormalBase;
            // The linear coefficient is the normal base's one (all bits
            // set); the constant term is the AES `0x20` transported into
            // normal coordinates.
            const A: u64 = 0xFF;
            const B: u64 = normal_transport(0x20);
            const NAME: &'static str = "custom normal-base tower";
        }

        type Field = Gf<16, crate::field::binary::tower::Tower<OverNormal>>;
        assert_eq!(Elem::<NormalBase>::ONE.to_raw(), 0xFF);

        assert_eq!(
            <Field as crate::kernel::FieldKernels>::backend(),
            crate::kernel::Backend::Scalar
        );

        let row_len = 34;
        let src = noise(row_len, 0x3a1);
        let coeffs: Vec<Elem<Field>> = (0..5u16)
            .map(|j| Elem::<Field>::from_raw(j.wrapping_mul(8191)))
            .collect();
        let mut got = noise(row_len * 5, 0x3b2);
        let mut want = got.clone();
        <Field as KernelDispatch>::mul_add_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
            scalar::mul_add::<Field>(row, coeff, &src);
        }
        assert_eq!(got, want, "custom normal-base tower scatter");

        let prepared = <Field as KernelDispatch>::prepare(RawDispatch, coeffs[0]);
        let mut got = noise(row_len, 0x3c3);
        let mut want = got.clone();
        <Field as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
        scalar::mul_add::<Field>(&mut want, coeffs[0], &src);
        assert_eq!(got, want, "custom normal-base tower mul_add");
    }

    /// The multi-row dispatched shapes for the AES-base general-relation
    /// tower, raw-element and prepared forms, against the portable
    /// scalar reference.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn aes_tower_multi_row_matches_scalar() {
        if !host_supports(&[Backend::V2]) {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        // Multi-row shapes over raw elements through dispatch.
        let row_len = 34;
        let nrows = 5;
        let src = noise(row_len, 0x28d);
        let coeffs: Vec<AesElem> = (0..nrows as u16)
            .map(|j| AesElem::from_raw(j * 8191))
            .collect();
        let mut got = noise(row_len * nrows, 0x29e);
        let mut want = got.clone();
        <AesField as KernelDispatch>::mul_add_scatter(
            RawDispatch,
            &mut got,
            row_len,
            &coeffs,
            &src,
        );
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
            scalar::mul_add::<AesField>(row, coeff, &src);
        }
        assert_eq!(got, want, "aes16 dispatch scatter");

        let srcs: Vec<Vec<u8>> = (0u16..3)
            .map(|t| noise(row_len, 0x2af + u64::from(t)))
            .collect();
        let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        let gather_coeffs: Vec<AesElem> = (0u16..3)
            .map(|t| AesElem::from_raw(t.wrapping_mul(40_961).wrapping_add(7)))
            .collect();
        let mut got = noise(row_len, 0x2c0);
        let mut want = got.clone();
        <AesField as KernelDispatch>::mul_add_gather(
            RawDispatch,
            &mut got,
            &gather_coeffs,
            &src_refs,
        );
        for (&coeff, &s) in gather_coeffs.iter().zip(&src_refs) {
            scalar::mul_add::<AesField>(&mut want, coeff, s);
        }
        assert_eq!(got, want, "aes16 dispatch gather");

        let matrix_coeffs: Vec<Vec<AesElem>> = (0u16..3)
            .map(|t| {
                (0..nrows)
                    .map(|j| {
                        AesElem::from_raw(
                            t.wrapping_mul(7919)
                                .wrapping_add((j as u16).wrapping_mul(613)),
                        )
                    })
                    .collect()
            })
            .collect();
        let terms: Vec<(&[AesElem], &[u8])> = (0..3)
            .map(|t| (matrix_coeffs[t].as_slice(), src_refs[t]))
            .collect();
        let mut got = noise(row_len * nrows, 0x2d1);
        let mut want = got.clone();
        <AesField as KernelDispatch>::mul_add_matrix(RawDispatch, &mut got, row_len, nrows, &terms);
        for &(coeffs, s) in &terms {
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                scalar::mul_add::<AesField>(row, coeff, s);
            }
        }
        assert_eq!(got, want, "aes16 dispatch matrix");
        // The `_with` and `_plan` forms cover the same routes through
        // prepared state; the multi-row defaults apply prepared AXPY per
        // row and term.
        #[cfg(feature = "alloc")]
        {
            let values: Vec<AesElem> = (0..nrows as u16)
                .map(|j| AesElem::from_raw(j.wrapping_mul(7411).wrapping_add(3)))
                .collect();
            let prepared: Vec<<AesField as KernelDispatch>::Prepared> = values
                .iter()
                .map(|&c| <AesField as KernelDispatch>::prepare(RawDispatch, c))
                .collect();
            let mut got = noise(row_len * nrows, 0x2e2);
            let mut want = got.clone();
            <AesField as KernelDispatch>::mul_add_scatter_with(
                RawDispatch,
                &mut got,
                row_len,
                &prepared,
                &src,
            );
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&values) {
                scalar::mul_add::<AesField>(row, coeff, &src);
            }
            assert_eq!(got, want, "aes16 dispatch scatter with");
            let mut got = noise(row_len * nrows, 0x2e3);
            let mut want = got.clone();
            <AesField as KernelDispatch>::mul_add_scatter_plan(
                RawDispatch,
                &mut got,
                row_len,
                &values,
                &prepared,
                &src,
            );
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&values) {
                scalar::mul_add::<AesField>(row, coeff, &src);
            }
            assert_eq!(got, want, "aes16 dispatch scatter plan");

            let mut got = noise(row_len, 0x2e4);
            let mut want = got.clone();
            <AesField as KernelDispatch>::mul_add_gather_with(
                RawDispatch,
                &mut got,
                &prepared[..src_refs.len()],
                &src_refs,
            );
            for (&coeff, &s) in values.iter().zip(&src_refs) {
                scalar::mul_add::<AesField>(&mut want, coeff, s);
            }
            assert_eq!(got, want, "aes16 dispatch gather with");
            let mut got = noise(row_len, 0x2e5);
            let mut want = got.clone();
            <AesField as KernelDispatch>::mul_add_gather_plan(
                RawDispatch,
                &mut got,
                &values[..src_refs.len()],
                &prepared[..src_refs.len()],
                &src_refs,
            );
            for (&coeff, &s) in values.iter().zip(&src_refs) {
                scalar::mul_add::<AesField>(&mut want, coeff, s);
            }
            assert_eq!(got, want, "aes16 dispatch gather plan");

            let nterms = src_refs.len();
            let flat: Vec<AesElem> = (0..nterms * nrows)
                .map(|k| AesElem::from_raw((k as u16).wrapping_mul(911).wrapping_add(17)))
                .collect();
            let flat_prepared: Vec<<AesField as KernelDispatch>::Prepared> = flat
                .iter()
                .map(|&c| <AesField as KernelDispatch>::prepare(RawDispatch, c))
                .collect();
            let mut got = noise(row_len * nrows, 0x2e6);
            let mut want = got.clone();
            <AesField as KernelDispatch>::mul_add_matrix_plan(
                RawDispatch,
                &mut got,
                row_len,
                nrows,
                &flat,
                &flat_prepared,
                &src_refs,
            );
            for (term, &s) in src_refs.iter().enumerate() {
                for (row, &coeff) in want
                    .chunks_exact_mut(row_len)
                    .zip(&flat[term * nrows..(term + 1) * nrows])
                {
                    scalar::mul_add::<AesField>(row, coeff, s);
                }
            }
            assert_eq!(got, want, "aes16 dispatch matrix plan");
        }
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
        check_gf8_mul_add("gf8 neon", &gf8_aes_coeffs(), |dst, table, src| {
            aarch64::gf8::mul_add_neon(token, dst, table, src)
        });
        check_gf8_mul_assign("gf8 neon", &gf8_aes_coeffs(), |dst, table| {
            aarch64::gf8::mul_assign_neon(token, dst, table)
        });
        check_gf16_mul_add_tables("gf16 neon", |dst, tables, src| {
            aarch64::gf16::mul_add_neon(token, dst, tables, src)
        });
        check_gf16_mul_assign_tables("gf16 neon", |dst, tables| {
            aarch64::gf16::mul_assign_neon(token, dst, tables)
        });
        check_gf8_mul_into("gf8 neon mul_into", &gf8_aes_coeffs(), |dst, table, src| {
            aarch64::gf8::mul_into_neon(token, dst, table, src)
        });
        check_gf16_mul_into_tables("gf16 neon mul_into", |dst, tables, src| {
            aarch64::gf16::mul_into_neon(token, dst, tables, src)
        });

        check_scatter(
            "gf8 neon scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                aarch64::gf8::mul_add_scatter_neon(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs),
                    src,
                )
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
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                aarch64::gf8::mul_add_matrix_neon(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                )
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
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                aarch64::gf8::mul_add_gather_neon(token, dst, &prepared_coeffs(coeffs), srcs)
            },
        );
        check_gather(
            "gf16 neon gather",
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| aarch64::gf16::mul_add_gather_neon(token, dst, coeffs, srcs),
        );
        check_gf8_elementwise::<AES>("gf8 neon elementwise", |dst, a, b| {
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
        check_gf8_elementwise::<AES>("gf8 pmull elementwise", |dst, a, b| {
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
        check_gf8_mul_add("gf8 simd128", &gf8_aes_coeffs(), |dst, table, src| {
            wasm32::gf8::mul_add_simd128(token, dst, table, src)
        });
        check_gf8_mul_assign("gf8 simd128", &gf8_aes_coeffs(), |dst, table| {
            wasm32::gf8::mul_assign_simd128(token, dst, table)
        });
        check_gf16_mul_add_tables("gf16 simd128", |dst, tables, src| {
            wasm32::gf16::mul_add_simd128(token, dst, tables, src)
        });
        check_gf16_mul_assign_tables("gf16 simd128", |dst, tables| {
            wasm32::gf16::mul_assign_simd128(token, dst, tables)
        });
        check_gf8_mul_into(
            "gf8 simd128 mul_into",
            &gf8_aes_coeffs(),
            |dst, table, src| wasm32::gf8::mul_into_simd128(token, dst, table, src),
        );
        check_gf16_mul_into_tables("gf16 simd128 mul_into", |dst, tables, src| {
            wasm32::gf16::mul_into_simd128(token, dst, tables, src)
        });
        check_scatter(
            "gf8 simd128 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                wasm32::gf8::mul_add_scatter_simd128(
                    token,
                    rows,
                    row_len,
                    &prepared_coeffs(coeffs),
                    src,
                )
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
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                wasm32::gf8::mul_add_gather_simd128(token, dst, &prepared_coeffs(coeffs), srcs)
            },
        );
        check_gather(
            "gf16 simd128 gather",
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| wasm32::gf16::mul_add_gather_simd128(token, dst, coeffs, srcs),
        );
        check_matrix(
            "gf8 simd128 matrix",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms| {
                wasm32::gf8::mul_add_matrix_simd128(
                    token,
                    rows,
                    row_len,
                    nrows,
                    &prepared_matrix(terms),
                )
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
        check_gf8_elementwise::<AES>("gf8 simd128 elementwise", |dst, a, b| {
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
fn check_binary_broadcast_dispatch<F: KernelDispatch>(values: &[Elem<F>]) {
    const ELEMS: [usize; 4] = [0, 17, 33, 49];
    for &n in &ELEMS {
        let len = n * F::BYTES;
        let base = noise(len, 0x2f00 + u64::from(F::STORAGE_BITS));
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
    check_binary_broadcast_dispatch::<Gf8<Poly<AES>>>(&[
        Elem::<Gf<8, Poly<AES>>>::from_raw(0),
        Elem::<Gf<8, Poly<AES>>>::from_raw(1),
        Elem::<Gf<8, Poly<AES>>>::from_raw(0x53),
    ]);
    check_binary_broadcast_dispatch::<Gf8<Poly<REED_SOLOMON>>>(&[
        Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0),
        Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(1),
        Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(0x53),
    ]);
    check_binary_broadcast_dispatch::<Gf16>(&[
        Elem::<Gf16>::from_raw(0),
        Elem::<Gf16>::from_raw(1),
        Elem::<Gf16>::from_raw(0x53a7),
    ]);
    check_binary_broadcast_dispatch::<Gf32>(&[
        Elem::<Gf32>::from_raw(0),
        Elem::<Gf32>::from_raw(1),
        Elem::<Gf32>::from_raw(0xdead_beef),
    ]);
    check_binary_broadcast_dispatch::<Gf64>(&[
        Elem::<Gf64>::from_raw(0),
        Elem::<Gf64>::from_raw(1),
        Elem::<Gf64>::from_raw(0x0123_4567_89ab_cdef),
    ]);
    check_binary_broadcast_dispatch::<FanPaar8>(&[
        Elem::<FanPaar8>::from_raw(0),
        Elem::<FanPaar8>::from_raw(1),
        Elem::<FanPaar8>::from_raw(0xa5),
    ]);
    check_binary_broadcast_dispatch::<FanPaar16>(&[
        Elem::<FanPaar16>::from_raw(0),
        Elem::<FanPaar16>::from_raw(1),
        Elem::<FanPaar16>::from_raw(0xa55a),
    ]);
    check_binary_broadcast_dispatch::<FanPaar32>(&[
        Elem::<FanPaar32>::from_raw(0),
        Elem::<FanPaar32>::from_raw(1),
        Elem::<FanPaar32>::from_raw(0xa55a_1234),
    ]);
    check_binary_broadcast_dispatch::<FanPaar64>(&[
        Elem::<FanPaar64>::from_raw(0),
        Elem::<FanPaar64>::from_raw(1),
        Elem::<FanPaar64>::from_raw(0xa55a_1234_dead_beef),
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
        let coeffs = [gf16_coeff_at(1), gf16_coeff_at(2)];
        x86::gf16::mul_add_scatter_avx2(v3, &mut got, len, &coeffs, &src);
        scalar::mul_add::<Gf16>(&mut want[..len], gf16_coeff_at(1), &src);
        scalar::mul_add::<Gf16>(&mut want[len..], gf16_coeff_at(2), &src);
        assert_eq!(got, want, "plain-prepared scatter at len {len}");

        // The plain prepared form reaches the dispatched single-row path.
        let prepared = crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(1).to_raw());
        let mut dispatched = noise(len, 0x63);
        let mut reference = dispatched.clone();
        <Gf16 as KernelDispatch>::mul_add(RawDispatch, &mut dispatched, &prepared, &src);
        scalar::mul_add::<Gf16>(&mut reference, gf16_coeff_at(1), &src);
        assert_eq!(
            dispatched, reference,
            "plain-prepared dispatch at len {len}"
        );
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
    let coeffs = [Prepared::new(Elem::<Gf<8, Poly<AES>>>::from_raw(3))];
    let empty_aes: &[(&[Prepared], &[u8])] = &[];
    x86::gf8::mul_add_scatter_ssse3(v2, &mut [], 0, &coeffs, &[]);
    let gf16_coeffs = [Elem::<Gf16>::from_raw(5)];
    x86::gf16::mul_add_scatter_ssse3(v2, &mut [], 0, &gf16_coeffs, &[]);

    if let Some(v3) = X64V3Token::summon() {
        x86::gf8::mul_add_scatter_avx2(v3, &mut [], 0, &coeffs, &[]);
        x86::gf16::mul_add_scatter_avx2(v3, &mut [], 0, &gf16_coeffs, &[]);
        x86::gf8::mul_add_matrix_avx2_with(v3, &mut [], 0, 1, empty_aes);
    }
    x86::gf8::mul_add_matrix_ssse3_with(v2, &mut [], 0, 1, empty_aes);
    let no_terms: [(&[Elem<Gf16>], &[u8]); 0] = [];
    x86::gf16::mul_add_matrix_ssse3(v2, &mut [], 0, 1, &no_terms);

    // Empty term lists must return before any store, in the checked
    // scattered wrappers as well. The GFNI wrapper runs only where its
    // token summons; the guarded empty-terms differential below covers
    // GFNI hosts.
    if let Some(token) = X64V3GfniCryptoToken::summon() {
        let mut dst = [0u8; 8];
        let no_terms: &[(&[Prepared], &[u8])] = &[];
        x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(token, &mut dst, 8, &[0], no_terms);
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
        x86::gf8::mul_add_scatter_gfni::<true>(
            X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here"),
            &mut [0u8; 4],
            4,
            &[
                Prepared::new(Elem::<Gf<8, Poly<AES>>>::from_raw(1)),
                Prepared::new(Elem::<Gf<8, Poly<AES>>>::from_raw(2)),
            ],
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
    let coeffs = [
        Elem::<FanPaar8>::from_raw(1),
        Elem::<FanPaar8>::from_raw(0x8d),
    ];
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

fn quad_zero() -> Elem<quad_mersenne31::QuadMersenne31> {
    Elem::<quad_mersenne31::QuadMersenne31>::from_raw(0, 0)
}

fn quad_one() -> Elem<quad_mersenne31::QuadMersenne31> {
    Elem::<quad_mersenne31::QuadMersenne31>::from_raw(1, 0)
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
    // both selection constants.
    let empty: FlatMatrix<'_, Prepared> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    let mut rows = [0u8; 8];
    x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut rows, 8, 0, &empty);
    x86::gf8::mul_add_matrix_gfni_with::<false, _>(token, &mut rows, 8, 0, &empty);
    let gf16_empty: FlatMatrix<'_, TowerCoeff> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &gf16_empty);
    assert_eq!(rows, [0; 8]);
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn scatter_gfni_rs_rejects_short_rows_buffer() {
    use crate::kernel::x86;
    use archmage::{SimdToken as _, X64V3GfniCryptoToken};

    if !host_supports(&[crate::kernel::Backend::V3GfniCrypto]) {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    }
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::mul_add_scatter_gfni::<false>(
            X64V3GfniCryptoToken::summon().expect("guard passed: GFNI summons here"),
            &mut [0u8; 4],
            4,
            &[
                Prepared::new(Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(1)),
                Prepared::new(Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(2)),
            ],
            &[0u8; 4],
        );
    })
    .is_err();
    assert!(
        panicked,
        "mul_add_scatter_gfni must reject a short rows buffer"
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
    x86::gf16::mul_add_scatter_gfni(token, &mut [], 0, &[Elem::<Gf16>::from_raw(1)], &[]);
    let mut rows = [0u8; 8];
    x86::gf16::mul_add_scatter_gfni(token, &mut rows, 0, &[Elem::<Gf16>::from_raw(1)], &[]);
    assert_eq!(rows, [0; 8]);

    let empty: crate::kernel::FlatMatrix<'_, TowerCoeff> = crate::kernel::FlatMatrix {
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
/// per-bit scalar oracle over `gf1::Elem` — a genuinely different path from
/// the word loops (no packing, no masks). Geometry sweeps the same tail and
/// boundary cases the public suite drives, minus the wrapper checks, so a
/// wrapper bug cannot mask a kernel bug.
mod gf1_bits {
    use super::Vec;
    use crate::field::Elem;
    use crate::field::Gf1;
    use crate::kernel::gf1 as k;

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

            let mut acc = Elem::<Gf1>::ZERO;
            for i in 0..bits_len {
                acc = acc.add(Elem::<Gf1>::from_raw(u8::from(bit(&a, i) && bit(&b, i))));
            }
            assert_eq!(
                Elem::<Gf1>::from_raw(k::parity(&a, &b, bits_len) as u8),
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
