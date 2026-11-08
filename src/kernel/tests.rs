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
//!
//! The suite splits on two axes, family first: [`x86`] groups the
//! x86-compiled field families, each holding one module per instruction set
//! its differentials drive; [`isa`] holds the cross-family suites whose one
//! test spans several families per instruction set. Shared geometry,
//! coefficient classes, and differential drivers stay here so every family
//! sweeps the same shapes. Every arithmetic kernel invocation earns a
//! per-entry receipt through [`tier_gate`]; geometry and allocation checks
//! summon through the same gate without earning one.
//!
//! # Representation
//!
//! [`fixtures`] supplies the shared noise streams; the coefficient and
//! geometry lists below are the shared classes every family sweeps.
#![allow(clippy::cast_possible_truncation)]

extern crate std;

use std::vec;
use std::vec::Vec;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::field::FieldElem as _;
use crate::field::binary::tower::{
    FanPaar8Field, FanPaar16Field, FanPaar32Field, FanPaar64Field, Rijndael16Field,
    Rijndael32Field, Rijndael64Field,
};
use crate::field::binary::{AES, Polynomial, RS};
use crate::field::extension::{Packed, Relation};
use crate::field::goldilocks::Goldilocks;
use crate::field::{Elem, FiniteField};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::gf8::Prepared;
use crate::kernel::scalar;

use crate::field::binary::byte::ScaleTable;
use crate::field::dispatch::{KernelDispatch, RawDispatch};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::tables::{FpTowerTables, TowerCoeff};
use crate::kernel::tables::{TowerTables, scale_table};

// Names the ISA modules reach through `use super::*`: the tier gate and the
// capability tokens its receipts are keyed on.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::FGF_TIERS;
#[cfg(all(feature = "simd", feature = "simd512", target_arch = "x86_64"))]
use archmage::X64V4xToken;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{X64V1Token, X64V2Token, X64V3GfniCryptoToken, X64V3Token};
use fixtures::noise_odd_seed as noise;

/// Resolve raw GF(2^8) terms into the prepared coefficient pairs the
/// blocked matrix entries take; the sets stay alive across the body.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

// A provider's borrowed type is instantiated only after its fixtures exist.
// Each receipt follows the corresponding case's arithmetic comparison.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
macro_rules! check_prepared_matrix {
    ($driver:ident, $label:expr, $coeff:expr, $reference:expr, $tier:ty, $entry:path) => {
        if tier_gate::summon::<$tier>().is_some() {
            $driver(
                $label,
                $coeff,
                $reference,
                |rows, row_len, nrows, terms, want| {
                    prepared_pairs!(terms, |pairs| {
                        tier_gate::run_entry($entry, |token, entry| {
                            entry(token, rows, row_len, nrows, &pairs[..]);
                            assert_eq!(rows, want, "{}", $label);
                        });
                    });
                },
            );
        }
    };
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
macro_rules! check_prepared_prefetch {
    ($label:expr, $coeff:expr, $reference:expr, $add:path, $into:path) => {
        if tier_gate::summon::<X64V3GfniCryptoToken>().is_some() {
            check_gfni_matrix_prefetch(
                $label,
                $coeff,
                $reference,
                |rows, row_len, nrows, terms, want| {
                    prepared_pairs!(terms, |pairs| {
                        tier_gate::run_entry($add, |token, entry| {
                            entry(token, rows, row_len, nrows, &pairs[..]);
                            assert_eq!(rows, want, "{}", $label);
                        });
                    });
                },
                |rows, row_len, nrows, terms, want| {
                    prepared_pairs!(terms, |pairs| {
                        tier_gate::run_entry($into, |token, entry| {
                            entry(token, rows, row_len, nrows, &pairs[..]);
                            assert_eq!(rows, want, "{}", $label);
                        });
                    });
                },
            );
        }
    };
}

/// Prefetch-boundary differential for the GFNI matrix row-group bodies.
///
/// Row lengths sit just below, at, and just above
/// [`MATRIX_PREFETCH_MIN`](crate::kernel::x86::gf8::MATRIX_PREFETCH_MIN),
/// so every case runs both the prefetching and the non-prefetching body
/// across each row-group shape and term count. The accumulate and
/// overwrite forms share each case against per-term scalar AXPY.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn check_gfni_matrix_prefetch<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: F,
    kernel_add: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
    kernel_overwrite: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    let floor = crate::kernel::x86::gf8::MATRIX_PREFETCH_MIN;
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
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                kernel_add(&mut got, row_len, nrows, &terms, &want);
                assert_eq!(
                    got, want,
                    "{name}: accumulate row_len {row_len}, nrows {nrows}, terms {nterms}"
                );

                let mut got = noise(row_len * nrows, 0xc12);
                let mut want = vec![0u8; row_len * nrows];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                kernel_overwrite(&mut got, row_len, nrows, &terms, &want);
                assert_eq!(
                    got, want,
                    "{name}: overwrite row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

mod fixtures;

mod bytes;
mod fan_paar;
#[cfg(feature = "alloc")]
mod gf1;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod isa;
mod quad_mersenne31;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod tier_gate;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86;

/// Lengths covering: empty, sub-lane, exact lanes, lane+1, several unroll
/// tiles, and a large odd size. All even so GF(2^16) can use the same list.
const LENGTHS: &[usize] = fixtures::EVEN_DIFFERENTIAL_LENGTHS;
/// Byte-XOR lengths around every 16- and 32-byte lane boundary, including
/// compound AVX2 tails and body-plus-tail cases.
#[allow(dead_code)]
const XOR_LENGTHS: &[usize] = fixtures::XOR_LENGTHS;

/// GF(2^8) coefficients worth testing: the two short-circuits, the extremes,
/// and a spread through the field.
fn gf8_aes_coeffs() -> Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>> {
    gf8_coeffs_of::<AES>()
}

/// The Reed–Solomon field coefficients, same byte spread under `0x11D`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn gf8_rs_coeffs() -> Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>> {
    gf8_coeffs_of::<RS>()
}

fn gf8_coeffs_of<const POLY: u128>() -> Vec<Elem<FiniteField<2, 8, Polynomial<POLY>>>> {
    fixtures::GF8_DIFFERENTIAL_COEFFICIENTS
        .iter()
        .copied()
        .map(Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw)
        .collect()
}

/// The Rijndael extension's relation constant and base reduction byte: the raw
/// facts the elementwise kernels receive as plain values.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const DELTA_BYTE: u8 = 0x20;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn delta_table() -> &'static ScaleTable {
    scale_table(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x20))
}
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const AES_REDUCTION: u8 = Polynomial::<AES>::REDUCTION_LOW;

/// A custom degree-16 extension over the Reed-Solomon base: the relation
/// `t^2 + 2*t + 0x80` (the absolute trace of `0x80 / 2^2` is one). It is
/// not a pinned presentation, so every bulk operation routes through the
/// typed scalar fallback and `backend_for` reports `Scalar`.
#[derive(Clone, Copy)]
pub(crate) struct RsExtension;

impl crate::field::ExtensionSpec<2> for RsExtension {
    type Base = FiniteField<2, 8, Polynomial<RS>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/0x11D";
    const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x80, 2]);
}

pub(crate) type RsExtField = FiniteField<2, 16, crate::field::extension::Extension<RsExtension, 2>>;
#[allow(dead_code)]
pub(crate) type RsExtElem = Elem<RsExtField>;

/// A custom degree-16 extension over the AES base with a non-unit linear
/// coefficient: `t^2 + 2*t + 0x01`.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct AesExtension;

impl crate::field::ExtensionSpec<2> for AesExtension {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/0x11B/A2";
    const RELATION: Relation = Relation::from_binary_words::<Self::Base>(&[0x01, 2]);
}

#[allow(dead_code)]
pub(crate) type AesExtField =
    FiniteField<2, 16, crate::field::extension::Extension<AesExtension, 2>>;
#[allow(dead_code)]
pub(crate) type AesExtElem = Elem<AesExtField>;

/// GF(2^16) coefficients: pure base-field, pure extension, and mixed. The
/// extension kernels have distinct code paths for the `same` and `cross` factors,
/// and a coefficient with a zero component silently masks one of them.
fn gf16_coeffs() -> Vec<Elem<Rijndael16Field>> {
    fixtures::GF16_DIFFERENTIAL_COEFFICIENTS
        .iter()
        .copied()
        .map(Elem::<Rijndael16Field>::from_raw)
        .collect()
}

/// Row geometries for the multi-row kernels. Row counts straddle the
/// four-row and two-row grouping boundaries the blocked kernels use.
#[allow(dead_code)]
const ROW_COUNTS: &[usize] = fixtures::DIFFERENTIAL_ROW_COUNTS;
#[allow(dead_code)]
const ROW_LENS: &[usize] = fixtures::DIFFERENTIAL_ROW_LENGTHS;

/// Gather lengths exercise each side of the native 16/32/64/128-byte ladder,
/// compound remainders, and a body-plus-tail case.
#[allow(dead_code)]
const GATHER_LENGTHS: &[usize] = fixtures::GATHER_LENGTHS;

// ---------------------------------------------------------------------------
// Generic differential drivers, shared by every architecture below.
// ---------------------------------------------------------------------------

/// Compare a GF(2^8) `mul_add` kernel against the reference at every length
/// and coefficient.
fn check_gf8_mul_add<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
    kernel: impl Fn(&mut [u8], &ScaleTable, &[u8]),
) {
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for &coeff in coeffs {
            let mut got = noise(len, 0x62);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_assign<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
    kernel: impl Fn(&mut [u8], &ScaleTable),
) {
    for &len in LENGTHS {
        for &coeff in coeffs {
            let mut got = noise(len, 0x73);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff));
            scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff);
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
            scalar::mul_add::<Rijndael16Field>(&mut want, coeff, &src);
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
            scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn check_gf8_mul_into<const POLY: u128>(
    name: &str,
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
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
            scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff);
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
            scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
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
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
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

                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                kernel(&mut got, row_len, nrows, &terms, &want);
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
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
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

                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                kernel(&mut got, row_len, nrows, &terms, &want);
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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn for_each_gf8_coeff<const POLY: u128>(
    case: impl FnMut(Elem<FiniteField<2, 8, Polynomial<POLY>>>, &[u8]),
) {
    let mut case = case;
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for c in 0..=u8::MAX {
            case(
                Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw(c),
                &src,
            );
        }
    }
    let every_byte: Vec<u8> = (0..=u8::MAX).collect();
    for c in 0..=u8::MAX {
        case(
            Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw(c),
            &every_byte,
        );
    }
}

/// Exhaustive `mul_add` sweep through `prepare`'s coefficient form — the raw
/// element for the blocked single-buffer entries, [`Prepared`](crate::kernel::gf8::Prepared)
/// for the form dispatch hands them.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn check_gf8_exhaustive_mul_add<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<FiniteField<2, 8, Polynomial<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C, &[u8]),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x62);
        let mut want = got.clone();
        kernel(&mut got, prepare(coeff), src);
        scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff, src);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Exhaustive `mul_assign` sweep through `prepare`'s coefficient form.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn check_gf8_exhaustive_mul_assign<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<FiniteField<2, 8, Polynomial<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x73);
        let mut want = got.clone();
        kernel(&mut got, prepare(coeff));
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff);
        assert_eq!(got, want, "{name}: len {}, coeff {coeff:?}", src.len());
    });
}

/// Exhaustive `mul_into` sweep through `prepare`'s coefficient form: the
/// destination carries noise the fused kernel must overwrite.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn check_gf8_exhaustive_mul_into<C: Copy + core::fmt::Debug, const POLY: u128>(
    name: &str,
    prepare: impl Fn(Elem<FiniteField<2, 8, Polynomial<POLY>>>) -> C,
    kernel: impl Fn(&mut [u8], C, &[u8]),
) {
    for_each_gf8_coeff(|coeff, src| {
        let mut got = noise(src.len(), 0x147);
        let mut want = src.to_vec();
        kernel(&mut got, prepare(coeff), src);
        scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, coeff);
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
        scalar::mul_elementwise::<FiniteField<2, 8, Polynomial<POLY>>>(&mut want, &a, &b);
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
        scalar::mul_elementwise::<Rijndael16Field>(&mut want, &a, &b);
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
            // Noise in the destination: an overwriting kernel must replace
            // it, not accumulate. `want` starts at zero, so an accumulating
            // reference yields the product an overwriting kernel must produce.
            let mut want = vec![0; len];
            let mut got = noise(len, 0x95);
            reference(&mut want, coeff, &src);
            kernel(&mut got, coeff, &src);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

#[allow(dead_code)]
fn gf8_coeff_at<const POLY: u128>(j: usize) -> Elem<FiniteField<2, 8, Polynomial<POLY>>> {
    // Includes 0 and 1 as j sweeps, which is what we want: the blocked
    // kernels must handle degenerate coefficients per row, not per call.
    Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw((j as u8).wrapping_mul(29))
}

#[allow(dead_code)]
fn gf8_coeff_at2<const POLY: u128>(
    t: usize,
    j: usize,
) -> Elem<FiniteField<2, 8, Polynomial<POLY>>> {
    Elem::<FiniteField<2, 8, Polynomial<POLY>>>::from_raw(((t * 31 + j * 29) % 256) as u8)
}

#[allow(dead_code)]
fn gf16_coeff_at(j: usize) -> Elem<Rijndael16Field> {
    Elem::<Rijndael16Field>::from_raw((j as u16).wrapping_mul(7411))
}

#[allow(dead_code)]
fn gf16_coeff_at2(t: usize, j: usize) -> Elem<Rijndael16Field> {
    Elem::<Rijndael16Field>::from_raw(((t * 7919 + j * 613) % 65536) as u16)
}

#[allow(dead_code)]
fn gf8_reference<const POLY: u128>(
    dst: &mut [u8],
    coeff: Elem<FiniteField<2, 8, Polynomial<POLY>>>,
    src: &[u8],
) {
    scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(dst, coeff, src);
}

#[allow(dead_code)]
fn gf16_reference(dst: &mut [u8], coeff: Elem<Rijndael16Field>, src: &[u8]) {
    scalar::mul_add::<Rijndael16Field>(dst, coeff, src);
}

/// Fan–Paar GF(2^16) coefficients: short-circuits, extremes, the tower
/// generator and its components, and a deterministic spread.
#[allow(dead_code)]
fn fp16_coeffs() -> Vec<Elem<FanPaar16Field>> {
    let mut v = vec![
        Elem::<FanPaar16Field>::ZERO,
        Elem::<FanPaar16Field>::ONE,
        Elem::<FanPaar16Field>::from_raw(u16::MAX),
        // Pure base-field and pure extension.
        Elem::<FanPaar16Field>::from_components([
            Elem::<FanPaar8Field>::ONE,
            Elem::<FanPaar8Field>::ZERO,
        ]),
        Elem::<FanPaar16Field>::from_components([
            Elem::<FanPaar8Field>::ZERO,
            Elem::<FanPaar8Field>::ONE,
        ]),
        // The tower generator and its subfield tower generator.
        Elem::<FanPaar16Field>::GENERATOR,
        Elem::<FanPaar16Field>::from_raw(0x0100),
    ];
    let mut s = 0xf491u16;
    for _ in 0..16 {
        s = s.wrapping_mul(2057).wrapping_add(13849);
        v.push(Elem::<FanPaar16Field>::from_raw(s));
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
fn fp16_reference(dst: &mut [u8], coeff: Elem<FanPaar16Field>, src: &[u8]) {
    fp_mul_add(dst, u64::from(coeff.to_raw()), src, 16);
}
#[allow(dead_code)]
fn fp16_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar16Field>) {
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 16);
}
#[allow(dead_code)]
fn fp16_into_reference(dst: &mut [u8], coeff: Elem<FanPaar16Field>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 16);
}

/// Fan–Paar GF(2^32) coefficients, the same shape as the polynomial towers.
#[allow(dead_code)]
fn fp32_coeffs() -> Vec<Elem<FanPaar32Field>> {
    let mut v = vec![
        Elem::<FanPaar32Field>::ZERO,
        Elem::<FanPaar32Field>::ONE,
        Elem::<FanPaar32Field>::from_raw(u32::MAX),
        Elem::<FanPaar32Field>::from_components([
            Elem::<FanPaar16Field>::ONE,
            Elem::<FanPaar16Field>::ZERO,
        ]),
        Elem::<FanPaar32Field>::from_components([
            Elem::<FanPaar16Field>::ZERO,
            Elem::<FanPaar16Field>::ONE,
        ]),
        Elem::<FanPaar32Field>::from_raw(0x0001_0000),
        Elem::<FanPaar32Field>::from_components([
            Elem::<FanPaar16Field>::ZERO,
            Elem::<FanPaar16Field>::from_raw(0x0100),
        ]),
        Elem::<FanPaar32Field>::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(Elem::<FanPaar32Field>::from_raw(s));
    }
    v
}

/// Fan–Paar GF(2^64) coefficients.
#[allow(dead_code)]
fn fp64_coeffs() -> Vec<Elem<FanPaar64Field>> {
    let mut v = vec![
        Elem::<FanPaar64Field>::ZERO,
        Elem::<FanPaar64Field>::ONE,
        Elem::<FanPaar64Field>::from_raw(u64::MAX),
        Elem::<FanPaar64Field>::from_components([
            Elem::<FanPaar32Field>::ONE,
            Elem::<FanPaar32Field>::ZERO,
        ]),
        Elem::<FanPaar64Field>::from_raw(0x0000_0001_0000_0000),
        Elem::<FanPaar64Field>::from_components([
            Elem::<FanPaar32Field>::ZERO,
            Elem::<FanPaar32Field>::from_raw(0x0001_0000),
        ]),
        Elem::<FanPaar64Field>::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(Elem::<FanPaar64Field>::from_raw(s));
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
fn fp32_reference(dst: &mut [u8], coeff: Elem<FanPaar32Field>, src: &[u8]) {
    fp_mul_add(dst, u64::from(coeff.to_raw()), src, 32);
}
#[allow(dead_code)]
fn fp32_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar32Field>) {
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
#[allow(dead_code)]
fn fp32_into_reference(dst: &mut [u8], coeff: Elem<FanPaar32Field>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, u64::from(coeff.to_raw()), 32);
}
#[allow(dead_code)]
fn fp64_reference(dst: &mut [u8], coeff: Elem<FanPaar64Field>, src: &[u8]) {
    fp_mul_add(dst, coeff.to_raw(), src, 64);
}
#[allow(dead_code)]
fn fp64_assign_reference(dst: &mut [u8], coeff: Elem<FanPaar64Field>) {
    fp_mul_assign(dst, coeff.to_raw(), 64);
}
#[allow(dead_code)]
fn fp64_into_reference(dst: &mut [u8], coeff: Elem<FanPaar64Field>, src: &[u8]) {
    dst.copy_from_slice(src);
    fp_mul_assign(dst, coeff.to_raw(), 64);
}

/// GF(2^32) coefficients: the short-circuits, the extremes, pure tower
/// components, the tower constant, the generator, and a deterministic spread.
#[allow(dead_code)]
fn gf32_coeffs() -> Vec<Elem<Rijndael32Field>> {
    let mut v = vec![
        Elem::<Rijndael32Field>::ZERO,
        Elem::<Rijndael32Field>::ONE,
        Elem::<Rijndael32Field>::from_raw(u32::MAX),
        Elem::<Rijndael32Field>::from_raw(0x0000_ffff),
        Elem::<Rijndael32Field>::from_raw(0xffff_0000),
        // Pure base field and pure extension.
        Elem::<Rijndael32Field>::from_components([
            Elem::<Rijndael16Field>::ONE,
            Elem::<Rijndael16Field>::ZERO,
        ]),
        Elem::<Rijndael32Field>::from_components([
            Elem::<Rijndael16Field>::ZERO,
            Elem::<Rijndael16Field>::ONE,
        ]),
        // The tower constant in each component.
        Elem::<Rijndael32Field>::from_components([
            Elem::<Rijndael16Field>::from_raw(0x2000),
            Elem::<Rijndael16Field>::ZERO,
        ]),
        Elem::<Rijndael32Field>::from_components([
            Elem::<Rijndael16Field>::ZERO,
            Elem::<Rijndael16Field>::from_raw(0x2000),
        ]),
        Elem::<Rijndael32Field>::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(Elem::<Rijndael32Field>::from_raw(s));
    }
    v
}

/// GF(2^64) coefficients, the same shape one level up.
#[allow(dead_code)]
fn gf64_coeffs() -> Vec<Elem<Rijndael64Field>> {
    let mut v = vec![
        Elem::<Rijndael64Field>::ZERO,
        Elem::<Rijndael64Field>::ONE,
        Elem::<Rijndael64Field>::from_raw(u64::MAX),
        Elem::<Rijndael64Field>::from_raw(0x0000_0000_ffff_ffff),
        Elem::<Rijndael64Field>::from_raw(0xffff_ffff_0000_0000),
        Elem::<Rijndael64Field>::from_components([
            Elem::<Rijndael32Field>::ONE,
            Elem::<Rijndael32Field>::ZERO,
        ]),
        Elem::<Rijndael64Field>::from_components([
            Elem::<Rijndael32Field>::ZERO,
            Elem::<Rijndael32Field>::ONE,
        ]),
        Elem::<Rijndael64Field>::from_components([
            Elem::<Rijndael32Field>::from_raw(0x2000_0000),
            Elem::<Rijndael32Field>::ZERO,
        ]),
        Elem::<Rijndael64Field>::from_components([
            Elem::<Rijndael32Field>::ZERO,
            Elem::<Rijndael32Field>::from_raw(0x2000_0000),
        ]),
        Elem::<Rijndael64Field>::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(Elem::<Rijndael64Field>::from_raw(s));
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
fn gf32_reference(dst: &mut [u8], coeff: Elem<Rijndael32Field>, src: &[u8]) {
    scalar::mul_add::<Rijndael32Field>(dst, coeff, src);
}
#[allow(dead_code)]
fn gf32_assign_reference(dst: &mut [u8], coeff: Elem<Rijndael32Field>) {
    scalar::mul_assign::<Rijndael32Field>(dst, coeff);
}
#[allow(dead_code)]
fn gf32_into_reference(dst: &mut [u8], coeff: Elem<Rijndael32Field>, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Rijndael32Field>(dst, coeff);
}
#[allow(dead_code)]
fn gf64_reference(dst: &mut [u8], coeff: Elem<Rijndael64Field>, src: &[u8]) {
    scalar::mul_add::<Rijndael64Field>(dst, coeff, src);
}
#[allow(dead_code)]
fn gf64_assign_reference(dst: &mut [u8], coeff: Elem<Rijndael64Field>) {
    scalar::mul_assign::<Rijndael64Field>(dst, coeff);
}
#[allow(dead_code)]
fn gf64_into_reference(dst: &mut [u8], coeff: Elem<Rijndael64Field>, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Rijndael64Field>(dst, coeff);
}

// ---------------------------------------------------------------------------
// Backend-independent
// ---------------------------------------------------------------------------

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
            scalar::mul_add::<Rijndael16Field>(&mut want, coeff, &src);
            assert_eq!(got, want, "gf16 scalar: len {len}, coeff {coeff:?}");
        }
    }
}

/// Resolve raw GF(2^8) coefficients into the prepared form the blocked
/// entries take.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
fn prepared_coeffs<const POLY: u128>(
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
) -> Vec<Prepared> {
    coeffs.iter().map(|&c| Prepared::new(c)).collect()
}

/// A [`Matrix`](crate::kernel::Matrix) of prepared coefficients over raw
/// element terms, resolving each coefficient on construction: the provider
/// shape the dispatch layer hands the blocked matrix entries.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
struct PreparedMatrix<'a, const POLY: u128> {
    sets: Vec<Vec<Prepared>>,
    srcs: Vec<&'a [u8]>,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(dead_code)]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn prepared_matrix<'a, const POLY: u128>(
    terms: &[(&'a [Elem<FiniteField<2, 8, Polynomial<POLY>>>], &'a [u8])],
) -> PreparedMatrix<'a, POLY> {
    PreparedMatrix {
        sets: terms
            .iter()
            .map(|(coeffs, _)| prepared_coeffs(coeffs))
            .collect(),
        srcs: terms.iter().map(|&(_, src)| src).collect(),
    }
}

/// Canonical fixtures the prime drives share: source, base, and a second
/// elementwise operand, all canonicalized through the portable field so
/// both sides start from the canonical bytes the kernels are contracted
/// to produce.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn prime_fixtures<F: crate::field::FieldBuffer>(len: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let canon = |buf: &mut Vec<u8>| {
        let z = vec![0u8; buf.len()];
        crate::kernel::prime::add_assign::<F>(buf, &z);
    };
    let mut src = noise(len, 0xa1);
    canon(&mut src);
    let mut base = noise(len, 0xb2);
    canon(&mut base);
    let mut b2 = noise(len, 0xc3);
    canon(&mut b2);
    (src, base, b2)
}

/// Runs one prime entry's differential at every length: the entry writes
/// `got`, the portable reference writes `want`, both starting from `base`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn drive_prime_len<F: crate::field::FieldBuffer>(
    lens: &[usize],
    mut check: impl FnMut(&mut Vec<u8>, &mut Vec<u8>, &[u8], &[u8], &[u8]),
    label: &str,
    op_name: &str,
) {
    for &len in lens {
        let (src, base, b2) = prime_fixtures::<F>(len);
        let mut got = base.clone();
        let mut want = base.clone();
        check(&mut got, &mut want, &base, &src, &b2);
        assert_eq!(got, want, "{label} {op_name} len {len}");
    }
}

/// Runs one prime entry's coefficient differential at every length and
/// coefficient.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn drive_prime_scalar<F: crate::field::FieldBuffer, C: Copy + core::fmt::Debug>(
    lens: &[usize],
    coeffs: &[C],
    to_elem: impl Fn(C) -> Elem<F>,
    mut check: impl FnMut(&mut Vec<u8>, &mut Vec<u8>, C, &Elem<F>, &[u8]),
    label: &str,
    op_name: &str,
) {
    for &len in lens {
        let (src, base, _) = prime_fixtures::<F>(len);
        for &c in coeffs {
            let mut got = base.clone();
            let mut want = base.clone();
            check(&mut got, &mut want, c, &to_elem(c), &src);
            assert_eq!(got, want, "{label} {op_name} len {len} coeff {c:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage of dispatch-level contracts the differential drivers skip
// ---------------------------------------------------------------------------

#[test]
fn default_gather_kernels_zero_the_destination_on_empty_terms() {
    // The defaulted `KernelDispatch::mul_into_gather` (used by the prime and
    // Fan-Paar fields) zeroes the destination when there are no terms, and
    // the defaulted prepared `mul_into_gather_with` empty arm (the scalar
    // and Fan-Paar fields) does the same.
    let mut dst = noise(16, 0x5a);
    <Goldilocks as KernelDispatch>::mul_into_gather(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
    let mut dst = noise(16, 0x5b);
    <FanPaar32Field as KernelDispatch>::mul_into_gather(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
    let mut dst = noise(16, 0x66);
    <FanPaar8Field as KernelDispatch>::mul_into_gather_with(RawDispatch, &mut dst, &[], &[]);
    assert!(dst.iter().all(|&b| b == 0), "empty terms must zero dst");
}

/// The dispatch-level binary broadcast for every characteristic-two field —
/// including the wider fields that have no hand-written broadcast overrides
/// and reach the field-independent primitive through the trait default —
/// against the portable byte-cyclic reference, at element counts that
/// straddle the 16- and 32-byte lane boundaries.
fn check_binary_broadcast_dispatch<F: crate::field::FieldBuffer>(values: &[Elem<F>]) {
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
    check_binary_broadcast_dispatch::<FiniteField<2, 8, Polynomial<AES>>>(&[
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0),
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1),
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
    ]);
    check_binary_broadcast_dispatch::<FiniteField<2, 8, Polynomial<RS>>>(&[
        Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0),
        Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(1),
        Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0x53),
    ]);
    check_binary_broadcast_dispatch::<Rijndael16Field>(&[
        Elem::<Rijndael16Field>::from_raw(0),
        Elem::<Rijndael16Field>::from_raw(1),
        Elem::<Rijndael16Field>::from_raw(0x53a7),
    ]);
    check_binary_broadcast_dispatch::<Rijndael32Field>(&[
        Elem::<Rijndael32Field>::from_raw(0),
        Elem::<Rijndael32Field>::from_raw(1),
        Elem::<Rijndael32Field>::from_raw(0xdead_beef),
    ]);
    check_binary_broadcast_dispatch::<Rijndael64Field>(&[
        Elem::<Rijndael64Field>::from_raw(0),
        Elem::<Rijndael64Field>::from_raw(1),
        Elem::<Rijndael64Field>::from_raw(0x0123_4567_89ab_cdef),
    ]);
    check_binary_broadcast_dispatch::<FanPaar8Field>(&[
        Elem::<FanPaar8Field>::from_raw(0),
        Elem::<FanPaar8Field>::from_raw(1),
        Elem::<FanPaar8Field>::from_raw(0xa5),
    ]);
    check_binary_broadcast_dispatch::<FanPaar16Field>(&[
        Elem::<FanPaar16Field>::from_raw(0),
        Elem::<FanPaar16Field>::from_raw(1),
        Elem::<FanPaar16Field>::from_raw(0xa55a),
    ]);
    check_binary_broadcast_dispatch::<FanPaar32Field>(&[
        Elem::<FanPaar32Field>::from_raw(0),
        Elem::<FanPaar32Field>::from_raw(1),
        Elem::<FanPaar32Field>::from_raw(0xa55a_1234),
    ]);
    check_binary_broadcast_dispatch::<FanPaar64Field>(&[
        Elem::<FanPaar64Field>::from_raw(0),
        Elem::<FanPaar64Field>::from_raw(1),
        Elem::<FanPaar64Field>::from_raw(0xa55a_1234_dead_beef),
    ]);
}
