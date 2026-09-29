//! AVX-512 GFNI matrix updates over interleaved tower rows.
//!
//! Disjoint row slices carry the destination geometry through the tile loop.
//! The destination is loaded once per term block, and sub-lane tails descend
//! through the narrower token-proven kernel.

use crate::field::gf16::Elem;
use crate::kernel::Matrix;
use crate::kernel::tables::TowerCoeff;

use super::gfni::mul_add_gfni;
use super::wide::{scale512, swap512};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Accumulates every `(coefficients, source)` term into destination rows.
///
/// # Panics
/// Panics if row geometry overflows, a row span exceeds `rows`, or a term
/// does not provide the requested coefficients and source bytes.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix512(
    token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    super::check_elements("gf16::mul_add_matrix512", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix512: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix512: rows cannot hold requested rows"
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "gf16::mul_add_matrix512: coefficient count differs from rows"
        );
        assert_eq!(
            src.len(),
            row_len,
            "gf16::mul_add_matrix512: source length differs from row_len"
        );
    }
    mul_add_matrix512_with(token, rows, row_len, nrows, terms);
}

/// Accumulates a borrowed matrix provider into destination rows.
///
/// # Panics
/// Panics if the row length contains a partial element or the requested row
/// span overflows or exceeds `rows`.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix512_with<M: Matrix<Elem> + ?Sized>(
    token: archmage::X64V4xToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    super::check_elements("gf16::mul_add_matrix512_with", row_len);
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        return;
    }
    let used = nrows
        .checked_mul(row_len)
        .expect("gf16::mul_add_matrix512_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_matrix512_with: rows cannot hold requested rows"
    );
    let mut span = row_len;
    for term in 0..terms.len() {
        span = span.min(terms.source(term).len());
    }
    if span == 0 {
        return;
    }
    if span < 64 {
        return super::matrix::mul_add_matrix_gfni_with(
            token.v3_gfni_crypto(),
            rows,
            row_len,
            nrows,
            terms,
        );
    }
    let mut iter = rows[..used].chunks_exact_mut(row_len);
    let swap = swap512();
    let mut first = 0;
    while first + 4 <= nrows {
        let group: [&mut [u8]; 4] =
            core::array::from_fn(|_| iter.next().expect("checked row span"));
        matrix_group(token, group, first, span, terms, swap);
        first += 4;
    }
    if first + 2 <= nrows {
        let group: [&mut [u8]; 2] =
            core::array::from_fn(|_| iter.next().expect("checked row span"));
        matrix_group(token, group, first, span, terms, swap);
        first += 2;
    }
    if first < nrows {
        let group: [&mut [u8]; 1] = [iter.next().expect("checked row span")];
        matrix_group(token, group, first, span, terms, swap);
    }
}

#[archmage::rite(v4x, import_intrinsics)]
fn matrix_group<const N: usize, M: Matrix<Elem> + ?Sized>(
    token: archmage::X64V4xToken,
    mut rows: [&mut [u8]; N],
    first: usize,
    span: usize,
    terms: &M,
    swap: __m512i,
) {
    const TERM_BLOCK: usize = 16;
    let full = span / 64 * 64;
    for block in (0..terms.len()).step_by(TERM_BLOCK) {
        let count = (terms.len() - block).min(TERM_BLOCK);
        let mut factors: [[(__m512i, __m512i); N]; TERM_BLOCK] =
            core::array::from_fn(|_| [(_mm512_setzero_si512(), _mm512_setzero_si512()); N]);
        for (term, group) in factors.iter_mut().enumerate().take(count) {
            for (k, pair) in group.iter_mut().enumerate() {
                let (same, cross) = super::broadcast_words(TowerCoeff::new(
                    *terms.coefficient(block + term, first + k),
                ));
                // One broadcast per term and row for the whole block: the
                // offset loop reuses these instead of re-issuing them for
                // every 64-byte lane.
                *pair = (_mm512_set1_epi16(same), _mm512_set1_epi16(cross));
            }
        }
        for offset in (0..full).step_by(64) {
            let mut acc: [__m512i; N] = core::array::from_fn(|k| {
                let lane: &[u8; 64] = rows[k][offset..offset + 64]
                    .try_into()
                    .expect("complete destination lane");
                _mm512_loadu_si512(lane)
            });
            for (term, group) in factors.iter().enumerate().take(count) {
                let source: &[u8; 64] = terms.source(block + term)[offset..offset + 64]
                    .try_into()
                    .expect("complete source lane");
                let x = _mm512_loadu_si512(source);
                let swapped = _mm512_shuffle_epi8(x, swap);
                for (value, &(same, cross)) in acc.iter_mut().zip(group) {
                    *value = _mm512_xor_si512(*value, scale512(x, swapped, same, cross));
                }
            }
            for (row, value) in rows.iter_mut().zip(acc) {
                let lane: &mut [u8; 64] = (&mut row[offset..offset + 64])
                    .try_into()
                    .expect("complete destination lane");
                _mm512_storeu_si512(lane, value);
            }
        }
        for (k, row) in rows.iter_mut().enumerate() {
            for term in block..block + count {
                mul_add_gfni(
                    token.v3_gfni_crypto(),
                    &mut row[full..span],
                    TowerCoeff::new(*terms.coefficient(term, first + k)),
                    &terms.source(term)[full..span],
                );
            }
        }
    }
}
