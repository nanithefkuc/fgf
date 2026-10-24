//! Delayed-reduction matrix AVX2 kernels for four-byte residue rows.
//!
//! A shared-source tile of up to four outputs over four-byte rows. Each
//! source lane normalizes once, then multiplies by each output's canonical
//! coefficient into that output's pair of 64-bit accumulator vectors
//! without reduction for up to `K` sources; each group of `K` sources
//! reduces through narrow Barrett and folds into a running canonical value
//! modulo `P`. The shared helpers (`normalize_avx2`, `narrow_avx2`,
//! `combine_halves_avx2`, `widen_add_avx2`) live in the parent module.
//!
//! The delay bound: prepared coefficients resolve to canonical `Elem`
//! values, so each product sits below `(P-1)^2` and after `K` sources each
//! accumulator is below `K * (P-1)^2 < K * P * P <= 2^32 * P`, which is
//! the narrow-Barrett input bound and keeps the summed 64-bit lanes below
//! `2^64`.
//!
//! Entries are safe capability-token functions taking an `X64V3Token` and
//! validating their own geometry; reference loads/stores express every
//! lane, and scalar `u64` tails handle the sub-vector remainders.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{combine_halves_avx2, narrow_avx2, normalize_avx2, widen_add_avx2};
#[cfg(test)]
use crate::kernel::Backend;
use crate::kernel::Matrix;
use crate::kernel::proven_checks::check_equal;
use crate::kernel::residue::PreparedResidue;
#[cfg(test)]
use crate::kernel::trace::{Substage, Unit, emit};
use crate::kernel::x86::residue::SourceGroup;

/// Up to four outputs per delayed tile.
const TILE: usize = 4;

/// Check the shared geometry: `nrows` destination rows of `row_len` bytes
/// in `rows`, whole four-byte lanes.
fn check_geometry_matrix(name: &str, rows_bytes: usize, row_len: usize, nrows: usize) {
    assert!(
        row_len.is_multiple_of(4),
        "{name}: row lane width is 4 bytes but the row is {row_len} bytes"
    );
    let used = nrows
        .checked_mul(row_len)
        .unwrap_or_else(|| panic!("{name}: row geometry overflows"));
    assert!(
        rows_bytes >= used,
        "{name}: rows is {rows_bytes} bytes but {nrows} rows of {row_len} bytes need {used}"
    );
}

/// Check every term source against the row length: `terms.len()` sources
/// of exactly `row_len` bytes each.
fn check_terms<M: Matrix<PreparedResidue> + ?Sized>(name: &str, row_len: usize, terms: &M) {
    for term in 0..terms.len() {
        check_equal(
            name,
            "term source",
            terms.source(term).len(),
            "row_len",
            row_len,
        );
    }
}

/// Fold one group of `terms.len()` sources at column `offset` into the
/// running canonical sums: normalize each source once, widen-multiply by
/// each output's canonical coefficient into 64-bit accumulators without
/// reduction, then reduce through `narrow` and fold with `widen_add`.
///
/// The caller guarantees `terms.len() <= k` and the delay bound keeps
/// every accumulator below `2^32 * P`.
///
/// Coefficients resolve through `terms` as canonical low halves of the
/// prepared Shoup pairs.
#[allow(clippy::too_many_arguments)]
#[archmage::rite(v3, import_intrinsics)]
fn tile_group<M: Matrix<PreparedResidue> + ?Sized>(
    running: &mut [__m256i; TILE],
    terms: &M,
    group: usize,
    count: usize,
    offset: usize,
    norm_mu: __m256i,
    p32: __m256i,
    barrett_mu: __m256i,
    p64: __m256i,
    p64_m1: __m256i,
    shift: i32,
    #[cfg(test)] id: &'static str,
    #[cfg(test)] substage: Substage,
) {
    // One even/odd pair of 64-bit accumulator vectors per output.
    let mut acc_e = [_mm256_setzero_si256(); TILE];
    let mut acc_o = [_mm256_setzero_si256(); TILE];
    for t in 0..terms.len() {
        let column: &[u8; 32] = terms.source(t)[offset..offset + 32]
            .try_into()
            .expect("residue delayed: column lies inside its source");
        let v = _mm256_loadu_si256(column);
        let x = normalize_avx2(v, norm_mu, p32);
        let xe = _mm256_mul_epu32(x, _mm256_set1_epi64x(1));
        let xo = _mm256_mul_epu32(_mm256_shuffle_epi32(x, 0xB1), _mm256_set1_epi64x(1));
        for (slot, (even, odd)) in acc_e
            .iter_mut()
            .zip(acc_o.iter_mut())
            .take(count)
            .enumerate()
        {
            let c = _mm256_set1_epi64x(i64::from(
                (terms.coefficient(t, group + slot).word & 0xFFFF_FFFF) as u32,
            ));
            *even = _mm256_add_epi64(*even, _mm256_mul_epu32(xe, c));
            *odd = _mm256_add_epi64(*odd, _mm256_mul_epu32(xo, c));
        }
    }
    for (slot, sum) in running.iter_mut().take(count).enumerate() {
        let reduced = combine_halves_avx2(
            narrow_avx2(acc_e[slot], barrett_mu, p64, shift),
            narrow_avx2(acc_o[slot], barrett_mu, p64, shift),
        );
        *sum = widen_add_avx2(*sum, reduced, p64, p64_m1);
    }
    #[cfg(test)]
    emit(
        id,
        substage,
        Unit::Elements,
        count * 8 * terms.len(),
        Backend::V3,
    );
}

/// Many sources overwrite many four-byte rows with delayed reduction:
/// `rows[j] = sum_t coeffs[t][j] * src[t]` for the tile's outputs, AVX2.
///
/// Up to `K = floor(2^32 / P)` sources accumulate into 64-bit lanes
/// before each `narrow` reduction; the delay bound keeps every group sum
/// below `2^32 * P` and every 64-bit lane below `2^64`. Source lanes may
/// hold any 32-bit pattern; every stored lane is canonical. The depth
/// derives from `p` inside the kernel; callers pass no depth.
///
/// # Panics
///
/// Panics if `p` is even, if `p` is 1 or less, if the derived depth is
/// below two (any odd `p` above `2^31`), if any coefficient's low half
/// reaches `p`, if `rows` does not hold `nrows` rows of `row_len` bytes,
/// if any term source is not `row_len` bytes, or if any buffer holds a
/// partial four-byte lane.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::used_underscore_binding
)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix_u32_avx2_with<M: Matrix<PreparedResidue> + ?Sized>(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
    p: u32,
) {
    run_delayed::<M, false>(
        "residue::mul_into_matrix_u32_avx2_with",
        rows,
        row_len,
        nrows,
        terms,
        p,
    );
}

/// Many sources accumulate into many four-byte rows with delayed
/// reduction: `rows[j] += sum_t coeffs[t][j] * src[t]`, AVX2.
///
/// The raw destination normalizes once per column, then joins the delayed
/// sums exactly as the overwrite form; the output is canonical. The depth
/// derives from `p` inside the kernel; callers pass no depth.
///
/// # Panics
///
/// Panics if `p` is even, if `p` is 1 or less, if the derived depth is
/// below two (any odd `p` above `2^31`), if any coefficient's low half
/// reaches `p`, if `rows` does not hold `nrows` rows of `row_len` bytes,
/// if any term source is not `row_len` bytes, or if any buffer holds a
/// partial four-byte lane.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::used_underscore_binding
)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_u32_avx2_with<M: Matrix<PreparedResidue> + ?Sized>(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
    p: u32,
) {
    run_delayed::<M, true>(
        "residue::mul_add_matrix_u32_avx2_with",
        rows,
        row_len,
        nrows,
        terms,
        p,
    );
}

/// Shared tile driver: groups of up to four outputs, each column folded
/// group by group with at most `K` sources per reduction, vector body
/// then scalar tails, overwrite or accumulation per `ADD`.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
#[archmage::rite(v3, import_intrinsics)]
fn run_delayed<M: Matrix<PreparedResidue> + ?Sized, const ADD: bool>(
    name: &str,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
    p: u32,
) {
    check_geometry_matrix(name, rows.len(), row_len, nrows);
    check_terms(name, row_len, terms);
    assert!(
        p > 1,
        "{name}: modulus {p} is not a delayed-reduction modulus"
    );
    assert!(p % 2 == 1, "residue delayed: modulus {p} is even");
    // The delay depth derives from the modulus: any caller-supplied depth
    // above `floor(2^32 / P)` would silently break the narrow bound.
    #[allow(clippy::cast_possible_truncation)]
    let k = ((1u128 << 32) / u128::from(p)) as u32;
    assert!(
        k >= 2,
        "{name}: modulus {p} has delay depth {k} but delayed reduction needs at least 2"
    );
    // The delay bound holds only for canonical coefficients: each product
    // must sit below `(P-1)^2`, so every prepared low half stays below `p`.
    for t in 0..terms.len() {
        for row in 0..nrows {
            let c = (terms.coefficient(t, row).word & 0xFFFF_FFFF) as u32;
            assert!(
                c < p,
                "residue delayed: prepared coefficient {c} reaches modulus {p}"
            );
        }
    }
    if row_len == 0 || nrows == 0 || terms.len() == 0 {
        if !ADD {
            let used = nrows.saturating_mul(row_len).min(rows.len());
            rows[..used].fill(0);
        }
        return;
    }
    let k = (k as usize).min(terms.len()).max(1);
    let p64 = _mm256_set1_epi64x(i64::from(p));
    let p64_m1 = _mm256_set1_epi64x(i64::from(p.wrapping_sub(1)));
    let p32 = _mm256_set1_epi32(p as i32);
    #[allow(clippy::cast_possible_truncation)]
    let norm_mu = _mm256_set1_epi64x(u64::from(((1u128 << 32) / u128::from(p)) as u32) as i64);
    let shift = p.ilog2();
    let s = shift + 32;
    #[allow(clippy::cast_possible_truncation)]
    let barrett_mu = _mm256_set1_epi64x(u64::from(((1u128 << s) / u128::from(p)) as u32) as i64);
    let vector_len = row_len & !31;
    let modulus = u64::from(p);
    let mut group = 0;
    while group < nrows {
        let count = (nrows - group).min(TILE);
        let mut offset = 0;
        while offset < vector_len {
            let mut running = [_mm256_setzero_si256(); TILE];
            if ADD {
                for (slot, sum) in running.iter_mut().take(count).enumerate() {
                    let lane: &[u8; 32] = rows[(group + slot) * row_len + offset..][..32]
                        .try_into()
                        .expect("residue delayed: column lies inside its row");
                    *sum = normalize_avx2(_mm256_loadu_si256(lane), norm_mu, p32);
                }
            }
            let mut base = 0;
            while base < terms.len() {
                let len = (terms.len() - base).min(k);
                let window = SourceGroup { terms, base, len };
                tile_group(
                    &mut running,
                    &window,
                    group,
                    count,
                    offset,
                    norm_mu,
                    p32,
                    barrett_mu,
                    p64,
                    p64_m1,
                    shift as i32,
                    #[cfg(test)]
                    if ADD {
                        "residue.avx2.mul_add_matrix"
                    } else {
                        "residue.avx2.mul_into_matrix"
                    },
                    #[cfg(test)]
                    if ADD { Substage::MulAdd } else { Substage::Mul },
                );
                base += len;
            }
            for (slot, sum) in running.iter().take(count).enumerate() {
                let row = &mut rows[(group + slot) * row_len..(group + slot + 1) * row_len];
                let lane: &mut [u8; 32] = (&mut row[offset..offset + 32])
                    .try_into()
                    .expect("residue delayed: column lies inside its row");
                _mm256_storeu_si256(lane, *sum);
            }
            offset += 32;
        }
        // Scalar tails: whole lanes past the vector body, reduced per
        // group of `K` sources so the `u64` running sum keeps the same
        // delay bound as the vector lanes.
        for (slot, row) in rows
            .chunks_exact_mut(row_len)
            .skip(group)
            .take(count)
            .enumerate()
        {
            // Each tail lane accumulates its own `u64` sum, reduced per
            // group of `K` sources so the scalar lanes keep the same
            // delay bound as the vector lanes. The tail holds fewer than
            // eight lanes, so a fixed stack array keeps the kernel
            // allocation-free.
            let tail_lanes = row[vector_len..].len() / 4;
            let mut lane_sums = [0u64; 7];
            for (lane_sum, lane) in lane_sums.iter_mut().zip(row[vector_len..].chunks_exact(4)) {
                *lane_sum = if ADD {
                    u64::from(u32::from_le_bytes(lane.try_into().unwrap())) % modulus
                } else {
                    0
                };
            }
            let mut base = 0;
            while base < terms.len() {
                let len = (terms.len() - base).min(k);
                for t in base..base + len {
                    let c =
                        u64::from((terms.coefficient(t, group + slot).word & 0xFFFF_FFFF) as u32);
                    for (lane_sum, lane) in lane_sums[..tail_lanes]
                        .iter_mut()
                        .zip(terms.source(t)[vector_len..].chunks_exact(4))
                    {
                        let x = u64::from(u32::from_le_bytes(lane.try_into().unwrap())) % modulus;
                        *lane_sum += x * c;
                    }
                }
                for lane_sum in &mut lane_sums[..tail_lanes] {
                    *lane_sum %= modulus;
                }
                base += len;
            }
            #[allow(clippy::cast_possible_truncation)]
            for (lane, lane_sum) in row[vector_len..].chunks_exact_mut(4).zip(lane_sums.iter()) {
                lane.copy_from_slice(&(*lane_sum as u32).to_le_bytes());
            }
        }
        group += count;
    }
}
