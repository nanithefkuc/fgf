//! Tile-width and store-policy contrasts against the production tiles.
//!
//! Measurement scaffolding. Each body keeps the production multiply,
//! resolution and remainder and varies exactly one knob: how many 32-byte
//! lanes a main-loop iteration folds, and whether the destination stores are
//! non-temporal. The entries are safe [`archmage`] capability-token
//! functions; the bodies keep the non-temporal store and offset-row residue.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::rows::{matrix_tail, rows_resolved};
use super::super::{Affine8D, Blocked, Gfni, bfactor, bmul};
use crate::field::gf8b::Elem;
use crate::field::gf8d;

/// Experimental two-row overwrite matrix (`p2` erasure encode) with a tunable
/// main-tile width and store policy, for profiling against production's
/// four-lane temporal `matrix_rows2`.
///
/// `lanes` is the number of 32-byte lanes per row folded per main-loop
/// iteration (`2` = 64-byte "2-way" tile, `3` = ISA-L's 96-byte schedule,
/// `4` = production's 128-byte tile); `nt` selects non-temporal
/// (`vmovntdq`) destination stores. Benchmark evidence only; production
/// dispatch is unchanged.
///
/// # Panics
/// Panics unless `rows` holds two `row_len`-byte rows, each term supplies two
/// coefficients over a `row_len`-byte source, and `lanes` is `2` to `4`.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix2_8b(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[Elem], &[u8])],
    nt: bool,
    lanes: usize,
) {
    mul_into_matrix2_checked::<Gfni>(rows, row_len, terms, nt, lanes);
}

/// [`mul_into_matrix2_8b`] under the `Gf8D` polynomial `0x11D`.
///
/// # Panics
/// As [`mul_into_matrix2_8b`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix2_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
    nt: bool,
    lanes: usize,
) {
    mul_into_matrix2_checked::<Affine8D>(rows, row_len, terms, nt, lanes);
}

/// Shared geometry check and tile-width dispatch for the two-row contrasts.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_into_matrix2_checked<S: Blocked>(
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[S::Coeff], &[u8])],
    nt: bool,
    lanes: usize,
) {
    assert!(
        2usize
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "matrix_overwrite2: rows buffer does not hold two rows of {row_len} bytes"
    );
    for (coeffs, src) in terms {
        assert_eq!(src.len(), row_len);
        assert!(
            coeffs.len() >= 2,
            "matrix_overwrite2: term needs two coefficients"
        );
    }
    if nt {
        debug_assert_eq!(
            rows.as_ptr() as usize % 32,
            0,
            "matrix_overwrite2: non-temporal stores need a 32-byte-aligned buffer"
        );
        debug_assert_eq!(
            row_len % 32,
            0,
            "matrix_overwrite2: non-temporal stores need a 32-byte row"
        );
    }
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the asserts above bound two `row_len`-byte rows inside `rows`,
    //        and every term covers two coefficients over a `row_len`-byte
    //        source.
    // THUS: the staged row pointers address whole, in-bounds rows and every
    //       source load stays inside its slice.
    //
    // ALIASING
    // SINCE: the two rows sit `row_len` apart in one region, and `2 *
    //        row_len <= rows.len()` was asserted.
    // THUS: the two row windows are disjoint.
    unsafe {
        let ptr0 = base;
        let ptr1 = base.add(row_len);
        match (nt, lanes) {
            (false, 2) => matrix2_body::<S, false, 2>(ptr0, ptr1, row_len, terms),
            (false, 3) => matrix2_body::<S, false, 3>(ptr0, ptr1, row_len, terms),
            (false, 4) => matrix2_body::<S, false, 4>(ptr0, ptr1, row_len, terms),
            (true, 2) => matrix2_body::<S, true, 2>(ptr0, ptr1, row_len, terms),
            (true, 3) => matrix2_body::<S, true, 3>(ptr0, ptr1, row_len, terms),
            (true, 4) => matrix2_body::<S, true, 4>(ptr0, ptr1, row_len, terms),
            _ => panic!("matrix_overwrite2: lanes must be 2, 3, or 4"),
        }
    }
}

/// Two-row overwrite body: `LANES` 32-byte lanes per row per iteration, seeded
/// from zero, `NT`-selected stores.
///
/// Residue: the two rows are runtime-offset windows of one region bounded by
/// [`mul_into_matrix2_checked`], and the tile stores run through the
/// streaming-store primitive, whose alignment [`mul_into_matrix2_checked`]
/// debug-asserts under `NT`.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn matrix2_body<S: Blocked, const NT: bool, const LANES: usize>(
    ptr0: *mut u8,
    ptr1: *mut u8,
    row_len: usize,
    terms: &[(&[S::Coeff], &[u8])],
) {
    let step = 32 * LANES;
    let mut tile = 0;
    while tile + step <= row_len {
        let mut a0 = [_mm256_setzero_si256(); LANES];
        let mut a1 = [_mm256_setzero_si256(); LANES];
        for (coeffs, src) in terms {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: every source is `row_len` bytes, so `tile + step <=
            //        row_len` bounds each of the `LANES` loads.
            // THUS: the source loads stay inside the term's source slice.
            unsafe {
                let f0 = bfactor::<S>(coeffs[0]);
                let f1 = bfactor::<S>(coeffs[1]);
                let sp = src.as_ptr().add(tile);
                for l in 0..LANES {
                    let x = core::arch::x86_64::_mm256_loadu_si256(sp.add(32 * l).cast());
                    a0[l] = _mm256_xor_si256(a0[l], bmul::<S>(x, f0));
                    a1[l] = _mm256_xor_si256(a1[l], bmul::<S>(x, f1));
                }
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + step <= row_len` bounds the same offsets on both
        //        rows, each of which spans `row_len` bytes.
        // THUS: the stores stay inside their originating rows.
        //
        // NON-TEMPORAL STORE ALIGNMENT
        // SINCE: `NT` was selected only after the checked layer
        //        debug-asserted a 32-byte-aligned buffer and a 32-byte
        //        `row_len`, so every lane offset is 32-byte aligned.
        // THUS: each streaming store address is 32-byte aligned.
        //
        // ALIASING
        // SINCE: the two rows are disjoint windows staged by the checked
        //        layer.
        // THUS: the stores conflict with no other live row window.
        unsafe {
            for l in 0..LANES {
                crate::kernel::x86::store256::<NT>(ptr0.add(tile + 32 * l), a0[l]);
                crate::kernel::x86::store256::<NT>(ptr1.add(tile + 32 * l), a1[l]);
            }
        }
        tile += step;
    }
    while tile + 32 <= row_len {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        for (coeffs, src) in terms {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 32 <= row_len == src.len()` bounds the load.
            // THUS: the source load stays inside the term's source slice.
            unsafe {
                let f0 = bfactor::<S>(coeffs[0]);
                let f1 = bfactor::<S>(coeffs[1]);
                let x = core::arch::x86_64::_mm256_loadu_si256(src.as_ptr().add(tile).cast());
                a0 = _mm256_xor_si256(a0, bmul::<S>(x, f0));
                a1 = _mm256_xor_si256(a1, bmul::<S>(x, f1));
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 32 <= row_len`, and the two rows are disjoint.
        // THUS: the cleanup stores stay inside their originating rows.
        unsafe {
            core::arch::x86_64::_mm256_storeu_si256(ptr0.add(tile).cast(), a0);
            core::arch::x86_64::_mm256_storeu_si256(ptr1.add(tile).cast(), a1);
        }
        tile += 32;
    }
    // Distinct in-bounds rows; every term covers coefficient index 1. The
    // `sfence` orders the non-temporal stores before any later observer.
    matrix_tail::<S, [(&[S::Coeff], &[u8])], true>(&[ptr0, ptr1], row_len, 0, tile, terms);
    if NT {
        _mm_sfence();
    }
}

/// Experimental one-row overwrite matrix on the production resolved path with
/// a selectable main-tile width.
///
/// `lanes` 4 is production's 128-byte tile, `lanes` 3 is ISA-L's 96-byte
/// one-output tile. Same coefficient resolution, same remainder, same public
/// geometry checks: only the tile width differs, so the two measure the width
/// alone. Benchmark evidence only; production dispatch is unchanged.
///
/// # Panics
/// Panics unless `rows` holds one `row_len`-byte row, every term supplies at
/// least one coefficient over a `row_len`-byte source, and `lanes` is 3 or 4.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix1_tile_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
    lanes: usize,
) {
    assert!(
        row_len <= rows.len(),
        "matrix_overwrite1_tile: rows buffer does not hold one row of {row_len} bytes"
    );
    for (coeffs, src) in terms {
        assert_eq!(src.len(), row_len);
        assert!(
            !coeffs.is_empty(),
            "matrix_overwrite1_tile: term needs one coefficient"
        );
    }
    let ptr = rows.as_mut_ptr();
    // One in-bounds row of `row_len` bytes, as asserted above; every term
    // covers one coefficient over a `row_len`-byte source.
    match lanes {
        3 => rows_resolved::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 3, true>(
            [ptr],
            row_len,
            0,
            terms,
        ),
        4 => rows_resolved::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 4, true>(
            [ptr],
            row_len,
            0,
            terms,
        ),
        _ => panic!("matrix_overwrite1_tile: lanes must be 3 or 4"),
    }
}
