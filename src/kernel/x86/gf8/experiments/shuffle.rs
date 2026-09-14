//! Nibble-shuffle alternatives to the GFNI multiply in the multi-row shapes.
//!
//! Measurement scaffolding. These bodies ask whether spreading the multiply
//! across the two `PSHUFB` ports beats the single-port `GF2P8MULB` ceiling at
//! six and two outputs, including the prepacked 32-byte coefficient records
//! ISA-L's `gftbls` contract uses.
//!
//! The entries are safe [`archmage`] capability-token functions; the row
//! windows and the packed-table pointer walk keep the offset-addressed-row
//! residue in [`archmage::rite`] helpers.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::super::{Affine8D, Blocked, Gfni, brem};
use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::Matrix;

/// Experimental six-output overwrite matrix using AVX2 nibble shuffles for
/// `Gf8B`, even on a GFNI host.
///
/// This mirrors ISA-L's `gf_6vect_dot_prod_avx2`: one source load feeds six
/// output accumulators. It is benchmark evidence only; production dispatch is
/// unchanged.
///
/// # Panics
/// Panics unless `rows` holds six `row_len`-byte rows and each term supplies
/// six coefficients over a `row_len`-byte source.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix6_shuffle_8b(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[Elem], &[u8])],
) {
    mul_into_matrix6_shuffle_checked::<Gfni>(rows, row_len, terms);
}

/// [`mul_into_matrix6_shuffle_8b`] under the `Gf8D` polynomial `0x11D`.
///
/// # Panics
/// As [`mul_into_matrix6_shuffle_8b`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix6_shuffle_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
) {
    mul_into_matrix6_shuffle_checked::<Affine8D>(rows, row_len, terms);
}

/// Shared geometry check for the six-row shuffle contrasts.
#[archmage::rite(v3_gfni_crypto)]
fn mul_into_matrix6_shuffle_checked<S: Blocked>(
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[S::Coeff], &[u8])],
) {
    assert!(
        row_len
            .checked_mul(6)
            .is_some_and(|needed| needed <= rows.len()),
        "six-row shuffle matrix: rows buffer does not hold six rows of {row_len} bytes"
    );
    for &(coeffs, src) in terms {
        assert_eq!(coeffs.len(), 6);
        assert_eq!(src.len(), row_len);
    }
    mul_into_matrix6_shuffle_impl::<S, _>(rows, row_len, terms);
}

/// One source load feeding six output accumulators through `PSHUFB` lookups.
#[inline]
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn nibble_product<S: Blocked>(lo: __m256i, hi: __m256i, coeff: S::Coeff) -> __m256i {
    let table = S::table(coeff);
    // Each table half is exactly one 16-byte lookup bank.
    let lo_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
    let hi_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo_table, lo),
        _mm256_shuffle_epi8(hi_table, hi),
    )
}

/// Six-row shuffle body.
///
/// Residue: the six rows are runtime-offset windows of one region, bounded by
/// [`mul_into_matrix6_shuffle_checked`]; coefficient rows contain six entries,
/// checked there term by term.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_into_matrix6_shuffle_impl<S: Blocked, M: Matrix<S::Coeff> + ?Sized>(
    rows: &mut [u8],
    row_len: usize,
    terms: &M,
) {
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checked layer proved `6 * row_len <= rows.len()`, so all six
    //        staged pointers address whole in-bounds rows.
    // THUS: every row window access stays inside `rows`.
    //
    // ALIASING
    // SINCE: the six rows sit `row_len` apart in one region.
    // THUS: the row windows are pairwise disjoint.
    let ptrs = unsafe {
        [
            base,
            base.add(row_len),
            base.add(row_len * 2),
            base.add(row_len * 3),
            base.add(row_len * 4),
            base.add(row_len * 5),
        ]
    };
    let mask = _mm256_set1_epi8(0x0f);
    let len = row_len & !31;
    let mut offset = 0;
    while offset < len {
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        for term in 0..terms.len() {
            let src = terms.source(term);
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the checked layer proved every source is `row_len`
            //        bytes, and `offset + 32 <= len <= row_len`.
            // THUS: the source load stays inside the term's source slice.
            let x =
                unsafe { core::arch::x86_64::_mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
            let lo = _mm256_and_si256(x, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
            a0 = _mm256_xor_si256(a0, nibble_product::<S>(lo, hi, *terms.coefficient(term, 0)));
            a1 = _mm256_xor_si256(a1, nibble_product::<S>(lo, hi, *terms.coefficient(term, 1)));
            a2 = _mm256_xor_si256(a2, nibble_product::<S>(lo, hi, *terms.coefficient(term, 2)));
            a3 = _mm256_xor_si256(a3, nibble_product::<S>(lo, hi, *terms.coefficient(term, 3)));
            a4 = _mm256_xor_si256(a4, nibble_product::<S>(lo, hi, *terms.coefficient(term, 4)));
            a5 = _mm256_xor_si256(a5, nibble_product::<S>(lo, hi, *terms.coefficient(term, 5)));
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= len <= row_len`, the span of each of the six
        //        checked rows.
        // THUS: every store stays inside its originating row.
        //
        // ALIASING
        // SINCE: the six row windows are disjoint, as staged.
        // THUS: no store conflicts with another row's window.
        unsafe {
            core::arch::x86_64::_mm256_storeu_si256(ptrs[0].add(offset).cast(), a0);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[1].add(offset).cast(), a1);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[2].add(offset).cast(), a2);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[3].add(offset).cast(), a3);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[4].add(offset).cast(), a4);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[5].add(offset).cast(), a5);
        }
        offset += 32;
    }
    if offset == row_len {
        return;
    }
    let remaining = row_len - offset;
    for (row_index, &ptr) in ptrs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + remaining == row_len`, the span of each checked
        //        row.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the rows are disjoint and only one tail borrow is live at a
        //        time.
        // THUS: the mutable borrow conflicts with no live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(offset), remaining) };
        tail.fill(0);
        for term in 0..terms.len() {
            let coeff = *terms.coefficient(term, row_index);
            if S::byte(coeff) != 0 {
                // Equal remainder lengths; the checked layer bounded every
                // source at `row_len` bytes.
                brem::<S>(tail, coeff, &terms.source(term)[offset..]);
            }
        }
    }
}

/// Experimental six-output shuffle kernel over prepacked 32-byte coefficient
/// records for `Gf8B`.
///
/// `tables[term * 6 + row]` is `[lo_table, hi_table]`. Preparation and matrix
/// geometry are outside the hot loop, matching ISA-L's `gftbls` contract.
///
/// # Panics
/// Panics unless `rows` holds six rows, `tables.len() == srcs.len() * 6`, and
/// every source is `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix6_shuffle_packed_8b(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
) {
    mul_into_matrix6_shuffle_packed_checked(rows, row_len, tables, srcs);
}

/// [`mul_into_matrix6_shuffle_packed_8b`] under `Gf8D` polynomial `0x11D`.
///
/// # Panics
/// As [`mul_into_matrix6_shuffle_packed_8b`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix6_shuffle_packed_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
) {
    mul_into_matrix6_shuffle_packed_checked(rows, row_len, tables, srcs);
}

/// Shared geometry check for the packed six-row shuffle contrasts.
#[archmage::rite(v3_gfni_crypto)]
fn mul_into_matrix6_shuffle_packed_checked(
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
) {
    assert!(
        row_len
            .checked_mul(6)
            .is_some_and(|needed| needed <= rows.len()),
        "packed six-row shuffle matrix: rows buffer does not hold six rows of {row_len} bytes"
    );
    assert_eq!(tables.len(), srcs.len() * 6);
    for &src in srcs {
        assert_eq!(src.len(), row_len);
    }
    mul_into_matrix6_shuffle_packed_impl(rows, row_len, tables, srcs);
}

/// One packed-record product: both table halves broadcast from one record.
///
/// Residue: the record pointer is a runtime offset into the packed table
/// slice, bounded by the checked layer's `tables.len() == srcs.len() * 6`.
#[inline]
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn packed_nibble_product(lo: __m256i, hi: __m256i, table: *const [u8; 32]) -> __m256i {
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checked wrapper bounds every 32-byte table record, and the
    //        pointer is `tables.as_ptr()` advanced by whole records.
    // THUS: both 16-byte loads stay inside the packed table slice.
    let (lo_table, hi_table) = unsafe {
        let bytes = table.cast::<u8>();
        (
            _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(bytes.cast())),
            _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(bytes.add(16).cast())),
        )
    };
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo_table, lo),
        _mm256_shuffle_epi8(hi_table, hi),
    )
}

/// Packed six-row shuffle body.
///
/// Residue: the rows are runtime-offset windows of one region and the
/// source/table walk is pointer-cached to keep the measured term loop free of
/// bounds checks; the checked layer bounds every row, source, and table
/// record before this body runs.
#[allow(unsafe_code)]
#[archmage::rite(v3_gfni_crypto)]
fn mul_into_matrix6_shuffle_packed_impl(
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
) {
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checked wrapper proved `6 * row_len <= rows.len()`, so all
    //        six staged pointers address whole in-bounds rows.
    // THUS: every row window access stays inside `rows`.
    //
    // ALIASING
    // SINCE: the six rows sit `row_len` apart in one region.
    // THUS: the row windows are pairwise disjoint.
    let ptrs = unsafe {
        [
            base,
            base.add(row_len),
            base.add(row_len * 2),
            base.add(row_len * 3),
            base.add(row_len * 4),
            base.add(row_len * 5),
        ]
    };
    let mask = _mm256_set1_epi8(0x0f);
    let len = row_len & !31;
    let table_ptr = tables.as_ptr();
    let source_ptr = srcs.as_ptr();
    let mut offset = 0;
    while offset < len {
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        for term in 0..srcs.len() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `term < srcs.len()` and the checked wrapper bounds six
            //        packed table records for every term.
            // THUS: the source reference and the table record pointer stay
            //       inside their slices.
            let (src, term_tables) = unsafe { (*source_ptr.add(term), table_ptr.add(term * 6)) };
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the checked wrapper proved `src.len() == row_len` and
            //        `offset + 32 <= len <= row_len`.
            // THUS: the source load stays inside the term's source slice.
            let x =
                unsafe { core::arch::x86_64::_mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
            let lo = _mm256_and_si256(x, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: all six packed table records for this term are in
            //        bounds, per the checked wrapper.
            // THUS: every record read stays inside the packed table slice.
            unsafe {
                a0 = _mm256_xor_si256(a0, packed_nibble_product(lo, hi, term_tables));
                a1 = _mm256_xor_si256(a1, packed_nibble_product(lo, hi, term_tables.add(1)));
                a2 = _mm256_xor_si256(a2, packed_nibble_product(lo, hi, term_tables.add(2)));
                a3 = _mm256_xor_si256(a3, packed_nibble_product(lo, hi, term_tables.add(3)));
                a4 = _mm256_xor_si256(a4, packed_nibble_product(lo, hi, term_tables.add(4)));
                a5 = _mm256_xor_si256(a5, packed_nibble_product(lo, hi, term_tables.add(5)));
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= len <= row_len`, the span of each of the six
        //        checked rows.
        // THUS: every store stays inside its originating row.
        //
        // ALIASING
        // SINCE: the six row windows are disjoint, as staged.
        // THUS: no store conflicts with another row's window.
        unsafe {
            core::arch::x86_64::_mm256_storeu_si256(ptrs[0].add(offset).cast(), a0);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[1].add(offset).cast(), a1);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[2].add(offset).cast(), a2);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[3].add(offset).cast(), a3);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[4].add(offset).cast(), a4);
            core::arch::x86_64::_mm256_storeu_si256(ptrs[5].add(offset).cast(), a5);
        }
        offset += 32;
    }
    if offset == row_len {
        return;
    }
    let remaining = row_len - offset;
    for (row_index, &ptr) in ptrs.iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + remaining == row_len`, the span of each checked
        //        row.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the rows are disjoint and only one tail borrow is live at a
        //        time.
        // THUS: the mutable borrow conflicts with no live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(offset), remaining) };
        tail.fill(0);
        for (term, &src) in srcs.iter().enumerate() {
            // Packed records do not retain the coefficient byte; recover it
            // from the low table's `c * 1` entry.
            let coefficient = tables[term * 6 + row_index][1];
            if coefficient != 0 {
                // The benchmark target rows are lane-aligned. Leave arbitrary
                // tails to the scalar fallback by multiplying from the packed
                // table directly.
                for (dst, &value) in tail.iter_mut().zip(&src[offset..]) {
                    let low = tables[term * 6 + row_index][usize::from(value & 0x0f)];
                    let high = tables[term * 6 + row_index][16 + usize::from(value >> 4)];
                    *dst ^= low ^ high;
                }
            }
        }
    }
}

/// Experimental two-row overwrite matrix (`p2`) using AVX2 nibble shuffles
/// instead of the GFNI multiply, to test whether spreading the multiply across
/// the two `PSHUFB` ports beats production's single-port `GF2P8MULB` ceiling.
///
/// Field-agnostic: `tables[term * 2 + row]` is the packed `[lo;16, hi;16]`
/// scale table for that coefficient, so the same body serves `Gf8B` and
/// `Gf8D`. Each term broadcasts its two rows' four table halves once and reuses
/// them across `lanes` 32-byte lanes (`1` = ISA-L's 32-byte/iter geometry,
/// `2` = a 64-byte spatial unroll for more shuffle ILP). Benchmark evidence
/// only; production dispatch is unchanged.
///
/// # Panics
/// Panics unless `rows` holds two `row_len`-byte rows, `tables.len() ==
/// srcs.len() * 2`, every source is `row_len` bytes, and `lanes` is 1 or 2.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix2_shuffle_packed(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
    lanes: usize,
) {
    assert!(
        2usize
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "matrix_overwrite2_shuffle: rows buffer does not hold two rows of {row_len} bytes"
    );
    assert_eq!(tables.len(), srcs.len() * 2);
    for &src in srcs {
        assert_eq!(src.len(), row_len);
    }
    match lanes {
        1 => mul_into_matrix2_shuffle_body::<1>(rows, row_len, tables, srcs),
        2 => mul_into_matrix2_shuffle_body::<2>(rows, row_len, tables, srcs),
        _ => panic!("matrix_overwrite2_shuffle: lanes must be 1 or 2"),
    }
}

/// Two-row packed shuffle body.
///
/// `rows` holds two disjoint in-bounds rows of `row_len` bytes,
/// `tables.len() == srcs.len() * 2`, and every source is `row_len` bytes —
/// all asserted by the entry. The source/table walk is pointer-cached to keep
/// the measured term loop free of bounds checks, which with the row windows
/// is the residue this body keeps.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn mul_into_matrix2_shuffle_body<const LANES: usize>(
    rows: &mut [u8],
    row_len: usize,
    tables: &[[u8; 32]],
    srcs: &[&[u8]],
) {
    let base = rows.as_mut_ptr();
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the entry proved `2 * row_len <= rows.len()`, so both staged
    //        pointers address whole in-bounds rows.
    // THUS: every row window access stays inside `rows`.
    //
    // ALIASING
    // SINCE: the two rows sit `row_len` apart in one region.
    // THUS: the row windows are disjoint.
    let (ptr0, ptr1) = unsafe { (base, base.add(row_len)) };
    let mask = _mm256_set1_epi8(0x0f);
    let table_ptr = tables.as_ptr();
    let source_ptr = srcs.as_ptr();
    let step = 32 * LANES;
    let mut offset = 0;
    while offset + step <= row_len {
        let mut a0 = [_mm256_setzero_si256(); LANES];
        let mut a1 = [_mm256_setzero_si256(); LANES];
        for term in 0..srcs.len() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `term < srcs.len()` and the entry bounds both packed
            //        table records for every term.
            // THUS: the source reference and the table record pointer stay
            //       inside their slices.
            unsafe {
                let src = *source_ptr.add(term);
                let t = table_ptr.add(term * 2).cast::<u8>();
                // Broadcast both rows' table halves once, reused across lanes.
                let lo0 =
                    _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(t.cast()));
                let hi0 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(16).cast(),
                ));
                let lo1 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(32).cast(),
                ));
                let hi1 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(48).cast(),
                ));
                let sp = src.as_ptr().add(offset);
                for l in 0..LANES {
                    let x = core::arch::x86_64::_mm256_loadu_si256(sp.add(32 * l).cast());
                    let lo = _mm256_and_si256(x, mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
                    a0[l] = _mm256_xor_si256(
                        a0[l],
                        _mm256_xor_si256(
                            _mm256_shuffle_epi8(lo0, lo),
                            _mm256_shuffle_epi8(hi0, hi),
                        ),
                    );
                    a1[l] = _mm256_xor_si256(
                        a1[l],
                        _mm256_xor_si256(
                            _mm256_shuffle_epi8(lo1, lo),
                            _mm256_shuffle_epi8(hi1, hi),
                        ),
                    );
                }
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + step <= row_len`, the span of both checked rows.
        // THUS: every store stays inside its originating row.
        //
        // ALIASING
        // SINCE: the two row windows are disjoint, as staged.
        // THUS: no store conflicts with the other row's window.
        unsafe {
            for l in 0..LANES {
                core::arch::x86_64::_mm256_storeu_si256(ptr0.add(offset + 32 * l).cast(), a0[l]);
                core::arch::x86_64::_mm256_storeu_si256(ptr1.add(offset + 32 * l).cast(), a1[l]);
            }
        }
        offset += step;
    }
    while offset + 32 <= row_len {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        for term in 0..srcs.len() {
            // SAFETY: as the tile loop above, for a single 32-byte lane.
            unsafe {
                let src = *source_ptr.add(term);
                let t = table_ptr.add(term * 2).cast::<u8>();
                let x = core::arch::x86_64::_mm256_loadu_si256(src.as_ptr().add(offset).cast());
                let lo = _mm256_and_si256(x, mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
                let lo0 =
                    _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(t.cast()));
                let hi0 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(16).cast(),
                ));
                let lo1 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(32).cast(),
                ));
                let hi1 = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_loadu_si128(
                    t.add(48).cast(),
                ));
                a0 = _mm256_xor_si256(
                    a0,
                    _mm256_xor_si256(_mm256_shuffle_epi8(lo0, lo), _mm256_shuffle_epi8(hi0, hi)),
                );
                a1 = _mm256_xor_si256(
                    a1,
                    _mm256_xor_si256(_mm256_shuffle_epi8(lo1, lo), _mm256_shuffle_epi8(hi1, hi)),
                );
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 32 <= row_len`, the span of both checked rows.
        // THUS: the cleanup stores stay inside their originating rows.
        //
        // ALIASING
        // SINCE: the two row windows are disjoint, as staged.
        // THUS: no store conflicts with the other row's window.
        unsafe {
            core::arch::x86_64::_mm256_storeu_si256(ptr0.add(offset).cast(), a0);
            core::arch::x86_64::_mm256_storeu_si256(ptr1.add(offset).cast(), a1);
        }
        offset += 32;
    }
    if offset == row_len {
        return;
    }
    let remaining = row_len - offset;
    for (row, &ptr) in [ptr0, ptr1].iter().enumerate() {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + remaining == row_len`, the span of both checked
        //        rows.
        // THUS: the tail slice lies wholly within its originating row.
        //
        // ALIASING
        // SINCE: the two rows are disjoint and only one tail borrow is live
        //        at a time.
        // THUS: the mutable borrow conflicts with no live row window.
        let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(offset), remaining) };
        tail.fill(0);
        for (term, &src) in srcs.iter().enumerate() {
            let record = &tables[term * 2 + row];
            // Recover `c == c * 1` from the low table to skip zero terms.
            if record[1] != 0 || record[16] != 0 {
                for (dst, &value) in tail.iter_mut().zip(&src[offset..]) {
                    let low = record[usize::from(value & 0x0f)];
                    let high = record[16 + usize::from(value >> 4)];
                    *dst ^= low ^ high;
                }
            }
        }
    }
}
