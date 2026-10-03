//! SSSE3 GF(2^8) kernels over 16-byte lanes, on the nibble-shuffle multiply.
//!
//! Without a byte-wide multiply, `c * x` splits into
//! `c * (x & 0xf) ^ c * (x & 0xf0)`, two `PSHUFB` lookups against a
//! [`ScaleTable`]. The multi-row shapes group rows so one source load and one
//! pair of table halves serve several destinations. Lane-wise products of two
//! varying buffers carry the Russian-peasant multiply in registers.
//!
//! Every entry is a safe [`archmage`] capability-token function taking an
//! `X64V2Token`. The fused `mul_into` body keeps the non-temporal store
//! residue, and the row windows the offset-addressed-row residue, in
//! [`archmage::rite`] helpers with per-block proofs.

use super::check_gather;
use crate::field::{Elem, Poly};
use crate::kernel::Matrix;
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::tables::{ScaleTable, scale_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `dst ^= coeff * src` by nibble shuffle over 16-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm_loadu_si128(&table.lo);
    let hi_tbl = _mm_loadu_si128(&table.hi);
    let mask = _mm_set1_epi8(0x0f);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_rest) = src.as_chunks::<16>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(slane);
        let d = _mm_loadu_si128(dlane);
        let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
        let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
        let product = _mm_xor_si128(lo, hi);
        _mm_storeu_si128(dlane, _mm_xor_si128(d, product));
    }

    mul_add_nibble(dst_rest, table, src_rest);
}

/// `dst = coeff * dst` by nibble shuffle over 16-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], table: &ScaleTable) {
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm_loadu_si128(&table.lo);
    let hi_tbl = _mm_loadu_si128(&table.hi);
    let mask = _mm_set1_epi8(0x0f);
    let (lanes, rest) = dst.as_chunks_mut::<16>();
    for lane in lanes {
        let x = _mm_loadu_si128(&*lane);
        let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
        let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
        _mm_storeu_si128(lane, _mm_xor_si128(lo, hi));
    }

    mul_assign_nibble(rest, table);
}

/// `dst = coeff * src` by nibble shuffle over 16-byte lanes.
///
/// Fused out-of-place multiply: the `mul_add` body without the destination
/// read, so one pass instead of copy-then-scale.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_ssse3(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    table: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len());
    // The peel lands on a 32-byte — hence 16-byte — boundary.
    match crate::kernel::x86::nt_split(dst, 1) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_ssse3_impl::<false>(head, table, src_head);
            mul_into_ssse3_impl::<true>(body, table, src_body);
            _mm_sfence();
        }
        None => mul_into_ssse3_impl::<false>(dst, table, src),
    }
}

/// Fused nibble-shuffle body over equal-length slices, 16-byte lanes with
/// `NT`-selected stores.
///
/// As [`mul_into_impl`], at the 16-byte lane width.
#[archmage::rite(v2, import_intrinsics)]
#[allow(unsafe_code)]
pub(super) fn mul_into_ssse3_impl<const NT: bool>(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm_loadu_si128(&table.lo);
    let hi_tbl = _mm_loadu_si128(&table.hi);
    let mask = _mm_set1_epi8(0x0f);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_rest) = src.as_chunks::<16>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(slane);
        let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
        let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
        let product = _mm_xor_si128(lo, hi);
        if NT {
            // SAFETY:
            // NON-TEMPORAL STORE
            // SINCE: the entry selected `NT` only after `nt_split` peeled the
            //        destination to a 32-byte boundary, so every 16-byte lane
            //        of the peeled body starts on that boundary's grid.
            // THUS: the streaming store writes a 16-byte-aligned destination
            //       window that lies wholly inside `dst`.
            unsafe {
                crate::kernel::x86::store_sse2::<true>(dlane.as_mut_ptr(), product);
            }
        } else {
            _mm_storeu_si128(dlane, product);
        }
    }

    mul_into_nibble(dst_rest, table, src_rest);
}

/// One source into many rows using one SSSE3 source load per tile.
///
/// # Panics
/// As [`mul_add_scatter_avx2`](super::mul_add_scatter_avx2).
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_ssse3<const POLY: u32>(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem<8, Poly<POLY>>],
    src: &[u8],
) {
    assert_eq!(row_len, src.len());
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_scatter_ssse3: rows buffer does not hold {} rows of {row_len} bytes",
        coeffs.len()
    );
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !15;
    let base = rows.as_mut_ptr();
    let mask = _mm_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm_loadu_si128(&table.lo);
            hi[slot] = _mm_loadu_si128(&table.hi);
        }
        let (src_lanes, src_tail) = src.as_chunks::<16>();
        for (l, slane) in src_lanes.iter().enumerate() {
            let offset = l * 16;
            let x = _mm_loadu_si128(slane);
            let indices_lo = _mm_and_si128(x, mask);
            let indices_hi = _mm_and_si128(_mm_srli_epi16::<4>(x), mask);
            for slot in 0..count {
                let product = _mm_xor_si128(
                    _mm_shuffle_epi8(lo[slot], indices_lo),
                    _mm_shuffle_epi8(hi[slot], indices_hi),
                );
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `(group + slot) * row_len + offset + 16 <=
                //        coeffs.len() * row_len <= rows.len()`, asserted by
                //        this entry, and `offset + 16 <= vector_len <=
                //        row_len == src.len()`.
                // THUS: the row load and store stay inside the selected row
                //       window, and the destination read is in bounds.
                //
                // ALIASING
                // SINCE: distinct slots address rows `row_len` apart in one
                //        region, and only this row window is touched.
                // THUS: the store conflicts with no other live row window.
                unsafe {
                    let ptr = base.add((group + slot) * row_len + offset);
                    core::arch::x86_64::_mm_storeu_si128(
                        ptr.cast(),
                        _mm_xor_si128(core::arch::x86_64::_mm_loadu_si128(ptr.cast()), product),
                    );
                }
            }
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `vector_len <= row_len` and the row spans
            //        `coeffs.len() * row_len <= rows.len()` bytes overall, so
            //        `row_len - vector_len` bytes remain in this row.
            // THUS: the tail slice lies wholly within its originating row.
            //
            // ALIASING
            // SINCE: the rows are `row_len` apart and disjoint, and only one
            //        tail borrow is live at a time.
            // THUS: the mutable borrow conflicts with no live row window.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            mul_add_nibble(tail, scale_table(coeffs[group + slot]), src_tail);
        }
    }
}

/// Many sources into many rows using SSSE3 nibble shuffles.
///
/// # Panics
/// As [`mul_add_matrix_avx2`](super::mul_add_matrix_avx2).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3<const POLY: u32>(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<8, Poly<POLY>>], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_ssse3: term {t} needs {nrows} coefficients"
        );
    }
    mul_add_matrix_ssse3_with(_token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, SSSE3 backend, over a generic matrix source.
///
/// # Panics
/// As [`mul_add_matrix_avx2_with`](super::mul_add_matrix_avx2_with).
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3_with<const POLY: u32, M: Matrix<Elem<8, Poly<POLY>>> + ?Sized>(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_matrix_ssse3: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !15;
    let base = rows.as_mut_ptr();
    let mask = _mm_set1_epi8(0x0f);
    let mut group = 0;
    while group < nrows {
        let count = (nrows - group).min(4);
        let mut offset = 0;
        while offset < vector_len {
            let mut acc = [_mm_setzero_si128(); 4];
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `(group + slot) * row_len + offset + 16 <= nrows *
            //        row_len <= rows.len()` for every selected row.
            // THUS: each 16-byte window load stays inside its row.
            //
            // ALIASING
            // SINCE: distinct selected rows sit `row_len` apart in one
            //        region.
            // THUS: no two loaded windows overlap.
            unsafe {
                for (slot, value) in acc.iter_mut().take(count).enumerate() {
                    *value = core::arch::x86_64::_mm_loadu_si128(
                        base.add((group + slot) * row_len + offset).cast(),
                    );
                }
            }
            for term in 0..terms.len() {
                let src = terms.source(term);
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: the entry asserted every term source is exactly
                //        `row_len` bytes, and `offset + 16 <= vector_len <=
                //        row_len`.
                // THUS: the source load stays inside the term's source.
                let x =
                    unsafe { core::arch::x86_64::_mm_loadu_si128(src.as_ptr().add(offset).cast()) };
                for (slot, value) in acc.iter_mut().take(count).enumerate() {
                    let table = scale_table(*terms.coefficient(term, group + slot));
                    // Each table half is exactly one 16-byte lookup bank.
                    let lo = _mm_loadu_si128(&table.lo);
                    let hi = _mm_loadu_si128(&table.hi);
                    let product = _mm_xor_si128(
                        _mm_shuffle_epi8(lo, _mm_and_si128(x, mask)),
                        _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16::<4>(x), mask)),
                    );
                    *value = _mm_xor_si128(*value, product);
                }
            }
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the same row-window bounds that governed the loads.
            // THUS: each 16-byte window store stays inside its row.
            //
            // ALIASING
            // SINCE: the stored windows are the same disjoint windows loaded
            //        above.
            // THUS: no store conflicts with another row's window.
            unsafe {
                for (slot, &value) in acc.iter().take(count).enumerate() {
                    core::arch::x86_64::_mm_storeu_si128(
                        base.add((group + slot) * row_len + offset).cast(),
                        value,
                    );
                }
            }
            offset += 16;
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `(group + slot) * row_len + vector_len + (row_len -
            //        vector_len) = (group + slot + 1) * row_len <= nrows *
            //        row_len <= rows.len()`.
            // THUS: the tail slice lies wholly within its originating row.
            //
            // ALIASING
            // SINCE: the rows are `row_len` apart and disjoint, and only one
            //        tail borrow is live at a time.
            // THUS: the mutable borrow conflicts with no live row window.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            for term in 0..terms.len() {
                let src = terms.source(term);
                let table = scale_table(*terms.coefficient(term, group + slot));
                mul_add_nibble(tail, table, &src[vector_len..]);
            }
        }
        group += count;
    }
}

/// Many sources into one destination using SSSE3 nibble shuffles.
///
/// # Panics
/// As [`mul_add_gather_gfni`](super::mul_add_gather_gfni).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_ssse3<const POLY: u32>(
    token: archmage::X64V2Token,
    dst: &mut [u8],
    coeffs: &[Elem<8, Poly<POLY>>],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_ssse3", dst, coeffs.len(), srcs);
    let len = dst.len() & !63;
    let mask = _mm_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm_loadu_si128(&table.lo);
            hi[slot] = _mm_loadu_si128(&table.hi);
        }
        let (tiles, rest) = dst.as_chunks_mut::<64>();
        for (t, dtile) in tiles.iter_mut().enumerate() {
            let offset = t * 64;
            let (d4, _) = dtile.as_chunks_mut::<16>();
            let mut acc = [
                _mm_loadu_si128(&d4[0]),
                _mm_loadu_si128(&d4[1]),
                _mm_loadu_si128(&d4[2]),
                _mm_loadu_si128(&d4[3]),
            ];
            for slot in 0..count {
                let (s4, _) = srcs[group + slot][offset..offset + 64].as_chunks::<16>();
                for (lane, value) in acc.iter_mut().enumerate() {
                    let x = _mm_loadu_si128(&s4[lane]);
                    let product = _mm_xor_si128(
                        _mm_shuffle_epi8(lo[slot], _mm_and_si128(x, mask)),
                        _mm_shuffle_epi8(hi[slot], _mm_and_si128(_mm_srli_epi16::<4>(x), mask)),
                    );
                    *value = _mm_xor_si128(*value, product);
                }
            }
            for (lane, &value) in acc.iter().enumerate() {
                _mm_storeu_si128(&mut d4[lane], value);
            }
        }
        for slot in 0..count {
            // The destination and source remainders keep equal lengths.
            mul_add_ssse3(
                token,
                rest,
                scale_table(coeffs[group + slot]),
                &srcs[group + slot][len..],
            );
        }
    }
}

/// Reference GF(2^8) product under `POLY`, for the sub-lane elementwise
/// tail.
#[inline]
pub(super) fn gf_mul_ref<const POLY: u32>(mut a: u8, mut b: u8) -> u8 {
    let reduction = Poly::<POLY>::REDUCTION_LOW;
    let mut product = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= reduction;
        }
        b >>= 1;
    }
    product
}

/// The 16-byte form of [`multiply_vectors_avx2`](super::multiply_vectors_avx2).
#[archmage::rite(v2)]
pub fn multiply_vectors_ssse3<const POLY: u32>(mut a: __m128i, mut b: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let reduction = _mm_set1_epi8(Poly::<POLY>::REDUCTION_LOW.cast_signed());
    let low7 = _mm_set1_epi8(0x7f);
    let mut product = zero;
    for round in 0..8 {
        let active = _mm_cmpeq_epi8(_mm_and_si128(b, one), one);
        product = _mm_xor_si128(product, _mm_and_si128(a, active));
        if round == 7 {
            break;
        }
        let carry = _mm_cmpgt_epi8(zero, a);
        a = _mm_xor_si128(_mm_add_epi8(a, a), _mm_and_si128(reduction, carry));
        b = _mm_and_si128(_mm_srli_epi16::<1>(b), low7);
    }
    product
}

/// `dst[i] = a[i] * b[i]` by branchless shift/reduce over 16-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_ssse3<const POLY: u32>(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_ssse3: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_ssse3: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    let (a_lanes, a_rest) = a.as_chunks::<16>();
    let (b_lanes, b_rest) = b.as_chunks::<16>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm_loadu_si128(alane);
        let y = _mm_loadu_si128(blane);
        _mm_storeu_si128(dlane, multiply_vectors_ssse3::<POLY>(x, y));
    }
    for ((d, &x), &y) in dst_rest.iter_mut().zip(a_rest).zip(b_rest) {
        *d = gf_mul_ref::<POLY>(x, y);
    }
}

/// `dst[i] = dst[i] * src[i]` by branchless shift/reduce over 16-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_ssse3<const POLY: u32>(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_ssse3: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_rest) = src.as_chunks::<16>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(&*dlane);
        let y = _mm_loadu_si128(slane);
        _mm_storeu_si128(dlane, multiply_vectors_ssse3::<POLY>(x, y));
    }
    for (d, &s) in dst_rest.iter_mut().zip(src_rest) {
        *d = gf_mul_ref::<POLY>(*d, s);
    }
}
