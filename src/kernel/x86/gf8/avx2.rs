//! AVX2 GF(2^8) kernels over 32-byte lanes, on the nibble-shuffle multiply.
//!
//! The nibble split of [`super::ssse3`] at 32 bytes: `PSHUFB` indexes within
//! each 128-bit half, so every form broadcasts the 16-byte table into both
//! halves first. Sub-32-byte remainders descend to the SSSE3 entries through
//! the token's guaranteed downgrade.
//!
//! Every entry is a safe [`archmage`] capability-token function taking an
//! `X64V3Token`. The fused `mul_into` body keeps the non-temporal store
//! residue, and the row windows the offset-addressed-row residue, in
//! [`archmage::rite`] helpers with per-block proofs.

use super::check_gather;
use super::ssse3::{
    mul_add_ssse3, mul_assign_ssse3, mul_elementwise_assign_ssse3, mul_elementwise_ssse3,
    mul_into_ssse3_impl,
};
use crate::field::{Elem, Poly};
use crate::kernel::Matrix;
use crate::kernel::tables::{ScaleTable, scale_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `dst ^= coeff * src` by nibble shuffle over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(_token: archmage::X64V3Token, dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
    let mask = _mm256_set1_epi8(0x0f);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        let d = _mm256_loadu_si256(dlane);
        let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
        let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
        let product = _mm256_xor_si256(lo, hi);
        _mm256_storeu_si256(dlane, _mm256_xor_si256(d, product));
    }

    // AVX2 implies SSSE3; the remainders keep equal lengths.
    mul_add_ssse3(_token.v2(), dst_rest, table, src_rest);
}

/// `dst = coeff * dst` by nibble shuffle over 32-byte lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], table: &ScaleTable) {
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
    let mask = _mm256_set1_epi8(0x0f);
    let (lanes, rest) = dst.as_chunks_mut::<32>();
    for lane in lanes {
        let x = _mm256_loadu_si256(&*lane);
        let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
        let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
        _mm256_storeu_si256(lane, _mm256_xor_si256(lo, hi));
    }

    // AVX2 implies SSSE3; the remainders keep equal lengths.
    mul_assign_ssse3(_token.v2(), rest, table);
}

/// `dst = coeff * src` by nibble shuffle over 32-byte lanes.
///
/// Fused out-of-place multiply: the `mul_add` body without the destination
/// read, so one pass instead of copy-then-scale.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(_token: archmage::X64V3Token, dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    match crate::kernel::x86::nt_split(dst, 1) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_impl::<false>(head, table, src_head);
            mul_into_impl::<true>(body, table, src_body);
            _mm_sfence();
        }
        None => mul_into_impl::<false>(dst, table, src),
    }
}

/// Fused nibble-shuffle body over equal-length slices, 32-byte lanes with
/// `NT`-selected stores.
///
/// Loads and ordinary stores are reference-based; the streaming store is the
/// residue this body keeps.
#[archmage::rite(v3, import_intrinsics)]
#[allow(unsafe_code)]
fn mul_into_impl<const NT: bool>(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
    let mask = _mm256_set1_epi8(0x0f);
    // The cursor walks 64-byte pairs of lanes so the prefetch names every
    // line the body is about to store; see [`crate::kernel::x86::prefetch_dst`].
    let prefetch = crate::kernel::x86::prefetch_dst(dst, NT);
    let pairs = dst.len() / 64 * 64;
    let (mut drest, dst_rest) = dst.split_at_mut(pairs);
    let (mut srest, src_rest) = src.split_at(pairs);
    while !drest.is_empty() {
        if prefetch && drest.len() >= crate::kernel::x86::PREFETCH_AHEAD + 64 {
            crate::kernel::x86::prefetch_line(&drest[crate::kernel::x86::PREFETCH_AHEAD]);
        }
        let (dpair, dnext) = drest.split_at_mut(64);
        let (spair, snext) = srest.split_at(64);
        let (dst_lanes, _) = dpair.as_chunks_mut::<32>();
        let (src_lanes, _) = spair.as_chunks::<32>();
        for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
            let x = _mm256_loadu_si256(slane);
            let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
            let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
            let product = _mm256_xor_si256(lo, hi);
            if NT {
                // SAFETY:
                // NON-TEMPORAL STORE
                // SINCE: the entry selected `NT` only after `nt_split`
                //        peeled the destination to a 32-byte boundary, and
                //        every lane of the peeled body starts on that
                //        boundary's 32-byte grid.
                // THUS: the streaming store writes a 32-byte-aligned
                //       destination window that lies wholly inside `dst`.
                unsafe {
                    crate::kernel::x86::store_avx2::<true>(dlane.as_mut_ptr(), product);
                }
            } else {
                _mm256_storeu_si256(dlane, product);
            }
        }
        drest = dnext;
        srest = snext;
    }
    let (dst_lanes, dst_rest) = dst_rest.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src_rest.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(slane);
        let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
        let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
        let product = _mm256_xor_si256(lo, hi);
        if NT {
            // SAFETY:
            // NON-TEMPORAL STORE
            // SINCE: as above - the `nt_split` peel established the 32-byte
            //        boundary and every lane advances on its grid.
            // THUS: the streaming store writes a 32-byte-aligned window
            //       inside `dst`.
            unsafe {
                crate::kernel::x86::store_avx2::<true>(dlane.as_mut_ptr(), product);
            }
        } else {
            _mm256_storeu_si256(dlane, product);
        }
    }

    mul_into_ssse3_impl::<false>(dst_rest, table, src_rest);
}

/// One source into many rows using one AVX2 source load per tile.
///
/// # Panics
/// Panics unless `src.len() == row_len` and `rows` holds `coeffs.len()` rows
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_avx2<const POLY: u32>(
    _token: archmage::X64V3Token,
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
        "mul_add_scatter_avx2: rows buffer does not hold {} rows of {row_len} bytes",
        coeffs.len()
    );
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !31;
    let base = rows.as_mut_ptr();
    let mask = _mm256_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm256_setzero_si256(); 4];
        let mut hi = [_mm256_setzero_si256(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
            hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
        }
        let (src_lanes, src_tail) = src.as_chunks::<32>();
        for (l, slane) in src_lanes.iter().enumerate() {
            let offset = l * 32;
            let x = _mm256_loadu_si256(slane);
            let indices_lo = _mm256_and_si256(x, mask);
            let indices_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
            for slot in 0..count {
                let product = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], indices_lo),
                    _mm256_shuffle_epi8(hi[slot], indices_hi),
                );
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `(group + slot) * row_len + offset + 32 <=
                //        coeffs.len() * row_len <= rows.len()`, asserted by
                //        this entry, and `offset + 32 <= vector_len <=
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
                    core::arch::x86_64::_mm256_storeu_si256(
                        ptr.cast(),
                        _mm256_xor_si256(
                            core::arch::x86_64::_mm256_loadu_si256(ptr.cast()),
                            product,
                        ),
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
            // AVX2 implies SSSE3; the remainders keep equal lengths.
            mul_add_ssse3(
                _token.v2(),
                tail,
                scale_table(coeffs[group + slot]),
                src_tail,
            );
        }
    }
}

/// Many sources into many rows using AVX2 nibble shuffles.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies `nrows` coefficients for a source of `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx2<const POLY: u32>(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem<8, Poly<POLY>>], &[u8])],
) {
    for (t, (coeffs, _)) in terms.iter().enumerate() {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_avx2: term {t} needs {nrows} coefficients"
        );
    }
    mul_add_matrix_avx2_with(_token, rows, row_len, nrows, terms);
}

/// Many sources into many rows, AVX2 backend, over a generic matrix source.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies a source of `row_len` bytes. Coefficient counts per term are the
/// provider's contract.
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx2_with<const POLY: u32, M: Matrix<Elem<8, Poly<POLY>>> + ?Sized>(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &M,
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "mul_add_matrix_avx2: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for term in 0..terms.len() {
        assert_eq!(terms.source(term).len(), row_len);
    }
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !31;
    let tile_len = row_len & !63;
    let base = rows.as_mut_ptr();
    let mut group = 0;
    while group < nrows {
        let count = (nrows - group).min(4);
        matrix_tiles(base, row_len, group, count, tile_len, terms);
        // At most one 32-byte vector survives the 64-byte tile loop.
        if tile_len < vector_len {
            matrix_vector(base, row_len, group, count, tile_len, terms);
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `(group + slot) * row_len + vector_len + (row_len -
            //        vector_len) = (group + slot + 1) * row_len <= nrows *
            //        row_len <= rows.len()`, asserted above.
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
                // AVX2 implies SSSE3; every source remainder has the same
                // length as the tail.
                mul_add_ssse3(_token.v2(), tail, table, &src[vector_len..]);
            }
        }
        group += count;
    }
}

/// Fold every term into a group of up to four rows, 64 bytes of each row at
/// a time.
///
/// Two accumulator vectors per row, matching the 64-byte tile of the GFNI
/// four-row group body, so a coefficient's nibble tables are broadcast once
/// per (tile, term) and shuffled twice. Lifting the broadcast clear of the tile
/// loop the way the scatter kernels do is only possible there because a
/// scatter has a single source, so four tables cover the whole row group; a
/// matrix would need `4 * terms.len()` tables live at once, well past the
/// register file. Deepening the tile is the reachable form of the same
/// saving, and it cannot cost destination traffic: the tile still absorbs
/// every term before it is stored.
///
/// A full four-row group is the one shape that does not fit the register
/// file: eight accumulators, four nibble indices, a table pair and two
/// shuffle temporaries need seventeen YMM registers, so the last row's two
/// accumulators live on the stack. That still trades eight table broadcasts
/// for two reloads and two stores per (tile, term), and the round trip sits
/// off the critical path behind sixteen shuffles, so the tile is a net
/// sixteen-to-twelve reduction in memory operations. Narrower groups stay
/// entirely in registers.
///
/// Residue: rows `group..group + count` are runtime-offset windows of one
/// region, bounded by the entry's `nrows * row_len` assert; every term
/// source is exactly `row_len` bytes, asserted by the entry.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn matrix_tiles<const POLY: u32, M: Matrix<Elem<8, Poly<POLY>>> + ?Sized>(
    base: *mut u8,
    row_len: usize,
    group: usize,
    count: usize,
    tile_len: usize,
    terms: &M,
) {
    let mask = _mm256_set1_epi8(0x0f);
    let mut offset = 0;
    while offset < tile_len {
        let mut acc = [[_mm256_setzero_si256(); 2]; 4];
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `(group + slot) * row_len + offset + 64 <= nrows * row_len
        //        <= rows.len()` for every selected row, and the destination
        //        accumulators are this body's own registers.
        // THUS: each 64-byte window load stays inside its originating row.
        //
        // ALIASING
        // SINCE: distinct selected rows sit `row_len` apart in one region.
        // THUS: no two loaded windows overlap.
        unsafe {
            for (slot, value) in acc.iter_mut().take(count).enumerate() {
                let ptr = base.add((group + slot) * row_len + offset);
                value[0] = core::arch::x86_64::_mm256_loadu_si256(ptr.cast());
                value[1] = core::arch::x86_64::_mm256_loadu_si256(ptr.add(32).cast());
            }
        }
        for term in 0..terms.len() {
            let src = terms.source(term);
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the entry asserted every term source is exactly
            //        `row_len` bytes, and `offset + 64 <= tile_len <=
            //        row_len`.
            // THUS: both source loads stay inside the term's source slice.
            let (x0, x1) = unsafe {
                let sp = src.as_ptr().add(offset);
                (
                    core::arch::x86_64::_mm256_loadu_si256(sp.cast()),
                    core::arch::x86_64::_mm256_loadu_si256(sp.add(32).cast()),
                )
            };
            // Nibble indices depend on the source alone, so the whole row
            // group shares them.
            let indices = [
                _mm256_and_si256(x0, mask),
                _mm256_and_si256(_mm256_srli_epi16::<4>(x0), mask),
                _mm256_and_si256(x1, mask),
                _mm256_and_si256(_mm256_srli_epi16::<4>(x1), mask),
            ];
            for (slot, value) in acc.iter_mut().take(count).enumerate() {
                let table = scale_table(*terms.coefficient(term, group + slot));
                // Each table half is exactly one 16-byte lookup bank.
                let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
                let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
                value[0] = _mm256_xor_si256(
                    value[0],
                    _mm256_xor_si256(
                        _mm256_shuffle_epi8(lo, indices[0]),
                        _mm256_shuffle_epi8(hi, indices[1]),
                    ),
                );
                value[1] = _mm256_xor_si256(
                    value[1],
                    _mm256_xor_si256(
                        _mm256_shuffle_epi8(lo, indices[2]),
                        _mm256_shuffle_epi8(hi, indices[3]),
                    ),
                );
            }
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the same `(group + slot) * row_len + offset + 64 <= nrows *
        //        row_len <= rows.len()` relation that bounded the loads.
        // THUS: each 64-byte window store stays inside its originating row.
        //
        // ALIASING
        // SINCE: the stored windows are the same disjoint windows loaded
        //        above.
        // THUS: no store conflicts with another row's window.
        unsafe {
            for (slot, value) in acc.iter().take(count).enumerate() {
                let ptr = base.add((group + slot) * row_len + offset);
                core::arch::x86_64::_mm256_storeu_si256(ptr.cast(), value[0]);
                core::arch::x86_64::_mm256_storeu_si256(ptr.add(32).cast(), value[1]);
            }
        }
        offset += 64;
    }
}

/// Fold every term into one 32-byte window of a group of up to four rows.
///
/// Residue: as [`matrix_tiles`], with `offset + 32 <= row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn matrix_vector<const POLY: u32, M: Matrix<Elem<8, Poly<POLY>>> + ?Sized>(
    base: *mut u8,
    row_len: usize,
    group: usize,
    count: usize,
    offset: usize,
    terms: &M,
) {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = [_mm256_setzero_si256(); 4];
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `(group + slot) * row_len + offset + 32 <= nrows * row_len <=
    //        rows.len()` for every selected row.
    // THUS: each 32-byte window load stays inside its originating row.
    //
    // ALIASING
    // SINCE: distinct selected rows sit `row_len` apart in one region.
    // THUS: no two loaded windows overlap.
    unsafe {
        for (slot, value) in acc.iter_mut().take(count).enumerate() {
            *value = core::arch::x86_64::_mm256_loadu_si256(
                base.add((group + slot) * row_len + offset).cast(),
            );
        }
    }
    for term in 0..terms.len() {
        let src = terms.source(term);
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the entry asserted every term source is exactly `row_len`
        //        bytes, and `offset + 32 <= row_len`.
        // THUS: the source load stays inside the term's source slice.
        let x = unsafe { core::arch::x86_64::_mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
        let indices_lo = _mm256_and_si256(x, mask);
        let indices_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
        for (slot, value) in acc.iter_mut().take(count).enumerate() {
            let table = scale_table(*terms.coefficient(term, group + slot));
            // Each table half is exactly one 16-byte lookup bank.
            let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
            let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
            *value = _mm256_xor_si256(
                *value,
                _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo, indices_lo),
                    _mm256_shuffle_epi8(hi, indices_hi),
                ),
            );
        }
    }
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the same row-window bounds that governed the loads.
    // THUS: each 32-byte window store stays inside its originating row.
    //
    // ALIASING
    // SINCE: the stored windows are the same disjoint windows loaded above.
    // THUS: no store conflicts with another row's window.
    unsafe {
        for (slot, &value) in acc.iter().take(count).enumerate() {
            core::arch::x86_64::_mm256_storeu_si256(
                base.add((group + slot) * row_len + offset).cast(),
                value,
            );
        }
    }
}

/// Many sources into one destination using AVX2 nibble shuffles.
///
/// # Panics
/// As [`mul_add_gather_gfni`](super::mul_add_gather_gfni).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_gather_avx2<const POLY: u32>(
    token: archmage::X64V3Token,
    dst: &mut [u8],
    coeffs: &[Elem<8, Poly<POLY>>],
    srcs: &[&[u8]],
) {
    check_gather("mul_add_gather_avx2", dst, coeffs.len(), srcs);
    let len = dst.len() & !63;
    let mask = _mm256_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm256_setzero_si256(); 4];
        let mut hi = [_mm256_setzero_si256(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // Each table half is exactly one 16-byte lookup bank.
            lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
            hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
        }
        let (tiles, rest) = dst.as_chunks_mut::<64>();
        for (t, dtile) in tiles.iter_mut().enumerate() {
            let offset = t * 64;
            let (d2, _) = dtile.as_chunks_mut::<32>();
            let mut acc0 = _mm256_loadu_si256(&d2[0]);
            let mut acc1 = _mm256_loadu_si256(&d2[1]);
            for slot in 0..count {
                let (s2, _) = srcs[group + slot][offset..offset + 64].as_chunks::<32>();
                let x0 = _mm256_loadu_si256(&s2[0]);
                let x1 = _mm256_loadu_si256(&s2[1]);
                let p0 = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x0, mask)),
                    _mm256_shuffle_epi8(
                        hi[slot],
                        _mm256_and_si256(_mm256_srli_epi16::<4>(x0), mask),
                    ),
                );
                let p1 = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x1, mask)),
                    _mm256_shuffle_epi8(
                        hi[slot],
                        _mm256_and_si256(_mm256_srli_epi16::<4>(x1), mask),
                    ),
                );
                acc0 = _mm256_xor_si256(acc0, p0);
                acc1 = _mm256_xor_si256(acc1, p1);
            }
            _mm256_storeu_si256(&mut d2[0], acc0);
            _mm256_storeu_si256(&mut d2[1], acc1);
        }
        for slot in 0..count {
            // The destination and source remainders keep equal lengths.
            mul_add_avx2(
                token,
                rest,
                scale_table(coeffs[group + slot]),
                &srcs[group + slot][len..],
            );
        }
    }
}

/// Lane-parallel GF(2^8) multiplication of two varying byte vectors.
///
/// Without `GF2P8MULB` there is no fixed coefficient to build a nibble table
/// from, so the product comes from eight branchless shift/reduce rounds — the
/// sequence already proven on NEON and wasm. x86 has no byte shift: `PADDB`
/// doubles each lane (a left shift that drops the carry) and the signed
/// `PCMPGTB` against zero recovers the bit it dropped.
#[archmage::rite(v3)]
pub fn multiply_vectors_avx2<const POLY: u32>(mut a: __m256i, mut b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi8(1);
    let reduction = _mm256_set1_epi8(Poly::<POLY>::REDUCTION_LOW.cast_signed());
    let low7 = _mm256_set1_epi8(0x7f);
    let mut product = zero;
    for round in 0..8 {
        let active = _mm256_cmpeq_epi8(_mm256_and_si256(b, one), one);
        product = _mm256_xor_si256(product, _mm256_and_si256(a, active));
        if round == 7 {
            break;
        }
        let carry = _mm256_cmpgt_epi8(zero, a);
        a = _mm256_xor_si256(_mm256_add_epi8(a, a), _mm256_and_si256(reduction, carry));
        b = _mm256_and_si256(_mm256_srli_epi16::<1>(b), low7);
    }
    product
}

/// `dst[i] = a[i] * b[i]` by branchless shift/reduce over 32-byte lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2<const POLY: u32>(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    assert_eq!(
        dst.len(),
        a.len(),
        "mul_elementwise_avx2: dst and a differ in length"
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "mul_elementwise_avx2: dst and b differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    for ((dlane, alane), blane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(alane);
        let y = _mm256_loadu_si256(blane);
        _mm256_storeu_si256(dlane, multiply_vectors_avx2::<POLY>(x, y));
    }

    // AVX2 implies SSSE3; the remainders keep equal lengths.
    mul_elementwise_ssse3::<POLY>(_token.v2(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = dst[i] * src[i]` by branchless shift/reduce over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2<const POLY: u32>(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    src: &[u8],
) {
    assert_eq!(
        dst.len(),
        src.len(),
        "mul_elementwise_assign_avx2: dst and src differ in length"
    );
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dlane, slane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dlane);
        let y = _mm256_loadu_si256(slane);
        _mm256_storeu_si256(dlane, multiply_vectors_avx2::<POLY>(x, y));
    }

    // AVX2 implies SSSE3; the remainders keep equal lengths.
    mul_elementwise_assign_ssse3::<POLY>(_token.v2(), dst_rest, src_rest);
}
