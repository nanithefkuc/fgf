//! Multi-row GF(2^8) kernels on the nibble-shuffle multiply.
//!
//! The tier-1 fan shapes: one source into many rows, and many sources into
//! many rows, both grouping rows so that one source load and one pair of
//! broadcast table halves serve several destinations. These are what the
//! shuffle tiers dispatch to where the GFNI tiers use the blocked kernels.
//!
//! The entries are safe [`archmage`] capability-token functions carrying the
//! geometry contract. The row windows are addressed as runtime offsets into
//! one region — the offset-addressed-row residue — and stay in
//! [`archmage::rite`] helpers with per-block proofs.

use super::nibble::mul_add_ssse3;
use crate::field::gf8b::Elem;
use crate::kernel::Matrix;
use crate::kernel::gf8::mul_add_nibble;
use crate::kernel::tables::scale_table;

/// One source into many rows using one AVX2 source load per tile.
///
/// # Panics
/// Panics unless `src.len() == row_len` and `rows` holds `coeffs.len()` rows
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_avx2(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
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

/// One source into many rows using one SSSE3 source load per tile.
///
/// # Panics
/// As [`mul_add_scatter_avx2`].
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_ssse3(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
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

/// Many sources into many rows using AVX2 nibble shuffles.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies `nrows` coefficients for a source of `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_avx2(
    _token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
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
pub fn mul_add_matrix_avx2_with<M: Matrix<Elem> + ?Sized>(
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
        matrix_tiles_avx2(base, row_len, group, count, tile_len, terms);
        // At most one 32-byte vector survives the 64-byte tile loop.
        if tile_len < vector_len {
            matrix_vector_avx2(base, row_len, group, count, tile_len, terms);
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
/// Two accumulator vectors per row, matching the 64-byte tile of
/// [`matrix_rows4`](super::rows::matrix_rows4), so a coefficient's nibble tables are broadcast once per
/// (tile, term) and shuffled twice. Lifting the broadcast clear of the tile
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
fn matrix_tiles_avx2<M: Matrix<Elem> + ?Sized>(
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
/// Residue: as [`matrix_tiles_avx2`], with `offset + 32 <= row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn matrix_vector_avx2<M: Matrix<Elem> + ?Sized>(
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

/// Many sources into many rows using SSSE3 nibble shuffles.
///
/// # Panics
/// As [`mul_add_matrix_avx2`].
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3(
    _token: archmage::X64V2Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
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
/// As [`mul_add_matrix_avx2_with`].
#[allow(clippy::used_underscore_binding)]
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_matrix_ssse3_with<M: Matrix<Elem> + ?Sized>(
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
