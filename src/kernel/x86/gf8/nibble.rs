//! Single-buffer GF(2^8) AXPY kernels on the nibble-shuffle multiply.
//!
//! Without a byte-wide multiply, `c * x` splits into
//! `c * (x & 0xf) ^ c * (x & 0xf0)`, two `PSHUFB` lookups against a
//! [`ScaleTable`]. `PSHUFB` indexes within each 128-bit half, so the AVX2 form
//! broadcasts the 16-byte table into both halves first; the SSSE3 form uses it
//! as loaded.
//!
//! Every entry is a safe [`archmage`] capability-token function over
//! reference-based intrinsics. The AVX2 entries hand their sub-32-byte
//! remainders to the SSSE3 entries through the token's guaranteed downgrade;
//! the fused `mul_into` bodies keep the non-temporal store residue in
//! [`archmage::rite`] helpers.

use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::tables::ScaleTable;

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
    match super::super::nt_split(dst, 1) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_avx2_impl::<false>(head, table, src_head);
            mul_into_avx2_impl::<true>(body, table, src_body);
            _mm_sfence();
        }
        None => mul_into_avx2_impl::<false>(dst, table, src),
    }
}

/// Fused nibble-shuffle body over equal-length slices, 32-byte lanes with
/// `NT`-selected stores.
///
/// Loads and ordinary stores are reference-based; the streaming store is the
/// residue this body keeps.
#[archmage::rite(v3, import_intrinsics)]
#[allow(unsafe_code)]
fn mul_into_avx2_impl<const NT: bool>(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // Each table half is exactly one 16-byte lookup bank.
    let lo_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.lo));
    let hi_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.hi));
    let mask = _mm256_set1_epi8(0x0f);
    // The cursor walks 64-byte pairs of lanes so the prefetch names every
    // line the body is about to store; see [`super::super::prefetch_dst`].
    let prefetch = super::super::prefetch_dst(dst, NT);
    let pairs = dst.len() / 64 * 64;
    let (mut drest, dst_rest) = dst.split_at_mut(pairs);
    let (mut srest, src_rest) = src.split_at(pairs);
    while !drest.is_empty() {
        if prefetch && drest.len() >= super::super::PREFETCH_AHEAD + 64 {
            super::super::prefetch_line(&drest[super::super::PREFETCH_AHEAD]);
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
                    super::super::store256::<true>(dlane.as_mut_ptr(), product);
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
                super::super::store256::<true>(dlane.as_mut_ptr(), product);
            }
        } else {
            _mm256_storeu_si256(dlane, product);
        }
    }

    mul_into_ssse3_impl::<false>(dst_rest, table, src_rest);
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
    match super::super::nt_split(dst, 1) {
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
/// As [`mul_into_avx2_impl`], at the 16-byte lane width.
#[archmage::rite(v2, import_intrinsics)]
#[allow(unsafe_code)]
fn mul_into_ssse3_impl<const NT: bool>(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
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
                super::super::store128::<true>(dlane.as_mut_ptr(), product);
            }
        } else {
            _mm_storeu_si128(dlane, product);
        }
    }

    mul_into_nibble(dst_rest, table, src_rest);
}
