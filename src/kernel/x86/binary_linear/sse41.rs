//! SSE4.1 GF(2)-linear map kernels over 16-byte lanes.
//!
//! Each set input bit selects its column with a lane-width equality compare
//! against the bit's one-bit mask, so one map application costs an
//! AND-compare-AND-XOR chain per active bit and every column broadcast
//! stays loop-invariant. Entries are safe [`archmage`] capability-token
//! functions taking an `X64V2Token` and validating their geometry; the
//! lane builders are [`archmage::rite`] helpers.

use super::{
    LinearMap, PreparedBits, check_geometry, prepare_bits, tail_add, tail_assign, tail_into,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Per-call lane masks and column broadcasts of one map, in SSE registers.
struct LaneTables {
    /// Broadcast of each active input bit's one-bit mask.
    masks: [__m128i; 64],
    /// Broadcast of each active input bit's column, masked to the map's
    /// output width.
    cols: [__m128i; 64],
}

/// Broadcasts the active bit masks and output-masked columns of one map.
#[archmage::rite(v2, import_intrinsics)]
fn lane_tables<const BYTES: usize>(map: &LinearMap, prepared: &PreparedBits) -> LaneTables {
    let mut tables = LaneTables {
        masks: [_mm_setzero_si128(); 64],
        cols: [_mm_setzero_si128(); 64],
    };
    let mut slot = 0;
    while slot < prepared.len {
        let bit = usize::from(prepared.bits[slot]);
        tables.masks[slot] = splat::<BYTES>(1u64 << bit);
        tables.cols[slot] = splat::<BYTES>(map.columns[bit] & prepared.out_mask);
        slot += 1;
    }
    tables
}

/// XORs the column of every set bit of `x` into `acc` at lane width
/// `BYTES`.
#[archmage::rite(v2)]
fn fold<const BYTES: usize>(
    x: __m128i,
    acc: __m128i,
    tables: &LaneTables,
    prepared: &PreparedBits,
) -> __m128i {
    let mut acc = acc;
    let mut slot = 0;
    while slot < prepared.len {
        let hit = eq_lanes::<BYTES>(_mm_and_si128(x, tables.masks[slot]), tables.masks[slot]);
        acc = _mm_xor_si128(acc, _mm_and_si128(hit, tables.cols[slot]));
        slot += 1;
    }
    acc
}

/// Broadcasts one element word across every `BYTES`-byte lane.
///
/// # Panics
/// Panics on an element width other than two, four, or eight bytes; the
/// checked entries reject those widths before the lane builders run.
#[archmage::rite(v2, import_intrinsics)]
fn splat<const BYTES: usize>(word: u64) -> __m128i {
    match BYTES {
        // The word carries exactly one element's bytes, and the intrinsics
        // read lane words as signed integers of the same width.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        2 => _mm_set1_epi16(word as i16),
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        4 => _mm_set1_epi32(word as i32),
        #[allow(clippy::cast_possible_wrap)]
        8 => _mm_set1_epi64x(word as i64),
        _ => panic!("binary_linear: element width {BYTES} is unsupported"),
    }
}

/// Lane-width equality of two vectors: all-ones where lanes match.
///
/// # Panics
/// Panics on an element width other than two, four, or eight bytes; the
/// checked entries reject those widths before the lane builders run.
#[archmage::rite(v2)]
fn eq_lanes<const BYTES: usize>(left: __m128i, right: __m128i) -> __m128i {
    match BYTES {
        2 => _mm_cmpeq_epi16(left, right),
        4 => _mm_cmpeq_epi32(left, right),
        8 => _mm_cmpeq_epi64(left, right),
        _ => panic!("binary_linear: element width {BYTES} is unsupported"),
    }
}

/// `dst ^= map(src)` per `BYTES`-byte little-endian coordinate word.
///
/// Bits of a destination word above the map's output width pass through
/// the XOR untouched.
///
/// # Panics
/// Panics when the slices differ in length, a buffer ends with a partial
/// element, or the map width exceeds the element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_sse41<const BYTES: usize>(
    token: archmage::X64V2Token,
    dst: &mut [u8],
    map: &LinearMap,
    src: &[u8],
) {
    let _ = token;
    check_geometry::<BYTES>("binary_linear::mul_add_sse41", dst, Some(src), map);
    let prepared = prepare_bits(map);
    let tables = lane_tables::<BYTES>(map, &prepared);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, fold::<BYTES>(x, d, &tables, &prepared));
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "binary_linear.sse41.mul_add",
        crate::kernel::trace::Substage::MulAdd,
        crate::kernel::trace::Unit::Elements,
        dst_lanes.len() * (16 / BYTES),
        crate::kernel::Backend::V2,
    );
    tail_add::<BYTES>(dst_tail, src_tail, map);
}

/// `dst = map(src)` per `BYTES`-byte little-endian coordinate word; the
/// previous destination contents are ignored, and result bits above the
/// map's output width are zero.
///
/// # Panics
/// Panics when the slices differ in length, a buffer ends with a partial
/// element, or the map width exceeds the element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_sse41<const BYTES: usize>(
    token: archmage::X64V2Token,
    dst: &mut [u8],
    map: &LinearMap,
    src: &[u8],
) {
    let _ = token;
    check_geometry::<BYTES>("binary_linear::mul_into_sse41", dst, Some(src), map);
    let prepared = prepare_bits(map);
    let tables = lane_tables::<BYTES>(map, &prepared);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let written = fold::<BYTES>(x, _mm_setzero_si128(), &tables, &prepared);
        _mm_storeu_si128(dst_lane, written);
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "binary_linear.sse41.mul_into",
        crate::kernel::trace::Substage::Mul,
        crate::kernel::trace::Unit::Elements,
        dst_lanes.len() * (16 / BYTES),
        crate::kernel::Backend::V2,
    );
    tail_into::<BYTES>(dst_tail, src_tail, map);
}

/// `dst = map(dst)` per `BYTES`-byte little-endian coordinate word; the
/// old destination word is the map input, and result bits above the map's
/// output width are zero.
///
/// # Panics
/// Panics when a buffer ends with a partial element or the map width
/// exceeds the element.
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_sse41<const BYTES: usize>(
    token: archmage::X64V2Token,
    dst: &mut [u8],
    map: &LinearMap,
) {
    let _ = token;
    check_geometry::<BYTES>("binary_linear::mul_assign_sse41", dst, None, map);
    let prepared = prepare_bits(map);
    let tables = lane_tables::<BYTES>(map, &prepared);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes.iter_mut() {
        let d = _mm_loadu_si128(&*dst_lane);
        let written = fold::<BYTES>(d, _mm_setzero_si128(), &tables, &prepared);
        _mm_storeu_si128(dst_lane, written);
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "binary_linear.sse41.mul_assign",
        crate::kernel::trace::Substage::Mul,
        crate::kernel::trace::Unit::Elements,
        dst_lanes.len() * (16 / BYTES),
        crate::kernel::Backend::V2,
    );
    tail_assign::<BYTES>(dst_tail, map);
}
