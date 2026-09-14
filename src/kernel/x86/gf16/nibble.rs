//! Single-buffer GF(2^16) kernels over split-nibble `PSHUFB` lookups.
//!
//! AVX2 and SSSE3 have no field multiply, so each of the four base-field
//! factors becomes a `PSHUFB` pair against the prepared nibble tables and
//! the even and odd byte lanes are selected with a `0x00ff` halfword mask.
//!
//! The register-resident table sets and their `scale_*` cores are the part
//! other modules want: the Fan–Paar tower loads its own factors through the
//! same [`NibbleFactors`] loaders, and the blocked fan shapes reuse the
//! cores. The rest is single-buffer wiring.

use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::{NibbleFactors, TowerTables};

use super::{swap_mask128, swap_mask256};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Nibble tables and lane masks for one coefficient, held in AVX2 registers.
pub(crate) struct NibbleAvx2 {
    /// Low-nibble table of factor `i`, the same 16 entries in both halves.
    pub(crate) lo: [__m256i; 4],
    /// High-nibble table of factor `i`, the same 16 entries in both halves.
    pub(crate) hi: [__m256i; 4],
    /// `0x0f` in every byte: the nibble-extraction mask.
    pub(crate) nibble: __m256i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    pub(crate) even: __m256i,
    /// Adjacent-byte exchange control.
    pub(crate) swap: __m256i,
}

/// Widen the 16-byte factor tables of `tables` into AVX2 registers.
///
/// Generic over [`NibbleFactors`] so the GF(2^16) (AES-base) and Fan–Paar
/// (`fp8`-base) towers share this loader and the [`scale_avx2`] core.
#[archmage::rite(v3, import_intrinsics)]
pub(crate) fn nibble_avx2<T: NibbleFactors>(tables: &T) -> NibbleAvx2 {
    let mut lo = [_mm256_setzero_si256(); 4];
    let mut hi = [_mm256_setzero_si256(); 4];
    for slot in 0..4 {
        lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.lo(slot)));
        hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.hi(slot)));
    }
    NibbleAvx2 {
        lo,
        hi,
        nibble: _mm256_set1_epi8(0x0f),
        even: _mm256_set1_epi16(0x00ff),
        swap: swap_mask256(),
    }
}
/// Split `value` into its low and high nibbles, both as byte indices.
#[archmage::rite(v3)]
pub(crate) fn split_avx2(value: __m256i, nibble: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_and_si256(value, nibble),
        _mm256_and_si256(_mm256_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[archmage::rite(v3)]
pub(crate) fn lookup_avx2(lo: __m256i, hi: __m256i, split: (__m256i, __m256i)) -> __m256i {
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo, split.0),
        _mm256_shuffle_epi8(hi, split.1),
    )
}

/// `coeff * src` for one 32-byte lane, via four nibble-shuffle multiplies.
///
/// The even and odd contributions are summed *before* masking rather than
/// after: `AND` distributes over `XOR`, so grouping by lane parity halves the
/// mask operations relative to grouping by direct/crossed.
#[archmage::rite(v3)]
pub(crate) fn scale_avx2(src: __m256i, tables: &NibbleAvx2) -> __m256i {
    let swapped = _mm256_shuffle_epi8(src, tables.swap);
    let direct = split_avx2(src, tables.nibble);
    let crossed = split_avx2(swapped, tables.nibble);
    let even = _mm256_xor_si256(
        lookup_avx2(tables.lo[0], tables.hi[0], direct),
        lookup_avx2(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm256_xor_si256(
        lookup_avx2(tables.lo[1], tables.hi[1], direct),
        lookup_avx2(tables.lo[3], tables.hi[3], crossed),
    );
    _mm256_xor_si256(
        _mm256_and_si256(even, tables.even),
        _mm256_andnot_si256(tables.even, odd),
    )
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_avx2", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let vectors = nibble_avx2(tables);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        let d = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, _mm256_xor_si256(d, scale_avx2(x, &vectors)));
    }
    // One 128-bit step down before the scalar tail. SSSE3 is implied by AVX2,
    // so the 16-byte helpers are callable here; their tables are narrowed
    // only when the step is actually taken.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    if !(dst_lanes.is_empty() && src_lanes.is_empty()) {
        let narrow = nibble_ssse3(tables);
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let x = _mm_loadu_si128(src_lane);
            let d = _mm_loadu_si128(&*dst_lane);
            _mm_storeu_si128(dst_lane, _mm_xor_si128(d, scale_ssse3(x, &narrow)));
        }
    }
    // Both steps above are a whole number of elements, so the tail starts on
    // an element boundary.
    mul_add_scalar(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 32-byte lanes.
///
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_avx2(_token: archmage::X64V3Token, dst: &mut [u8], tables: &TowerTables) {
    super::check_elements("gf16::mul_assign_avx2", dst.len());
    let vectors = nibble_avx2(tables);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for dst_lane in dst_lanes {
        let x = _mm256_loadu_si256(&*dst_lane);
        _mm256_storeu_si256(dst_lane, scale_avx2(x, &vectors));
    }
    // One 128-bit step down before the scalar tail. SSSE3 is implied by AVX2,
    // so the 16-byte helpers are callable here; their tables are narrowed
    // only when the step is actually taken.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    if !dst_lanes.is_empty() {
        let narrow = nibble_ssse3(tables);
        for dst_lane in dst_lanes {
            let x = _mm_loadu_si128(&*dst_lane);
            _mm_storeu_si128(dst_lane, scale_ssse3(x, &narrow));
        }
    }
    // Both steps above are a whole number of elements, so the tail starts on
    // an element boundary.
    mul_assign_scalar(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    super::check_elements("gf16::mul_into_avx2", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    match super::nt_split(dst, 2) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_avx2_lane::<false>(head, tables, src_head);
            mul_into_avx2_lane::<true>(body, tables, src_body);
            _mm_sfence();
        }
        None => mul_into_avx2_lane::<false>(dst, tables, src),
    }
}

/// The `mul_into` body over 32-byte lanes, stored non-temporally when `NT`.
///
/// Only the streaming stores remain unsafe; every load and every ordinary
/// store goes through a checked slice window. A `NT` body is entered only
/// through the [`super::nt_split`] peel in [`mul_into_avx2`], which starts it
/// on a 32-byte boundary, and the caller issues the matching `_mm_sfence`.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn mul_into_avx2_lane<const NT: bool>(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let vectors = nibble_avx2(tables);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `dst_lane` is a live `&mut [u8; 32]` window of `dst`, and
        //        `store256` writes exactly 32 bytes at its pointer.
        // THUS: the store remains within `dst`.
        //
        // ALIGNMENT
        // SINCE: `NT` bodies are entered only through the `nt_split` peel in
        //        `mul_into_avx2`, whose contract starts the body destination
        //        on a 32-byte boundary, and the loop advances one whole lane.
        // THUS: the pointer meets `store256`'s 32-byte alignment requirement
        //       when `NT`.
        unsafe { super::store256::<NT>(dst_lane.as_mut_ptr(), scale_avx2(x, &vectors)) };
    }
    // One 128-bit step down before the scalar tail. SSSE3 is implied by AVX2,
    // so the 16-byte helpers are callable here; their tables are narrowed
    // only when the step is actually taken.
    let (dst_lanes, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src_rest.as_chunks::<16>();
    if !(dst_lanes.is_empty() && src_lanes.is_empty()) {
        let narrow = nibble_ssse3(tables);
        for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
            let x = _mm_loadu_si128(src_lane);
            _mm_storeu_si128(dst_lane, scale_ssse3(x, &narrow));
        }
    }
    // Both steps above are a whole number of elements, so the tail starts on
    // an element boundary.
    mul_into_scalar(dst_tail, tables.coeff, src_tail);
}

/// Nibble tables and lane masks for one coefficient, held in SSE registers.
pub(crate) struct NibbleSsse3 {
    /// Low-nibble table of factor `i`.
    pub(crate) lo: [__m128i; 4],
    /// High-nibble table of factor `i`.
    pub(crate) hi: [__m128i; 4],
    /// `0x0f` in every byte: the nibble-extraction mask.
    pub(crate) nibble: __m128i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    pub(crate) even: __m128i,
    /// Adjacent-byte exchange control.
    pub(crate) swap: __m128i,
}

/// Load the 16-byte factor tables of `tables` into SSE registers.
///
/// Generic over [`NibbleFactors`]; see [`nibble_avx2`].
#[archmage::rite(v2, import_intrinsics)]
pub(crate) fn nibble_ssse3<T: NibbleFactors>(tables: &T) -> NibbleSsse3 {
    let mut lo = [_mm_setzero_si128(); 4];
    let mut hi = [_mm_setzero_si128(); 4];
    for slot in 0..4 {
        lo[slot] = _mm_loadu_si128(tables.lo(slot));
        hi[slot] = _mm_loadu_si128(tables.hi(slot));
    }
    NibbleSsse3 {
        lo,
        hi,
        nibble: _mm_set1_epi8(0x0f),
        even: _mm_set1_epi16(0x00ff),
        swap: swap_mask128(),
    }
}

/// Split `value` into its low and high nibbles, both as byte indices.
#[archmage::rite(v2)]
pub(crate) fn split_ssse3(value: __m128i, nibble: __m128i) -> (__m128i, __m128i) {
    (
        _mm_and_si128(value, nibble),
        _mm_and_si128(_mm_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[archmage::rite(v2)]
pub(crate) fn lookup_ssse3(lo: __m128i, hi: __m128i, split: (__m128i, __m128i)) -> __m128i {
    _mm_xor_si128(_mm_shuffle_epi8(lo, split.0), _mm_shuffle_epi8(hi, split.1))
}
#[archmage::rite(v2)]
pub(crate) fn scale_ssse3(src: __m128i, tables: &NibbleSsse3) -> __m128i {
    let swapped = _mm_shuffle_epi8(src, tables.swap);
    let direct = split_ssse3(src, tables.nibble);
    let crossed = split_ssse3(swapped, tables.nibble);
    let even = _mm_xor_si128(
        lookup_ssse3(tables.lo[0], tables.hi[0], direct),
        lookup_ssse3(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm_xor_si128(
        lookup_ssse3(tables.lo[1], tables.hi[1], direct),
        lookup_ssse3(tables.lo[3], tables.hi[3], crossed),
    );
    _mm_xor_si128(
        _mm_and_si128(even, tables.even),
        _mm_andnot_si128(tables.even, odd),
    )
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 16-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_ssse3(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    super::check_elements("gf16::mul_add_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_add_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        let d = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, _mm_xor_si128(d, scale_ssse3(x, &vectors)));
    }
    mul_add_scalar(dst_tail, tables.coeff, src_tail);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 16-byte lanes.
///
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_assign_ssse3(_token: archmage::X64V2Token, dst: &mut [u8], tables: &TowerTables) {
    super::check_elements("gf16::mul_assign_ssse3", dst.len());
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    for dst_lane in dst_lanes {
        let x = _mm_loadu_si128(&*dst_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    mul_assign_scalar(dst_tail, tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
///
/// Unlike the GFNI and AVX2 forms this keeps ordinary stores at every size.
/// Non-temporal stores only pay when the loop is store-bound, and this one is
/// not: eight `PSHUFB` per 16 bytes hold it well under the host's write
/// bandwidth, and 16-byte non-temporal stores from a slow loop flush
/// write-combining buffers before a line fills. The GF(2^8) SSSE3 kernel does
/// reach the ceiling and does use them. See BENCHMARKS.md.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_ssse3(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    super::check_elements("gf16::mul_into_ssse3", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_ssse3: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let vectors = nibble_ssse3(tables);
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (src_lanes, src_tail) = src.as_chunks::<16>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm_loadu_si128(src_lane);
        _mm_storeu_si128(dst_lane, scale_ssse3(x, &vectors));
    }
    mul_into_scalar(dst_tail, tables.coeff, src_tail);
}
