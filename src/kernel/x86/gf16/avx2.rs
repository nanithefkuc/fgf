//! AVX2 GF(2^16) kernels over 32-byte lanes, on split-nibble `PSHUFB`
//! lookups.
//!
//! The nibble strategy of [`super::ssse3`] at 32 bytes. AVX2 has enough width
//! but not enough registers to keep several four-table coefficient sets
//! resident, so dispatch serves gather and matrix with repeated AXPY; scatter
//! shares one source load across a row group.
//!
//! Every entry is a safe [`archmage`] capability-token function taking an
//! `X64V3Token`. The fused `mul_into` body keeps the non-temporal store
//! residue, and the scatter group body the offset-addressed-rows residue.

use super::{TableCoefficient, check_elements, swap_mask_avx2};
use crate::field::gf8::AES;
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::{NibbleFactors, TowerTables, scale_table};
use crate::kernel::x86::gf8;
use crate::kernel::x86::{nt_split, peel_to_align, store_avx2};

use super::ssse3::{
    mul_elementwise_assign_ssse3, mul_elementwise_ssse3, nibble_ssse3, scale_ssse3,
};
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
        swap: swap_mask_avx2(),
    }
}

/// Split `value` into its low and high nibbles, both as byte indices.
#[archmage::rite(v3)]
fn split(value: __m256i, nibble: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_and_si256(value, nibble),
        _mm256_and_si256(_mm256_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[archmage::rite(v3)]
fn lookup(lo: __m256i, hi: __m256i, split: (__m256i, __m256i)) -> __m256i {
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
    let direct = split(src, tables.nibble);
    let crossed = split(swapped, tables.nibble);
    let even = _mm256_xor_si256(
        lookup(tables.lo[0], tables.hi[0], direct),
        lookup(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm256_xor_si256(
        lookup(tables.lo[1], tables.hi[1], direct),
        lookup(tables.lo[3], tables.hi[3], crossed),
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
    check_elements("gf16::mul_add_avx2", dst.len());
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
    check_elements("gf16::mul_assign_avx2", dst.len());
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
    check_elements("gf16::mul_into_avx2", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_into_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    match nt_split(dst, 2) {
        Some(peel) => {
            let (head, body) = dst.split_at_mut(peel);
            let (src_head, src_body) = src.split_at(peel);
            mul_into_impl::<false>(head, tables, src_head);
            mul_into_impl::<true>(body, tables, src_body);
            _mm_sfence();
        }
        None => mul_into_impl::<false>(dst, tables, src),
    }
}

/// The `mul_into` body over 32-byte lanes, stored non-temporally when `NT`.
///
/// Only the streaming stores remain unsafe; every load and every ordinary
/// store goes through a checked slice window. A `NT` body is entered only
/// through the [`nt_split`] peel in [`mul_into_avx2`], which starts it
/// on a 32-byte boundary, and the caller issues the matching `_mm_sfence`.
#[allow(unsafe_code)]
#[archmage::rite(v3, import_intrinsics)]
fn mul_into_impl<const NT: bool>(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let vectors = nibble_avx2(tables);
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    for (dst_lane, src_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(src_lane);
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `dst_lane` is a live `&mut [u8; 32]` window of `dst`, and
        //        `store_avx2` writes exactly 32 bytes at its pointer.
        // THUS: the store remains within `dst`.
        //
        // ALIGNMENT
        // SINCE: `NT` bodies are entered only through the `nt_split` peel in
        //        `mul_into_avx2`, whose contract starts the body destination
        //        on a 32-byte boundary, and the loop advances one whole lane.
        // THUS: the pointer meets `store_avx2`'s 32-byte alignment requirement
        //       when `NT`.
        unsafe { store_avx2::<NT>(dst_lane.as_mut_ptr(), scale_avx2(x, &vectors)) };
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

/// One source into many rows using four AVX2 table sets at a time.
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// As [`mul_add_scatter_gfni`](super::mul_add_scatter_gfni).
#[allow(unsafe_code)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_add_scatter_avx2<C: TableCoefficient>(
    token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    check_elements("gf16::mul_add_scatter_avx2", row_len);
    assert_eq!(
        row_len,
        src.len(),
        "gf16::mul_add_scatter_avx2: row_len is {row_len} bytes but src is {} bytes",
        src.len(),
    );
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("gf16::mul_add_scatter_avx2: row geometry overflows");
    assert!(
        rows.len() >= used,
        "gf16::mul_add_scatter_avx2: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY:
    // CPU FEATURES
    // SINCE: this entrypoint holds an `X64V3Token` (the `token` parameter),
    //        whose invariant guarantees AVX2 on the executing CPU.
    // THUS: the callee's CPU-feature precondition holds.
    //
    // MEMORY VALIDITY
    // SINCE: `rows.len() >= coeffs.len() * row_len` was asserted above, so
    //        every row at `base + j * row_len` lies inside `rows`, and
    //        `src.len() == row_len` was asserted above.
    // THUS: every row and source access in the callee stays within its
    //       originating slice.
    //
    // ALIASING
    // SINCE: the rows are `row_len` bytes apart and `row_len` bytes long, so
    //        they are pairwise disjoint, and `src` is an independent shared
    //        borrow.
    // THUS: no two row writes in the callee conflict.
    unsafe { scatter_rows(token, rows, row_len, coeffs, src) };
}

/// The AVX2 scatter body over offset-addressed rows of one region.
///
/// # Safety
/// The executing CPU must provide AVX2, `rows.len()` must be at least
/// `coeffs.len() * row_len`, and `src.len()` must equal `row_len`.
#[allow(unsafe_code)]
#[archmage::rite(v3)]
unsafe fn scatter_rows<C: TableCoefficient>(
    token: archmage::X64V3Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[C],
    src: &[u8],
) {
    let base = rows.as_mut_ptr();
    let src_ptr = src.as_ptr();
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let vectors: [NibbleAvx2; 4] = core::array::from_fn(|i| {
            coeffs[group + i.min(count - 1)].with_tables(|tables| nibble_avx2(tables))
        });
        // Bring this group's rows to a 32-byte boundary; see
        // `peel_to_align`.
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the function contract places row `group` inside `rows`, and
        //        `peel_to_align` returns at most `row_len`.
        // THUS: reading row `group`'s address stays within `rows`.
        let head = peel_to_align(unsafe { base.add(group * row_len) }, row_len, 2);
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the function contract bounds row `group + slot` by
            //        `row_len`, `head <= row_len`, and `src.len() == row_len`.
            // THUS: the `head`-byte lead slice lies inside that row, and
            //       `&src[..head]` inside `src`.
            unsafe {
                let lead =
                    core::slice::from_raw_parts_mut(base.add((group + slot) * row_len), head);
                coeffs[group + slot]
                    .with_tables(|tables| mul_add_avx2(token, lead, tables, &src[..head]));
            }
        }
        let mut offset = head;
        while offset + 32 <= row_len {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 32 <= row_len` bounds this iteration, every
            //        row is `row_len` bytes by the function contract, and
            //        `src.len() == row_len`.
            // THUS: the source load and each row's load and store land
            //       inside `src` and row `group + slot` respectively.
            //
            // ALIASING
            // SINCE: distinct slots address disjoint `row_len`-byte rows of
            //        one region, and `src` is an independent shared borrow.
            // THUS: a row store conflicts with no other live access.
            unsafe {
                let source = _mm256_loadu_si256(src_ptr.add(offset).cast());
                for (slot, vector) in vectors.iter().take(count).enumerate() {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm256_storeu_si256(
                        ptr.cast(),
                        _mm256_xor_si256(
                            _mm256_loadu_si256(ptr.cast()),
                            scale_avx2(source, vector),
                        ),
                    );
                }
            }
            offset += 32;
        }
        for slot in 0..count {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the loop above advanced `offset` past every whole
            //        32-byte window, so `offset..row_len` is the untouched
            //        remainder of row `group + slot`, and `src.len() ==
            //        row_len`.
            // THUS: the tail slice lies inside that row, and
            //       `&src[offset..]` inside `src`.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + offset),
                    row_len - offset,
                )
            };
            mul_add_scalar(tail, coeffs[group + slot].coefficient(), &src[offset..]);
        }
    }
}

/// `DELTA * v` for every byte lane, by split-nibble `PSHUFB`.
///
/// The tower product needs one multiply by the *constant* `DELTA`, which —
/// unlike the two varying-operand products around it — does have a nibble
/// table. Two shuffles beat eight shift/reduce rounds.
#[archmage::rite(v3)]
fn scale_delta(v: __m256i, lo: __m256i, hi: __m256i, nibble: __m256i) -> __m256i {
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo, _mm256_and_si256(v, nibble)),
        _mm256_shuffle_epi8(hi, _mm256_and_si256(_mm256_srli_epi16::<4>(v), nibble)),
    )
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements, AVX2.
///
/// Three base products, as everywhere else: `direct = [ac, bd]`,
/// `crossed = [ad, bc]`, and `DELTA*bd`. The first two have two varying
/// operands and use the shift/reduce vector multiply from
/// [`crate::kernel::x86::gf8`]; the third is a constant multiply and stays a nibble
/// shuffle.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_avx2(token: archmage::X64V3Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_elements("gf16::mul_elementwise_avx2", dst.len());
    assert_eq!(
        dst.len(),
        a.len(),
        "gf16::mul_elementwise_avx2: dst is {} bytes but a is {} bytes",
        dst.len(),
        a.len(),
    );
    assert_eq!(
        dst.len(),
        b.len(),
        "gf16::mul_elementwise_avx2: dst is {} bytes but b is {} bytes",
        dst.len(),
        b.len(),
    );
    let swap = swap_mask_avx2();
    let even = _mm256_set1_epi16(0x00ff);
    let nibble = _mm256_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.lo)),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.hi)),
    );
    let (a_lanes, a_rest) = a.as_chunks::<32>();
    let (b_lanes, b_rest) = b.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for ((dst_lane, x_lane), y_lane) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = _mm256_loadu_si256(x_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = gf8::multiply_vectors_avx2::<AES>(x, y);
        let crossed = gf8::multiply_vectors_avx2::<AES>(x, _mm256_shuffle_epi8(y, swap));
        let delta_bd = scale_delta(
            _mm256_shuffle_epi8(direct, swap),
            delta_lo,
            delta_hi,
            nibble,
        );
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    // AVX2 implies SSSE3, and the remainders keep equal lengths.
    mul_elementwise_ssse3(token.v2(), dst_rest, a_rest, b_rest);
}

/// `dst[i] = dst[i] * src[i]` over interleaved tower elements, AVX2.
///
/// # Panics
/// Panics if the slices differ in length.
#[archmage::arcane(import_intrinsics)]
pub fn mul_elementwise_assign_avx2(token: archmage::X64V3Token, dst: &mut [u8], src: &[u8]) {
    check_elements("gf16::mul_elementwise_assign_avx2", dst.len());
    assert_eq!(
        dst.len(),
        src.len(),
        "gf16::mul_elementwise_assign_avx2: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    let swap = swap_mask_avx2();
    let even = _mm256_set1_epi16(0x00ff);
    let nibble = _mm256_set1_epi8(0x0f);
    let delta = scale_table(crate::field::gf16::DELTA);
    let (delta_lo, delta_hi) = (
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.lo)),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&delta.hi)),
    );
    let (src_lanes, src_rest) = src.as_chunks::<32>();
    let (dst_lanes, dst_rest) = dst.as_chunks_mut::<32>();
    for (dst_lane, y_lane) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = _mm256_loadu_si256(&*dst_lane);
        let y = _mm256_loadu_si256(y_lane);
        let direct = gf8::multiply_vectors_avx2::<AES>(x, y);
        let crossed = gf8::multiply_vectors_avx2::<AES>(x, _mm256_shuffle_epi8(y, swap));
        let delta_bd = scale_delta(
            _mm256_shuffle_epi8(direct, swap),
            delta_lo,
            delta_hi,
            nibble,
        );
        let constant = _mm256_xor_si256(direct, delta_bd);
        let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
        let extension = _mm256_xor_si256(cross_sum, direct);
        let product = _mm256_xor_si256(
            _mm256_and_si256(constant, even),
            _mm256_andnot_si256(even, extension),
        );
        _mm256_storeu_si256(dst_lane, product);
    }
    // AVX2 implies SSSE3, and the remainders keep equal lengths.
    mul_elementwise_assign_ssse3(token.v2(), dst_rest, src_rest);
}
