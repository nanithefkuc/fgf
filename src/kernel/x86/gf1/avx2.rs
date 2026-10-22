//! AVX2 bit-packed GF(2) interop kernels over 32-byte lanes.
//!
//! The SSE4.2 algorithms at double width: each entry is a safe
//! capability-token function taking an `X64V3Token` and validating its own
//! geometry. Full 32-byte units run the vector body; `packs` needs the
//! `permute4x64` fix-up to restore element order before extraction, and any
//! remainder lane falls back to the portable scalar word loop.

use crate::kernel::gf1::LaneTest;

/// Expand packed bits into target lanes over 32-byte units.
///
/// Same replicate-select-compare-select algorithm as
/// [`expand_sse42`](super::sse42::expand_sse42) at double width.
///
/// # Panics
///
/// Panics if `dst.len() != bits * bytes`, if `bytes` is not 1, 2, 4, or 8,
/// or if `src` holds fewer than `bits.div_ceil(8)` bytes.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::used_underscore_binding
)]
#[archmage::arcane(import_intrinsics)]
pub fn expand_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    assert_eq!(dst.len(), bits.saturating_mul(bytes));
    assert!(matches!(bytes, 1 | 2 | 4 | 8));
    assert!(crate::bits::bytes_for(bits) <= src.len());
    let (dst_units, _) = dst.as_chunks_mut::<32>();
    let lanes = match bytes {
        1 => {
            let sel = _mm256_loadu_si256(&EXPAND_SEL_B1);
            let one = _mm256_set1_epi8(one_w as i8);
            let zero = _mm256_setzero_si256();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                let mut rep = [0u8; 32];
                for (k, slot) in rep.chunks_exact_mut(16).enumerate() {
                    // Unit `u` covers lanes `32 * u..32 * u + 32`, fed by
                    // source bytes `4 * u + 2 * k` and `4 * u + 2 * k + 1`.
                    slot[..8].fill(src[4 * u + 2 * k]);
                    slot[8..].fill(src[4 * u + 2 * k + 1]);
                }
                let masked = _mm256_and_si256(_mm256_loadu_si256(&rep), sel);
                let is_zero = _mm256_cmpeq_epi8(masked, zero);
                _mm256_storeu_si256(unit, _mm256_andnot_si256(is_zero, one));
            }
            dst_units.len() * 32
        }
        2 => {
            let sel = _mm256_loadu_si256(&EXPAND_SEL_B2);
            let one = _mm256_set1_epi16(one_w as i16);
            let zero = _mm256_setzero_si256();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `16 * u..16 * u + 16`, fed by
                // source bytes `2 * u` (low half) and `2 * u + 1`
                // (high half).
                let mut rep = [0u8; 32];
                rep[..16].fill(src[2 * u]);
                rep[16..].fill(src[2 * u + 1]);
                let masked = _mm256_and_si256(_mm256_loadu_si256(&rep), sel);
                let is_zero = _mm256_cmpeq_epi16(masked, zero);
                _mm256_storeu_si256(unit, _mm256_andnot_si256(is_zero, one));
            }
            dst_units.len() * 16
        }
        4 => {
            let sel = _mm256_loadu_si256(&EXPAND_SEL_B4);
            let one = _mm256_set1_epi32(one_w as i32);
            let zero = _mm256_setzero_si256();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `8 * u..8 * u + 8` (source byte
                // `u`): the low half's lanes own bits 0..3, the high
                // half's bits 4..7.
                let mut rep = [0u8; 32];
                rep.fill(src[u]);
                let masked = _mm256_and_si256(_mm256_loadu_si256(&rep), sel);
                let is_zero = _mm256_cmpeq_epi32(masked, zero);
                _mm256_storeu_si256(unit, _mm256_andnot_si256(is_zero, one));
            }
            dst_units.len() * 8
        }
        _ => {
            let sels = [
                _mm256_loadu_si256(&EXPAND_SEL_B8_0),
                _mm256_loadu_si256(&EXPAND_SEL_B8_4),
            ];
            let one = _mm256_set1_epi64x(one_w as i64);
            let zero = _mm256_setzero_si256();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `4 * u..4 * u + 4`; the base bit
                // `4u` cycles through 0 and 4 mod 8.
                let mut rep = [0u8; 32];
                rep.fill(src[(4 * u) / 8]);
                let masked = _mm256_and_si256(_mm256_loadu_si256(&rep), sels[u % 2]);
                let is_zero = _mm256_cmpeq_epi64(masked, zero);
                _mm256_storeu_si256(unit, _mm256_andnot_si256(is_zero, one));
            }
            dst_units.len() * 4
        }
    };
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx2.expand",
        crate::kernel::trace::Substage::Unpack,
        crate::kernel::trace::Unit::Bits,
        lanes,
        crate::kernel::Backend::V3,
    );
    // Remainder lanes run the portable indexing.
    for i in lanes..bits {
        let bit = (src[i / 8] >> (i % 8)) & 1;
        let lane = one_w & 0u64.wrapping_sub(u64::from(bit));
        let lane_bytes = lane.to_le_bytes();
        dst[i * bytes..(i + 1) * bytes].copy_from_slice(&lane_bytes[..bytes]);
    }
}

/// B=1 selector: byte lane `j` owns bit `j % 8` of its replicated byte.
const EXPAND_SEL_B1: [u8; 32] = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
];
/// B=2 selector: word lane `j` owns bit `j % 8` in both bytes.
const EXPAND_SEL_B2: [u8; 32] = [
    0x01, 0x01, 0x02, 0x02, 0x04, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80,
    0x01, 0x01, 0x02, 0x02, 0x04, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80,
];
/// B=4 selector: unit `u` covers lanes `8u..8u+8` (source byte `u`), so
/// the low half's dword lane `j` owns bit `j` and the high half's bit `4+j`.
const EXPAND_SEL_B4: [u8; 32] = [
    0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80,
];
/// B=8 selectors: qword lane `j` of each pair owns bit `phase + j`.
/// Unit `u` covers lanes `4u..4u+4`, so phase cycles 0, 4 mod 8 and the two
/// halves test consecutive pairs.
const EXPAND_SEL_B8_0: [u8; 32] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
];
const EXPAND_SEL_B8_4: [u8; 32] = [
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
];

/// Compress target lanes into a packed bitmap over 32-byte units.
///
/// Same width-parameterized extraction as
/// [`compress_sse42`](super::sse42::compress_sse42) at double width: B=1
/// reads the 32-bit movemask directly, B=2 packs words to bytes first, B=4
/// reads the float movemask, B=8 the double movemask. The 128-bit lanes of
/// `packs` are in the wrong order afterwards, so a `permute4x64` fix-up
/// restores element order before extraction.
///
/// # Panics
///
/// Panics if `src.len() != bits * bytes`, if `dst.len()` is not
/// `bits.div_ceil(8)`, or if `bytes` is not 1, 2, 4, or 8.
#[allow(clippy::cast_lossless, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn compress_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    src: &[u8],
    bits: usize,
    bytes: usize,
    test: &LaneTest,
) {
    assert_eq!(src.len(), bits.saturating_mul(bytes));
    assert_eq!(dst.len(), crate::bits::bytes_for(bits));
    assert!(matches!(bytes, 1 | 2 | 4 | 8));
    dst.fill(0);
    match *test {
        LaneTest::NonZero => compress_eq_inner(dst, src, bits, bytes, 0, false),
        LaneTest::ZeroOrOne(one) => compress_eq_inner(dst, src, bits, bytes, one, true),
        LaneTest::Parity(tau) => compress_parity_avx2(dst, src, bits, bytes, tau),
    }
}

/// Shared equality compress over 32-byte units.
///
/// `or_zero == true` selects the membership predicate `w == 0 || w == one`;
/// `false` selects the nonzero predicate `w != 0`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v3, import_intrinsics)]
fn compress_eq_inner(
    dst: &mut [u8],
    src: &[u8],
    bits: usize,
    bytes: usize,
    one: u64,
    or_zero: bool,
) {
    let zero = _mm256_setzero_si256();
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one & !width_mask == 0;
    let one_b = _mm256_set1_epi8(one as i8);
    let one_w16 = _mm256_set1_epi16(one as i16);
    let one_w32 = _mm256_set1_epi32(one as i32);
    let one_w64 = _mm256_set1_epi64x(one as i64);
    let (src_units, _) = src.as_chunks::<32>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 32 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm256_loadu_si256(unit);
        let is_zero = match bytes {
            1 => _mm256_cmpeq_epi8(v, zero),
            2 => _mm256_cmpeq_epi16(v, zero),
            4 => _mm256_cmpeq_epi32(v, zero),
            _ => _mm256_cmpeq_epi64(v, zero),
        };
        let is_one = if one_in_range {
            match bytes {
                1 => _mm256_cmpeq_epi8(v, one_b),
                2 => _mm256_cmpeq_epi16(v, one_w16),
                4 => _mm256_cmpeq_epi32(v, one_w32),
                _ => _mm256_cmpeq_epi64(v, one_w64),
            }
        } else {
            zero
        };
        let pass = if or_zero {
            _mm256_or_si256(is_zero, is_one)
        } else {
            _mm256_xor_si256(is_zero, _mm256_set1_epi8(-1))
        };
        let lane_bits = match bytes {
            1 => _mm256_movemask_epi8(pass) as u32,
            // `packs` concatenates the low halves then the high halves, so
            // lanes 8..16 land in bytes 16..24: permute the 64-bit quartets
            // back to element order first.
            2 => {
                let packed = _mm256_packs_epi16(pass, zero);
                let fixed = _mm256_permute4x64_epi64::<0xD8>(packed);
                _mm256_movemask_epi8(fixed) as u32 & 0xffff
            }
            4 => _mm256_movemask_ps(_mm256_castsi256_ps(pass)) as u32 & 0xff,
            _ => _mm256_movemask_pd(_mm256_castsi256_pd(pass)) as u32 & 0x0f,
        };
        for j in 0..lanes_here {
            if lane_bits >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx2.compress",
        crate::kernel::trace::Substage::Pack,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V3,
    );
    // Remainder lanes: scalar word compares.
    for i in lane..bits {
        let mut word = 0u64;
        for (shift, &byte) in src[i * bytes..(i + 1) * bytes].iter().enumerate() {
            word |= u64::from(byte) << (8 * shift);
        }
        let ok = if or_zero {
            word == 0 || word == one
        } else {
            word != 0
        };
        if ok {
            dst[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Parity predicate over 32-byte units: bit `i` is the parity of
/// `lane_word(i) & tau`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
#[archmage::rite(v3, import_intrinsics)]
fn compress_parity_avx2(dst: &mut [u8], src: &[u8], bits: usize, bytes: usize, tau: u64) {
    const PARITY_LUT: [u8; 16] = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];
    let lut_half = _mm_loadu_si128(&PARITY_LUT);
    let lut = _mm256_broadcastsi128_si256(lut_half);
    let tau_b = _mm256_set1_epi8(tau as i8);
    let tau_w16 = _mm256_set1_epi16(tau as i16);
    let tau_w32 = _mm256_set1_epi32(tau as i32);
    let tau_w64 = _mm256_set1_epi64x(tau as i64);
    let rev_d = _mm256_broadcastsi128_si256(_mm_loadu_si128(&REV_DWORD));
    let rev_q = _mm256_broadcastsi128_si256(_mm_loadu_si128(&REV_QWORD));
    let mask_lo = _mm256_set1_epi8(0x0f);
    let (src_units, _) = src.as_chunks::<32>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 32 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm256_loadu_si256(unit);
        let masked = match bytes {
            1 => _mm256_and_si256(v, tau_b),
            2 => _mm256_and_si256(v, tau_w16),
            4 => _mm256_and_si256(v, tau_w32),
            _ => _mm256_and_si256(v, tau_w64),
        };
        // Same lane-local reversal fold as the SSE4.2 body at double
        // width: `pshufb` indexes within each 128-bit half, and both
        // reversals are half-local, so the fold is lane-local too.
        let folded = match bytes {
            1 => masked,
            2 => _mm256_xor_si256(masked, _mm256_srli_epi16::<8>(masked)),
            4 => {
                let rev = _mm256_shuffle_epi8(masked, rev_d);
                let pairs = _mm256_xor_si256(masked, rev);
                _mm256_xor_si256(pairs, _mm256_srli_epi16::<8>(pairs))
            }
            _ => {
                let t1 = _mm256_xor_si256(masked, _mm256_shuffle_epi8(masked, rev_q));
                let t2 = _mm256_xor_si256(t1, _mm256_shuffle_epi8(t1, rev_d));
                _mm256_xor_si256(t2, _mm256_srli_epi16::<8>(t2))
            }
        };
        let lo_nib = _mm256_and_si256(folded, mask_lo);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(folded), mask_lo);
        let p = _mm256_xor_si256(
            _mm256_shuffle_epi8(lut, lo_nib),
            _mm256_shuffle_epi8(lut, hi_nib),
        );
        // Lanes run contiguously across the two 128-bit halves (the fold
        // above is lane-local, not half-local), so the parity bytes read
        // out in element order.
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(&mut tmp, p);
        for j in 0..lanes_here {
            if tmp[j * bytes] & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx2.compress_parity",
        crate::kernel::trace::Substage::BitParity,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V3,
    );
    for i in lane..bits {
        let mut word = 0u64;
        for (shift, &byte) in src[i * bytes..(i + 1) * bytes].iter().enumerate() {
            word |= u64::from(byte) << (8 * shift);
        }
        if (word & tau).count_ones() & 1 == 1 {
            dst[i / 8] |= 1 << (i % 8);
        }
    }
}

/// `pshufb` control reversing bytes within each 32-bit lane (one half).
const REV_DWORD: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
/// `pshufb` control reversing bytes within each 64-bit lane (one half).
const REV_QWORD: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Fused image-membership and preimage row over 32-byte units.
///
/// Same single-read algorithm as
/// [`restrict_sse42`](super::sse42::restrict_sse42) at double width, with
/// the same `packs` fix-up for B=2.
///
/// # Panics
///
/// Panics if `src.len() != bits * bytes`, if either output is not
/// `bits.div_ceil(8)` bytes, or if `bytes` is not 1, 2, 4, or 8.
#[allow(clippy::cast_lossless, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn restrict_avx2(
    _token: archmage::X64V3Token,
    dst: &mut [u8],
    valid: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    assert_eq!(src.len(), bits.saturating_mul(bytes));
    assert_eq!(dst.len(), crate::bits::bytes_for(bits));
    assert_eq!(valid.len(), crate::bits::bytes_for(bits));
    assert!(matches!(bytes, 1 | 2 | 4 | 8));
    dst.fill(0);
    valid.fill(0);
    restrict_inner(dst, valid, one_w, src, bits, bytes);
}

/// Shared restrict body over 32-byte units.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v3, import_intrinsics)]
fn restrict_inner(
    dst: &mut [u8],
    valid: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    let zero = _mm256_setzero_si256();
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one_w & !width_mask == 0;
    let one_b = _mm256_set1_epi8(one_w as i8);
    let one_w16 = _mm256_set1_epi16(one_w as i16);
    let one_w32 = _mm256_set1_epi32(one_w as i32);
    let one_w64 = _mm256_set1_epi64x(one_w as i64);
    let (src_units, _) = src.as_chunks::<32>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 32 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm256_loadu_si256(unit);
        let is_zero = match bytes {
            1 => _mm256_cmpeq_epi8(v, zero),
            2 => _mm256_cmpeq_epi16(v, zero),
            4 => _mm256_cmpeq_epi32(v, zero),
            _ => _mm256_cmpeq_epi64(v, zero),
        };
        let is_one = if one_in_range {
            match bytes {
                1 => _mm256_cmpeq_epi8(v, one_b),
                2 => _mm256_cmpeq_epi16(v, one_w16),
                4 => _mm256_cmpeq_epi32(v, one_w32),
                _ => _mm256_cmpeq_epi64(v, one_w64),
            }
        } else {
            zero
        };
        let valid_v = _mm256_or_si256(is_zero, is_one);
        let (valid_bits, one_bits) = match bytes {
            1 => (
                _mm256_movemask_epi8(valid_v) as u32,
                _mm256_movemask_epi8(is_one) as u32,
            ),
            2 => {
                let pv = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(valid_v, zero));
                let po = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(is_one, zero));
                (
                    _mm256_movemask_epi8(pv) as u32 & 0xffff,
                    _mm256_movemask_epi8(po) as u32 & 0xffff,
                )
            }
            4 => (
                _mm256_movemask_ps(_mm256_castsi256_ps(valid_v)) as u32 & 0xff,
                _mm256_movemask_ps(_mm256_castsi256_ps(is_one)) as u32 & 0xff,
            ),
            _ => (
                _mm256_movemask_pd(_mm256_castsi256_pd(valid_v)) as u32 & 0x0f,
                _mm256_movemask_pd(_mm256_castsi256_pd(is_one)) as u32 & 0x0f,
            ),
        };
        for j in 0..lanes_here {
            if valid_bits >> j & 1 == 1 {
                valid[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
            if one_bits >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx2.restrict",
        crate::kernel::trace::Substage::Linear,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V3,
    );
    for i in lane..bits {
        let mut word = 0u64;
        for (shift, &byte) in src[i * bytes..(i + 1) * bytes].iter().enumerate() {
            word |= u64::from(byte) << (8 * shift);
        }
        if word == 0 || word == one_w {
            valid[i / 8] |= 1 << (i % 8);
        }
        if word == one_w {
            dst[i / 8] |= 1 << (i % 8);
        }
    }
}
