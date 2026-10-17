//! SSE4.2 bit-packed GF(2) interop kernels over 16-byte lanes.
//!
//! `V2` tier (x86-64-v2 guarantees SSE4.1 `pshufb`/`blendv` and SSE4.2
//! `pcmpgtq`): each entry is a safe capability-token function taking an
//! `X64V2Token` and validating its own geometry. Full 16-byte units run the
//! vector body; any remainder lane falls back to the portable scalar loop,
//! which owns the bit indexing.

use crate::kernel::gf1::LaneTest;

/// Expand packed bits into target lanes over 16-byte units.
///
/// Element `i` is bit `i % 8` of `src[i / 8]`; lane `i` is the `bytes`-byte
/// little-endian truncation of `one_w` when the bit is set, zero otherwise.
///
/// One `and` with the selector mask keeps only the slot each lane's bit
/// owns, the lane-width compare against zero turns a set bit into an all-one
/// lane mask, and the final `and` selects the broadcast one.
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
pub fn expand_sse42(
    _token: archmage::X64V2Token,
    dst: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    assert_eq!(dst.len(), bits.saturating_mul(bytes));
    assert!(matches!(bytes, 1 | 2 | 4 | 8));
    assert!(crate::bits::bytes_for(bits) <= src.len());
    let (dst_units, _) = dst.as_chunks_mut::<16>();
    let lanes = match bytes {
        1 => {
            let sel = _mm_loadu_si128(&EXPAND_SEL_B1);
            let one = _mm_set1_epi8(one_w as i8);
            let zero = _mm_setzero_si128();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `16 * u..16 * u + 16`, fed by source
                // bytes `2 * u` and `2 * u + 1`.
                let mut rep = [0u8; 16];
                rep[..8].fill(src[2 * u]);
                rep[8..].fill(src[2 * u + 1]);
                let masked = _mm_and_si128(_mm_loadu_si128(&rep), sel);
                let is_zero = _mm_cmpeq_epi8(masked, zero);
                _mm_storeu_si128(unit, _mm_andnot_si128(is_zero, one));
            }
            dst_units.len() * 16
        }
        2 => {
            let sel = _mm_loadu_si128(&EXPAND_SEL_B2);
            let one = _mm_set1_epi16(one_w as i16);
            let zero = _mm_setzero_si128();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `8 * u..8 * u + 8`, fed by source
                // byte `u`.
                let mut rep = [0u8; 16];
                rep.fill(src[u]);
                let masked = _mm_and_si128(_mm_loadu_si128(&rep), sel);
                let is_zero = _mm_cmpeq_epi16(masked, zero);
                _mm_storeu_si128(unit, _mm_andnot_si128(is_zero, one));
            }
            dst_units.len() * 8
        }
        4 => {
            let sel_even = _mm_loadu_si128(&EXPAND_SEL_B4_LO);
            let sel_odd = _mm_loadu_si128(&EXPAND_SEL_B4_HI);
            let one = _mm_set1_epi32(one_w as i32);
            let zero = _mm_setzero_si128();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `4 * u..4 * u + 4`; the base bit is
                // `4 * u`, so even units test bits 0..3 of their source
                // byte and odd units bits 4..7.
                let mut rep = [0u8; 16];
                rep.fill(src[(4 * u) / 8]);
                let sel = if u % 2 == 0 { sel_even } else { sel_odd };
                let masked = _mm_and_si128(_mm_loadu_si128(&rep), sel);
                let is_zero = _mm_cmpeq_epi32(masked, zero);
                _mm_storeu_si128(unit, _mm_andnot_si128(is_zero, one));
            }
            dst_units.len() * 4
        }
        _ => {
            let sels = [
                _mm_loadu_si128(&EXPAND_SEL_B8_0),
                _mm_loadu_si128(&EXPAND_SEL_B8_2),
                _mm_loadu_si128(&EXPAND_SEL_B8_4),
                _mm_loadu_si128(&EXPAND_SEL_B8_6),
            ];
            let one = _mm_set1_epi64x(one_w as i64);
            let zero = _mm_setzero_si128();
            for (u, unit) in dst_units.iter_mut().enumerate() {
                // Unit `u` covers lanes `2 * u..2 * u + 2`; the base bit is
                // `2 * u`, cycling through 0, 2, 4, 6 mod 8.
                let mut rep = [0u8; 16];
                rep.fill(src[(2 * u) / 8]);
                let masked = _mm_and_si128(_mm_loadu_si128(&rep), sels[u % 4]);
                let is_zero = _mm_cmpeq_epi64(masked, zero);
                _mm_storeu_si128(unit, _mm_andnot_si128(is_zero, one));
            }
            dst_units.len() * 2
        }
    };
    // Remainder lanes run the portable indexing.
    for i in lanes..bits {
        let bit = (src[i / 8] >> (i % 8)) & 1;
        let lane = one_w & 0u64.wrapping_sub(u64::from(bit));
        let lane_bytes = lane.to_le_bytes();
        dst[i * bytes..(i + 1) * bytes].copy_from_slice(&lane_bytes[..bytes]);
    }
}

/// Selector masks for [`expand_sse42`]: slot `j` of each lane keeps only the
/// bit that lane's element owns.
///
/// B=1: byte lane `j` owns bit `j % 8` of its replicated source byte.
/// B=2: word lane `j` owns bit `j % 8`, tested in both bytes of the lane.
/// B=4: dword lane `j` owns bit `j` (even units) or `4 + j` (odd units).
/// B=8: qword lane `j` owns bit `phase + j` for the unit's phase.
const EXPAND_SEL_B1: [u8; 16] = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
];
const EXPAND_SEL_B2: [u8; 16] = [
    0x01, 0x01, 0x02, 0x02, 0x04, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80,
];
const EXPAND_SEL_B4_LO: [u8; 16] = [
    0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
];
const EXPAND_SEL_B4_HI: [u8; 16] = [
    0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80,
];
const EXPAND_SEL_B8_0: [u8; 16] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
];
const EXPAND_SEL_B8_2: [u8; 16] = [
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
];
const EXPAND_SEL_B8_4: [u8; 16] = [
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
];
const EXPAND_SEL_B8_6: [u8; 16] = [
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
];
/// `pshufb` control reversing bytes within each 32-bit lane.
const REV_DWORD: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
/// `pshufb` control reversing bytes within each 64-bit lane.
const REV_QWORD: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Compress target lanes into a packed bitmap over 16-byte units.
///
/// Lane `i` is the little-endian word of `bytes` bytes at
/// `src[i * bytes..]`; bit `i` of `dst` is the `LaneTest` predicate over
/// that word.
///
/// # Panics
///
/// Panics if `src.len() != bits * bytes`, if `dst.len()` is not
/// `bits.div_ceil(8)`, or if `bytes` is not 1, 2, 4, or 8.
#[allow(clippy::cast_lossless, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn compress_sse42(
    _token: archmage::X64V2Token,
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
        LaneTest::Parity(tau) => compress_parity_sse42(dst, src, bits, bytes, tau),
    }
}

/// Shared equality compress: bit `i` is `w == 0 || w == one` when `or_zero`
/// is set, else `w != 0`.
///
/// Compares each lane against zero and `one` at the lane width, then
/// extracts one bit per lane: B=1 reads the byte movemask directly, B=2
/// packs words to bytes first, B=4 reads the float movemask, and B=8 the
/// double movemask.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v2, import_intrinsics)]
fn compress_eq_inner(
    dst: &mut [u8],
    src: &[u8],
    bits: usize,
    bytes: usize,
    one: u64,
    or_zero: bool,
) {
    let zero = _mm_setzero_si128();
    let one_b = _mm_set1_epi8(one as i8);
    // The portable body compares the truncated lane word against the full
    // `one`: when `one` has bits above the lane width no lane can match, so
    // the equality side is all-false rather than the truncated compare.
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one & !width_mask == 0;
    let one_w16 = _mm_set1_epi16(one as i16);
    let one_w32 = _mm_set1_epi32(one as i32);
    let one_w64 = _mm_set1_epi64x(one as i64);
    let (src_units, _) = src.as_chunks::<16>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 16 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm_loadu_si128(unit);
        // Equality against zero at the lane width.
        let is_zero = match bytes {
            1 => _mm_cmpeq_epi8(v, zero),
            2 => _mm_cmpeq_epi16(v, zero),
            4 => _mm_cmpeq_epi32(v, zero),
            _ => _mm_cmpeq_epi64(v, zero),
        };
        // Equality against one at the lane width, gated on `one` fitting
        // the width (see above).
        let is_one = if one_in_range {
            match bytes {
                1 => _mm_cmpeq_epi8(v, one_b),
                2 => _mm_cmpeq_epi16(v, one_w16),
                4 => _mm_cmpeq_epi32(v, one_w32),
                _ => _mm_cmpeq_epi64(v, one_w64),
            }
        } else {
            zero
        };
        // `or_zero == false` selects the nonzero predicate: a lane passes
        // when it is not zero.
        let pass = if or_zero {
            _mm_or_si128(is_zero, is_one)
        } else {
            _mm_xor_si128(is_zero, _mm_set1_epi8(-1))
        };
        // One bit per lane, LSB-first: the movemask reads the sign bit of
        // each byte, so collapse each lane's comparison to its low byte.
        let lane_bits = match bytes {
            1 => _mm_movemask_epi8(pass) as u32,
            2 => _mm_movemask_epi8(_mm_packs_epi16(pass, zero)) as u32 & 0xff,
            4 => _mm_movemask_ps(_mm_castsi128_ps(pass)) as u32 & 0x0f,
            _ => _mm_movemask_pd(_mm_castsi128_pd(pass)) as u32 & 0x03,
        };
        for j in 0..lanes_here {
            if lane_bits >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
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

/// Parity predicate: bit `i` is the parity of `lane_word(i) & tau`.
///
/// ANDs the broadcast tau, folds each lane's bytes to the lane's low byte
/// with lane-local `pshufb` reversals, then looks up each nibble's parity
/// with `pshufb`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
#[archmage::rite(v2, import_intrinsics)]
fn compress_parity_sse42(dst: &mut [u8], src: &[u8], bits: usize, bytes: usize, tau: u64) {
    // Nibble parity: entry `n` is the parity of the 4-bit value `n`.
    const PARITY_LUT: [u8; 16] = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];
    let lut = _mm_loadu_si128(&PARITY_LUT);
    let tau_b = _mm_set1_epi8(tau as i8);
    let tau_w16 = _mm_set1_epi16(tau as i16);
    let tau_w32 = _mm_set1_epi32(tau as i32);
    let tau_w64 = _mm_set1_epi64x(tau as i64);
    let mask_lo = _mm_set1_epi8(0x0f);
    let (src_units, _) = src.as_chunks::<16>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 16 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm_loadu_si128(unit);
        let masked = match bytes {
            1 => _mm_and_si128(v, tau_b),
            2 => _mm_and_si128(v, tau_w16),
            4 => _mm_and_si128(v, tau_w32),
            _ => _mm_and_si128(v, tau_w64),
        };
        // Fold each lane's bytes into the lane's low byte. The shifts are
        // element-granular, so bytes never cross a lane edge — but a pure
        // shift pyramid leaves the full XOR in a nonzero slot for 4- and
        // 8-byte lanes. `pshufb` reversals are lane-local too, so fold wide
        // lanes through reversals that accumulate at byte 0.
        let folded = match bytes {
            1 => masked,
            2 => _mm_xor_si128(masked, _mm_srli_epi16::<8>(masked)),
            4 => {
                let rev = _mm_shuffle_epi8(masked, _mm_loadu_si128(&REV_DWORD));
                let pairs = _mm_xor_si128(masked, rev);
                _mm_xor_si128(pairs, _mm_srli_epi16::<8>(pairs))
            }
            _ => {
                let rev_q = _mm_shuffle_epi8(masked, _mm_loadu_si128(&REV_QWORD));
                let t1 = _mm_xor_si128(masked, rev_q);
                let rev_d = _mm_shuffle_epi8(t1, _mm_loadu_si128(&REV_DWORD));
                let t2 = _mm_xor_si128(t1, rev_d);
                _mm_xor_si128(t2, _mm_srli_epi16::<8>(t2))
            }
        };
        // Nibble parity lookup over the folded bytes.
        let lo_nib = _mm_and_si128(folded, mask_lo);
        let hi_nib = _mm_and_si128(_mm_srli_epi16::<4>(folded), mask_lo);
        let p = _mm_xor_si128(_mm_shuffle_epi8(lut, lo_nib), _mm_shuffle_epi8(lut, hi_nib));
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(&mut tmp, p);
        for j in 0..lanes_here {
            if tmp[j * bytes] & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
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

/// Fused image-membership and preimage row over 16-byte units.
///
/// `valid` bit `i` is `w == 0 || w == one_w`, `dst` bit `i` is
/// `w == one_w`; each lane is read once.
///
/// # Panics
///
/// Panics if `src.len() != bits * bytes`, if either output is not
/// `bits.div_ceil(8)` bytes, or if `bytes` is not 1, 2, 4, or 8.
#[allow(clippy::cast_lossless, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn restrict_sse42(
    _token: archmage::X64V2Token,
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

/// Shared restrict body: one vector pass reads each lane once, extracting
/// the membership and equality bit rows through the width's movemask.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v2, import_intrinsics)]
fn restrict_inner(
    dst: &mut [u8],
    valid: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    let zero = _mm_setzero_si128();
    let one_b = _mm_set1_epi8(one_w as i8);
    // Same width-fit gate as [`compress_eq_inner`]: an out-of-range `one`
    // matches no lane.
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one_w & !width_mask == 0;
    let one_w16 = _mm_set1_epi16(one_w as i16);
    let one_w32 = _mm_set1_epi32(one_w as i32);
    let one_w64 = _mm_set1_epi64x(one_w as i64);
    let (src_units, _) = src.as_chunks::<16>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 16 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm_loadu_si128(unit);
        let is_zero = match bytes {
            1 => _mm_cmpeq_epi8(v, zero),
            2 => _mm_cmpeq_epi16(v, zero),
            4 => _mm_cmpeq_epi32(v, zero),
            _ => _mm_cmpeq_epi64(v, zero),
        };
        let is_one = if one_in_range {
            match bytes {
                1 => _mm_cmpeq_epi8(v, one_b),
                2 => _mm_cmpeq_epi16(v, one_w16),
                4 => _mm_cmpeq_epi32(v, one_w32),
                _ => _mm_cmpeq_epi64(v, one_w64),
            }
        } else {
            zero
        };
        let valid_v = _mm_or_si128(is_zero, is_one);
        let valid_bits = match bytes {
            1 => _mm_movemask_epi8(valid_v) as u32,
            2 => _mm_movemask_epi8(_mm_packs_epi16(valid_v, zero)) as u32 & 0xff,
            4 => _mm_movemask_ps(_mm_castsi128_ps(valid_v)) as u32 & 0x0f,
            _ => _mm_movemask_pd(_mm_castsi128_pd(valid_v)) as u32 & 0x03,
        };
        let one_bits = match bytes {
            1 => _mm_movemask_epi8(is_one) as u32,
            2 => _mm_movemask_epi8(_mm_packs_epi16(is_one, zero)) as u32 & 0xff,
            4 => _mm_movemask_ps(_mm_castsi128_ps(is_one)) as u32 & 0x0f,
            _ => _mm_movemask_pd(_mm_castsi128_pd(is_one)) as u32 & 0x03,
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
