//! AVX-512 bit-packed GF(2) interop kernels over 64-byte lanes.
//!
//! `V4x` tier (AVX-512F+BW+VL+DQ plus BITALG and VPOPCNTDQ): each entry is a
//! safe capability-token function taking an `X64V4xToken` and validating its
//! own geometry. Expand broadcasts the one word through a mask-selected
//! move; compress compares (or popcounts, for parity) at the lane width and
//! stores the low element-count mask bits LSB-first. Full 64-byte units run
//! the vector body; any remainder lane falls back to the portable scalar
//! word loop.

use crate::kernel::gf1::LaneTest;

/// Expand packed bits into target lanes over 64-byte units.
///
/// Lanes arrive in groups of 64: the group's source bits form the opmask,
/// and `_mm512_maskz_mov_epi{8,16,32,64}` moves the broadcast one word into
/// the selected lanes.
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
pub fn expand_avx512(
    _token: archmage::X64V4xToken,
    dst: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    assert_eq!(dst.len(), bits.saturating_mul(bytes));
    assert!(matches!(bytes, 1 | 2 | 4 | 8));
    assert!(crate::bits::bytes_for(bits) <= src.len());
    let (dst_units, _) = dst.as_chunks_mut::<64>();
    let one_b = _mm512_set1_epi8(one_w as i8);
    let one_w16 = _mm512_set1_epi16(one_w as i16);
    let one_w32 = _mm512_set1_epi32(one_w as i32);
    let one_w64 = _mm512_set1_epi64(one_w as i64);
    let lanes = match bytes {
        1 => {
            for (u, unit) in dst_units.iter_mut().enumerate() {
                let base = 64 * u;
                if base + 64 > bits {
                    break;
                }
                let mut mask = 0u64;
                for j in 0..64 {
                    mask |= u64::from((src[(base + j) / 8] >> ((base + j) % 8)) & 1) << j;
                }
                _mm512_storeu_si512(unit, _mm512_maskz_mov_epi8(mask, one_b));
            }
            dst_units.len() * 64
        }
        2 => {
            for (u, unit) in dst_units.iter_mut().enumerate() {
                let base = 32 * u;
                if base + 32 > bits {
                    break;
                }
                let mut mask = 0u32;
                for j in 0..32 {
                    mask |= u32::from((src[(base + j) / 8] >> ((base + j) % 8)) & 1) << j;
                }
                _mm512_storeu_si512(unit, _mm512_maskz_mov_epi16(mask, one_w16));
            }
            dst_units.len() * 32
        }
        4 => {
            for (u, unit) in dst_units.iter_mut().enumerate() {
                let base = 16 * u;
                if base + 16 > bits {
                    break;
                }
                let mut mask = 0u16;
                for j in 0..16 {
                    mask |= u16::from((src[(base + j) / 8] >> ((base + j) % 8)) & 1) << j;
                }
                _mm512_storeu_si512(unit, _mm512_maskz_mov_epi32(mask, one_w32));
            }
            dst_units.len() * 16
        }
        _ => {
            for (u, unit) in dst_units.iter_mut().enumerate() {
                let base = 8 * u;
                if base + 8 > bits {
                    break;
                }
                let mut mask = 0u8;
                for j in 0..8 {
                    mask |= ((src[(base + j) / 8] >> ((base + j) % 8)) & 1) << j;
                }
                _mm512_storeu_si512(unit, _mm512_maskz_mov_epi64(mask, one_w64));
            }
            dst_units.len() * 8
        }
    };
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx512.expand",
        crate::kernel::trace::Substage::Unpack,
        crate::kernel::trace::Unit::Bits,
        lanes,
        crate::kernel::Backend::V4x,
    );
    // Remainder lanes run the portable indexing.
    for i in lanes..bits {
        let bit = (src[i / 8] >> (i % 8)) & 1;
        let lane = one_w & 0u64.wrapping_sub(u64::from(bit));
        let lane_bytes = lane.to_le_bytes();
        dst[i * bytes..(i + 1) * bytes].copy_from_slice(&lane_bytes[..bytes]);
    }
}

/// Compress target lanes into a packed bitmap over 64-byte units.
///
/// Lane `i` is the little-endian word of `bytes` bytes at
/// `src[i * bytes..]`; bit `i` of `dst` is the `LaneTest` predicate over
/// that word. Equality compares through `_mm512_cmpeq_epi*_mask`; parity
/// popcounts `w & tau` through the BITALG/VPOPCNTDQ entries and tests bit 0.
/// The low element-count mask bits store LSB-first.
///
/// # Panics
///
/// Panics if `src.len() != bits * bytes`, if `dst.len()` is not
/// `bits.div_ceil(8)`, or if `bytes` is not 1, 2, 4, or 8.
#[allow(clippy::cast_lossless, clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn compress_avx512(
    _token: archmage::X64V4xToken,
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
        LaneTest::Parity(tau) => compress_parity_avx512(dst, src, bits, bytes, tau),
    }
}

/// Shared equality compress over 64-byte units.
///
/// `or_zero == true` selects the membership predicate `w == 0 || w == one`;
/// `false` selects the nonzero predicate `w != 0`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v4x, import_intrinsics)]
fn compress_eq_inner(
    dst: &mut [u8],
    src: &[u8],
    bits: usize,
    bytes: usize,
    one: u64,
    or_zero: bool,
) {
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one & !width_mask == 0;
    let one_b = _mm512_set1_epi8(one as i8);
    let one_w16 = _mm512_set1_epi16(one as i16);
    let one_w32 = _mm512_set1_epi32(one as i32);
    let one_w64 = _mm512_set1_epi64(one as i64);
    let zero = _mm512_setzero_si512();
    let (src_units, _) = src.as_chunks::<64>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 64 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm512_loadu_si512(unit);
        let is_zero = match bytes {
            1 => _mm512_cmpeq_epi8_mask(v, zero) as u64,
            2 => _mm512_cmpeq_epi16_mask(v, zero) as u64,
            4 => _mm512_cmpeq_epi32_mask(v, zero) as u64,
            _ => _mm512_cmpeq_epi64_mask(v, zero) as u64,
        };
        let is_one = if one_in_range {
            match bytes {
                1 => _mm512_cmpeq_epi8_mask(v, one_b) as u64,
                2 => _mm512_cmpeq_epi16_mask(v, one_w16) as u64,
                4 => _mm512_cmpeq_epi32_mask(v, one_w32) as u64,
                _ => _mm512_cmpeq_epi64_mask(v, one_w64) as u64,
            }
        } else {
            0
        };
        let pass = if or_zero { is_zero | is_one } else { !is_zero };
        for j in 0..lanes_here {
            if pass >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx512.compress",
        crate::kernel::trace::Substage::Pack,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V4x,
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

/// Parity predicate over 64-byte units: bit `i` is the parity of
/// `lane_word(i) & tau`.
///
/// Popcounts `w & tau` through the BITALG 8/16-bit entries or the VPOPCNTDQ
/// 32/64-bit entries, then tests bit 0 of each count.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
#[archmage::rite(v4x, import_intrinsics)]
fn compress_parity_avx512(dst: &mut [u8], src: &[u8], bits: usize, bytes: usize, tau: u64) {
    let tau_b = _mm512_set1_epi8(tau as i8);
    let tau_w16 = _mm512_set1_epi16(tau as i16);
    let tau_w32 = _mm512_set1_epi32(tau as i32);
    let tau_w64 = _mm512_set1_epi64(tau as i64);
    let one_b = _mm512_set1_epi8(1);
    let one_w16 = _mm512_set1_epi16(1);
    let one_w32 = _mm512_set1_epi32(1);
    let one_w64 = _mm512_set1_epi64(1);
    let (src_units, _) = src.as_chunks::<64>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 64 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm512_loadu_si512(unit);
        let pass = match bytes {
            1 => {
                let counts = _mm512_popcnt_epi8(_mm512_and_si512(v, tau_b));
                _mm512_test_epi8_mask(counts, one_b) as u64
            }
            2 => {
                let counts = _mm512_popcnt_epi16(_mm512_and_si512(v, tau_w16));
                _mm512_test_epi16_mask(counts, one_w16) as u64
            }
            4 => {
                let counts = _mm512_popcnt_epi32(_mm512_and_si512(v, tau_w32));
                _mm512_test_epi32_mask(counts, one_w32) as u64
            }
            _ => {
                let counts = _mm512_popcnt_epi64(_mm512_and_si512(v, tau_w64));
                _mm512_test_epi64_mask(counts, one_w64) as u64
            }
        };
        for j in 0..lanes_here {
            if pass >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx512.compress_parity",
        crate::kernel::trace::Substage::BitParity,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V4x,
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

/// Fused image-membership and preimage row over 64-byte units.
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
pub fn restrict_avx512(
    _token: archmage::X64V4xToken,
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

/// Shared restrict body over 64-byte units.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[archmage::rite(v4x, import_intrinsics)]
fn restrict_inner(
    dst: &mut [u8],
    valid: &mut [u8],
    one_w: u64,
    src: &[u8],
    bits: usize,
    bytes: usize,
) {
    let width_mask: u64 = match bytes {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    let one_in_range = one_w & !width_mask == 0;
    let one_b = _mm512_set1_epi8(one_w as i8);
    let one_w16 = _mm512_set1_epi16(one_w as i16);
    let one_w32 = _mm512_set1_epi32(one_w as i32);
    let one_w64 = _mm512_set1_epi64(one_w as i64);
    let zero = _mm512_setzero_si512();
    let (src_units, _) = src.as_chunks::<64>();
    let mut lane = 0usize;
    for unit in src_units {
        let lanes_here = 64 / bytes;
        if lane + lanes_here > bits {
            break;
        }
        let v = _mm512_loadu_si512(unit);
        let is_zero = match bytes {
            1 => _mm512_cmpeq_epi8_mask(v, zero) as u64,
            2 => _mm512_cmpeq_epi16_mask(v, zero) as u64,
            4 => _mm512_cmpeq_epi32_mask(v, zero) as u64,
            _ => _mm512_cmpeq_epi64_mask(v, zero) as u64,
        };
        let is_one = if one_in_range {
            match bytes {
                1 => _mm512_cmpeq_epi8_mask(v, one_b) as u64,
                2 => _mm512_cmpeq_epi16_mask(v, one_w16) as u64,
                4 => _mm512_cmpeq_epi32_mask(v, one_w32) as u64,
                _ => _mm512_cmpeq_epi64_mask(v, one_w64) as u64,
            }
        } else {
            0
        };
        let valid_bits = is_zero | is_one;
        for j in 0..lanes_here {
            if valid_bits >> j & 1 == 1 {
                valid[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
            if is_one >> j & 1 == 1 {
                dst[(lane + j) / 8] |= 1 << ((lane + j) % 8);
            }
        }
        lane += lanes_here;
    }
    #[cfg(test)]
    crate::kernel::trace::emit(
        "gf1.avx512.restrict",
        crate::kernel::trace::Substage::Linear,
        crate::kernel::trace::Unit::Bits,
        lane,
        crate::kernel::Backend::V4x,
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
