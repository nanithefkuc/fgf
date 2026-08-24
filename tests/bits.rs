//! Bit-packed GF(2) surface against an independent per-bit oracle.
//!
//! The oracle reads and writes single bits with explicit `i / 8` / `i % 8`
//! arithmetic — no word packing, no masks — so a kernel bug in the word loop,
//! the endianness assembly, or the tail masking cannot hide behind a
//! self-consistent implementation. The padding-bits-zero invariant is
//! re-asserted after every operation on every geometry: a kernel that reads
//! or writes one bit past the live region fails here.

use fgf::bits;

/// The shared fixed-seed LCG noise, as in the other suites.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Bit `i` of a packed buffer.
fn bit(buf: &[u8], i: usize) -> bool {
    buf[i / 8] >> (i % 8) & 1 == 1
}

/// Set or clear bit `i` of a packed buffer.
fn put(buf: &mut [u8], i: usize, value: bool) {
    if value {
        buf[i / 8] |= 1 << (i % 8);
    } else {
        buf[i / 8] &= !(1 << (i % 8));
    }
}

/// Zero every bit at or past `bits`, padding and surplus bytes alike.
fn clean_padding(buf: &mut [u8], bits: usize) {
    for i in bits..8 * buf.len() {
        put(buf, i, false);
    }
}

/// Assert the maintained invariant: all bits at or past `bits` are zero.
fn assert_padding_zero(buf: &[u8], bits: usize) {
    for i in bits..8 * buf.len() {
        assert!(!bit(buf, i), "padding bit {i} is set (byte {})", buf[i / 8]);
    }
}

/// A live-bit buffer with clean padding, in the geometry the kernels see:
/// sometimes exactly `bytes_for(bits)` long, sometimes word-padded the way
/// an aligned row consumer passes its storage.
fn live(bits: usize, seed: u64, pad_to_word: bool) -> Vec<u8> {
    let len = if pad_to_word {
        bits.div_ceil(64) * 8
    } else {
        bits.div_ceil(8)
    };
    let mut buf = noise(len, seed);
    clean_padding(&mut buf, bits);
    buf
}

/// Bit counts straddling every boundary: empty, sub-byte, partial bytes,
/// exact words, word-plus-partial-byte, and multi-word.
const BIT_LENGTHS: &[usize] = &[
    0, 1, 2, 3, 5, 6, 7, 8, 9, 13, 16, 17, 31, 32, 33, 63, 64, 65, 70, 100, 127, 128, 129, 191,
    192, 193, 255, 256, 257, 300, 1024,
];

/// Range pairs worth probing: the plan's frozen cross-byte case, byte and
/// word boundaries, single bits at both ends, and the whole live span.
fn ranges(bits: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for &(from, to) in &[
        (0, 0),
        (0, 1),
        (5, 13),
        (5, 5),
        (5, 6),
        (7, 8),
        (8, 16),
        (60, 70),
        (63, 64),
        (63, 65),
        (64, 65),
        (64, 128),
        (0, 64),
        (1, 100),
        (127, 128),
    ] {
        if to <= bits {
            pairs.push((from, to));
        }
    }
    if bits > 0 {
        pairs.push((0, bits));
        pairs.push((bits - 1, bits));
        if bits >= 3 {
            pairs.push((bits - 3, bits - 1));
        }
    }
    pairs
}

// ---------------------------------------------------------------------------
// Frozen layout fixtures (the interop fingerprint)
// ---------------------------------------------------------------------------

/// The wire convention, frozen: element `i` is bit `i % 8` of byte `i / 8`,
/// LSB-first. The expected bytes are spelled out independently of `set_range`
/// — changing the bit order is a format break that must fail this test.
#[test]
fn lsb_first_layout_fingerprint_is_frozen() {
    // Elements set at indices 0, 3, 8, 13, 31, 64, 69 over 70 bits (9 bytes).
    let set: &[usize] = &[0, 3, 8, 13, 31, 64, 69];
    let mut buf = vec![0u8; 9];
    for &i in set {
        bits::set_range(&mut buf, 70, i, i + 1);
    }
    let expected: [u8; 9] = [
        0b0000_1001, // bits 0 and 3
        0b0010_0001, // bits 8 (byte 1 bit 0) and 13 (byte 1 bit 5)
        0,
        0b1000_0000, // bit 31 (byte 3 bit 7)
        0,
        0,
        0,
        0,
        0b0010_0001, // bits 64 (byte 8 bit 0) and 69 (byte 8 bit 5)
    ];
    assert_eq!(buf, expected);
    assert_padding_zero(&buf, 70);

    // And the scalar view of the same convention: element i is byte i/8,
    // bit i%8, read back per-bit.
    for i in 0..70 {
        assert_eq!(bit(&buf, i), set.contains(&i), "element {i}");
    }
}

/// The plan's named known-answer: a cross-byte `xor_range(_, _, 5, 13)`
/// touching two bytes and leaving bits `0..5` and `13..` untouched.
#[test]
fn xor_range_cross_byte_kat() {
    let src = [0xffu8, 0xff];
    let mut dst = [0b1010_1010u8, 0b0101_0101];
    bits::xor_range(&mut dst, &src, 16, 5, 13);

    // Byte 0: bits 0..5 keep 01010, bits 5..8 flip 101.
    assert_eq!(dst[0], 0b1010_1010 ^ 0b1110_0000);
    // Byte 1: elements 8..13 (bits 0..5) flip, elements 13..16 keep 010.
    assert_eq!(dst[1], 0b0101_0101 ^ 0b0001_1111);
}

/// `clear_range` leaves padding zero and `weight` counts only live bits.
#[test]
fn clear_and_weight_tail_kat() {
    let mut buf = [0b0011_1111u8]; // six live bits, padding already zero
    bits::clear_range(&mut buf, 6, 2, 4);
    assert_eq!(buf[0], 0b0011_0011);
    assert_eq!(bits::weight(&buf, 6), 4);
    // A full byte seen as three live bits counts three, not eight.
    assert_eq!(bits::weight(&[0xff], 3), 3);

    // Empty intersection under AND.
    let a = [0x0fu8];
    let b = [0xf0u8];
    let mut dst = [0u8; 1];
    bits::and_into(&mut dst, &a, &b);
    assert_eq!(dst[0], 0);
}

// ---------------------------------------------------------------------------
// Differential sweeps against the per-bit oracle
// ---------------------------------------------------------------------------

#[test]
fn xor_matches_per_bit_oracle() {
    for &bits_len in BIT_LENGTHS {
        for &word_padded in &[false, true] {
            let src = live(bits_len, 0x100 + bits_len as u64, word_padded);
            let mut dst = live(bits_len, 0x200 + bits_len as u64, word_padded);
            let mut oracle = dst.clone();
            for i in 0..bits_len {
                if bit(&src, i) {
                    let flipped = !bit(&oracle, i);
                    put(&mut oracle, i, flipped);
                }
            }
            bits::xor(&mut dst, &src);
            assert_eq!(dst, oracle, "xor over {bits_len} bits");
            assert_padding_zero(&dst, bits_len);
        }
    }
}

#[test]
fn and_ops_match_per_bit_oracle() {
    for &bits_len in BIT_LENGTHS {
        for &word_padded in &[false, true] {
            let a = live(bits_len, 0x300 + bits_len as u64, word_padded);
            let b = live(bits_len, 0x400 + bits_len as u64, word_padded);

            let mut dst = vec![0xa5u8; a.len()];
            bits::and_into(&mut dst, &a, &b);
            for i in 0..8 * dst.len() {
                assert_eq!(
                    bit(&dst, i),
                    bit(&a, i) && bit(&b, i),
                    "and_into {bits_len}/{i}"
                );
            }

            let mut dst = live(bits_len, 0x500 + bits_len as u64, word_padded);
            let mut oracle = dst.clone();
            for i in 0..8 * dst.len() {
                let kept = bit(&oracle, i) && bit(&b, i);
                put(&mut oracle, i, kept);
            }
            bits::and_assign(&mut dst, &b);
            assert_eq!(dst, oracle, "and_assign over {bits_len} bits");

            let mut dst = live(bits_len, 0x600 + bits_len as u64, word_padded);
            let mut oracle = dst.clone();
            for i in 0..8 * dst.len() {
                let kept = bit(&oracle, i) && !bit(&b, i);
                put(&mut oracle, i, kept);
            }
            bits::andnot_assign(&mut dst, &b);
            assert_eq!(dst, oracle, "andnot_assign over {bits_len} bits");
            assert_padding_zero(&dst, bits_len);
        }
    }
}

#[test]
fn range_ops_match_per_bit_oracle() {
    for &bits_len in BIT_LENGTHS {
        for &(from, to) in &ranges(bits_len) {
            let src = live(bits_len, 0x700 + bits_len as u64, false);
            let base = live(bits_len, 0x800 + bits_len as u64, false);

            // xor_range, and the split prepare/apply form of the same op
            let mut dst = base.clone();
            let mut oracle = base.clone();
            for i in from..to {
                if bit(&src, i) {
                    let flipped = !bit(&oracle, i);
                    put(&mut oracle, i, flipped);
                }
            }
            bits::xor_range(&mut dst, &src, bits_len, from, to);
            assert_eq!(dst, oracle, "xor_range {from}..{to} of {bits_len}");
            assert_padding_zero(&dst, bits_len);

            let mut split = base.clone();
            let plan = bits::RangeXor::new(bits_len, from, to);
            bits::xor_range_with(&mut split, &src, &plan);
            assert_eq!(split, dst, "xor_range_with {from}..{to} of {bits_len}");
            // The plan applies repeatedly and to buffers of different
            // lengths: a longer `src` contributes nothing past its live
            // bits, exactly like the one-shot form.
            let mut long_src = src.clone();
            long_src.resize(src.len() + 8, 0xA5);
            bits::xor_range_with(&mut split, &long_src, &plan);
            bits::xor_range(&mut dst, &src, bits_len, from, to);
            assert_eq!(split, dst, "repeated apply {from}..{to} of {bits_len}");
            assert_padding_zero(&split, bits_len);

            // clear_range
            let mut dst = base.clone();
            bits::clear_range(&mut dst, bits_len, from, to);
            for i in 0..bits_len {
                let expected = if (from..to).contains(&i) {
                    false
                } else {
                    bit(&base, i)
                };
                assert_eq!(bit(&dst, i), expected, "clear_range {from}..{to}/{i}");
            }
            assert_padding_zero(&dst, bits_len);

            // set_range
            let mut dst = base.clone();
            bits::set_range(&mut dst, bits_len, from, to);
            for i in 0..bits_len {
                let expected = if (from..to).contains(&i) {
                    true
                } else {
                    bit(&base, i)
                };
                assert_eq!(bit(&dst, i), expected, "set_range {from}..{to}/{i}");
            }
            assert_padding_zero(&dst, bits_len);
        }
    }
}

#[test]
fn weight_and_parity_match_per_bit_oracle() {
    for &bits_len in BIT_LENGTHS {
        for &word_padded in &[false, true] {
            let a = live(bits_len, 0x900 + bits_len as u64, word_padded);
            let b = live(bits_len, 0xa00 + bits_len as u64, word_padded);

            let expected_weight: u32 = (0..bits_len).map(|i| u32::from(bit(&a, i))).sum();
            assert_eq!(
                bits::weight(&a, bits_len),
                expected_weight,
                "weight {bits_len}"
            );

            let mut acc = 0u32;
            for i in 0..bits_len {
                acc += u32::from(bit(&a, i) && bit(&b, i));
            }
            let expected = u8::from(!acc.is_multiple_of(2));
            assert_eq!(
                bits::parity_dot(&a, &b, bits_len).to_raw(),
                expected,
                "parity_dot {bits_len}"
            );
        }
    }
}

#[test]
fn xor_gather_matches_per_bit_oracle() {
    for &bits_len in &[0, 1, 8, 13, 64, 65, 100, 300] {
        const SOURCES: usize = 5;
        let srcs: Vec<Vec<u8>> = (0..SOURCES)
            .map(|index| live(bits_len, 0xb00 + index as u64 + bits_len as u64, false))
            .collect();
        let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        let selectors = [0u64, 1, (1 << SOURCES) - 1, 0b01010, 0b10101, 0b10000];

        for &selector in &selectors {
            let mut dst = live(bits_len, 0xc00 + selector + bits_len as u64, false);
            let mut oracle = dst.clone();
            for (index, src) in srcs.iter().enumerate() {
                if selector >> index & 1 == 1 {
                    for i in 0..bits_len {
                        if bit(src, i) {
                            let flipped = !bit(&oracle, i);
                            put(&mut oracle, i, flipped);
                        }
                    }
                }
            }
            bits::xor_gather(&mut dst, &refs, selector, bits_len);
            assert_eq!(dst, oracle, "xor_gather {selector:#b} over {bits_len} bits");
            assert_padding_zero(&dst, bits_len);
        }
    }
}

/// A word-padded consumer (the `gfm` row shape: 100 columns in 16 aligned
/// bytes) sees its surplus bytes honored as padding, not live elements.
#[test]
fn surplus_bytes_are_padding_not_elements() {
    let mut buf = live(100, 0xd00, true);
    assert_eq!(buf.len(), 16);

    // weight and parity stop at 100 bits.
    let expected_weight: u32 = (0..100).map(|i| u32::from(bit(&buf, i))).sum();
    assert_eq!(bits::weight(&buf, 100), expected_weight);
    let mut acc = 0u32;
    for i in 0..100 {
        acc += u32::from(bit(&buf, i));
    }
    assert_eq!(
        u32::from(bits::parity_dot(&buf, &buf, 100).to_raw()),
        acc % 2
    );

    // set_range stays inside the live span.
    bits::set_range(&mut buf, 100, 90, 100);
    assert_eq!(buf[13..], [0, 0, 0]);
    assert_padding_zero(&buf, 100);
}

/// A full-width selector drives all 64 possible sources — the arm where
/// `srcs.len() < 64` no longer holds.
#[test]
fn xor_gather_accepts_a_full_u64_selector() {
    let srcs: Vec<Vec<u8>> = (0..64)
        .map(|index| vec![(index as u8) & 0x0f | 0x10])
        .collect();
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let mut dst = vec![0u8; 1];

    bits::xor_gather(&mut dst, &refs, 1 << 63, 8);
    assert_eq!(dst[0], srcs[63][0], "only source 63 contributes");

    let mut dst = vec![0u8; 1];
    bits::xor_gather(&mut dst, &refs, u64::MAX, 8);
    let expected = srcs.iter().fold(0u8, |acc, src| acc ^ src[0]);
    assert_eq!(dst[0], expected, "every source contributes");
}

// ---------------------------------------------------------------------------
// Geometry panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "bits::xor: dst is 2 bytes but src is 3 bytes")]
fn xor_rejects_length_mismatch() {
    bits::xor(&mut [0u8; 2], &[0u8; 3]);
}

#[test]
#[should_panic(expected = "bits::xor_range: dst is 2 bytes but src is 3 bytes")]
fn xor_range_rejects_length_mismatch() {
    bits::xor_range(&mut [0u8; 2], &[0u8; 3], 16, 0, 8);
}

#[test]
#[should_panic(expected = "bits::xor_range: range start 9 exceeds end 5")]
fn xor_range_rejects_inverted_range() {
    bits::xor_range(&mut [0u8; 2], &[0u8; 2], 16, 9, 5);
}

#[test]
#[should_panic(expected = "bits::xor_range: range end 13 exceeds the 8 live bits")]
fn xor_range_rejects_range_past_live_bits() {
    bits::xor_range(&mut [0u8; 1], &[0u8; 1], 8, 5, 13);
}

#[test]
#[should_panic(expected = "RangeXor::new: range start 9 exceeds end 5")]
fn range_xor_rejects_inverted_range() {
    let _ = bits::RangeXor::new(16, 9, 5);
}

#[test]
#[should_panic(expected = "RangeXor::new: range end 13 exceeds the 8 live bits")]
fn range_xor_rejects_range_past_live_bits() {
    let _ = bits::RangeXor::new(8, 5, 13);
}

#[test]
#[should_panic(
    expected = "bits::xor_range_with: the planned range needs 2 bytes but dst holds 1 and src holds 1"
)]
fn xor_range_with_rejects_short_buffers() {
    let plan = bits::RangeXor::new(13, 5, 13);
    bits::xor_range_with(&mut [0u8; 1], &[0u8; 1], &plan);
}
#[test]
#[should_panic(expected = "bits::xor_range: 13 bits need at least 2 bytes but dst holds 1")]
fn xor_range_rejects_short_buffer() {
    bits::xor_range(&mut [0u8; 1], &[0u8; 1], 13, 5, 13);
}

#[test]
#[should_panic(expected = "bits::and_into: dst is 2 bytes but a is 3 bytes")]
fn and_into_rejects_dst_mismatch() {
    bits::and_into(&mut [0u8; 2], &[0u8; 3], &[0u8; 2]);
}

#[test]
#[should_panic(expected = "bits::and_into: a is 3 bytes but b is 2 bytes")]
fn and_into_rejects_input_mismatch() {
    bits::and_into(&mut [0u8; 3], &[0u8; 3], &[0u8; 2]);
}

#[test]
#[should_panic(expected = "bits::and_assign: dst is 2 bytes but src is 4 bytes")]
fn and_assign_rejects_length_mismatch() {
    bits::and_assign(&mut [0u8; 2], &[0u8; 4]);
}

#[test]
#[should_panic(expected = "bits::andnot_assign: dst is 2 bytes but mask is 1 bytes")]
fn andnot_assign_rejects_length_mismatch() {
    bits::andnot_assign(&mut [0u8; 2], &[0u8; 1]);
}

#[test]
#[should_panic(expected = "bits::clear_range: range start 7 exceeds end 2")]
fn clear_range_rejects_inverted_range() {
    bits::clear_range(&mut [0u8; 1], 8, 7, 2);
}

#[test]
#[should_panic(expected = "bits::clear_range: range end 9 exceeds the 8 live bits")]
fn clear_range_rejects_range_past_live_bits() {
    bits::clear_range(&mut [0u8; 1], 8, 0, 9);
}

#[test]
#[should_panic(expected = "bits::clear_range: 9 bits need at least 2 bytes but dst holds 1")]
fn clear_range_rejects_short_buffer() {
    bits::clear_range(&mut [0u8; 1], 9, 0, 9);
}

#[test]
#[should_panic(expected = "bits::set_range: range start 4 exceeds end 2")]
fn set_range_rejects_inverted_range() {
    bits::set_range(&mut [0u8; 1], 8, 4, 2);
}

#[test]
#[should_panic(expected = "bits::set_range: range end 12 exceeds the 11 live bits")]
fn set_range_rejects_range_past_live_bits() {
    bits::set_range(&mut [0u8; 2], 11, 0, 12);
}

#[test]
#[should_panic(expected = "bits::set_range: 17 bits need at least 3 bytes but dst holds 2")]
fn set_range_rejects_short_buffer() {
    bits::set_range(&mut [0u8; 2], 17, 0, 17);
}

#[test]
#[should_panic(expected = "bits::weight: 9 bits need at least 2 bytes but buf holds 1")]
fn weight_rejects_short_buffer() {
    let _ = bits::weight(&[0u8; 1], 9);
}

#[test]
#[should_panic(expected = "bits::parity_dot: a is 2 bytes but b is 3 bytes")]
fn parity_dot_rejects_length_mismatch() {
    let _ = bits::parity_dot(&[0u8; 2], &[0u8; 3], 16);
}

#[test]
#[should_panic(expected = "bits::parity_dot: 12 bits need at least 2 bytes but a holds 1")]
fn parity_dot_rejects_short_buffer() {
    let _ = bits::parity_dot(&[0u8; 1], &[0u8; 1], 12);
}

#[test]
#[should_panic(expected = "bits::xor_gather: source 1 is 2 bytes but dst is 3 bytes")]
fn xor_gather_rejects_source_length_mismatch() {
    let srcs: [&[u8]; 2] = [&[0u8; 3], &[0u8; 2]];
    bits::xor_gather(&mut [0u8; 3], &srcs, 0b11, 24);
}

#[test]
#[should_panic(expected = "bits::xor_gather: selector names source 3 but only 3 sources exist")]
fn xor_gather_rejects_selector_past_sources() {
    let srcs: [&[u8]; 3] = [&[0u8; 1], &[0u8; 1], &[0u8; 1]];
    bits::xor_gather(&mut [0u8; 1], &srcs, 0b1000, 8);
}

#[test]
#[should_panic(expected = "bits::xor_gather: 17 bits need at least 3 bytes but dst holds 2")]
fn xor_gather_rejects_short_dst() {
    let srcs: [&[u8]; 1] = [&[0u8; 2]];
    bits::xor_gather(&mut [0u8; 2], &srcs, 1, 17);
}
