//! GF(2) bit-packed kernel differentials against the per-bit scalar
//! oracle.

/// The differential partner of the GF(2) kernels is the
/// per-bit scalar oracle over `gf1::Elem` — a genuinely different path from
/// the word loops (no packing, no masks). Geometry sweeps the same tail and
/// boundary cases the public suite drives, minus the wrapper checks, so a
/// wrapper bug cannot mask a kernel bug.
use super::Vec;

use crate::field::binary::Polynomial;
use crate::field::{Elem, FiniteField};
use crate::kernel::gf1 as k;

fn bit(buf: &[u8], i: usize) -> bool {
    buf[i / 8] >> (i % 8) & 1 == 1
}

fn put(buf: &mut [u8], i: usize, value: bool) {
    let mask = 1u8 << (i % 8);
    if value {
        buf[i / 8] |= mask;
    } else {
        buf[i / 8] &= !mask;
    }
}

fn zeros(len: usize) -> Vec<u8> {
    (0..len).map(|_| 0u8).collect()
}

fn noise_bits(bits: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut buf = zeros(bits.div_ceil(8));
    for byte in &mut buf {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *byte = (state >> 33) as u8;
    }
    for i in bits..8 * buf.len() {
        put(&mut buf, i, false);
    }
    buf
}

fn assert_padding_zero(buf: &[u8], bits: usize) {
    for i in bits..8 * buf.len() {
        assert!(!bit(buf, i), "padding bit {i} set");
    }
}

/// Element counts across sub-byte, byte, word, and word+tail boundaries.
const BITS: &[usize] = &[0, 1, 3, 7, 8, 9, 13, 63, 64, 65, 100, 128, 200, 257];

#[test]
fn whole_buffer_kernels_match_oracle() {
    for &bits_len in BITS {
        let a = noise_bits(bits_len, 0x10 + bits_len as u64);
        let b = noise_bits(bits_len, 0x20 + bits_len as u64);

        let mut dst = a.clone();
        k::xor(&mut dst, &b);
        for i in 0..bits_len {
            assert_eq!(bit(&dst, i), bit(&a, i) ^ bit(&b, i), "xor {bits_len}/{i}");
        }

        let mut dst = noise_bits(8 * a.len(), 0);
        k::and_into(&mut dst, &a, &b);
        for i in 0..8 * dst.len() {
            assert_eq!(
                bit(&dst, i),
                bit(&a, i) && bit(&b, i),
                "and_into {bits_len}/{i}"
            );
        }

        let mut dst = a.clone();
        k::and_assign(&mut dst, &b);
        for i in 0..bits_len {
            assert_eq!(
                bit(&dst, i),
                bit(&a, i) && bit(&b, i),
                "and_assign {bits_len}/{i}"
            );
        }

        let mut dst = a.clone();
        k::andnot_assign(&mut dst, &b);
        for i in 0..bits_len {
            assert_eq!(
                bit(&dst, i),
                bit(&a, i) && !bit(&b, i),
                "andnot {bits_len}/{i}"
            );
        }
        assert_padding_zero(&dst, bits_len);
    }
}

#[test]
fn range_kernels_match_oracle() {
    for &bits_len in BITS {
        for from in [0, 1, 5, 8, 63, 64] {
            for span in [1, 3, 64, 65] {
                let to = from + span;
                if to > bits_len {
                    continue;
                }
                let src = noise_bits(bits_len, 0x30 + bits_len as u64);

                let mut dst = noise_bits(bits_len, 0x40 + bits_len as u64);
                let mut oracle = dst.clone();
                k::xor_window(&mut dst, &src, &k::window(from, to));
                for i in from..to {
                    if bit(&src, i) {
                        let flipped = !bit(&oracle, i);
                        put(&mut oracle, i, flipped);
                    }
                }
                assert_eq!(dst, oracle, "xor_window {from}..{to} of {bits_len}");

                let mut dst = noise_bits(bits_len, 0x50 + bits_len as u64);
                let original = dst.clone();
                k::clear_range(&mut dst, from, to);
                for i in 0..bits_len {
                    let expected = !(from..to).contains(&i) && bit(&original, i);
                    assert_eq!(bit(&dst, i), expected, "clear {from}..{to}/{i}");
                }

                let mut dst = zeros(bits_len.div_ceil(8));
                k::set_range(&mut dst, from, to);
                for i in 0..bits_len {
                    assert_eq!(
                        bit(&dst, i),
                        (from..to).contains(&i),
                        "set {from}..{to}/{i}"
                    );
                }
                assert_padding_zero(&dst, bits_len);
            }
        }
    }
}

#[test]
fn weight_parity_and_gather_match_oracle() {
    for &bits_len in BITS {
        let a = noise_bits(bits_len, 0x60 + bits_len as u64);
        let b = noise_bits(bits_len, 0x70 + bits_len as u64);

        let expected_weight: u32 = (0..bits_len).map(|i| u32::from(bit(&a, i))).sum();
        assert_eq!(
            k::weight(&a, bits_len),
            expected_weight,
            "weight {bits_len}"
        );

        let mut acc = Elem::<FiniteField<2, 1, Polynomial<3>>>::ZERO;
        for i in 0..bits_len {
            acc = acc.add(Elem::<FiniteField<2, 1, Polynomial<3>>>::from_raw(
                u8::from(bit(&a, i) && bit(&b, i)),
            ));
        }
        assert_eq!(
            Elem::<FiniteField<2, 1, Polynomial<3>>>::from_raw(k::parity(&a, &b, bits_len) as u8),
            acc,
            "parity {bits_len}"
        );

        let srcs: Vec<Vec<u8>> = (0..4)
            .map(|index| noise_bits(bits_len, 0x80 + index + bits_len as u64))
            .collect();
        let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        for selector in [0u64, 0b1, 0b1111, 0b1010] {
            let mut dst = noise_bits(bits_len, 0x90 + selector + bits_len as u64);
            let mut oracle = dst.clone();
            k::xor_gather(&mut dst, &refs, selector);
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
            assert_eq!(dst, oracle, "xor_gather {selector:#b}/{bits_len}");
            assert_padding_zero(&dst, bits_len);
        }
    }
}

/// Empty ranges are no-ops through the kernel layer too.
#[test]
fn empty_ranges_are_no_ops() {
    let mut dst = [0b1010_1010u8];
    let src = [0b0101_0101u8];
    k::xor_window(&mut dst, &src, &k::window(3, 3));
    k::clear_range(&mut dst, 4, 4);
    k::set_range(&mut dst, 5, 5);
    assert_eq!(dst, [0b1010_1010]);
    k::xor_gather(&mut dst, &[], 0);
    assert_eq!(dst, [0b1010_1010]);
}
