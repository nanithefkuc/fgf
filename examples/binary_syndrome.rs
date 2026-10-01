//! Compute a Hamming-code syndrome from bit-packed binary rows.
//!
//! Run with `just example binary_syndrome`. Coordinates use LSB-first packing:
//! codeword position 1 is bit 0. A parity-check row contains a one at position
//! j when that position's binary index has the corresponding bit set.

#![forbid(unsafe_code)]

use fgf::bits;

const LENGTH: usize = 7;
const CHECKS: [[u8; 1]; 3] = [[0b0101_0101], [0b0110_0110], [0b0111_1000]];

fn main() {
    // Fixed generator rows for a systematic (7,4) Hamming code. The message
    // bits occupy codeword positions 3, 5, 6, 7; positions 1, 2, 4 are parity.
    let generator = [[0b0000_0111], [0b0001_1001], [0b0010_1010], [0b0100_1011]];
    let sources: [&[u8]; 4] = generator.each_ref().map(|row| row.as_slice());
    let message = 0b1101u64;
    let mut codeword = [0u8; bits::bytes_for(LENGTH)];
    bits::xor_gather(&mut codeword, LENGTH, message, &sources);
    assert_eq!(syndrome(&codeword), 0);
    assert_eq!((codeword[0] >> 2) & 1, 1);
    assert_eq!((codeword[0] >> 4) & 1, 0);
    assert_eq!((codeword[0] >> 5) & 1, 1);
    assert_eq!((codeword[0] >> 6) & 1, 1);
    println!("Message {message:04b} -> codeword {:07b}", codeword[0]);

    // A single error produces the binary label of its coordinate. Multiple
    // errors do not have this interpretation; this is not a general decoder.
    let mut received = codeword;
    received[0] ^= 1 << 4;
    let error_position = syndrome(&received);
    assert_eq!(error_position, 5);
    let mut difference = received;
    bits::xor_assign(&mut difference, &codeword);
    assert_eq!(bits::weight(&difference, LENGTH), 1);

    received[0] ^= 1 << (error_position - 1);
    assert_eq!(received, codeword);
    println!(
        "Syndrome {error_position:03b} locates position {error_position}; correction matches the codeword."
    );

    // Logical length excludes padding even if a received byte has it set.
    let mut with_padding = codeword;
    with_padding[0] |= 0x80;
    assert_eq!(syndrome(&with_padding), 0);
    assert_eq!(
        bits::weight(&with_padding, LENGTH),
        bits::weight(&codeword, LENGTH)
    );
    // Range operations change only their window, preserving the padding bit.
    let live_bits = bits::XorRange::new(LENGTH, 0, LENGTH);
    bits::xor_range_with(&mut with_padding, &codeword, &live_bits);
    assert_eq!(with_padding, [0x80]);
    println!("Logical-length queries ignore padding; range XOR preserves it.");
}

fn syndrome(word: &[u8]) -> u8 {
    CHECKS.iter().enumerate().fold(0, |value, (bit, check)| {
        value | (u8::from(bits::dot_product(word, check, LENGTH).is_one()) << bit)
    })
}
