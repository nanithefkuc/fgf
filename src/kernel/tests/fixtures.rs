//! Shared deterministic fixtures for the kernel test suites.
//!
//! Suites that need fixture data include this file through a path attribute
//! (`#[path = "../src/kernel/tests/fixtures.rs"] mod fixtures;`, adjusted for
//! the includer's depth), so one definition serves the in-crate tests and
//! every integration suite. The file is dependency-free: it uses only `std`,
//! so every feature configuration compiles it.
//!
//! # Streams
//!
//! Two fixed-step LCG byte streams, named by their seeding rule so an
//! includer states which convention it builds on:
//!
//! - [`noise_odd_seed`] promotes the seed with `seed | 1`, the crate's
//!   original convention.
//! - [`noise_distinct_seed`] maps the seed with `2 * seed + 1`, which keeps
//!   the LCG state odd without collapsing neighbouring seeds onto one
//!   stream.
//!
//! Includers import one under the local name `noise` (`use
//! fixtures::noise_odd_seed as noise;`) and keep their call sites unchanged.
//! The known-answer tests freeze the first outputs of both streams, so a
//! seeding change cannot slip past a suite that regenerates frozen fixture
//! bytes.
//!
//! # Shared geometry and coefficients
//!
//! The raw arrays are the common byte-length ladders, row geometries, and
//! coefficient words of the differential suites. An includer maps raw
//! coefficient words to its own field types and keeps its own boundary
//! geometries, coefficient classes, and Miri reductions; items one suite
//! does not use carry a per-item `allow(dead_code)`.

// The library's own `cfg(test)` tree is `no_std` unless the `std` feature is
// enabled; the explicit extern keeps this file includable there and in every
// integration suite.
extern crate std;

use std::vec::Vec;

/// One step of the fixed LCG every stream shares.
fn step(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 33) as u8
}

/// Deterministic byte noise from the `seed | 1` stream.
pub fn noise_odd_seed(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len).map(|_| step(&mut state)).collect()
}

/// Deterministic byte noise from the `2 * seed + 1` stream.
pub fn noise_distinct_seed(len: usize, seed: u64) -> Vec<u8> {
    // `2 * seed + 1` keeps the LCG state odd without collapsing neighbouring
    // seeds onto one stream.
    let mut state = seed.wrapping_mul(2).wrapping_add(1);
    (0..len).map(|_| step(&mut state)).collect()
}

/// Freezes the first outputs of the `seed | 1` stream.
#[test]
fn odd_seed_stream_first_outputs_are_frozen() {
    assert_eq!(
        noise_odd_seed(12, 1),
        [
            0x16, 0x67, 0x23, 0x94, 0x82, 0xDA, 0xBC, 0xEC, 0x34, 0x53, 0xFF, 0x1F
        ]
    );
    assert_eq!(
        noise_odd_seed(12, 3),
        [
            0x43, 0x09, 0x9A, 0x78, 0x5C, 0x8A, 0x81, 0x4B, 0xC4, 0x92, 0x57, 0x5F
        ]
    );
    assert_eq!(
        noise_odd_seed(12, 5),
        [
            0x71, 0xAB, 0x11, 0x5C, 0x36, 0x3A, 0x45, 0xAA, 0x55, 0xD0, 0xAF, 0xA0
        ]
    );
    assert_eq!(noise_odd_seed(0, 1), [] as [u8; 0]);
}

/// Freezes the first outputs of the `2 * seed + 1` stream.
#[test]
fn distinct_seed_stream_first_outputs_are_frozen() {
    assert_eq!(
        noise_distinct_seed(12, 0),
        [
            0x16, 0x67, 0x23, 0x94, 0x82, 0xDA, 0xBC, 0xEC, 0x34, 0x53, 0xFF, 0x1F
        ]
    );
    assert_eq!(
        noise_distinct_seed(12, 2),
        [
            0x71, 0xAB, 0x11, 0x5C, 0x36, 0x3A, 0x45, 0xAA, 0x55, 0xD0, 0xAF, 0xA0
        ]
    );
    assert_eq!(
        noise_distinct_seed(12, 9),
        [
            0xAE, 0x19, 0x51, 0x95, 0x2F, 0x0A, 0xA6, 0x45, 0x4A, 0x86, 0x18, 0x61
        ]
    );
    assert_eq!(noise_distinct_seed(0, 4), [] as [u8; 0]);
}

// The rest is the shared raw data. A suite takes the arrays its geometries
// need and keeps everything else local; unused arrays stay per-item allowed.

/// Byte lengths straddling the 16/32-byte lane boundaries and tails.
#[allow(dead_code)]
pub const LANE_BOUNDARY_LENGTHS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 64, 96, 130, 300];

/// Even byte lengths for GF(2^16)-shaped kernels.
#[allow(dead_code)]
pub const EVEN_BYTE_LENGTHS: &[usize] = &[0, 2, 16, 30, 32, 34, 64, 66, 128, 130, 300];

/// Row counts straddling the blocked kernels' four/two/one row groups.
#[allow(dead_code)]
pub const ROW_GROUP_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 8, 9];

/// GF(2^8) row lengths: tiny, one lane, lane plus tail, network payload, and
/// six 96-byte three-lane tiles with a whole lane and a sub-lane tail.
#[allow(dead_code)]
pub const ROW_LENGTHS: &[usize] = &[2, 32, 34, 300, 617];

/// GF(2^16) row lengths: even, tiny through network payload.
#[allow(dead_code)]
pub const EVEN_ROW_LENGTHS: &[usize] = &[2, 30, 32, 34];

/// Term counts straddling the blocked kernels' resolve chunk.
#[allow(dead_code)]
pub const TERM_COUNTS: &[usize] = &[1, 2, 3, 8, 9, 17];

/// GF(2^8) coefficient words: both short-circuits, the extremes, a spread.
/// Includers map the raw words to their own field types.
#[allow(dead_code)]
pub const GF8_COEFFICIENTS: [u8; 8] = [0x00, 0x01, 0x02, 0x03, 0x07, 0x53, 0xD3, 0xFF];

/// GF(2^16) coefficient words: short-circuits, pure components, mixed,
/// extremes. Includers map the raw words to their own field types.
#[allow(dead_code)]
pub const GF16_COEFFICIENTS: [u16; 8] = [
    0x0000, 0x0001, 0x0100, 0x00FF, 0x1234, 0xBEEF, 0x7411, 0xFFFF,
];

/// Even lengths shared by the direct arithmetic differentials.
#[allow(dead_code)]
pub const EVEN_DIFFERENTIAL_LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
];

/// Byte lengths straddling lane and compound-tail boundaries.
#[allow(dead_code)]
pub const XOR_LENGTHS: &[usize] = &[
    0, 1, 2, 7, 8, 15, 16, 17, 30, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129,
    255, 256, 257, 511, 512, 513, 1023,
];

/// Row counts covering the direct kernels' blocked groups.
#[allow(dead_code)]
pub const DIFFERENTIAL_ROW_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 13];

/// Even row lengths shared by the direct fan differentials.
#[allow(dead_code)]
pub const DIFFERENTIAL_ROW_LENGTHS: &[usize] = &[2, 16, 32, 34, 64, 66, 128, 300];

/// Gather lengths covering native lanes and compound remainders.
#[allow(dead_code)]
pub const GATHER_LENGTHS: &[usize] = &[
    0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97, 111, 112, 113,
    127, 128, 129, 143, 144, 145, 255, 256, 257, 300,
];

/// Byte-field words shared by the exhaustive direct entry drivers.
#[allow(dead_code)]
pub const GF8_DIFFERENTIAL_COEFFICIENTS: &[u8] = &[
    0, 1, 2, 255, 0, 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253,
];

/// Extension-field words shared by the direct entry drivers.
#[allow(dead_code)]
pub const GF16_DIFFERENTIAL_COEFFICIENTS: &[u16] = &[
    0x0000, 0x0001, 0x0002, 0x00ff, 0x0100, 0xff00, 0xffff, 0x0108, 0x72a2, 0x7b68, 0xa4f7, 0xaa9f,
    0x1424, 0xf1ac, 0xda51, 0xee90, 0x00a7, 0x7ae0, 0xdbcc, 0x457b, 0xea46, 0x414b, 0x606f, 0xc632,
    0xa385, 0x655f, 0x4fce, 0x0141, 0x637e, 0x5c18, 0x2298, 0x6e5c,
];
