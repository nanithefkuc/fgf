//! GF(2^64) tower kernels for `x86_64`, split by instruction set.
//!
//! The half-swap mask is shared with the Fan–Paar family.

mod gfni;

pub use gfni::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};

/// `PSHUFB` control that exchanges the two GF(2^32) halves of every
/// GF(2^64) element — a 4-byte-granular swap, period 8.
pub(crate) const SWAP4: [u8; 32] = [
    4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15,
    8, 9, 10, 11,
];

/// `SWAP1` composed with `SWAP4`.
const SWAP1_4: [u8; 32] = [
    5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14,
    9, 8, 11, 10,
];

/// `SWAP2` composed with `SWAP4`.
const SWAP2_4: [u8; 32] = [
    6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13,
    10, 11, 8, 9,
];

/// `REV4` composed with `SWAP4` — a full reversal of every 8-byte element.
const REV8: [u8; 32] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
    11, 10, 9, 8,
];
