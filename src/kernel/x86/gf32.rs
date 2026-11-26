//! GF(2^32) tower kernels for `x86_64`, split by instruction set.
//!
//! Shuffle masks and their AVX2 load are shared with the wider tower and Fan–Paar families.

mod gfni;

pub use gfni::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `PSHUFB` control that exchanges adjacent bytes within every 2-byte
/// element — the GF(2^16) tower swap. The 16-byte pattern repeats because
/// `PSHUFB` indexes within each 128-bit lane independently.
pub(crate) const SWAP1: [u8; 32] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13,
    12, 15, 14,
];

/// `PSHUFB` control that exchanges the two GF(2^16) halves of every GF(2^32)
/// element — a 2-byte-granular swap, period 4.
pub(crate) const SWAP2: [u8; 32] = [
    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14,
    15, 12, 13,
];

/// `PSHUFB` control that reverses every 4-byte group — `SWAP1` composed
/// with `SWAP2`, the swap the second GF(2^16) lane multiply applies to its
/// already-half-swapped input.
pub(crate) const REV4: [u8; 32] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15,
    14, 13, 12,
];

/// Load a 32-byte shuffle control. The mask arrays are exactly one vector
/// wide, so the load is of a complete, in-bounds lane.
///
/// A V3 [`archmage::rite`] helper: the Fan–Paar lane builders hold exactly
/// V3, and the GFNI tower lane builders cover V3 in turn.
#[archmage::rite(v3, import_intrinsics)]
pub(crate) fn load_mask(m: &[u8; 32]) -> __m256i {
    _mm256_loadu_si256(m)
}
