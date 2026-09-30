//! Goldilocks (`u64` lanes) kernels for x86 / `x86_64`.
//!
//! A 64×64→128 product is built from four `vpmuludq` 32-bit half-products,
//! then reduced by the split fold (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`).
//!
//! Each file is one instruction set: `avx512` under `simd512`, `avx2`, and
//! `sse42`. Every entry processes whole vector lanes and hands the sub-lane
//! remainder to the next narrower entry, ending at the portable
//! `crate::kernel::prime` path, which is also the differential oracle.
//! Entries are safe capability-token functions taking the exact token their
//! instructions require and validating their own geometry before the first
//! lane; lane helpers are `#[rite]` functions.

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;
mod sse42;

pub use avx2::{
    add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
#[cfg(feature = "simd512")]
pub use avx512::{
    add_assign_avx512, add_assign_scalar_avx512, mul_add_avx512, mul_assign_avx512,
    mul_elementwise_assign_avx512, mul_elementwise_avx512, mul_into_avx512, sub_assign_avx512,
    sub_assign_scalar_avx512,
};
pub use sse42::{
    add_assign_scalar_sse42, add_assign_sse42, mul_add_sse42, mul_assign_sse42,
    mul_elementwise_assign_sse42, mul_elementwise_sse42, mul_into_sse42, sub_assign_scalar_sse42,
    sub_assign_sse42,
};

use crate::field::goldilocks;

/// The Goldilocks modulus as a signed lane constant.
#[allow(clippy::cast_possible_wrap)]
const P: i64 = goldilocks::MODULUS as i64;
/// `p - 1`, the largest canonical lane.
#[allow(clippy::cast_possible_wrap)]
const PM1: i64 = (goldilocks::MODULUS - 1) as i64;
/// `2^32 - 1`, the residue of `2^64`.
const EPS: i64 = 0xFFFF_FFFF;
