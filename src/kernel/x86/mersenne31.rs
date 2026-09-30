//! Mersenne31 (`u32` lanes) kernels for x86 / `x86_64`.
//!
//! `2^31 ≡ 1 (mod p)`, so a multiply folds the 62-bit product with a
//! mask/shift/add and one conditional subtract; the products come from
//! `vpmuludq` on the even and odd 32-bit lanes.
//!
//! Each file is one instruction set: `avx512` under `simd512`, `avx2`, and
//! `sse42`. Every entry processes whole vector lanes and hands the sub-lane
//! remainder to the next narrower entry, ending at the portable
//! `crate::kernel::prime` path, which is also the differential oracle.
//! Entries are safe capability-token functions taking the exact token their
//! instructions require and validating their own geometry before the first
//! lane; lane helpers are `#[rite]` functions.
//!
//! The AVX2 lane helpers are shared with the
//! [`QuadMersenne31`](crate::field::QuadMersenne31) kernels, whose limbs are
//! these lanes.

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

pub(super) use avx2::{fold_avx2, min_chain_avx2};

use crate::field::mersenne31;

/// The Mersenne31 modulus as a signed lane constant.
#[allow(clippy::cast_possible_wrap)]
pub(super) const P: i32 = mersenne31::MODULUS as i32;
