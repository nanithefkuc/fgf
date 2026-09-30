//! `QuadMersenne31` (`(re, im)` Mersenne31 limb pairs) kernels for x86 /
//! `x86_64`.
//!
//! Each file is one instruction set: `avx512` under `simd512` and `avx2`.
//! Both share the limb arithmetic of the [`super::mersenne31`] x86
//! kernels, at `(re, im)` limb pairs.

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;

pub use avx2::{
    add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
#[cfg(feature = "simd512")]
pub use avx512::{mul_add_avx512, mul_elementwise_avx512, mul_into_avx512};
