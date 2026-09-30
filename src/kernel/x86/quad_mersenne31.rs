//! `QuadMersenne31` (`(re, im)` Mersenne31 limb pairs) kernels for x86 /
//! `x86_64`.
//!
//! Each file is one instruction set; `avx2` is the only one. Its limb
//! arithmetic is the [`super::mersenne31`] AVX2 lane discipline.

mod avx2;

pub use avx2::{
    add_assign_avx2, add_assign_scalar_avx2, mul_add_avx2, mul_assign_avx2,
    mul_elementwise_assign_avx2, mul_elementwise_avx2, mul_into_avx2, sub_assign_avx2,
    sub_assign_scalar_avx2,
};
