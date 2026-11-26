//! Fan–Paar GF(2^16) kernels, split by instruction set.

mod avx2;
mod ssse3;

pub use avx2::{mul_add_avx2, mul_assign_avx2, mul_into_avx2};
pub use ssse3::{mul_add_ssse3, mul_assign_ssse3, mul_into_ssse3};
