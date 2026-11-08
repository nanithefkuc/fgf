//! Fan–Paar GF(2^64) kernels, split by instruction set.

mod avx2;

pub use avx2::{mul_add_avx2, mul_assign_avx2, mul_into_avx2};
