//! Fan–Paar GF(2^32) kernels, split by instruction set.

pub(super) mod avx2;

pub use avx2::{mul_add_avx2, mul_assign_avx2, mul_into_avx2};
