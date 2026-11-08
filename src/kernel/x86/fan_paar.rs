//! Canonical Fan–Paar tower kernels for `x86_64`.
//!
//! Subtrees split by field width, then instruction set. The wider AVX2
//! kernels reuse the narrower width's concrete-feature lane scale.

pub mod gf16;
pub mod gf32;
pub mod gf64;
