//! Canonical Fan–Paar tower kernels for x86 / `x86_64`.
//!
//! A Fan–Paar GF(2^16) multiply is the same four-nibble-shuffle shape as the
//! polynomial GF(2^16) kernel: two alternating `fp8` byte multiplies of the
//! source and its adjacent-swap, under the period-2 coefficient pair from
//! [`crate::kernel::tables::FpTowerTables`]. The only difference is the base
//! field — the canonical Fan–Paar byte field is *not* the AES field, so the
//! nibble tables are filled from `fp8` arithmetic instead of GF(2^8), and
//! there is no GFNI fast path. The shuffle multiply core in `kernel::x86::gf16`
//! reads only the nibble tables, so this module is just the per-kernel dispatch
//! loops over a pre-built
//! [`FpTowerTables`](crate::kernel::tables::FpTowerTables).
//!
//! Each file is one instruction set, `avx2` or `ssse3`. Every kernel is a
//! safe [`archmage`] capability-token function taking the exact token its
//! instructions require — `X64V3Token` for the AVX2 lanes, `X64V2Token` for
//! the SSSE3 lanes — asserting its own buffer geometry before the lane loop
//! and walking the buffer with reference-based loads and stores over chunk
//! arrays; the lane builders are `#[rite]` helpers. Sub-lane tails go to the
//! portable scalar kernel, which is also the differential oracle.
//!
//! `crate::kernel::fan_paar` holds the dispatch and the algebraic fold that
//! keeps `mul_alpha` in coefficient preparation.
//!
//! [`archmage`]: https://docs.rs/archmage

mod avx2;
mod ssse3;

pub use avx2::{
    mul_add_avx2_fp16, mul_add_avx2_fp32, mul_add_avx2_fp64, mul_assign_avx2_fp16,
    mul_assign_avx2_fp32, mul_assign_avx2_fp64, mul_into_avx2_fp16, mul_into_avx2_fp32,
    mul_into_avx2_fp64,
};
pub use ssse3::{mul_add_ssse3_fp16, mul_assign_ssse3_fp16, mul_into_ssse3_fp16};
