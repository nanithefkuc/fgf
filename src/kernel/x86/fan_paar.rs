//! Canonical Fan–Paar tower kernels for `x86_64`.
//!
//! Subtrees split by field width, then instruction set. The wider AVX2
//! kernels reuse the narrower width's concrete-feature lane scale.
//!
//! The field arithmetic uses nibble tables over the canonical Fan–Paar byte
//! field rather than the AES field. The quadratic tower multiply combines
//! alternating byte products with an adjacent-byte swap under the prepared
//! coefficient pair in [`crate::kernel::tables::FpTowerTables`]. Wider towers
//! compose the narrower lane arithmetic.
//!
//! Concrete entries take the capability token required by their instructions
//! and validate buffer geometry before reference-based lane loads and stores.
//! Sub-lane tails use the portable scalar kernel, which also supplies the
//! differential oracle. `crate::kernel::tower` holds runtime dispatch and
//! folds tower constants into coefficient preparation.

pub mod gf16;
pub mod gf32;
pub mod gf64;
