//! Single-row AVX2 kernels for the generic residue rows.
//!
//! The child modules split by arithmetic lane width: `u8` holds nibble-table
//! products at `P <= 127`, `u32` holds Shoup fixed-coefficient and Barrett
//! elementwise products, and `u64` holds Montgomery products at odd `P`.
//! The multi-row delayed matrix lives in the sibling `matrix` module.
//!
//! Entries are safe capability-token functions taking an `X64V3Token` and
//! validating their own geometry; tails run the scalar forms.

pub mod u32;
pub mod u64;
pub mod u8;

/// The sub-vector tail of bytes past the 32-byte vector body.
pub(super) fn src_tail_avx2(src: &[u8]) -> &[u8] {
    let (_, tail) = src.as_chunks::<32>();
    tail
}
