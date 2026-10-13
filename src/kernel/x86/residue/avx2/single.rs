//! Single-row AVX2 kernels for the generic residue rows.
//!
//! The child modules split by arithmetic lane width: `u8` holds nibble-table
//! products at `P <= 127`, `u32` holds Shoup fixed-coefficient and Barrett
//! elementwise products, and `u64` holds Montgomery products at odd `P`.
//! The multi-row delayed matrix lives in the sibling `matrix` module.
//!
//! Entries are safe capability-token functions taking an `X64V3Token` and
//! validating their own geometry; tails run the scalar forms.

mod u32;
mod u64;
mod u8;

pub use u8::{mul_add_u8_avx2, mul_assign_u8_avx2, mul_into_u8_avx2};
pub use u32::{
    mul_add_u32_avx2, mul_assign_u32_avx2, mul_elementwise_assign_u32_avx2,
    mul_elementwise_u32_avx2, mul_into_u32_avx2,
};
pub use u64::{mul_add_u64_avx2, mul_assign_u64_avx2, mul_into_u64_avx2};

/// The sub-vector tail of bytes past the 32-byte vector body.
pub(super) fn src_tail_avx2(src: &[u8]) -> &[u8] {
    let (_, tail) = src.as_chunks::<32>();
    tail
}
