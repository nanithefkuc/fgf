//! Single-row AVX-512 kernels for the generic residue rows.
//!
//! The child modules split by arithmetic lane width: `u8` holds nibble-table
//! products at `P <= 127`, `u32` holds Shoup fixed-coefficient and Barrett
//! elementwise products, and `u64` holds Montgomery products at odd `P`.
//! The multi-row delayed matrix lives in the sibling `matrix` module.
//!
//! Direct entries take an `X64V4Token` and validate their own geometry;
//! tails run the scalar forms. Field adapters bind prepared coefficients
//! and the cached capability proof after routing selects `V4x`.

use crate::field::FieldBuffer;
use crate::field::finite_field::{FiniteField, Residue};
use crate::kernel::residue::PreparedResidue;

mod u32;
mod u64;
mod u8;

pub use u8::{mul_add_u8_avx512, mul_assign_u8_avx512, mul_into_u8_avx512};
pub use u32::{
    mul_add_u32_avx512, mul_assign_u32_avx512, mul_elementwise_assign_u32_avx512,
    mul_elementwise_u32_avx512, mul_into_u32_avx512,
};
pub use u64::{mul_add_u64_avx512, mul_assign_u64_avx512, mul_into_u64_avx512};

/// Apply a prepared coefficient through this field's AVX-512 overwrite entry.
///
/// The caller selects `V4x` for a supported lane class and handles zero and
/// unit coefficients before this adapter. Lane layout belongs to `FieldBuffer`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mul_into_with<const P: u64>(dst: &mut [u8], coeff: PreparedResidue, src: &[u8]) {
    let token = crate::kernel::x86_v4_token();
    match <FiniteField<P, 1, Residue> as FieldBuffer>::BYTES {
        1 => mul_into_u8_avx512(token, dst, src, coeff.word as u8, P as u8),
        4 => mul_into_u32_avx512(
            token,
            dst,
            src,
            coeff.word as u32,
            (coeff.word >> 32) as u32,
            P as u32,
        ),
        8 => mul_into_u64_avx512(token, dst, src, coeff.word, P),
        _ => unreachable!("selected residue lane width"),
    }
}

/// Accumulate a prepared coefficient through this field's AVX-512 entry.
///
/// The caller selects `V4x` for a supported lane class and handles zero and
/// unit coefficients before this adapter. Lane layout belongs to `FieldBuffer`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mul_add_with<const P: u64>(dst: &mut [u8], coeff: PreparedResidue, src: &[u8]) {
    let token = crate::kernel::x86_v4_token();
    match <FiniteField<P, 1, Residue> as FieldBuffer>::BYTES {
        1 => mul_add_u8_avx512(token, dst, src, coeff.word as u8, P as u8),
        4 => mul_add_u32_avx512(
            token,
            dst,
            src,
            coeff.word as u32,
            (coeff.word >> 32) as u32,
            P as u32,
        ),
        8 => mul_add_u64_avx512(token, dst, src, coeff.word, P),
        _ => unreachable!("selected residue lane width"),
    }
}

/// Scale in place through this field's prepared AVX-512 entry.
///
/// The caller selects `V4x` for a supported lane class and handles zero and
/// unit coefficients before this adapter. Lane layout belongs to `FieldBuffer`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mul_assign_with<const P: u64>(dst: &mut [u8], coeff: PreparedResidue) {
    let token = crate::kernel::x86_v4_token();
    match <FiniteField<P, 1, Residue> as FieldBuffer>::BYTES {
        1 => mul_assign_u8_avx512(token, dst, coeff.word as u8, P as u8),
        4 => mul_assign_u32_avx512(
            token,
            dst,
            coeff.word as u32,
            (coeff.word >> 32) as u32,
            P as u32,
        ),
        8 => mul_assign_u64_avx512(token, dst, coeff.word, P),
        _ => unreachable!("selected residue lane width"),
    }
}

/// Multiply four-byte lanes after routing selects this field's `V4x` entry.
#[inline]
pub(crate) fn mul_elementwise<const P: u64>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    mul_elementwise_u32_avx512(
        crate::kernel::x86_v4_token(),
        dst,
        a,
        b,
        FiniteField::<P, 1, Residue>::U32_PARAMS,
    );
}

/// Multiply four-byte lanes in place after routing selects this field's `V4x` entry.
#[inline]
pub(crate) fn mul_elementwise_assign<const P: u64>(dst: &mut [u8], src: &[u8]) {
    mul_elementwise_assign_u32_avx512(
        crate::kernel::x86_v4_token(),
        dst,
        src,
        FiniteField::<P, 1, Residue>::U32_PARAMS,
    );
}

/// The sub-vector tail of bytes past the 64-byte vector body.
pub(super) fn src_tail_avx512(src: &[u8]) -> &[u8] {
    let (_, tail) = src.as_chunks::<64>();
    tail
}
