//! Generic residue-lane kernels for x86 / `x86_64`.
//!
//! The field buffer owns little-endian lanes sized by the modulus. The
//! `avx2` and `avx512` subtrees split by operation shape into `single`
//! and delayed-reduction `matrix` kernels; `avx512` requires `simd512`.
//! Single-row kernels split again into `u8`, `u32`, and `u64` lanes:
//! nibble-table products at `P <= 127`, Shoup and Barrett products over
//! u32 lanes, and Montgomery products over u64 lanes at odd `P`.
//! Two-byte rows, byte rows above 127, and `P = 2` stay portable.
//!
//! Both instruction sets share reduction parameters and source-provider
//! windows here. Preparation belongs to `kernel::residue`; entries take
//! the exact capability token and validate geometry before the first lane.

mod avx2;
#[cfg(feature = "simd512")]
pub(crate) mod avx512;

pub use avx2::{
    mul_add_matrix_u32_avx2_with, mul_add_u8_avx2, mul_add_u32_avx2, mul_add_u64_avx2,
    mul_assign_u8_avx2, mul_assign_u32_avx2, mul_assign_u64_avx2, mul_elementwise_assign_u32_avx2,
    mul_elementwise_u32_avx2, mul_into_matrix_u32_avx2_with, mul_into_u8_avx2, mul_into_u32_avx2,
    mul_into_u64_avx2,
};

/// Inline coefficient representation consumed by prepared matrix kernels.
#[allow(unused_imports)]
pub use crate::kernel::residue::PreparedResidue;

use crate::field::finite_field::{FiniteField, Residue};
use crate::kernel::Matrix;

/// Reduction parameters for one four-byte modulus, built once per row.
///
/// `norm_mu` reduces an arbitrary 32-bit lane, `barrett_mu` with `shift`
/// reduces a 64-bit product below `2^32 * P` through the narrow Barrett
/// form, and `p` is the modulus itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U32Params {
    /// The modulus, fitting the four-byte lane.
    pub p: u32,
    /// `floor(2^32 / P)`: the reciprocal normalizing raw lanes.
    pub norm_mu: u32,
    /// `floor(2^s / P)` with `s = floor(log2 P) + 32`: the narrow quotient
    /// reciprocal.
    pub barrett_mu: u32,
    /// `s - 32`: the quotient-estimate shift.
    pub shift: u32,
}

impl U32Params {
    /// Build the parameters for `p` (`1 < p < 2^32`).
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn new(p: u32) -> Self {
        let shift = p.ilog2();
        let s = shift + 32;
        Self {
            p,
            norm_mu: ((1u128 << 32) / p as u128) as u32,
            barrett_mu: ((1u128 << s) / p as u128) as u32,
            shift,
        }
    }
}

impl<const P: u64> FiniteField<P, 1, Residue> {
    /// This row's [`U32Params`], derived from its modulus. The constant is
    /// named in every row's monomorphized elementwise body and narrows
    /// `P as u32`; the narrowing is harmless off the four-byte class
    /// because those rows never dispatch through the u32 kernels.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const U32_PARAMS: U32Params = U32Params::new(P as u32);
}

/// A borrowed view of one provider over a consecutive source range: the
/// tile driver reuses one `tile_group` over groups of up to `K` sources
/// without copying coefficients or lanes.
struct SourceGroup<'a, M: Matrix<PreparedResidue> + ?Sized> {
    terms: &'a M,
    base: usize,
    len: usize,
}

impl<M: Matrix<PreparedResidue> + ?Sized> Matrix<PreparedResidue> for SourceGroup<'_, M> {
    fn len(&self) -> usize {
        self.len
    }

    fn coefficient(&self, term: usize, row: usize) -> PreparedResidue {
        self.terms.coefficient(self.base + term, row)
    }

    fn source(&self, term: usize) -> &[u8] {
        self.terms.source(self.base + term)
    }
}
