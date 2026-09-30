//! GFNI GF(2^16) kernels over 32-byte lanes.
//!
//! Each byte multiply of the tower identity is one `GF2P8MULB`, and the two
//! alternating coefficients come straight out of
//! [`TowerCoeff`](crate::kernel::tables::TowerCoeff) as one `vpbroadcastw`
//! each, so no shape needs a table. Submodules split on fan shape: `single`,
//! `scatter`, `gather`, `matrix`, and `elementwise`. Every entry takes an
//! `X64V3GfniCryptoToken`; this file holds the lane multiply every shape
//! shares.

mod elementwise;
mod gather;
mod matrix;
mod scatter;
mod single;

pub use elementwise::{mul_elementwise_assign_gfni, mul_elementwise_gfni};
pub use gather::mul_add_gather_gfni;
pub use matrix::{mul_add_matrix_gfni, mul_add_matrix_gfni_with};
pub use scatter::mul_add_scatter_gfni;
pub use single::{mul_add_gfni, mul_assign_gfni, mul_into_gfni};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
/// `coeff * src` for one 32-byte lane, given the source and its
/// adjacent-exchanged self.
///
/// The exchange is the caller's job so that the multi-row kernels can shuffle
/// once and reuse the result for every destination row.
#[archmage::rite(v3_gfni_crypto)]
fn scale_gfni(src: __m256i, swapped: __m256i, same: __m256i, cross: __m256i) -> __m256i {
    _mm256_xor_si256(
        _mm256_gf2p8mul_epi8(src, same),
        _mm256_gf2p8mul_epi8(swapped, cross),
    )
}
