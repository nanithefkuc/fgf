//! GF(2^8) kernels for x86 and `x86_64`.
//!
//! Submodules split along multiply strategy, then fan shape. `gfni` and
//! `nibble` hold the single-buffer AXPY kernels of the two strategies;
//! `scatter` takes one source into many rows, `gather` many sources into one,
//! `matrix` many sources into many rows over the register-blocked row-group
//! bodies in `rows`, and `nibble_rows` holds the multi-row shapes of the
//! shuffle strategy. `elementwise` multiplies two varying buffers lane-wise.
//! `experiments` holds measurement variants, which reach
//! into the production bodies above without duplicating them.
//!
//! Two multiply strategies, selected by the
//! [`Backend`](crate::kernel::Backend) the caller already resolved:
//!
//! - **GFNI.** `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply across 32
//!   byte lanes, so a coefficient is nothing but a broadcast byte. It is
//!   pipelined but not single-cycle, which shapes the loops below: the
//!   single-buffer AXPY keeps four independent multiply chains in flight, and
//!   the blocked kernels hold a destination tile in registers so the
//!   multiplier, not memory, is the limit.
//! - **Nibble shuffle (AVX2, SSSE3).** With no byte-wide multiply, `c * x`
//!   splits into `c * (x & 0xf) ^ c * (x & 0xf0)`, two `PSHUFB` lookups
//!   against a [`ScaleTable`]. `PSHUFB` indexes within each 128-bit half, so
//!   the AVX2 form broadcasts the 16-byte table into both halves first.
//!
//! Buffer lengths are arbitrary, so x86 kernels descend through 32-byte and
//! 16-byte SIMD lanes before handing only the sub-XMM remainder to the scalar
//! nibble kernels in `crate::kernel::gf8`.
//!
//! Every public entry is a safe [`archmage`] capability-token function:
//! `X64V3GfniCryptoToken` for the GFNI multiplies, `X64V3Token` for the
//! AVX2 nibble forms, `X64V2Token` for the SSSE3 ones. Tokens prove the ISA;
//! each entry asserts its own buffer geometry before the loops run.
//!
//! What stays here is the seam the multi-row submodules share. The
//! register-blocked scatter/gather/matrix kernels differ between the two byte
//! fields in exactly three places: how a coefficient becomes a factor vector,
//! which one-instruction multiply folds it in, and which nibble bank the
//! sub-lane tail reads. `GF2P8MULB` is the native `0x11B` multiply for
//! `Gf8B`; `VGF2P8AFFINEQB` applies an arbitrary GF(2) map and so multiplies
//! under `0x11D` for `Gf8D`. Everything else — the tiling, the register
//! budget, the remainder ladder — is identical, so the kernels are generic
//! over this seam and monomorphize back to the code each field had alone.

mod elementwise;
#[allow(dead_code)]
mod experiments;
mod gather;
mod gfni;
mod matrix;
mod nibble;
mod nibble_rows;
mod rows;
mod scatter;

#[allow(unused_imports)]
pub use elementwise::*;
#[allow(unused_imports)]
pub use experiments::*;
pub use gather::*;
pub use gfni::*;
pub use matrix::*;
pub use nibble::*;
pub use nibble_rows::*;
pub use scatter::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::field::gf8b::Elem;
use crate::field::gf8d;
use crate::kernel::Matrix;
use crate::kernel::tables::affine_8b;
use crate::kernel::tables::{ScaleTable, affine_8d, scale_table, scale_table_8d};

/// The per-field multiply seam for the blocked GF(2^8) kernels.
pub(super) trait Blocked {
    /// Coefficient form selected by the strategy.
    type Coeff: Copy;
    /// The additive identity, for staging arrays.
    fn zero() -> Self::Coeff;
    /// `true` when the multiply is `VGF2P8AFFINEQB`, `false` for `GF2P8MULB`.
    const AFFINE: bool;
    /// Raw coefficient byte.
    fn byte(coeff: Self::Coeff) -> u8;
    /// The `VGF2P8AFFINEQB` matrix qword (unused when `AFFINE` is false).
    fn map(coeff: Self::Coeff) -> u64;
    /// The coefficient's sub-lane nibble tables, from this field's bank.
    fn table(coeff: Self::Coeff) -> &'static ScaleTable;
}

/// Native GFNI multiply in the AES field `0x11B` (`Gf8B`).
pub(super) enum Gfni {}
impl Blocked for Gfni {
    type Coeff = Elem;
    const AFFINE: bool = false;
    #[inline]
    fn zero() -> Elem {
        Elem(0)
    }
    #[inline]
    fn byte(coeff: Elem) -> u8 {
        coeff.0
    }
    #[inline]
    fn map(_coeff: Elem) -> u64 {
        0
    }
    #[inline]
    fn table(coeff: Elem) -> &'static ScaleTable {
        scale_table(coeff)
    }
}

/// Prepared fixed multiply for the experimental `Gf8B` affine gather.
///
/// Map lookup stays outside the timed gather body. The nibble table remains
/// attached only for sub-XMM remainders.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Affine8BFactor {
    map: u64,
    table: &'static ScaleTable,
}

/// Prepare one `Gf8B` coefficient for [`mul_add_gather_affine_8b`].
#[inline]
#[must_use]
#[allow(dead_code)]
pub fn prepare_affine_8b(coeff: Elem) -> Affine8BFactor {
    Affine8BFactor {
        map: affine_8b(coeff),
        table: scale_table(coeff),
    }
}

/// Experimental affine-map multiply in the AES field `0x11B` (`Gf8B`).
#[allow(dead_code)]
enum Affine8B {}
impl Blocked for Affine8B {
    type Coeff = Affine8BFactor;
    const AFFINE: bool = true;
    #[inline]
    fn zero() -> Affine8BFactor {
        prepare_affine_8b(Elem(0))
    }
    #[inline]
    fn byte(coeff: Affine8BFactor) -> u8 {
        coeff.table.coeff.0
    }
    #[inline]
    fn map(coeff: Affine8BFactor) -> u64 {
        coeff.map
    }
    #[inline]
    fn table(coeff: Affine8BFactor) -> &'static ScaleTable {
        coeff.table
    }
}

/// Affine-map multiply in the Reed–Solomon field `0x11D` (`Gf8D`).
pub(super) enum Affine8D {}
impl Blocked for Affine8D {
    type Coeff = gf8d::Elem;
    const AFFINE: bool = true;
    #[inline]
    fn zero() -> gf8d::Elem {
        gf8d::Elem(0)
    }
    #[inline]
    fn byte(coeff: gf8d::Elem) -> u8 {
        coeff.0
    }
    #[inline]
    fn map(coeff: gf8d::Elem) -> u64 {
        affine_8d(coeff)
    }
    #[inline]
    fn table(coeff: gf8d::Elem) -> &'static ScaleTable {
        scale_table_8d(coeff)
    }
}

/// [`Affine8D`] over coefficients already prepared by the caller.
///
/// The multiply word is the stored affine map, so the resolve loop reads a
/// struct field instead of recomputing the bank lookup. The prepared
/// operations use this to consume a [`crate::ops::CoeffVec`]'s coefficients
/// without rebuilding them per call; the sub-lane remainder still reads the
/// attached nibble table.
#[allow(dead_code)]
pub(super) enum Affine8DPrepared {}
impl Blocked for Affine8DPrepared {
    type Coeff = crate::kernel::gf8::Prepared8D;
    const AFFINE: bool = true;
    #[inline]
    fn zero() -> Self::Coeff {
        crate::kernel::gf8::Prepared8D {
            table: scale_table_8d(gf8d::Elem(0)),
            affine: affine_8d(gf8d::Elem(0)),
        }
    }
    #[inline]
    fn byte(coeff: Self::Coeff) -> u8 {
        coeff.table.coeff.0
    }
    #[inline]
    fn map(coeff: Self::Coeff) -> u64 {
        coeff.affine
    }
    #[inline]
    fn table(coeff: Self::Coeff) -> &'static ScaleTable {
        coeff.table
    }
}

/// A row-major coefficient matrix over already-prepared coefficients, in the
/// same term-major order a [`crate::ops::CoeffMatrix`] stores: index
/// `term * nrows + row`.
#[allow(dead_code)]
pub(super) struct PreparedMatrix<'a> {
    /// Prepared coefficients, `terms * nrows` entries.
    pub(super) prepared: &'a [crate::kernel::gf8::Prepared8D],
    /// Destination row count.
    pub(super) nrows: usize,
    /// Source buffer of each term.
    pub(super) sources: &'a [&'a [u8]],
}

impl Matrix<crate::kernel::gf8::Prepared8D> for PreparedMatrix<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }
    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &crate::kernel::gf8::Prepared8D {
        &self.prepared[term * self.nrows + row]
    }
    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

/// Broadcast a coefficient into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor<S: Blocked>(coeff: S::Coeff) -> __m256i {
    if S::AFFINE {
        _mm256_set1_epi64x(S::map(coeff).cast_signed())
    } else {
        _mm256_set1_epi8(S::byte(coeff).cast_signed())
    }
}

/// Broadcast a coefficient into a 128-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bfactor128<S: Blocked>(coeff: S::Coeff) -> __m128i {
    if S::AFFINE {
        _mm_set1_epi64x(S::map(coeff).cast_signed())
    } else {
        _mm_set1_epi8(S::byte(coeff).cast_signed())
    }
}

/// `coeff * x` over a 256-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul<S: Blocked>(x: __m256i, factor: __m256i) -> __m256i {
    if S::AFFINE {
        _mm256_gf2p8affine_epi64_epi8::<0>(x, factor)
    } else {
        _mm256_gf2p8mul_epi8(x, factor)
    }
}

/// `coeff * x` over a 128-bit lane, one instruction.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn bmul128<S: Blocked>(x: __m128i, factor: __m128i) -> __m128i {
    if S::AFFINE {
        _mm_gf2p8affine_epi64_epi8::<0>(x, factor)
    } else {
        _mm_gf2p8mul_epi8(x, factor)
    }
}

/// Single-row remainder AXPY (`dst ^= coeff * src`) for this field: whole
/// 32/16-byte lanes on the one-instruction multiply, sub-XMM tail on the
/// nibble bank.
///
/// Runs in the row-group bodies' matching feature context, so it calls the
/// shared `#[rite]` multiply bodies directly rather than the token-bearing
/// entries.
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn brem<S: Blocked>(dst: &mut [u8], coeff: S::Coeff, src: &[u8]) {
    if S::AFFINE {
        gfni::mul_add_affine_impl(dst, S::map(coeff), S::table(coeff), src);
    } else {
        gfni::mul_add_gfni_impl(dst, Elem(S::byte(coeff)), src);
    }
}

/// The multiply factor of one coefficient, as a word the tile loop can
/// broadcast without touching a table: the `VGF2P8AFFINEQB` map on the affine
/// fields, the raw coefficient byte on the native `GF2P8MULB` one.
#[inline]
fn factor_word<S: Blocked>(coeff: S::Coeff) -> u64 {
    if S::AFFINE {
        S::map(coeff)
    } else {
        u64::from(S::byte(coeff))
    }
}

/// Broadcast a resolved factor word into a 256-bit multiply factor.
#[inline]
#[archmage::rite(v3_gfni_crypto)]
pub(super) fn wfactor<S: Blocked>(word: u64) -> __m256i {
    if S::AFFINE {
        _mm256_set1_epi64x(word.cast_signed())
    } else {
        // The byte fields park the coefficient in the word's low byte.
        _mm256_set1_epi8(word.to_le_bytes()[0].cast_signed())
    }
}
