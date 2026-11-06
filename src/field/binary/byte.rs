//! Byte-representation data shared with the kernel engine.
//!
//! This module holds the two data shapes the byte field rows and the kernel
//! engine share: the [`ScaleTable`] declaration, one split-nibble
//! multiplication table for a GF(2^8) coefficient, and the
//! [`ByteReprData`] contract a byte representation carries for the
//! dispatch layer. Both are data carriers — the engine owns every builder
//! and bank, and implements [`ByteReprData`] once per presentation family.

/// Split-nibble multiplication tables for one GF(2^8) coefficient.
///
/// The shuffle kernels read only `lo` and `hi`, so one type serves every
/// polynomial; the polynomial decides which bank an entry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleTable {
    /// The raw coefficient byte these tables multiply by.
    pub coeff: u8,
    /// `lo[i] = coeff * i` for the low nibble.
    pub lo: [u8; 16],
    /// `hi[i] = coeff * (i << 4)` for the high nibble.
    pub hi: [u8; 16],
}

/// The byte-representation data the degree-eight dispatch reads.
///
/// Polynomial presentations expose their real banks; basis presentations
/// expose placeholder banks the scalar fallback never reads. Dispatch
/// branches on the descriptor first, so no kernel applies polynomial
/// tables to basis coordinates. The engine implements this contract once
/// per presentation family, and the AES isomorphism travels as its two
/// matrix qwords rather than an engine type.
pub(crate) trait ByteReprData {
    /// Whether `GF2P8MULB` multiplies in this encoding natively.
    const AES_NATIVE: bool;
    /// Nibble tables per coefficient, or placeholders for basis roots.
    const SCALE: &'static [ScaleTable; 256];
    /// Affine map words per coefficient, or placeholders for basis roots.
    const AFFINE: &'static [u64; 256];
    /// Forward `VGF2P8AFFINEQB` matrix of the isomorphism onto the AES
    /// field, or the identity map's for basis roots.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    const AES_MAP_FORWARD: u64;
    /// Inverse `VGF2P8AFFINEQB` matrix of the isomorphism onto the AES
    /// field, or the identity map's for basis roots.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    const AES_MAP_INVERSE: u64;
    /// Reduction byte for shift/reduce tails, or zero for basis roots.
    ///
    /// Consumed only by the x86 elementwise kernels; other targets keep
    /// the contract without reading it.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    const REDUCTION_LOW: u8;
}
