//! Implementation details for experimentation, with no compatibility guarantees.
//!
//! Paths, types, signatures, and behavior may change or disappear in any release.
//! Production consumers should use the supported API in [`crate::ops`],
//! [`crate::bits`], [`crate::field`], and [`crate::kernel`].

/// Direct kernel implementations, preparation tables, and capability tokens.
pub mod kernel {
    pub use crate::kernel::byte_ops::xor;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    pub use crate::kernel::matrix_provider::{FlatMatrix, Matrix};

    /// Byte-field kernels and preparation.
    pub mod gf8 {
        pub use crate::kernel::gf8::*;
    }

    /// Quadratic tower kernels over GF(2¹⁶).
    pub mod gf16 {
        pub use crate::kernel::tower::gf16::*;
    }

    /// Quadratic tower kernels over GF(2³²).
    pub mod gf32 {
        pub use crate::kernel::tower::gf32::*;
    }

    /// Quadratic tower kernels over GF(2⁶⁴).
    pub mod gf64 {
        pub use crate::kernel::tower::gf64::*;
    }

    /// Rijndael-rooted quadratic tower kernel family.
    pub mod tower {
        pub use crate::kernel::tower::*;
    }

    /// Fan–Paar tower kernels and preparation.
    pub mod fan_paar {
        pub use crate::kernel::fan_paar::*;
    }

    /// Portable prime-field kernels.
    pub mod prime {
        pub use crate::kernel::prime::*;
    }

    /// Portable binary-field kernels and tails.
    pub mod scalar {
        pub use crate::kernel::scalar::*;
    }

    /// Coefficient tables shared by field and architecture kernels.
    pub mod tables {
        pub use crate::kernel::tables::*;
    }

    /// Token-bearing x86 kernels and experiments.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod x86 {
        pub use crate::kernel::x86::*;
    }

    /// AArch64 kernels and token-proven compatibility entries.
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    pub mod aarch64 {
        pub use crate::kernel::aarch64::*;
    }

    /// WebAssembly kernels and token-proven compatibility entries.
    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    pub mod wasm32 {
        pub use crate::kernel::wasm32::*;
    }

    #[cfg(feature = "simd")]
    pub use archmage::SimdToken;
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    pub use archmage::Wasm128Token;
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    pub use archmage::{NeonAesToken, NeonToken};
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    pub use archmage::{
        X64V1Token, X64V2Token, X64V3GfniCryptoToken, X64V3Token, X64V4Token, X64V4xToken,
    };
}

/// Independent scalar field oracles.
pub mod field {
    /// Wiedemann recurrence for differential checks of the Fan–Paar tower.
    pub mod wiedemann {
        pub use crate::field::wiedemann::*;
    }
}
