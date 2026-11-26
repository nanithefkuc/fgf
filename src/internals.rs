//! Implementation details for experimentation, with no compatibility guarantees.
//!
//! Paths, types, signatures, and behavior may change or disappear in any release.
//! Production consumers should use the supported API in [`crate::ops`],
//! [`crate::ops::bits`], [`crate::prime`], [`crate::binary`],
//! [`crate::extension`], and [`crate::maps`].

/// Direct kernel implementations, preparation tables, and capability tokens.
pub mod kernel {
    pub use crate::kernel::bytes::xor;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub use crate::kernel::matrix_provider::{FlatMatrix, Matrix};

    /// The declared tier ladder.
    pub use crate::kernel::FGF_TIERS;
    pub use simdispatch::ParseBackendError;

    /// Byte-field kernels and preparation.
    pub mod gf8 {
        pub use crate::kernel::gf8::*;
    }

    /// Packed-extension kernels over GF(2¹⁶).
    pub mod gf16 {
        pub use crate::kernel::tower::gf16::*;
    }

    /// Packed-extension kernels over GF(2³²).
    pub mod gf32 {
        pub use crate::kernel::tower::gf32::*;
    }

    /// Packed-extension kernels over GF(2⁶⁴).
    pub mod gf64 {
        pub use crate::kernel::tower::gf64::*;
    }

    /// Packed-extension kernel families of the shipped quadratic towers.
    pub mod tower {
        pub use crate::kernel::tower::*;
    }

    /// Portable prime-field kernels.
    pub mod prime {
        pub use crate::kernel::prime::*;
    }

    /// Portable binary-field kernels and tails.
    pub mod scalar {
        pub use crate::kernel::scalar::*;
    }

    /// The kernel class a residue row's vectorizable operations take, for
    /// routing reports; the label a selector's result reports as the
    /// process tier.
    pub use crate::kernel::ResidueStrategy;

    /// Row routing: the threshold constants and selector functions the
    /// production dispatch runs, for benchmark routing reports that must
    /// not copy constants. Available only where the x86 SIMD dispatch
    /// exists; other targets route every residue shape through the
    /// portable path.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub mod routing {
        pub use crate::kernel::routing::{
            DELAYED_ADD_MIN_DEPTH, DELAYED_INTO_MIN_DEPTH, DELAYED_MIN_SOURCES, VectorOp,
            avx2_floor_op_len, delay_depth, delayed_matrix_backend, routes_delayed, routes_vector,
            single_row_backend,
        };
    }

    /// Coefficient tables shared by field and architecture kernels.
    pub mod tables {
        // `ScaleTable` is declared in the field layer's byte module; the
        // unstable `tables` path keeps serving it by re-export.
        pub use crate::field::binary::byte::ScaleTable;
        pub use crate::kernel::tables::*;
    }

    /// Token-bearing x86 kernels.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub mod x86 {
        pub use crate::kernel::x86::*;
    }

    #[cfg(feature = "simd")]
    pub use archmage::SimdToken;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub use archmage::{
        X64V1Token, X64V2Token, X64V3GfniCryptoToken, X64V3Token, X64V4Token, X64V4xToken,
    };
}

/// Field-implementation plumbing: descriptors, reductions, and scalar oracles.
pub mod field {
    /// Binary presentation descriptors and the degree-eight log/exp tables.
    pub mod description {
        pub use crate::field::binary::description::{BinaryDescription, ByteLogExp};
    }

    /// Fan–Paar recurrence for differential checks of the tower family.
    pub mod fan_paar {
        pub use crate::field::binary::tower::{fp_invert, fp_mul_alpha, fp_multiply, fp_square};
    }

    /// The Goldilocks lane folds.
    pub mod goldilocks {
        pub use crate::field::goldilocks::{reduce, reduce_wide};
    }

    /// The Mersenne31 lane fold.
    pub mod mersenne31 {
        pub use crate::field::mersenne31::reduce;
    }

    /// Validated integer reductions modulo a checked prime.
    pub mod prime {
        pub use crate::field::prime::{reduce, reduce_wide};
    }
}
