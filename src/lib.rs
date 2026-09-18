#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Arch intrinsics are imported wholesale by universal convention; naming
    // each of the ~200 used here would be unmaintainable and would have to be
    // duplicated across the x86 and x86_64 cfg arms.
    clippy::wildcard_imports,
    clippy::inline_always,
    // `chunks_exact` → `as_chunks` is a toolchain-drift lint (not in the
    // MSRV 1.89); the hot kernels use `chunks_exact_mut` for const-unchecked
    // reasons and changing every site right before release is riskier than
    // silencing it. `unknown_lints` keeps the allow portable across toolchains
    // that predate the lint. Revisit when the MSRV catches up.
    unknown_lints,
    clippy::chunks_exact_to_as_chunks,
)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod bits;
pub mod field;
pub mod kernel;
pub mod ops;

#[cfg(feature = "internals")]
pub mod internals;

pub use field::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Field, Gf2, Gf8B, Gf8D, Gf16, Gf32, Gf64,
    Goldilocks, Mersenne31, QuadMersenne31, fan_paar, gf2, gf8b, gf8d, gf16, gf32, gf64,
    goldilocks, mersenne31, quad_mersenne31,
};
pub use kernel::{
    Backend, FGF_TIERS, FieldKernels, KernelBackend, ParseBackendError, backend, backend_for,
    has_vector_elementwise,
};
