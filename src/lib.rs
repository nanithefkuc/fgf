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
    AES, BinaryDegree, BinaryDescription, BinaryField, BinaryRepr, ByteLogExp, Cantor, Elem,
    FanPaar2, FanPaar2Spec, FanPaar4, FanPaar4Spec, FanPaar8, FanPaar8Spec, FanPaar16,
    FanPaar16Spec, FanPaar32, FanPaar32Spec, FanPaar64, FanPaar64Spec, Field, FieldBuffer,
    FieldElem, Gf, Gf1, Gf8, Gf16, Gf32, Gf64, Goldilocks, HasGenerator, Mersenne31, Normal, Poly,
    PrimeCharacteristic, PrimeIdentity, QuadMersenne31, REED_SOLOMON, Rijndael16, Rijndael32,
    Rijndael64, Tower, TowerGeneratorSpec, TowerSpec, cantor, description, goldilocks, mersenne31,
    normal, poly, quad_mersenne31, tower,
};
pub use kernel::{
    Backend, FGF_TIERS, FieldKernels, KernelBackend, ParseBackendError, backend, backend_for,
    has_vector_elementwise,
};
