#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Architecture kernels conventionally import their intrinsic namespace;
    // enumerating the hundreds of instruction names obscures the kernel body.
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

mod field;
mod kernel;

pub mod binary;
pub mod extension;
pub mod maps;
pub mod ops;
pub mod prime;

#[cfg(feature = "internals")]
pub mod internals;

pub use crate::field::finite_field::{FiniteField, FiniteFieldRepr};
pub use crate::field::{Elem, Field, FieldBuffer, FieldElem, HasGenerator, RangeError};
pub use crate::kernel::{Backend, FieldKernels, backend, backend_for};
