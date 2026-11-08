//! x86-compiled field families.
//!
//! One module per field family, each holding one module per instruction
//! set its differentials drive. The cross-family instruction-set suites
//! live in [`isa`](super::isa); everything here is family-pure. Shared
//! geometry, coefficient classes, and differential drivers come from the
//! suite parent through the glob import.

use super::*;

mod extension;
mod gf16;
mod gf8;
#[cfg(feature = "simd512")]
mod goldilocks;
mod prime;
