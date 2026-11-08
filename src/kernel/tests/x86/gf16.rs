//! GF(2^16) family differentials, one module per instruction set.

use super::*;
use crate::kernel::x86;

mod avx2;
#[cfg(feature = "simd512")]
mod avx512;
mod gfni;
