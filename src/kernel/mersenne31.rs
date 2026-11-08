//! GF(2^31 − 1) kernel dispatch.
//!
//! Prime-field arithmetic over 32-bit lanes. On x86 with AVX-512 (`V4x`,
//! under `simd512`), AVX2 (`V3`), or SSE4.1 (`V2`) the add/sub/multiply loops
//! run over packed integer lanes with a Mersenne fold (`2^31 ≡ 1`); every
//! other target runs the portable [`crate::kernel::prime`] path, so
//! the backend query reports `Scalar` there. The coefficient's
//! prepared form is simply its canonical lane word, which the kernels
//! broadcast on entry.
//!
//! The contract implementation is the shared prime-lane generator in
//! [`crate::kernel`], which matches the cached prime strategy once per call
//! and names this module's kernels; the field's own arithmetic, lane width,
//! and fallback loops stay here.

use crate::field::mersenne31::Mersenne31;

// The shared generator's fallback arms resolve `Elem` at this expansion
// site.
use crate::field::Elem;

prime_dispatch!(Mersenne31, mersenne31, u32);
