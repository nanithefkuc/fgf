// expect-rejection
use fgf::{Extension, FanPaar4, FieldKernels, FiniteField};

// The shipped GF(2^4) extension row sits below one byte: it has no
// packed kernels.
fn requires_kernels<F: FieldKernels>() {}

pub fn call() {
    requires_kernels::<FiniteField<2, 4, Extension<FanPaar4, 2>>>();
}
