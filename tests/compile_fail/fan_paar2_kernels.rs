// expect-rejection
use fgf::{FieldKernels, FiniteField};
use fgf::binary::FanPaar2;
use fgf::extension::Extension;

// The shipped GF(2^2) extension row sits below one byte: it has no
// packed kernels.
fn requires_kernels<F: FieldKernels>() {}

pub fn call() {
    requires_kernels::<FiniteField<2, 2, Extension<FanPaar2, 2>>>();
}
