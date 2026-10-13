// expect-rejection
use fgf::{FieldKernels, FiniteField, Polynomial};

// Binary rows below degree eight have no packed kernels.
fn requires_kernels<F: FieldKernels>() {}

pub fn call() {
    requires_kernels::<FiniteField<2, 1, Polynomial<3>>>();
}
