// expect-rejection
use fgf::{Extension, FanPaar2, FieldBuffer, FiniteField};

// The shipped GF(2^2) extension row sits below one byte: it keeps its
// inherent byte constructors but has no byte-buffer layout.
fn requires_buffer<F: FieldBuffer>() {}

pub fn call() {
    requires_buffer::<FiniteField<2, 2, Extension<FanPaar2, 2>>>();
}
