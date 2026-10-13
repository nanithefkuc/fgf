// expect-rejection
use fgf::{Extension, FanPaar4, FieldBuffer, FiniteField};

// The shipped GF(2^4) extension row sits below one byte: it keeps its
// inherent byte constructors but has no byte-buffer layout.
fn requires_buffer<F: FieldBuffer>() {}

pub fn call() {
    requires_buffer::<FiniteField<2, 4, Extension<FanPaar4, 2>>>();
}
