// expect-rejection
use fgf::{FieldBuffer, FiniteField, Polynomial};

// Binary rows below degree eight have no byte-buffer layout.
fn requires_buffer<F: FieldBuffer>() {}

pub fn call() {
    requires_buffer::<FiniteField<2, 4, Polynomial<0x13>>>();
}
