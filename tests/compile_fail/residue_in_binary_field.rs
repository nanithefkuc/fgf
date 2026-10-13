// expect-rejection
use fgf::{BinaryField, FiniteField, Residue};

// The residue row at characteristic two is a prime presentation, not a
// binary one: it has no coordinate descriptor.
fn requires_binary<F: BinaryField>() {}

pub fn call() {
    requires_binary::<FiniteField<2, 1, Residue>>();
}
