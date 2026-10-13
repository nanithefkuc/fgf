// expect-rejection
use fgf::{BinaryDegree, FiniteField, Residue};

// Extension bases name an exact binary degree; the residue row is not a binary
// degree row even at characteristic two.
fn requires_degree_one<F: BinaryDegree<1>>() {}

pub fn call() {
    requires_degree_one::<FiniteField<2, 1, Residue>>();
}
