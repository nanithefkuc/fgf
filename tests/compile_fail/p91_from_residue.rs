// expect: characteristic is not prime
use fgf::{FieldElem, FiniteField};

// Residue conversion validates the characteristic before narrowing.
pub fn call() -> Result<fgf::Elem<FiniteField<91, 1>>, fgf::RangeError> {
    <fgf::Elem<FiniteField<91, 1>> as FieldElem>::from_residue(1)
}
