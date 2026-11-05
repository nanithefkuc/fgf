// expect: characteristic is not prime
use fgf::{FieldElem, FiniteField};

// The public constructor validates the characteristic before storing.
pub fn call() -> fgf::Elem<FiniteField<91, 1>> {
    <fgf::Elem<FiniteField<91, 1>> as FieldElem>::from_raw(5)
}
