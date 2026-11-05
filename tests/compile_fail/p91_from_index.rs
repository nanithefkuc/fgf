// expect: characteristic is not prime
use fgf::{FieldElem, FiniteField};

// Index decoding validates the row before reading its characteristic.
pub fn call() -> Result<fgf::Elem<FiniteField<91, 1>>, fgf::RangeError> {
    <fgf::Elem<FiniteField<91, 1>> as FieldElem>::from_index(1)
}
