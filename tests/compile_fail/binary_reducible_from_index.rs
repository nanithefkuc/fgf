// expect: polynomial is reducible
use fgf::{FieldElem, FiniteField};
use fgf::binary::Polynomial;

// Index decoding validates the representation before storing.
pub fn call() -> Result<fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>>, fgf::RangeError> {
    <fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>> as FieldElem>::from_index(5)
}
