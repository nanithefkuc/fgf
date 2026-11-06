// expect: polynomial is reducible
use fgf::binary::Polynomial;
use fgf::{FieldElem, FiniteField};

// An out-of-range residue still validates the representation before the
// range error.
pub fn call() -> Result<fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>>, fgf::RangeError> {
    <fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>> as FieldElem>::from_residue(2)
}
