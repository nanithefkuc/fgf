// expect: polynomial is reducible
use fgf::{FieldElem, FiniteField};
use fgf::binary::Polynomial;

// The public constructor validates the representation's defining data.
pub fn call() -> fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>> {
    <fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>> as FieldElem>::from_raw(5)
}
