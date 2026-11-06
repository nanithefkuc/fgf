// expect: polynomial is reducible
use fgf::{FieldBuffer, FiniteField};
use fgf::binary::Polynomial;

// Byte decoding validates the representation before storing.
pub fn call() -> fgf::Elem<FiniteField<2, 8, Polynomial<0x101>>> {
    <FiniteField<2, 8, Polynomial<0x101>> as FieldBuffer>::decode(&[5])
}
