// expect: characteristic is not prime
// expect-code: E0080
//! Parameter-based encoding on a composite-characteristic residue row
//! validates the row before the byte-width check, without constructing the
//! supplied element.
use fgf::{Elem, FieldBuffer, FiniteField};

pub type Row = FiniteField<9, 1>;

pub fn encode_parameter(bytes: &mut [u8], value: Elem<Row>) {
    <Row as FieldBuffer>::encode(bytes, value)
}
