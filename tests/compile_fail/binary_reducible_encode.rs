// expect: polynomial is reducible
// expect-code: E0080
use fgf::{Elem, FieldBuffer, FiniteField};
pub type Row = FiniteField<2, 8, fgf::binary::Polynomial<0x101>>;
pub fn encode_parameter(bytes: &mut [u8], value: Elem<Row>) { <Row as FieldBuffer>::encode(bytes, value) }
