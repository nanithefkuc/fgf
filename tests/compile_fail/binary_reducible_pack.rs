// expect: polynomial is reducible
// expect-code: E0080
//! An empty pack on a reducible polynomial validates the row before the
//! width arithmetic.
use fgf::binary::Polynomial;
use fgf::{Elem, FiniteField, ops};

pub type Row = FiniteField<2, 8, Polynomial<0x101>>;

pub fn pack_empty(dst: &mut [u8], elems: &[Elem<Row>]) {
    ops::pack::<Row>(dst, elems);
}
