// expect: polynomial is reducible
// expect-code: E0080
//! An empty allocating pack on a reducible polynomial validates the row
//! before the allocation.
extern crate alloc;
use fgf::binary::Polynomial;
use fgf::{Elem, FiniteField, ops};

pub type Row = FiniteField<2, 8, Polynomial<0x101>>;

pub fn pack_to_vec_empty(elems: &[Elem<Row>]) -> alloc::vec::Vec<u8> {
    ops::pack_to_vec::<Row>(elems)
}
