// expect: characteristic is not prime
// expect-code: E0080
//! An empty allocating pack on a composite-characteristic residue row
//! validates the row before the allocation.
extern crate alloc;
use fgf::{Elem, FiniteField, ops};

pub type Row = FiniteField<9, 1>;

pub fn pack_to_vec_empty(elems: &[Elem<Row>]) -> alloc::vec::Vec<u8> {
    ops::pack_to_vec::<Row>(elems)
}
