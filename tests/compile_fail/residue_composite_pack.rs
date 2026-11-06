// expect: characteristic is not prime
// expect-code: E0080
//! An empty pack on a composite-characteristic residue row validates the row
//! before the width arithmetic.
use fgf::{Elem, FiniteField, ops};

pub type Row = FiniteField<9, 1>;

pub fn pack_empty(dst: &mut [u8], elems: &[Elem<Row>]) {
    ops::pack::<Row>(dst, elems);
}
