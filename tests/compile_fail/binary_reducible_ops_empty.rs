// expect: polynomial is reducible
// expect-code: E0080
//! An empty byte addition on a reducible polynomial validates the row
//! before the zero-work return.
use fgf::binary::Polynomial;
use fgf::{FiniteField, ops};

pub type Row = FiniteField<2, 8, Polynomial<0x101>>;

pub fn add_empty() {
    ops::add_assign::<Row>(&mut [], &[]);
}
