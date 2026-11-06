// expect: polynomial is reducible
// expect-code: E0080
//! An empty offset gather on a reducible polynomial validates the row
//! before the zero-work return.
use fgf::binary::Polynomial;
use fgf::{FiniteField, ops};

pub type Row = FiniteField<2, 8, Polynomial<0x101>>;

pub fn gather_empty() {
    ops::add_gather_offsets::<Row>(&mut [], &[], &[]);
}
