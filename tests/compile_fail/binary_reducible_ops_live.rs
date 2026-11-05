// expect: polynomial is reducible
// expect-code: E0080
//! A live byte addition on a reducible degree-eight polynomial validates
//! the row before the byte work, without constructing an element.
use fgf::binary::Polynomial;
use fgf::{FiniteField, ops};

pub type Row = FiniteField<2, 8, Polynomial<0x101>>;

pub fn add_live(dst: &mut [u8], src: &[u8]) {
    ops::add_assign::<Row>(dst, src);
}
