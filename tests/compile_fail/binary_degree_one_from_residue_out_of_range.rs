// expect: Polynomial<3> is the only degree-one form
use fgf::binary::Polynomial;
use fgf::{Elem, FiniteField};

// The inherent GF(2) residue constructor validates the representation
// before rejecting an out-of-range value.
pub fn call() -> Result<Elem<FiniteField<2, 1, Polynomial<0x7>>>, fgf::RangeError> {
    Elem::<FiniteField<2, 1, Polynomial<0x7>>>::from_residue(2)
}
