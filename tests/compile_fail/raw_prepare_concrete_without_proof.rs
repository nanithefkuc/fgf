// expect: cannot find method or associated constant `prepare` in trait
//! Concrete public-trait lookup does not reach raw preparation: the entry
//! is not a member of `FieldBuffer`.
use fgf::prime::Mersenne31;
use fgf::{Elem, FieldBuffer};

pub fn prepare(coeff: Elem<Mersenne31>) {
    let _ = <Mersenne31 as FieldBuffer>::prepare(coeff);
}
