// expect: cannot find method or associated constant `backend` in trait
//! Concrete public-trait lookup does not reach the backend query: it is not
//! a member of `FieldBuffer`.
use fgf::prime::Mersenne31;
use fgf::{Backend, FieldBuffer};

pub fn query() -> Backend {
    <Mersenne31 as FieldBuffer>::backend()
}
