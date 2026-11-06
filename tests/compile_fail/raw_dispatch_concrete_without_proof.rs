// expect: cannot find method or associated constant `add_assign` in trait
//! Concrete public-trait lookup does not reach the private dispatch entry:
//! the method is not a member of `FieldBuffer`, only of its private
//! supertrait.
use fgf::prime::Mersenne31;
use fgf::FieldBuffer;

pub fn drive(dst: &mut [u8], src: &[u8]) {
    <Mersenne31 as FieldBuffer>::add_assign(dst, src);
}
