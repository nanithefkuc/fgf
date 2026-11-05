// expect: error[E0576]
use fgf::FieldBuffer;
use fgf::prime::Mersenne31;
pub fn prepared() -> Option<<Mersenne31 as FieldBuffer>::Prepared> {
    None
}
