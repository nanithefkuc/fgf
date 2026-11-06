// expect: error[E0576]
use fgf::Field;
use fgf::prime::Mersenne31;

const VALID: () = <Mersenne31 as Field>::VALID;
