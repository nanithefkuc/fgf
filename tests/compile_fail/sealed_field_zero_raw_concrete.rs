// expect: error[E0576]
use fgf::Field;
use fgf::prime::Mersenne31;

const ZERO: u32 = <Mersenne31 as Field>::ZERO_RAW;
