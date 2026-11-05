// expect: error[E0576]
use fgf::Field;
use fgf::prime::Mersenne31;

const ONE: u32 = <Mersenne31 as Field>::ONE_RAW;
