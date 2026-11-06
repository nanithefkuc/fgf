// expect: error[E0576]
use fgf::HasGenerator;
use fgf::prime::Mersenne31;

const GENERATOR: u32 = <Mersenne31 as HasGenerator>::GENERATOR_RAW;
