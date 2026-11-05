// expect: error[E0599]
use fgf::prime::Mersenne31;
use fgf::HasGenerator;

const GENERATOR: u32 = Mersenne31::GENERATOR_RAW;
