// expect: error[E0576]
use fgf::Field;
use fgf::prime::Mersenne31;

fn probe(left: u32, right: u32) -> u32 {
    <Mersenne31 as Field>::add_raw(left, right)
}
