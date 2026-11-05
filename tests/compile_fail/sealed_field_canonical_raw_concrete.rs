// expect: error[E0576]
use fgf::Field;
use fgf::prime::Mersenne31;

fn probe(raw: u32) -> u32 {
    <Mersenne31 as Field>::canonical_raw(raw)
}
