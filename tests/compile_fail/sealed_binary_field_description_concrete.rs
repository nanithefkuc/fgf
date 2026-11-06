// expect: error[E0576]
use fgf::binary::{BinaryField, Polynomial};
use fgf::FiniteField;

fn probe() {
    let _ = <FiniteField<2, 8, Polynomial<0x11B>> as BinaryField>::DESCRIPTION;
}
