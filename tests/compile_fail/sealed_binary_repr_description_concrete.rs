// expect: error[E0576]
use fgf::binary::{BinaryRepr, Polynomial};

fn probe() {
    let _ = <Polynomial<0x11B> as BinaryRepr<8>>::DESCRIPTION;
}
