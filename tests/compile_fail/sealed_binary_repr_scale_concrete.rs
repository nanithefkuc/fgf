// expect: error[E0576]
use fgf::binary::{AES, BinaryRepr, Polynomial};
pub fn scale() -> usize {
    <Polynomial<AES> as BinaryRepr<8>>::SCALE.len()
}
