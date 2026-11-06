// expect: is private
use fgf::binary::BinaryRepr;

fn probe<R: BinaryRepr<8>>() {
    let _ = R::DESCRIPTION;
}
