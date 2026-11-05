// expect: associated constant `SCALE` is private
// expect-code: E0624
use fgf::binary::BinaryRepr;
pub fn scale<R: BinaryRepr<8>>() -> usize {
    R::SCALE.len()
}
