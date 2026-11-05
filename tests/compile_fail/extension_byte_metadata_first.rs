// expect: characteristic is not prime
// expect-code: E0080
use fgf::{FiniteField, ops};
use fgf::binary::Polynomial;
use fgf::extension::{Extension, ExtensionSpec, Packed, Relation};
type Base = FiniteField<2, 4, Polynomial<0x13>>;
pub struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Packed<u8>;
    const NAME: &'static str = "invalid metadata and relation";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0, 1]);
}
pub fn add(dst: &mut [u8], src: &[u8]) {
    ops::add_assign::<FiniteField<4, 8, Extension<Spec, 2>>>(dst, src);
}
