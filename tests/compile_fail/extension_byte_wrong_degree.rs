// expect: extension degree is not the base degree times the relative degree
// expect-code: E0080
use fgf::{FiniteField, ops};
use fgf::binary::Polynomial;
use fgf::extension::{Extension, ExtensionSpec, Packed, Relation};
type Base = FiniteField<2, 4, Polynomial<0x13>>;
pub struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Packed<u8>;
    const NAME: &'static str = "wrong-degree extension";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0, 1]);
}
pub fn add_empty() {
    ops::add_assign::<FiniteField<2, 7, Extension<Spec, 2>>>(&mut [], &[]);
}
