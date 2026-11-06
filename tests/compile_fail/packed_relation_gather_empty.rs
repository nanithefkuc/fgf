// expect: extension relation is reducible
// expect-code: E0080
//! An empty offset gather on a packed row with a reducible relation
//! validates the row before the zero-work return.
use fgf::binary::Polynomial;
use fgf::extension::{Extension, ExtensionSpec, Packed, Relation};
use fgf::{FiniteField, ops};

pub type Base = FiniteField<2, 4, Polynomial<0x13>>;

pub struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Packed<u8>;
    const NAME: &'static str = "reducible tower";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0, 1]);
}

pub type Row = FiniteField<2, 8, Extension<Spec, 2>>;

pub fn gather_empty() {
    ops::add_gather_offsets::<Row>(&mut [], &[], &[]);
}
