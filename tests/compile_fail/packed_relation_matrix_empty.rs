// expect: extension relation is reducible
// expect-code: E0080
//! Matrix preparation over a packed row with a reducible relation
//! validates the row before allocation.
use fgf::binary::Polynomial;
use fgf::extension::{Extension, ExtensionSpec, Packed, Relation};
use fgf::ops::CoeffMatrix;
use fgf::FiniteField;

pub type Base = FiniteField<2, 4, Polynomial<0x13>>;

pub struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Packed<u8>;
    const NAME: &'static str = "reducible tower";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0, 1]);
}

pub type Row = FiniteField<2, 8, Extension<Spec, 2>>;

pub fn matrix() -> CoeffMatrix<Row> {
    CoeffMatrix::from_source_major(0, 0, &[])
}
