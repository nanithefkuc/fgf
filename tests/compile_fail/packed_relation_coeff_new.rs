// expect: extension relation is reducible
// expect-code: E0080
//! Single-coefficient preparation over a packed row with a reducible
//! relation validates the row before preparation, without constructing the
//! supplied element.
use fgf::binary::Polynomial;
use fgf::extension::{Extension, ExtensionSpec, Packed, Relation};
use fgf::ops::Coeff;
use fgf::{Elem, FiniteField};

pub type Base = FiniteField<2, 4, Polynomial<0x13>>;

pub struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Packed<u8>;
    const NAME: &'static str = "reducible tower";
    // t^2 + t = t * (t + 1) over GF(16).
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[0, 1]);
}

pub type Row = FiniteField<2, 8, Extension<Spec, 2>>;

pub fn coeff_new(value: Elem<Row>) -> Coeff<Row> {
    Coeff::new(value)
}
