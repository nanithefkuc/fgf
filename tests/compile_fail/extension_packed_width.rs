// expect: packed storage width differs from the field degree
//! `Packed<u16>` over a degree-four row: the packed word must fill exactly.
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, Packed, Polynomial, Relation};

type Gf4 = FiniteField<2, 2, Polynomial<7>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Gf4;
    type Storage = Packed<u16>;
    const NAME: &'static str = "packed width";
    const RELATION: Relation = Relation::from_binary_words::<Gf4>(&[1, 1]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 4, Extension<Spec, 2>>>::ONE;
}
