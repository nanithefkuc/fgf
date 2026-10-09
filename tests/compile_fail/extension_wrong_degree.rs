// expect: extension degree is not the base degree times the relative degree
//! A degree-two relation over GF(7) named as a degree-three field.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "wrong degree";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 3, Extension<Spec, 2>>>::ONE;
}
