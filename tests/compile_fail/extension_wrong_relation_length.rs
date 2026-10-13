// expect: relation coordinate count differs from the field degree
//! One coordinate where the degree-two row needs two.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "short relation";
    const RELATION: Relation = Relation::from_coordinates(&[4]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 2, Extension<Spec, 2>>>::ONE;
}
