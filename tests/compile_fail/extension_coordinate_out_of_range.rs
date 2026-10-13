// expect: relation coordinate is not below the characteristic
//! The coordinate seven is not below the characteristic seven.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "coordinate range";
    const RELATION: Relation = Relation::from_coordinates(&[7, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 2, Extension<Spec, 2>>>::ONE;
}
