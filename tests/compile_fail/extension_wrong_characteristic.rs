// expect: extension base characteristic differs from the field characteristic
//! An odd base named at characteristic two.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "wrong characteristic";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 2, Extension<Spec, 2>>>::ONE;
}
