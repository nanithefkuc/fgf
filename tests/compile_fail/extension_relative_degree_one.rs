// expect: extension relative degree is below two
//! A marker at relative degree one: extensions are proper.
use fgf::{Elem, FiniteField};
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};

struct Spec;
impl ExtensionSpec<1> for Spec {
    type Base = FiniteField<2, 1, fgf::binary::Polynomial<3>>;
    type Storage = Components;
    const NAME: &'static str = "degree one";
    const RELATION: Relation = Relation::from_coordinates(&[1]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 1, Extension<Spec, 1>>>::ONE;
}
