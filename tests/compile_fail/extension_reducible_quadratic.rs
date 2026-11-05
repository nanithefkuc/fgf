// expect: extension relation is reducible
//! `t^2 - 2` over GF(7): two is a square, so the relation factors.
use fgf::{Elem, FiniteField};
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "reducible";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 2, Extension<Spec, 2>>>::ONE;
}
