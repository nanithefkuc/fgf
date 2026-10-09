// expect-rejection
//! A marker whose `D` matches no spec implementation: `ExtensionSpec<2>`
//! cannot name a relative-degree-three row.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "marker degree";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 2, Extension<Spec, 3>>>::ONE;
}
