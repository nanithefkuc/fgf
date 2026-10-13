// expect: polynomial is reducible
//! An extension over `x^2 + 1` over GF(2): the base polynomial factors, so
//! the base's own validation rejects the row before the relation is read.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Polynomial, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<2, 2, Polynomial<5>>;
    type Storage = Components;
    const NAME: &'static str = "over a reducible base";
    // Flattened coordinates of the two coefficients over the base.
    const RELATION: Relation = Relation::from_coordinates(&[1, 1, 0, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 4, Extension<Spec, 2>>>::ONE;
}
