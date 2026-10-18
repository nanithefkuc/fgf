// expect: field degree exceeds 64
//! A degree-65 row over GF(7): the total degree cap rejects the row at
//! first value use, before the descriptor is read. The relation stays a
//! well-formed five-coordinate value so the degree check is the failure.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation};

type R7 = FiniteField<7, 1>;

struct Spec;
impl ExtensionSpec<5> for Spec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "degree sixty-five over GF(7)";
    const RELATION: Relation = Relation::from_coordinates(&[6, 0, 0, 0, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 65, Extension<Spec, 5>>>::ONE;
}
