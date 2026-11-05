// expect: characteristic is not prime
//! A direct validity query on an extension row whose characteristic is not
//! prime: family metadata rejects the row before base validation runs.
use fgf::{Field, FiniteField};
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};
use fgf::prime::Residue;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1, Residue>;
    type Storage = Components;
    const NAME: &'static str = "quadratic over a composite characteristic";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

fn main() {
    let () = <FiniteField<4, 2, Extension<Spec, 2>> as Field>::VALID;
}
