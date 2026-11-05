// expect: characteristic is not prime
//! An extension row whose characteristic is not prime: central metadata
//! validation rejects it before any base or descriptor validation.
use fgf::{Elem, FiniteField};
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
    let _ = Elem::<FiniteField<4, 2, Extension<Spec, 2>>>::ZERO;
}
