// expect: field order overflows u128
//! Goldilocks `t^3 - t - 1`: the cube of the Goldilocks order exceeds
//! `u128`.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Goldilocks, Relation};

struct Spec;
impl ExtensionSpec<3> for Spec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "overflowing";
    const RELATION: Relation =
        Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 1, 0xFFFF_FFFF_0000_0001 - 1, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<0xFFFF_FFFF_0000_0001, 3, Extension<Spec, 3>>>::ONE;
}
