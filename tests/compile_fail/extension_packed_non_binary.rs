// expect-rejection
//! `Packed<u16>` over an odd base: packed storage requires a binary base,
//! so the spec's storage bound fails.
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, Packed, Relation};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "packed over odd";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<7, 2, Extension<Spec, 2>>>::ONE;
}
