// expect: extension relation is reducible
//! `t^2 + t + 1` over the `0x11D` byte field: GF(4) is a subfield, so the
//! relation splits.
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, Packed, Polynomial, Relation};

type Rs8 = FiniteField<2, 8, Polynomial<0x11D>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Rs8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "reducible";
    const RELATION: Relation = Relation::from_binary_words::<Rs8>(&[1, 1]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 16, Extension<Spec, 2>>>::ONE;
}
