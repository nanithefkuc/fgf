// expect: extension relation is reducible
//! Relation words `[3, 3]` over the degree-two normal basis, whose
//! identity word is `3`, so the relation is `t^2 + t + 1` and splits over
//! GF(4).
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, Normal, PackedPadded, Relation};

type Normal4 = FiniteField<2, 2, Normal<7, 2>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Normal4;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "reducible over normal";
    const RELATION: Relation = Relation::from_binary_words::<Normal4>(&[3, 3]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 4, Extension<Spec, 2>>>::ONE;
}
