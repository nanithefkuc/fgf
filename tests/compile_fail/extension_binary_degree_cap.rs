// expect: binary field degree exceeds 64
//! A quadratic over the shipped GF(2^64) extension row: the absolute degree
//! one hundred twenty-eight sits above the binary degree cap, so the field's
//! metadata rejects the row before the relation is read.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation, Rijndael64};

type Base = FiniteField<2, 64, Extension<Rijndael64, 2>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Components;
    const NAME: &'static str = "quadratic over GF(2^64)";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[1, 1]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 128, Extension<Spec, 2>>>::ONE;
}
