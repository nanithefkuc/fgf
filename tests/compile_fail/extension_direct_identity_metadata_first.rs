// expect: binary field degree exceeds 64
//! A direct identity query on a degree-128 quadratic over the shipped
//! GF(2^64) row: the binary degree cap rejects the row before the relation
//! capacity check runs.
use fgf::{Components, Elem, Extension, ExtensionSpec, FiniteField, Relation, Rijndael64};

type Base = FiniteField<2, 64, Extension<Rijndael64, 2>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Base;
    type Storage = Components;
    const NAME: &'static str = "quadratic over GF(2^64)";
    const RELATION: Relation = Relation::from_binary_words::<Base>(&[1, 1]);
}

type Row = FiniteField<2, 128, Extension<Spec, 2>>;

fn main() {
    let _ = Elem::<Row>::ZERO;
}
