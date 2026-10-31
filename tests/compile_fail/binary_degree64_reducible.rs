// expect: extension relation is reducible
//! The square-free relation t^64 + t has a zero root over GF(2).
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, Packed, Polynomial, Relation, ops};

type Gf2 = FiniteField<2, 1, Polynomial<3>>;
struct Spec;
impl ExtensionSpec<64> for Spec {
    type Base = Gf2;
    type Storage = Packed<u64>;
    const NAME: &'static str = "reducible degree-64 relation";
    const RELATION: Relation = {
        let mut coefficients = [0; 64];
        coefficients[1] = 1;
        Relation::from_binary_words::<Gf2>(&coefficients)
    };
}

fn main() {
    type Row = FiniteField<2, 64, Extension<Spec, 64>>;
    ops::mul_add::<Row>(&mut [0; 8], Elem::<Row>::ONE, &[0; 8]);
}
