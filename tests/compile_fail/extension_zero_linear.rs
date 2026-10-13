// expect: extension relation is reducible
//! `t^2 + 0x20` over the AES field: the zero linear coefficient makes the
//! quadratic reducible in characteristic two.
use fgf::{AES, Elem, Extension, ExtensionSpec, FiniteField, Packed, Polynomial, Relation};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Aes8;
    type Storage = Packed<u16>;
    const NAME: &'static str = "zero linear";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x20, 0]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 16, Extension<Spec, 2>>>::ONE;
}
