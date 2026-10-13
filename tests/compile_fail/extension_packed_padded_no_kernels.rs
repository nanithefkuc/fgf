// expect-rejection
//! A padded packed custom row has no kernel capability, so the packed
//! operation surface rejects it.
use fgf::{Elem, Extension, ExtensionSpec, FiniteField, PackedPadded, Relation};

struct Spec;
impl ExtensionSpec<3> for Spec {
    type Base = FiniteField<2, 1, fgf::Polynomial<3>>;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "GF(8)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 1, fgf::Polynomial<3>>>(&[1, 1, 0]);
}

type Row = FiniteField<2, 3, Extension<Spec, 3>>;

fn main() {
    let mut dst = vec![0u8; 4];
    let src = vec![0u8; 4];
    fgf::ops::mul_add::<Row>(&mut dst, Elem::<Row>::ONE, &src);
}
