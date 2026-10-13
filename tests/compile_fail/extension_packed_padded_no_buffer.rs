// expect-rejection
//! A padded packed custom row has no byte form, so the buffer capability
//! rejects it.
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
    let mut bytes = [0u8; 1];
    fgf::FieldBuffer::encode(&mut bytes, Elem::<Row>::ONE);
}
