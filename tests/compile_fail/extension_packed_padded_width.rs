// expect: padded packed storage is not narrower than its word
//! `PackedPadded<u8>` over a degree-sixteen row: the coordinate word must
//! stay below the storage word.
use fgf::{AES, Elem, Extension, ExtensionSpec, FiniteField, PackedPadded, Polynomial, Relation};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = Aes8;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "padded width";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[0x20, 1]);
}

fn main() {
    let _ = Elem::<FiniteField<2, 16, Extension<Spec, 2>>>::ONE;
}
