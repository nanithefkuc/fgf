// expect-rejection
//! An extension over `FiniteField<2, 1, Residue>`: the base is a prime
//! residue row, not a binary presentation, so the row never implements
//! `BinaryField`.
use fgf::{Components, Elem, Extension, ExtensionSpec, FieldElem, FiniteField, Relation, Residue};

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = FiniteField<2, 1, Residue>;
    type Storage = Components;
    const NAME: &'static str = "GF(4) over residues";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
}

type Row = FiniteField<2, 2, Extension<Spec, 2>>;

fn takes_binary<F: fgf::BinaryField>(value: Elem<F>) -> u128 {
    value.to_index()
}

fn main() {
    let _ = takes_binary(Elem::<Row>::ONE);
}
