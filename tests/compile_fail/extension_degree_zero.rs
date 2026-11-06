// expect: field degree must be positive
// expect-code: E0080
//! A zero absolute degree on a well-formed extension presentation: the
//! family metadata check rejects the degree at first value use, before
//! the presentation is read.
use fgf::{Elem, FiniteField};
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};

type R7 = FiniteField<7, 1>;

struct Spec;
impl ExtensionSpec<2> for Spec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "zero absolute degree over GF(7)";
    const RELATION: Relation = Relation::from_coordinates(&[6, 0]);
}

const BAD: Elem<FiniteField<7, 0, Extension<Spec, 2>>> =
    Elem::<FiniteField<7, 0, Extension<Spec, 2>>>::ZERO;
