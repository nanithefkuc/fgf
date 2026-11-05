// expect: Embedding requires the source degree to divide the target degree
// expect-code: E0080
//! A valid cubic presentation the reference ladder does not cover, into
//! GF(2^4): the degree contract rejects the pair before any
//! presentation-coverage question, so unsupported coverage never turns
//! into a runtime outcome.
use fgf::FiniteField;
use fgf::binary::Polynomial;
use fgf::extension::{Components, Extension, ExtensionSpec, Relation};
use fgf::maps::Embedding;

type Prime = FiniteField<2, 1, Polynomial<3>>;
type Target = FiniteField<2, 4, Polynomial<0x13>>;

pub struct CubicSpec;
impl ExtensionSpec<3> for CubicSpec {
    type Base = Prime;
    type Storage = Components;
    const NAME: &'static str = "GF(8) cubic over GF(2)";
    const RELATION: Relation = Relation::from_binary_words::<Prime>(&[1, 1, 0]);
}
type Gf8Cubic = FiniteField<2, 3, Extension<CubicSpec, 3>>;

pub fn probe() -> Embedding<Gf8Cubic, Target> {
    Embedding::<Gf8Cubic, Target>::new()
}
