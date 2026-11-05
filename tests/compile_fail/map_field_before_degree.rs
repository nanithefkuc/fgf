// expect: extension relation is reducible
// expect-code: E0080
// reject: evaluation panicked: Embedding requires the source degree to divide the target degree
//! Field validity precedes the map-degree contract: the source row's
//! reducible relation rejects the pair, and the nondividing degrees of
//! the same pair never reach their own assertion.
use fgf::{FiniteField, maps::Embedding, binary::Polynomial};
use fgf::extension::{Extension, ExtensionSpec, PackedPadded, Relation};
type Prime = FiniteField<2, 1, Polynomial<3>>;
type Target = FiniteField<2, 4, Polynomial<0x13>>;
pub struct Spec;
impl ExtensionSpec<3> for Spec {
    type Base = Prime;
    type Storage = PackedPadded<u8>;
    const NAME: &'static str = "cubic";
    const RELATION: Relation = Relation::from_binary_words::<Prime>(&[1, 0, 0]);
}
type Source = FiniteField<2, 3, Extension<Spec, 3>>;
pub fn probe() { let _ = Embedding::<Source, Target>::new(); }
