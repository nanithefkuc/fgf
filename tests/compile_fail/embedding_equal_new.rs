// expect: Embedding requires distinct source and target degrees
// expect-code: E0080
//! Two equal-degree presentations of GF(2^16): a proper inclusion of
//! either into the other is rejected when the crate is compiled, before
//! construction can run. `Isomorphism` names the equal-degree map.
use fgf::FiniteField;
use fgf::binary::{FanPaar16, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type FanPaarRow = FiniteField<2, 16, Extension<FanPaar16, 2>>;
type RijndaelRow = FiniteField<2, 16, Extension<Rijndael16, 2>>;

pub fn probe() -> Embedding<FanPaarRow, RijndaelRow> {
    Embedding::<FanPaarRow, RijndaelRow>::new()
}
