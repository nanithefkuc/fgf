// expect: Embedding requires distinct source and target degrees
// expect-code: E0080
//! The degree contract of `from_images` rejects the pair before any
//! image is read: the malformed empty sequence below never reaches the
//! count check.
use fgf::FiniteField;
use fgf::binary::{FanPaar16, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type FanPaarRow = FiniteField<2, 16, Extension<FanPaar16, 2>>;
type RijndaelRow = FiniteField<2, 16, Extension<Rijndael16, 2>>;

pub fn probe() {
    let _ = Embedding::<FanPaarRow, RijndaelRow>::from_images(&[]);
}
