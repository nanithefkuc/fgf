// expect: Embedding requires the source degree to divide the target degree
// expect-code: E0080
//! The degree contract of `from_images` rejects the reverse pair before
//! any image is read, whatever sequence the caller supplies.
use fgf::FiniteField;
use fgf::binary::{AES, Polynomial, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;

pub fn probe() {
    let _ = Embedding::<Gf16, Aes>::from_images(&[]);
}
