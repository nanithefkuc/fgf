// expect: Isomorphism requires equal source and target degrees
// expect-code: E0080
//! The degree contract of `from_images` rejects unequal degrees before
//! any image is read, whatever sequence the caller supplies.
use fgf::FiniteField;
use fgf::binary::{AES, Polynomial, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Isomorphism;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;

pub fn probe() {
    let _ = Isomorphism::<Aes, Gf16>::from_images(&[]);
}
