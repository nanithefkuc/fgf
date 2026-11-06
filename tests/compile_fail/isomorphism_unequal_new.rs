// expect: Isomorphism requires equal source and target degrees
// expect-code: E0080
//! The AES byte field against the GF(2^16) tower row: unequal degrees
//! reject the isomorphism when the crate is compiled.
use fgf::FiniteField;
use fgf::binary::{AES, Polynomial, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Isomorphism;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;

pub fn probe() -> Isomorphism<Aes, Gf16> {
    Isomorphism::<Aes, Gf16>::new()
}
