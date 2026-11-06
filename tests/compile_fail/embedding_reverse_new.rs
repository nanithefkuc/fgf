// expect: Embedding requires the source degree to divide the target degree
// expect-code: E0080
//! A larger source into a smaller target: the tower row over GF(2^16)
//! cannot include into the AES byte field, and the rejection happens
//! when the crate is compiled.
use fgf::FiniteField;
use fgf::binary::{AES, Polynomial, Rijndael16};
use fgf::extension::Extension;
use fgf::maps::Embedding;

type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Aes = FiniteField<2, 8, Polynomial<AES>>;

pub fn probe() -> Embedding<Gf16, Aes> {
    Embedding::<Gf16, Aes>::new()
}
