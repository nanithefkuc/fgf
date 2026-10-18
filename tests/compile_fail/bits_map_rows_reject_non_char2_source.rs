// expect-rejection
//! The bit bridge takes a characteristic-two source endpoint: a wider
//! binary field carries coordinates the packed bit cannot name, so the
//! source bound excludes it by construction.
use fgf::{Extension, FiniteField, Isomorphism, PreparedIsomorphism, Rijndael16, ops};

type Aes8 = FiniteField<2, 8, fgf::Polynomial<fgf::poly::AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;

fn main() {
    let map = PreparedIsomorphism::<Aes8, Gf16>::new(&Isomorphism::<Aes8, Gf16>::new().unwrap());
    ops::map_bits::apply_into_with(&mut [0u8; 32], &map, &[0u8; 2], 16);
}
