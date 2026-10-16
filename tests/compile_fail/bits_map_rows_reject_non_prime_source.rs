// expect-rejection
//! `GF(2^8)` is not a prime field: the bit-packed rows take the GF(2)
//! source, whose elements are the bits themselves.
use fgf::{
    Embedding, Extension, FiniteField, Polynomial, PreparedEmbedding, Rijndael16, ops, poly::AES,
};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;

fn main() {
    let map = PreparedEmbedding::<Aes8, Gf16>::new(&Embedding::<Aes8, Gf16>::new().unwrap());
    ops::map_bits::embed_into_with(&mut [0u8; 32], &map, &[0u8; 2], 16);
}
