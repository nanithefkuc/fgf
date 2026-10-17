// expect-rejection
//! Residue GF(2) is a `FieldBuffer` but not a `BinaryField` presentation:
//! its lanes alias (a lane holding 2 decodes to 0), so the bit-packed rows
//! exclude it from the target side by construction.
use fgf::{Embedding, FiniteField, Polynomial, PreparedEmbedding, Residue, ops};

type Gf1 = FiniteField<2, 1, Polynomial<3>>;
type Res2 = FiniteField<2, 1, Residue>;

fn main() {
    let map = PreparedEmbedding::<Gf1, Res2>::new(&Embedding::<Gf1, Res2>::new().unwrap());
    ops::map_bits::trace_into_with(&mut [0u8; 1], &map, &[0u8; 8], 1);
}
