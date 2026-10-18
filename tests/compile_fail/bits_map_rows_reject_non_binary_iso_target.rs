// expect-rejection
//! Residue GF(2) is a `FieldBuffer` but not a `BinaryField` presentation:
//! its lanes alias (a lane holding 2 decodes to 0), so the bit-packed rows
//! exclude it from the target side by construction.
use fgf::{FiniteField, Isomorphism, Polynomial, PreparedIsomorphism, Residue, ops};

type Gf1 = FiniteField<2, 1, Polynomial<3>>;
type Res2 = FiniteField<2, 1, Residue>;

fn main() {
    let map = PreparedIsomorphism::<Gf1, Res2>::new(&Isomorphism::<Gf1, Res2>::new().unwrap());
    ops::map_bits::apply_into_with(&mut [0u8; 8], &map, &[0u8; 1], 1);
}
