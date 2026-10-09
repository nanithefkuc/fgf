// expect: relation exceeds the coordinate capacity
//! Eighty-one one-bit words expand to eighty-one coordinates, past capacity.
use fgf::{FiniteField, Polynomial, Relation};

type Gf2 = FiniteField<2, 1, Polynomial<3>>;

const WORDS: [u64; 81] = [1; 81];
const BAD: Relation = Relation::from_binary_words::<Gf2>(&WORDS);
const CHECK: usize = BAD.to_coordinates().len();

fn main() {
    let _ = CHECK;
}
