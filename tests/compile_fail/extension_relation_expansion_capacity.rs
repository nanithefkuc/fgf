// expect: relation exceeds the coordinate capacity
//! Sixty-five one-bit words expand to sixty-five coordinates, past the
//! capacity of sixty-four.
use fgf::{FiniteField, Polynomial, Relation};

type Gf2 = FiniteField<2, 1, Polynomial<3>>;

const WORDS: [u64; 65] = [1; 65];
const BAD: Relation = Relation::from_binary_words::<Gf2>(&WORDS);
const CHECK: usize = BAD.to_coordinates().len();

fn main() {
    let _ = CHECK;
}
