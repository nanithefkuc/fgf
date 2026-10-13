// expect: relation word has bits at or above the base degree
//! Word `0x100` holds bit eight of a degree-eight base.
use fgf::{AES, FiniteField, Polynomial, Relation};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;

const WORDS: [u64; 2] = [0x100, 1];
const BAD: Relation = Relation::from_binary_words::<Aes8>(&WORDS);
const CHECK: usize = BAD.to_coordinates().len();

fn main() {
    let _ = CHECK;
}
