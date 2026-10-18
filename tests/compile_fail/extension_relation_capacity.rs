// expect: relation exceeds the coordinate capacity
//! Sixty-five flattened coordinates exceed the relation capacity of
//! sixty-four.
use fgf::Relation;

const OVER: [u64; 65] = [0; 65];
const BAD: Relation = Relation::from_coordinates(&OVER);
const CHECK: usize = BAD.to_coordinates().len();

fn main() {
    let _ = CHECK;
}
