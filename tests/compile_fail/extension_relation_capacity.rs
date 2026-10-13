// expect: relation exceeds the coordinate capacity
//! Eighty-one flattened coordinates exceed the relation capacity of eighty.
use fgf::Relation;

const OVER: [u64; 81] = [0; 81];
const BAD: Relation = Relation::from_coordinates(&OVER);
const CHECK: usize = BAD.to_coordinates().len();

fn main() {
    let _ = CHECK;
}
