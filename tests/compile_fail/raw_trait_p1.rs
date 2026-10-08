// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// One is rejected by the same primality proof on the trait path.
const BAD: u64 = <FiniteField<1, 1> as Field>::ONE_RAW;
