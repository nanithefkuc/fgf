// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// The full word is rejected on the raw trait path too.
const BAD: u64 = <FiniteField<{ u64::MAX }, 1> as Field>::ONE_RAW;
