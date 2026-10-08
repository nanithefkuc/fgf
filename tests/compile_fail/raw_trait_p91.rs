// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Ninety-one never reaches the remainder arithmetic behind the raw words.
const BAD: u64 = <FiniteField<91, 1> as Field>::ZERO_RAW;
