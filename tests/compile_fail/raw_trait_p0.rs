// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Reading a raw identity word through the trait evaluates the row's
// validation, so an invalid characteristic never answers.
const BAD: u64 = <FiniteField<0, 1> as Field>::ZERO_RAW;
