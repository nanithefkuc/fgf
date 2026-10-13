// expect: characteristic is not prime
use fgf::field::prime;

// The reduction refuses a zero modulus before the remainder runs.
const BAD: u64 = prime::reduce::<0>(7);
