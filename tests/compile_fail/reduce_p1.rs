// expect: characteristic is not prime
use fgf::field::prime;

// Reducing modulo one is not a field reduction.
const BAD: u64 = prime::reduce::<1>(7);
