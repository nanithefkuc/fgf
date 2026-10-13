// expect: characteristic is not prime
use fgf::field::prime;

// The full word modulus fails before any remainder arithmetic.
const BAD: u64 = prime::reduce_wide::<{ u64::MAX }>(7);
