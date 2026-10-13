// expect: characteristic is not prime
use fgf::field::prime;

// The widened reduction carries the same primality guard.
const BAD: u64 = prime::reduce_wide::<91>(7);
