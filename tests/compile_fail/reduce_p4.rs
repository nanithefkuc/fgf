// expect: characteristic is not prime
use fgf::field::prime;

// Four is composite, so the validated reduction fails.
const BAD: u64 = prime::reduce::<4>(7);
