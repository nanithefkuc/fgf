// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// 1,373,653 = 829 × 1657 passes the trial-division stage (the primes up
// to 37) and base 2 is a strong liar for it, so rejection depends on a
// later Miller–Rabin witness.
const BAD: Elem<FiniteField<1_373_653, 1>> = Elem::<FiniteField<1_373_653, 1>>::from_raw(1);
