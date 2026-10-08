// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The full word is not prime.
const BAD: Elem<FiniteField<{ u64::MAX }, 1>> = Elem::<FiniteField<{ u64::MAX }, 1>>::from_raw(1);
