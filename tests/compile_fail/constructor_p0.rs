// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// A non-prime characteristic must fail before any value can exist.
const BAD: Elem<FiniteField<0, 1>> = Elem::<FiniteField<0, 1>>::from_raw(1);
