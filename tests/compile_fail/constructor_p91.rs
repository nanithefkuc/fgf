// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// Ninety-one is composite.
const BAD: Elem<FiniteField<91, 1>> = Elem::<FiniteField<91, 1>>::from_raw(5);
