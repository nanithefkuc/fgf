// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// One is not a valid characteristic.
const BAD: Elem<FiniteField<1, 1>> = Elem::<FiniteField<1, 1>>::from_raw(0);
