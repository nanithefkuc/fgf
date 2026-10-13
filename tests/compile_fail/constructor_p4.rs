// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// Four is composite.
const BAD: Elem<FiniteField<4, 1>> = Elem::<FiniteField<4, 1>>::from_raw(3);
