// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The additive identity evaluates the row's validation first.
const BAD: Elem<FiniteField<0, 1>> = Elem::<FiniteField<0, 1>>::ZERO;
