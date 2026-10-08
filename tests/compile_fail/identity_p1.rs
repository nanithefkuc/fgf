// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The additive identity of a unit modulus is not a field.
const BAD: Elem<FiniteField<1, 1>> = Elem::<FiniteField<1, 1>>::ZERO;
