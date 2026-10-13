// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// Ninety-one has factors three and thirty.
const BAD: Elem<FiniteField<91, 1>> = Elem::<FiniteField<91, 1>>::ZERO;
