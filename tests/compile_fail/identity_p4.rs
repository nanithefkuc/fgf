// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The multiplicative identity validates before reading row-local data.
const BAD: Elem<FiniteField<4, 1>> = Elem::<FiniteField<4, 1>>::ONE;
