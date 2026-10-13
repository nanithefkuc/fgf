// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The full word fails the primality proof like any composite.
const BAD: Elem<FiniteField<{ u64::MAX }, 1>> = Elem::<FiniteField<{ u64::MAX }, 1>>::ONE;
