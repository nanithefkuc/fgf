// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The inherent residue constructor validates the characteristic before
// rejecting an out-of-range value.
pub fn call() -> Result<Elem<FiniteField<91, 1>>, fgf::RangeError> {
    Elem::<FiniteField<91, 1>>::from_residue(91)
}
