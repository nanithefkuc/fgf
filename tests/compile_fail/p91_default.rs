// expect: characteristic is not prime
use fgf::{Elem, FiniteField};

// The default element is the additive identity and validates the row.
pub fn call() -> Elem<FiniteField<91, 1>> {
    Default::default()
}
