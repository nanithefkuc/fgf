// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Every raw trait method validates the row at entry, so a call at an
// invalid characteristic fails when the call is instantiated.
pub fn call() -> u64 {
    <FiniteField<4, 1> as Field>::canonical_raw(5)
}
