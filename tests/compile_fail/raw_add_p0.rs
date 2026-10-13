// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Every raw trait method validates the row at entry, so a call at an
// invalid characteristic fails when the call is instantiated.
pub fn call() -> u64 {
    <FiniteField<0, 1> as Field>::add_raw(1, 2)
}
