// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Every raw trait method validates the row at entry, so a call at an
// invalid characteristic fails when the call is instantiated.
pub fn call() -> u64 {
    <FiniteField<18446744073709551615, 1> as Field>::mul_raw(2, 3)
}
