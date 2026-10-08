// expect: polynomial is reducible
use fgf::{Field, FiniteField, Polynomial};

// Every raw entry point validates the representation's defining data, so a
// reducible polynomial never answers.
pub fn call() -> u8 {
    <FiniteField<2, 8, Polynomial<0x101>> as Field>::sub_raw(1, 2)
}
