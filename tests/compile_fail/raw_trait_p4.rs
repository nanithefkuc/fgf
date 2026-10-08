// expect: characteristic is not prime
use fgf::{Field, FiniteField};

// Composite characteristics fail the use-time validity proof.
const BAD: () = <FiniteField<4, 1> as Field>::VALID;
