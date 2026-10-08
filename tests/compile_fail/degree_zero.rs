// expect-rejection
use fgf::{Elem, FiniteField};

// Degree zero is rejected at the type level: no representation is defined
// over a zero-degree base, and the default presentation exists only at
// degree one.
const BAD: Elem<FiniteField<7, 0>> = Elem::<FiniteField<7, 0>>::ZERO;
