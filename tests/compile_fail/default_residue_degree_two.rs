// expect-code: E0277
use fgf::{Elem, FiniteField};

// Omitting the presentation at degree two must not silently select an
// extension: the default residue presentation is degree-one only.
const BAD: Elem<FiniteField<7, 2>> = Elem::<FiniteField<7, 2>>::ZERO;
