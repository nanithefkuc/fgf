// expect-rejection
use fgf::{Elem, FiniteField};
use fgf::binary::Polynomial;

// No binary presentation exists above degree 64.
pub fn call() -> Elem<FiniteField<2, 128, Polynomial<3>>> {
    Elem::<FiniteField<2, 128, Polynomial<3>>>::ZERO
}
