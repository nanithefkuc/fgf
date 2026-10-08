// expect-rejection
use fgf::{Elem, FiniteField, Polynomial, AES};

// Binary presentations exist only at characteristic two.
pub fn call() -> Elem<FiniteField<3, 8, Polynomial<AES>>> {
    Elem::<FiniteField<3, 8, Polynomial<AES>>>::ZERO
}
