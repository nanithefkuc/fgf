// expect: polynomial is reducible
use fgf::{Elem, FiniteField};
use fgf::binary::Polynomial;

// The identity constant validates the representation's defining data.
pub fn call() -> Elem<FiniteField<2, 8, Polynomial<0x101>>> {
    Elem::<FiniteField<2, 8, Polynomial<0x101>>>::ONE
}
