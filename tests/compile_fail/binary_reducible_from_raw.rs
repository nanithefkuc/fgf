// expect: polynomial is reducible
use fgf::{Elem, FiniteField};
use fgf::binary::Polynomial;

// The inherent const constructor validates the representation's defining
// data at the use site.
pub fn call() -> Elem<FiniteField<2, 8, Polynomial<0x101>>> {
    Elem::<FiniteField<2, 8, Polynomial<0x101>>>::from_raw(5)
}
