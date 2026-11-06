// expect: polynomial is reducible
use fgf::binary::{AES, Polynomial};
use fgf::maps::Embedding;
use fgf::{Elem, FiniteField};

type Bad = FiniteField<2, 4, Polynomial<0x11>>;

// A generator query over an invalid source row validates the row before
// the level bounds check.
pub fn generator() -> Elem<Bad> {
    Embedding::<Bad, FiniteField<2, 8, Polynomial<AES>>>::generator(9)
}
