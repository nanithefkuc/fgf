// expect: polynomial is reducible
use fgf::binary::{AES, Polynomial};
use fgf::FiniteField;
use fgf::maps::Embedding;

// Generator metadata validates the source row.
pub fn count() -> usize {
    Embedding::<FiniteField<2, 4, Polynomial<0x11>>, FiniteField<2, 8, Polynomial<AES>>>::generator_count()
}
