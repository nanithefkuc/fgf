// expect: is private
use fgf::Field;

fn probe<F: Field>(raw: F::Raw) -> F::Raw {
    F::canonical_raw(raw)
}
