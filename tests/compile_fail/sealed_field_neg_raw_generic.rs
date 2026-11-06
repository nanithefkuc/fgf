// expect: is private
use fgf::Field;

fn probe<F: Field>(value: F::Raw) -> F::Raw {
    F::neg_raw(value)
}
