// expect: is private
use fgf::Field;

fn probe<F: Field>(left: F::Raw, right: F::Raw) -> F::Raw {
    F::mul_raw(left, right)
}
