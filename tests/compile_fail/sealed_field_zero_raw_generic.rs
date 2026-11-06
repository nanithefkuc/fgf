// expect: is private
use fgf::Field;

fn probe<F: Field>() -> F::Raw {
    F::ZERO_RAW
}
