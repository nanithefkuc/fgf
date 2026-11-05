// expect: is private
use fgf::HasGenerator;

fn probe<F: HasGenerator>() -> F::Raw {
    F::GENERATOR_RAW
}
