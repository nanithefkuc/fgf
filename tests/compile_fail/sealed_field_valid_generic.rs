// expect: is private
use fgf::Field;

fn probe<F: Field>() {
    let () = F::VALID;
}
