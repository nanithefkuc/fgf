// expect: is private
use fgf::binary::BinaryField;

fn probe<F: BinaryField>() {
    let _ = F::DESCRIPTION;
}
