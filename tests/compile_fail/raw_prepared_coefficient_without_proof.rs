// expect: method `repr` is private
// expect-code: E0624
use fgf::FieldBuffer;
use fgf::ops::PreparedCoefficient;
pub fn representation<F: FieldBuffer, C: PreparedCoefficient<F>>(coefficient: &C) {
    let _ = coefficient.repr(());
}
