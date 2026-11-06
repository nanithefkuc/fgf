// expect: associated type `Prepared` is private
// expect-code: E0624
use fgf::FieldBuffer;
pub fn prepared<F: FieldBuffer>() -> Option<F::Prepared> {
    None
}
