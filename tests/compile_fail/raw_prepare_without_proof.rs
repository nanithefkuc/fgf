// expect: mismatched types
//! Raw preparation needs the private dispatch proof: a generic `()`
//! argument cannot stand in for it.
pub fn prepare<F: fgf::FieldBuffer>(coeff: fgf::Elem<F>) {
    let _ = F::prepare((), coeff);
}
