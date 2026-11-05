// expect: mismatched types
//! The backend query needs the private dispatch proof: a generic `()`
//! argument cannot stand in for it.
fn query<F: fgf::FieldBuffer>() -> fgf::Backend {
    F::backend(())
}
