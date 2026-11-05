// expect: cannot find associated type `Prepared` in trait
//! The prepared coefficient form is not a member of the public bound: the
//! backend representation stays unreachable through the trait itself.
pub fn prepared<F: fgf::FieldBuffer>() -> <F as fgf::FieldBuffer>::Prepared {
    unreachable!()
}
