// expect: cannot find method or associated constant `backend` in trait
//! The backend query is not a member of the public bound: naming the trait
//! explicitly does not reach the private dispatch contract.
pub fn query<F: fgf::FieldBuffer>() -> fgf::Backend {
    <F as fgf::FieldBuffer>::backend()
}
