// expect: error[E0277]
// `HasGenerator` is sealed: only the crate's shipped rows carry a selected
// generator, so a downstream type cannot implement it.
#[derive(Clone, Copy, Debug)]
pub struct Downstream;

impl fgf::HasGenerator for Downstream {}
