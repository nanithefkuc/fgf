// expect: is private
use fgf::field::Field;

// The row hooks sit in a crate-private supertrait: generic downstream code
// bounded by `Field` cannot read a row's descriptor.
fn descriptor_degree<F: Field>() -> u32 {
    F::DESCRIPTOR.degree()
}
