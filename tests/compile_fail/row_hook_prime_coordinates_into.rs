// expect: is private
use fgf::field::Field;

// The coordinate hook sits in the crate-private supertrait: generic
// downstream code bounded by `Field` cannot call it.
fn hook<F: Field>(raw: F::Raw) {
    let mut coordinates = [0u64; 8];
    F::prime_coordinates_into(&mut coordinates, raw);
}
