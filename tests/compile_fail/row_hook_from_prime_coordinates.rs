// expect: is private
use fgf::field::Field;

// The coordinate constructor hook sits in the crate-private supertrait:
// generic downstream code bounded by `Field` cannot call it.
fn hook<F: Field>() -> F::Raw {
    F::from_prime_coordinates(&[0u64; 8])
}
