// expect: is private
use fgf::field::Field;

// The prime-embedding hook sits in the crate-private supertrait: generic
// downstream code bounded by `Field` cannot call it.
fn hook<F: Field>() -> F::Raw {
    F::from_prime(3)
}
