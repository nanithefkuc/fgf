// expect: is private
use fgf::field::Field;

// The prime-readout hook sits in the crate-private supertrait: generic
// downstream code bounded by `Field` cannot call it.
fn hook<F: Field>(raw: F::Raw) -> Option<u64> {
    F::to_prime(raw)
}
