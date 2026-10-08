// expect-rejection
use fgf::{FiniteField, Polynomial, AES, PrimeField};

// The prime-field marker covers degree-one rows only; a degree-eight binary
// row is an extension, not GF(P).
fn requires_prime_field<F: PrimeField>() {}

pub fn call() {
    requires_prime_field::<FiniteField<2, 8, Polynomial<AES>>>();
}
