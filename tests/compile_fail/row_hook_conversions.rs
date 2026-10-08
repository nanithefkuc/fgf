// expect-rejection
use fgf::field::Field;

// Every conversion hook sits beside the descriptor in the crate-private
// supertrait: generic downstream code bounded by `Field` cannot call them.
fn hooks<F: Field>(raw: F::Raw) {
    let mut coordinates = [0u64; 8];
    F::prime_coordinates_into(&mut coordinates, raw);
    let _ = F::from_prime_coordinates(&coordinates);
    let _ = F::from_prime(3);
    let _ = F::to_prime(raw);
}
