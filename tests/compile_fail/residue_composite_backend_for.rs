// expect: characteristic is not prime
// expect-code: E0080
use fgf::{Backend, FiniteField, backend_for};
pub fn query() -> Backend { backend_for::<FiniteField<9, 1>>() }
