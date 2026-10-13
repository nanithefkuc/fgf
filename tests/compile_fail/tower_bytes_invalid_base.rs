// expect: polynomial is reducible
use fgf::{Elem, FiniteField, Polynomial, Tower, TowerSpec};

#[derive(Clone, Copy)]
pub struct Spec;
impl TowerSpec for Spec {
    type Base = FiniteField<2, 8, Polynomial<0x101>>;
    const LINEAR_COEFFICIENT: u64 = 1;
    const CONSTANT_COEFFICIENT: u64 = 0x20;
    const NAME: &'static str = "invalid";
}

// Byte construction validates the tower like every other constructor.
pub const BAD: Elem<FiniteField<2, 16, Tower<Spec>>> =
    Elem::<FiniteField<2, 16, Tower<Spec>>>::from_bytes([1, 0]);
