// expect: field order overflows u128
use fgf::{Elem, Field, PrimeCharacteristic};

// The checked u128 cardinality computation overflows before this custom
// row's misdeclared order is ever compared: `P^3` leaves the u128 range for
// the widest prime, so central validation fails the const evaluation.
#[derive(Clone, Copy, Debug)]
struct Widest3;
impl Field for Widest3 {
    type Raw = u128;
    type Characteristic = PrimeCharacteristic<0xFFFF_FFFF_FFFF_FFC5>;
    const NAME: &'static str = "widest3";
    const DEGREE: u32 = 3;
    const ORDER: u128 = 0;
    const ZERO_RAW: u128 = 0;
    const ONE_RAW: u128 = 1;
    const VALID: () = ();
    fn canonical_raw(raw: u128) -> u128 {
        raw
    }
    fn add_raw(left: u128, right: u128) -> u128 {
        left ^ right
    }
    fn sub_raw(left: u128, right: u128) -> u128 {
        left ^ right
    }
    fn neg_raw(value: u128) -> u128 {
        value
    }
    fn mul_raw(left: u128, right: u128) -> u128 {
        left & right
    }
    fn inv_raw(value: u128) -> u128 {
        value
    }
}

const BAD: Elem<Widest3> = Elem::<Widest3>::ZERO;
