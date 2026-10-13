// expect-rejection
use fgf::field::{Field, PrimeCharacteristic};

// `Field` is sealed: the row hooks live in a crate-private supertrait, so a
// downstream implementation cannot satisfy the bound.
#[derive(Clone, Copy, Debug)]
struct DownstreamSeven;

impl Field for DownstreamSeven {
    type Raw = u8;
    type Characteristic = PrimeCharacteristic<7>;
    const NAME: &'static str = "seven";
    const DEGREE: u32 = 1;
    const ORDER: u128 = 7;
    const ZERO_RAW: u8 = 0;
    const ONE_RAW: u8 = 1;
    const VALID: () = ();
    fn canonical_raw(raw: u8) -> u8 {
        raw % 7
    }
    fn add_raw(left: u8, right: u8) -> u8 {
        (left + right) % 7
    }
    fn sub_raw(left: u8, right: u8) -> u8 {
        (left + 7 - right) % 7
    }
    fn neg_raw(value: u8) -> u8 {
        if value == 0 { 0 } else { 7 - value }
    }
    fn mul_raw(left: u8, right: u8) -> u8 {
        (left * right) % 7
    }
    fn inv_raw(value: u8) -> u8 {
        if value == 0 { 0 } else { 1 }
    }
}

const ONE: fgf::Elem<DownstreamSeven> = fgf::Elem::<DownstreamSeven>::ONE;
