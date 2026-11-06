// expect: error[E0277]
use fgf::Field;
use fgf::prime::PrimeCharacteristic;

// `Field` is sealed: the row hooks live in crate-private supertraits, so a
// downstream implementation cannot satisfy the bound.
#[derive(Clone, Copy, Debug)]
struct DownstreamSeven;

impl Field for DownstreamSeven {
    type Raw = u8;
    type Characteristic = PrimeCharacteristic<7>;
    const NAME: &'static str = "seven";
    const DEGREE: u32 = 1;
    const ORDER: u128 = 7;
}

const ONE: fgf::Elem<DownstreamSeven> = fgf::Elem::<DownstreamSeven>::ONE;
