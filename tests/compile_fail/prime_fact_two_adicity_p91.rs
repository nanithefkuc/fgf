// expect: characteristic is not prime
use fgf::prime::{PrimeCharacteristic, PrimeIdentity};

// Each characteristic fact re-proves primality before answering.
const FACT: u32 = <PrimeCharacteristic<91> as PrimeIdentity>::TWO_ADICITY;
