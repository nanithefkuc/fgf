// expect: characteristic is not prime
use fgf::{PrimeCharacteristic, PrimeIdentity};

// Each characteristic fact re-proves primality before answering.
const FACT: u32 = <PrimeCharacteristic<4> as PrimeIdentity>::BITS;
