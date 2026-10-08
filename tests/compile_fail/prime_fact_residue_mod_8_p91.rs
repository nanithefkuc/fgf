// expect: characteristic is not prime
use fgf::{PrimeCharacteristic, PrimeIdentity};

// Each characteristic fact re-proves primality before answering.
const FACT: u8 = <PrimeCharacteristic<91> as PrimeIdentity>::RESIDUE_MOD_8;
