// expect: characteristic is not prime
use fgf::{PrimeCharacteristic, PrimeIdentity};

// The characteristic facts describe a prime: every read re-proves primality
// before answering, so a composite modulus fails const evaluation.
const BITS: u32 = <PrimeCharacteristic<91> as PrimeIdentity>::BITS;
const TWO_ADICITY: u32 = <PrimeCharacteristic<91> as PrimeIdentity>::TWO_ADICITY;
const RESIDUE_MOD_8: u8 = <PrimeCharacteristic<91> as PrimeIdentity>::RESIDUE_MOD_8;
