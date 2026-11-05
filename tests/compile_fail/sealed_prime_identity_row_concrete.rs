// expect: error[E0576]
use fgf::prime::{PrimeCharacteristic, PrimeIdentity};

type Row = <PrimeCharacteristic<7> as PrimeIdentity>::Row;
