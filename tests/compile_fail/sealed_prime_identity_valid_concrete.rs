// expect: error[E0576]
use fgf::prime::{PrimeCharacteristic, PrimeIdentity};

fn probe() {
    let () = <PrimeCharacteristic<7> as PrimeIdentity>::VALID;
}
