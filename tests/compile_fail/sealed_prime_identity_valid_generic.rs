// expect: is private
use fgf::prime::PrimeIdentity;

fn probe<C: PrimeIdentity>() {
    let () = C::VALID;
}
