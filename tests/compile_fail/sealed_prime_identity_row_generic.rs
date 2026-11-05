// expect: error[E0220]
use fgf::prime::PrimeIdentity;

fn probe<C: PrimeIdentity>() {
    let _: Option<C::Row> = None;
}
