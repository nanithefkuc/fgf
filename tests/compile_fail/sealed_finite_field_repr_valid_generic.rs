// expect: is private
use fgf::FiniteFieldRepr;

fn probe<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>>() {
    let () = R::VALID;
}
