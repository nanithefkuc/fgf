// expect: error[E0576]
use fgf::FiniteFieldRepr;
use fgf::binary::Polynomial;

fn probe() {
    let () = <Polynomial<0x11B> as FiniteFieldRepr<2, 8>>::VALID;
}
