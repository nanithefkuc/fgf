// expect: error[E0576]
use fgf::{Elem, Field};
use fgf::maps::{Embedding, Postcompose};
use fgf::binary::Polynomial;
use fgf::FiniteField;

type Mid = FiniteField<2, 4, Polynomial<0x13>>;
type Out = FiniteField<2, 8, Polynomial<0x11B>>;

fn probe(map: &Embedding<Mid, Out>, image: Elem<Mid>) -> Elem<Out> {
    <Embedding<Mid, Out> as Postcompose<Mid, Out>>::postcompose(map, image)
}
