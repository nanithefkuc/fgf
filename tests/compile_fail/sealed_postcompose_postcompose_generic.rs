// expect: is private
use fgf::{Elem, Field};
use fgf::maps::Postcompose;

fn probe<M, Mid: Field, Out: Field>(map: &M, image: Elem<Mid>) -> Elem<Out>
where
    M: Postcompose<Mid, Out>,
{
    M::postcompose(map, image)
}
