//! Adapters for the reduction-shaped public operations.

use super::*;

pub(super) fn ops_dot_product<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let a = elements::<F>(case, 0, case.geometry.len, false);
    let b = elements::<F>(case, 0x51ed_2701, case.geometry.len, false);
    let mut expected = Elem::<F>::ZERO;
    for (x, y) in a.chunks_exact(F::BYTES).zip(b.chunks_exact(F::BYTES)) {
        expected = expected.add(F::decode(x).mul(F::decode(y)));
    }
    let (actual, entries) = observe(|| crate::ops::dot_product::<F>(&a, &b));
    let mut result = outcome::<F>(Vec::new(), entries);
    result
        .values
        .push(("dot_product", actual.to_index(), expected.to_index()));
    result
}
