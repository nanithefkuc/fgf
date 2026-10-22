//! Adapters for the broadcast-scalar public operations.

use super::*;

fn scalar<F: FieldKernels>(
    case: &Case,
    call: impl Fn(&mut [u8], Elem<F>),
    op: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let value = coefficient::<F>(case, 0);
    let original = dst.clone();
    let mut expected = original.clone();
    let used = layout.contiguous();
    unary::<F>(
        &mut expected[layout.start..layout.start + used],
        &original[layout.start..layout.start + used],
        |element| op(element, value),
    );
    let entries = observe_unit(|| {
        call(&mut dst[layout.start..layout.start + used], value);
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn ops_add_assign_scalar<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scalar::<F>(case, crate::ops::add_assign_scalar::<F>, Elem::<F>::add)
}

pub(super) fn ops_sub_assign_scalar<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scalar::<F>(case, crate::ops::sub_assign_scalar::<F>, Elem::<F>::sub)
}
