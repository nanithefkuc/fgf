//! Adapters for the elementwise and reciprocal public operations.

use super::*;

/// A guarded single row filled from noise.
fn row<F: FieldKernels>(case: &Case, salt: u64, nonzero: bool) -> (Vec<u8>, Layout) {
    let (mut buffer, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut buffer, case, &layout, salt, nonzero);
    (buffer, layout)
}

fn two_source<F: FieldKernels>(
    case: &Case,
    left: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
    call: impl Fn(&mut [u8], &[u8], &[u8]),
) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    let (a, a_layout) = row::<F>(case, 0, false);
    let coeff = coefficient::<F>(case, 0);
    let b = constant_row(coeff, case.geometry.len);
    let used = layout.contiguous();
    let mut expected = vec![0u8; used];
    binary::<F>(
        &mut expected,
        &a[a_layout.start..a_layout.start + used],
        &b,
        left,
    );
    let mut expected_buffer = dst.clone();
    expected_buffer[layout.start..layout.start + used].copy_from_slice(&expected);
    let entries = observe_unit(|| {
        call(
            &mut dst[layout.start..layout.start + used],
            &a[a_layout.start..a_layout.start + used],
            &b,
        );
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected: expected_buffer,
        }],
        entries,
    )
}

fn assign_source<F: FieldKernels>(
    case: &Case,
    combine: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
    call: impl Fn(&mut [u8], &[u8]),
) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let coeff = coefficient::<F>(case, 0);
    let src = constant_row(coeff, case.geometry.len);
    let original = dst.clone();
    let used = layout.contiguous();
    let mut expected = original.clone();
    binary::<F>(
        &mut expected[layout.start..layout.start + used],
        &original[layout.start..layout.start + used],
        &src,
        combine,
    );
    let entries = observe_unit(|| {
        call(&mut dst[layout.start..layout.start + used], &src);
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

fn unary_row<F: FieldKernels>(
    case: &Case,
    call: impl Fn(&mut [u8], &[u8]),
    op: impl Fn(Elem<F>) -> Elem<F>,
    in_place: bool,
) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    let (src, src_layout) = row::<F>(case, 0x51ed_2701, false);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let used = layout.contiguous();
    let original = dst.clone();
    let mut expected = original.clone();
    let source = if in_place {
        &original[layout.start..layout.start + used]
    } else {
        &src[src_layout.start..src_layout.start + used]
    };
    unary::<F>(&mut expected[layout.start..layout.start + used], source, op);
    let entries = observe_unit(|| {
        if in_place {
            call(&mut dst[layout.start..layout.start + used], &[]);
        } else {
            call(
                &mut dst[layout.start..layout.start + used],
                &src[src_layout.start..src_layout.start + used],
            );
        }
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

pub(super) fn ops_mul_elementwise<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    two_source::<F>(case, Elem::<F>::mul, crate::ops::mul_elementwise::<F>)
}

pub(super) fn ops_mul_elementwise_assign<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    assign_source::<F>(
        case,
        Elem::<F>::mul,
        crate::ops::mul_elementwise_assign::<F>,
    )
}

pub(super) fn ops_div_elementwise<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    two_source::<F>(case, Elem::<F>::div, crate::ops::div_elementwise::<F>)
}

pub(super) fn ops_div_elementwise_assign<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    assign_source::<F>(
        case,
        Elem::<F>::div,
        crate::ops::div_elementwise_assign::<F>,
    )
}

pub(super) fn ops_inv_into<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    unary_row::<F>(
        case,
        |dst, src| crate::ops::inv_into::<F>(dst, src),
        Elem::<F>::inv,
        false,
    )
}

pub(super) fn ops_inv_assign<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    unary_row::<F>(
        case,
        |dst, _| crate::ops::inv_assign::<F>(dst),
        Elem::<F>::inv,
        true,
    )
}
