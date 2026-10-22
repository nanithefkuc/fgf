//! Adapters for the single-row and row-pair public operations.

use super::*;

fn binary_rows<F: FieldKernels>(
    case: &Case,
    row_elems: Option<usize>,
    call: impl Fn(&mut [u8], usize, &[u8]),
    op: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    let (mut src, src_layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    fill_rows::<F>(&mut src, case, &src_layout, 0x51ed_2701, false);
    let original = dst.clone();
    let mut expected = original.clone();
    let used = layout.contiguous();
    for row in 0..layout.rows {
        let start = layout.row(row);
        let span = start..start + layout.row_bytes;
        binary::<F>(
            &mut expected[span.clone()],
            &original[span.clone()],
            &src[src_layout.row(row)..src_layout.row(row) + src_layout.row_bytes],
            &op,
        );
    }
    let entries = observe_unit(|| {
        let row_len = row_elems.map_or(0, |elems| elems * F::BYTES);
        call(
            &mut dst[layout.start..layout.start + used],
            row_len,
            &src[src_layout.start..src_layout.start + used],
        );
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

macro_rules! row_binary {
    ($name:ident, $plain:expr, $grouped:expr, $grouped_flag:literal, $op:expr) => {
        pub(super) fn $name<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
            let grouped = $grouped_flag;
            binary_rows::<F>(
                case,
                grouped.then(|| case.geometry.len),
                |dst, row_len, src| {
                    if grouped {
                        ($grouped)(dst, row_len, src);
                    } else {
                        let _ = row_len;
                        ($plain)(dst, src);
                    }
                },
                $op,
            )
        }
    };
}

row_binary!(
    ops_add_assign,
    |dst: &mut [u8], src: &[u8]| crate::ops::add_assign::<F>(dst, src),
    |dst: &mut [u8], row_len: usize, src: &[u8]| crate::ops::add_assign_rows::<F>(
        dst, row_len, src
    ),
    false,
    |a: Elem<F>, b: Elem<F>| a.add(b)
);
row_binary!(
    ops_sub_assign,
    |dst: &mut [u8], src: &[u8]| crate::ops::sub_assign::<F>(dst, src),
    |dst: &mut [u8], row_len: usize, src: &[u8]| crate::ops::sub_assign_rows::<F>(
        dst, row_len, src
    ),
    false,
    |a: Elem<F>, b: Elem<F>| a.sub(b)
);
row_binary!(
    ops_add_assign_rows,
    |dst: &mut [u8], src: &[u8]| crate::ops::add_assign::<F>(dst, src),
    |dst: &mut [u8], row_len: usize, src: &[u8]| crate::ops::add_assign_rows::<F>(
        dst, row_len, src
    ),
    true,
    |a: Elem<F>, b: Elem<F>| a.add(b)
);
row_binary!(
    ops_sub_assign_rows,
    |dst: &mut [u8], src: &[u8]| crate::ops::sub_assign::<F>(dst, src),
    |dst: &mut [u8], row_len: usize, src: &[u8]| crate::ops::sub_assign_rows::<F>(
        dst, row_len, src
    ),
    true,
    |a: Elem<F>, b: Elem<F>| a.sub(b)
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Add,
    Into,
    Assign,
}

fn scaled<F: FieldKernels>(case: &Case, shape: Shape, prepared: bool) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    let (mut src, src_layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    fill_rows::<F>(&mut src, case, &src_layout, 0x51ed_2701, false);
    let coeff = coefficient::<F>(case, 0);
    let held = crate::ops::Coeff::<F>::new(coeff);
    let original = dst.clone();
    let mut expected = original.clone();
    let used = layout.contiguous();
    match shape {
        Shape::Add => accumulate::<F>(
            &mut expected[layout.start..layout.start + used],
            coeff,
            &src[src_layout.start..src_layout.start + used],
        ),
        Shape::Into => scale_into::<F>(
            &mut expected[layout.start..layout.start + used],
            coeff,
            &src[src_layout.start..src_layout.start + used],
        ),
        Shape::Assign => {
            let payload = &mut expected[layout.start..layout.start + used];
            scale_assign::<F>(payload, coeff);
        }
    }
    let dst_payload = &mut dst[layout.start..layout.start + used];
    let src_payload = &src[src_layout.start..src_layout.start + used];
    let entries = observe_unit(|| match (shape, prepared) {
        (Shape::Add, false) => crate::ops::mul_add::<F>(dst_payload, coeff, src_payload),
        (Shape::Add, true) => crate::ops::mul_add_with::<F>(dst_payload, &held, src_payload),
        (Shape::Into, false) => crate::ops::mul_into::<F>(dst_payload, coeff, src_payload),
        (Shape::Into, true) => crate::ops::mul_into_with::<F>(dst_payload, &held, src_payload),
        (Shape::Assign, false) => crate::ops::mul_assign::<F>(dst_payload, coeff),
        (Shape::Assign, true) => crate::ops::mul_assign_with::<F>(dst_payload, &held),
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

pub(super) fn ops_mul_add<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Add, false)
}

pub(super) fn ops_mul_add_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Add, true)
}

pub(super) fn ops_mul_into<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Into, false)
}

pub(super) fn ops_mul_into_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Into, true)
}

pub(super) fn ops_mul_assign<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Assign, false)
}

pub(super) fn ops_mul_assign_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scaled::<F>(case, Shape::Assign, true)
}
