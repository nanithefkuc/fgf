//! Adapters for the scatter, gather, and matrix public operations.

use super::*;

/// One source row per source slot, each salted separately.
fn sources<F: FieldKernels>(case: &Case, elem_bytes: usize) -> (Vec<Vec<u8>>, Vec<Layout>) {
    (0..case.geometry.sources)
        .map(|index| {
            let (mut buffer, layout) = guarded(case, elem_bytes);
            fill_rows::<F>(
                &mut buffer,
                case,
                &layout,
                0x51ed_2701 + index as u64,
                false,
            );
            (buffer, layout)
        })
        .unzip()
}

fn scatter<F: FieldKernels>(case: &Case, into: bool, prepared: bool) -> Outcome {
    let (mut rows, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut rows, case, &layout, 0, true);
    let (mut src, src_layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut src, case, &src_layout, 0x51ed_2701, false);
    let coeffs: Vec<Elem<F>> = coefficients(case);
    let held = crate::ops::CoeffVec::<F>::new(&coeffs);
    let view = held.as_ref();
    let original = rows.clone();
    let mut expected = original.clone();
    let row_bytes = layout.row_bytes;
    for (index, &coeff) in coeffs.iter().enumerate() {
        let start = layout.row(index);
        if into {
            scale_into::<F>(
                &mut expected[start..start + row_bytes],
                coeff,
                &src[src_layout.start..src_layout.start + row_bytes],
            );
        } else {
            accumulate::<F>(
                &mut expected[start..start + row_bytes],
                coeff,
                &src[src_layout.start..src_layout.start + row_bytes],
            );
        }
    }
    let used = layout.contiguous();
    let dst = &mut rows[layout.start..layout.start + used];
    let source = &src[src_layout.start..src_layout.start + row_bytes];
    let entries = observe_unit(|| match (into, prepared) {
        (false, false) => crate::ops::mul_add_scatter::<F>(dst, row_bytes, &coeffs, source),
        (false, true) => crate::ops::mul_add_scatter_with::<F>(dst, row_bytes, view, source),
        (true, false) => crate::ops::mul_into_scatter::<F>(dst, row_bytes, &coeffs, source),
        (true, true) => unreachable!("the prepared scatter form is accumulation-only"),
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "rows",
            actual: rows,
            expected,
        }],
        entries,
    )
}

pub(super) fn ops_mul_add_scatter<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scatter::<F>(case, false, false)
}

pub(super) fn ops_mul_add_scatter_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scatter::<F>(case, false, true)
}

pub(super) fn ops_mul_into_scatter<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    scatter::<F>(case, true, false)
}

fn gather<F: FieldKernels>(case: &Case, into: bool, prepared: bool) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let (srcs, src_layouts) = sources::<F>(case, F::BYTES);
    let refs: Vec<&[u8]> = srcs
        .iter()
        .zip(&src_layouts)
        .map(|(buffer, layout)| &buffer[layout.start..layout.start + layout.row_bytes])
        .collect();
    let coeffs: Vec<Elem<F>> = coefficients(case);
    let held = crate::ops::CoeffVec::<F>::new(&coeffs);
    let view = held.as_ref();
    let original = dst.clone();
    let mut expected = original.clone();
    let row_bytes = layout.row_bytes;
    let payload = &mut expected[layout.start..layout.start + row_bytes];
    if into {
        payload.fill(0);
    }
    for (&coeff, source) in coeffs.iter().zip(&refs) {
        accumulate::<F>(payload, coeff, source);
    }
    let dst_row = &mut dst[layout.start..layout.start + row_bytes];
    let entries = observe_unit(|| match (into, prepared) {
        (false, false) => crate::ops::mul_add_gather::<F>(dst_row, &coeffs, &refs),
        (false, true) => crate::ops::mul_add_gather_with::<F>(dst_row, view, &refs),
        (true, false) => crate::ops::mul_into_gather::<F>(dst_row, &coeffs, &refs),
        (true, true) => crate::ops::mul_into_gather_with::<F>(dst_row, view, &refs),
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

pub(super) fn ops_mul_add_gather<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    gather::<F>(case, false, false)
}

pub(super) fn ops_mul_add_gather_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    gather::<F>(case, false, true)
}

pub(super) fn ops_mul_into_gather<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    gather::<F>(case, true, false)
}

pub(super) fn ops_mul_into_gather_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    gather::<F>(case, true, true)
}

pub(super) fn ops_add_gather_offsets<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let (mut region, region_layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut region, case, &region_layout, 0x51ed_2701, false);
    let row_bytes = layout.row_bytes;
    let offsets: Vec<u32> = (0..case.geometry.sources)
        .map(|index| u32::try_from(index * region_layout.pitch).expect("pitch fits u32"))
        .collect();
    let original = dst.clone();
    let mut expected = original.clone();
    let mut folded = expected[layout.start..layout.start + row_bytes].to_vec();
    for &offset in &offsets {
        let offset = offset as usize;
        let previous = folded.clone();
        binary::<F>(
            &mut folded,
            &previous,
            &region[region_layout.start + offset..region_layout.start + offset + row_bytes],
            Elem::<F>::add,
        );
    }
    expected[layout.start..layout.start + row_bytes].copy_from_slice(&folded);
    let region_slice = &region
        [region_layout.start..region_layout.start + region_layout.pitch * case.geometry.sources];
    let entries = observe_unit(|| {
        crate::ops::add_gather_offsets::<F>(
            &mut dst[layout.start..layout.start + row_bytes],
            region_slice,
            &offsets,
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

fn matrix<F: FieldKernels>(case: &Case, into: bool, prepared: bool) -> Outcome {
    let rows = case.geometry.rows;
    let count = case.geometry.sources;
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let (srcs, src_layouts) = sources::<F>(case, F::BYTES);
    let row_bytes = layout.row_bytes;
    let mut by_source: Vec<Vec<Elem<F>>> = Vec::with_capacity(count);
    for source in 0..count {
        by_source.push(
            (0..rows)
                .map(|row| coefficient::<F>(case, row * count + source))
                .collect(),
        );
    }
    let refs: Vec<&[u8]> = srcs
        .iter()
        .zip(&src_layouts)
        .map(|(buffer, layout)| &buffer[layout.start..layout.start + layout.row_bytes])
        .collect();
    let terms: Vec<(&[Elem<F>], &[u8])> = by_source
        .iter()
        .map(Vec::as_slice)
        .zip(&refs)
        .map(|(coeffs, source)| (coeffs, *source))
        .collect();
    let flat: Vec<Elem<F>> = (0..count)
        .flat_map(|source| by_source[source].iter().copied())
        .collect();
    let held = crate::ops::CoeffMatrix::<F>::from_source_major(count, rows, &flat);
    let original = dst.clone();
    let mut expected = original.clone();
    if into {
        for row in 0..rows {
            let start = layout.row(row);
            expected[start..start + row_bytes].fill(0);
        }
    }
    for (source, coeffs) in by_source.iter().enumerate() {
        for (&coeff, row) in coeffs.iter().zip(0..rows) {
            let start = layout.row(row);
            accumulate::<F>(&mut expected[start..start + row_bytes], coeff, refs[source]);
        }
    }
    let used = layout.contiguous();
    let dst_rows = &mut dst[layout.start..layout.start + used];
    let entries = observe_unit(|| match (into, prepared) {
        (false, false) => crate::ops::mul_add_matrix::<F>(dst_rows, row_bytes, rows, &terms),
        (false, true) => {
            crate::ops::mul_add_matrix_with::<F>(dst_rows, row_bytes, &held, &refs);
        }
        (true, false) => crate::ops::mul_into_matrix::<F>(dst_rows, row_bytes, rows, &terms),
        (true, true) => {
            crate::ops::mul_into_matrix_with::<F>(dst_rows, row_bytes, &held, &refs);
        }
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "rows",
            actual: dst,
            expected,
        }],
        entries,
    )
}

fn matrix_at<F: FieldKernels>(case: &Case, into: bool, prepared: bool) -> Outcome {
    let rows = case.geometry.rows;
    let count = case.geometry.sources;
    let (mut dst, layout) = guarded(case, F::BYTES);
    fill_rows::<F>(&mut dst, case, &layout, 0, true);
    let (srcs, src_layouts) = sources::<F>(case, F::BYTES);
    let row_bytes = layout.row_bytes;
    let mut by_source: Vec<Vec<Elem<F>>> = Vec::with_capacity(count);
    for source in 0..count {
        by_source.push(
            (0..rows)
                .map(|row| coefficient::<F>(case, row * count + source))
                .collect(),
        );
    }
    let refs: Vec<&[u8]> = srcs
        .iter()
        .zip(&src_layouts)
        .map(|(buffer, layout)| &buffer[layout.start..layout.start + layout.row_bytes])
        .collect();
    let terms: Vec<(&[Elem<F>], &[u8])> = by_source
        .iter()
        .map(Vec::as_slice)
        .zip(&refs)
        .map(|(coeffs, source)| (coeffs, *source))
        .collect();
    let flat: Vec<Elem<F>> = (0..count)
        .flat_map(|source| by_source[source].iter().copied())
        .collect();
    let held = crate::ops::CoeffMatrix::<F>::from_source_major(count, rows, &flat);
    let starts: Vec<usize> = (0..rows).map(|row| row * layout.pitch).collect();
    let original = dst.clone();
    let mut expected = original.clone();
    if into {
        for &start in &starts {
            let at = layout.start + start;
            expected[at..at + row_bytes].fill(0);
        }
    }
    for (source, coeffs) in by_source.iter().enumerate() {
        for (&coeff, &row_start) in coeffs.iter().zip(&starts) {
            let at = layout.start + row_start;
            accumulate::<F>(&mut expected[at..at + row_bytes], coeff, refs[source]);
        }
    }
    let span = layout.span();
    let dst_rows = &mut dst[layout.start..span];
    let entries = observe_unit(|| match (into, prepared) {
        (false, false) => {
            crate::ops::mul_add_matrix_at::<F>(dst_rows, row_bytes, &starts, &terms);
        }
        (false, true) => {
            crate::ops::mul_add_matrix_at_with::<F>(dst_rows, row_bytes, &starts, &held, &refs);
        }
        (true, false) => {
            crate::ops::mul_into_matrix_at::<F>(dst_rows, row_bytes, &starts, &terms);
        }
        (true, true) => {
            crate::ops::mul_into_matrix_at_with::<F>(dst_rows, row_bytes, &starts, &held, &refs);
        }
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "rows",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn ops_mul_add_matrix<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix::<F>(case, false, false)
}

pub(super) fn ops_mul_add_matrix_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix::<F>(case, false, true)
}

pub(super) fn ops_mul_into_matrix<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix::<F>(case, true, false)
}

pub(super) fn ops_mul_into_matrix_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix::<F>(case, true, true)
}

pub(super) fn ops_mul_add_matrix_at<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix_at::<F>(case, false, false)
}

pub(super) fn ops_mul_add_matrix_at_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix_at::<F>(case, false, true)
}

pub(super) fn ops_mul_into_matrix_at<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix_at::<F>(case, true, false)
}

pub(super) fn ops_mul_into_matrix_at_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    matrix_at::<F>(case, true, true)
}
