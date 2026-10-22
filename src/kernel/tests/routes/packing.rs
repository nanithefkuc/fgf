//! Adapters for the canonical pack and unpack operations.

use super::*;

fn pattern<F: FieldKernels>(case: &Case) -> Vec<Elem<F>> {
    let coeff = coefficient::<F>(case, 0);
    let order = <F as crate::field::Field>::ORDER;
    (0..case.geometry.len)
        .map(|index| {
            let raw = (index as u128) % (order - 1) + 1;
            coeff.mul(Elem::<F>::from_index(raw).expect("index below the field order"))
        })
        .collect()
}

fn encode_all<F: FieldKernels>(values: &[Elem<F>]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * F::BYTES];
    for (lane, &value) in out.chunks_exact_mut(F::BYTES).zip(values) {
        F::encode(lane, value);
    }
    out
}

pub(super) fn ops_pack<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let values = pattern::<F>(case);
    let length = values.len() * F::BYTES;
    let (mut dst, start) = guarded_bytes(length, case.geometry.misalign);
    let (mut expected, _) = guarded_bytes(length, case.geometry.misalign);
    expected[start..start + length].copy_from_slice(&encode_all::<F>(&values));
    let entries = observe_unit(|| {
        crate::ops::pack::<F>(&mut dst[start..start + length], &values);
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

pub(super) fn ops_unpack<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let values = pattern::<F>(case);
    let src = encode_all::<F>(&values);
    let mut dst = vec![Elem::<F>::ZERO; values.len()];
    let entries = observe_unit(|| {
        crate::ops::unpack::<F>(&mut dst, &src);
    });
    outcome::<F>(
        vec![BufferCheck {
            name: "dst",
            actual: encode_all::<F>(&dst),
            expected: encode_all::<F>(&values),
        }],
        entries,
    )
}

pub(super) fn ops_pack_to_vec<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let values = pattern::<F>(case);
    let (actual, entries) = observe(|| crate::ops::pack_to_vec::<F>(&values));
    outcome::<F>(
        vec![BufferCheck {
            name: "dst",
            actual,
            expected: encode_all::<F>(&values),
        }],
        entries,
    )
}

pub(super) fn ops_unpack_to_vec<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    let values = pattern::<F>(case);
    let src = encode_all::<F>(&values);
    let (actual, entries) = observe(|| crate::ops::unpack_to_vec::<F>(&src));
    outcome::<F>(
        vec![BufferCheck {
            name: "dst",
            actual: encode_all::<F>(&actual),
            expected: encode_all::<F>(&values),
        }],
        entries,
    )
}
