//! Adapters for the prepared field-map rows over stable byte lanes.

use super::*;
use crate::field::{
    Embedding, EmbeddingError, Frobenius, Isomorphism, PreparedEmbedding, PreparedIsomorphism,
};

type EmbeddingPair<S, T> = (Embedding<S, T>, PreparedEmbedding<S, T>);
type IsomorphismPair<S, T> = (Isomorphism<S, T>, PreparedIsomorphism<S, T>);

fn images<T: FieldBuffer>(spec: &MapSpec) -> Vec<Elem<T>> {
    spec.images
        .iter()
        .map(|image| {
            assert_eq!(
                image.len(),
                T::BYTES,
                "a caller image is one target element encoding"
            );
            T::decode(image)
        })
        .collect()
}

fn embedding<S, T>(case: &Case) -> Result<EmbeddingPair<S, T>, EmbeddingError>
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = match case.map.as_ref() {
        Some(spec) if !spec.images.is_empty() => Embedding::from_images(&images::<T>(spec))?,
        _ => Embedding::new()?,
    };
    let prepared = PreparedEmbedding::new(&compact);
    Ok((compact, prepared))
}

fn isomorphism<S, T>(case: &Case) -> Result<IsomorphismPair<S, T>, EmbeddingError>
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = match case.map.as_ref() {
        Some(spec) if !spec.images.is_empty() => Isomorphism::from_images(&images::<T>(spec))?,
        _ => Isomorphism::new()?,
    };
    let prepared = PreparedIsomorphism::new(&compact);
    Ok((compact, prepared))
}

fn source<S: FieldBuffer>(case: &Case, salt: u64) -> Vec<u8> {
    elements::<S>(case, salt, case.geometry.len, false)
}

fn destination<T: FieldBuffer>(case: &Case) -> (Vec<u8>, Layout) {
    guarded(case, T::BYTES)
}

/// Report the fixture source side's kernel backend.
///
/// A width-changing map's receipt names the source row's
/// [`backend_for`](crate::kernel::backend_for); the source is the fixture
/// side the ledger rule and its work basis describe.
fn map_outcome<S, T>(checks: Vec<BufferCheck>, entries: Vec<Entry>) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer,
{
    let mut outcome = Outcome::new(backend(), backend_for::<S>()).with_trace(entries);
    outcome.source_bytes = S::BYTES;
    outcome.target_bytes = T::BYTES;
    outcome.checks = checks;
    outcome
}

/// The independent Frobenius image: `x^(p^k)` with `k` normalized modulo
/// the field degree, computed by repeated `pow(p)` rather than the prepared
/// map under test.
pub(super) fn frobenius_oracle<F: FieldKernels>(value: Elem<F>, power: u32) -> Elem<F> {
    use crate::field::PrimeIdentity;
    let characteristic =
        u128::from(<<F as crate::field::Field>::Characteristic as PrimeIdentity>::CHARACTERISTIC);
    let mut image = value;
    for _ in 0..(power % F::DEGREE) {
        image = image.pow(characteristic);
    }
    image
}

fn frobenius<F: FieldKernels>(case: &Case, in_place: bool) -> Outcome {
    let spec = case
        .map
        .as_ref()
        .expect("a Frobenius case carries its power");
    let map = Frobenius::<F>::new(spec.power);
    let source = source::<F>(case, 0x51ed_2701);
    let (mut dst, layout) = destination::<F>(case);
    let mut expected = dst.clone();
    let payload = &mut expected[layout.start..layout.start + layout.row_bytes];
    if in_place {
        payload.copy_from_slice(&source);
    }
    unary::<F>(payload, &source, |value| {
        frobenius_oracle::<F>(value, spec.power)
    });
    let dst_payload = &mut dst[layout.start..layout.start + layout.row_bytes];
    let entries = observe_unit(|| {
        if in_place {
            dst_payload.copy_from_slice(&source);
            crate::ops::map::frobenius_assign_with::<F>(dst_payload, &map);
        } else {
            crate::ops::map::frobenius_into_with::<F>(dst_payload, &map, &source);
        }
    });
    map_outcome::<F, F>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_frobenius_into_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    frobenius::<F>(case, false)
}

pub(super) fn map_frobenius_assign_with<F: FieldKernels>(case: &Case, _cell: &Cell) -> Outcome {
    frobenius::<F>(case, true)
}

fn embed<S, T>(case: &Case) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let source = source::<S>(case, 0x51ed_2701);
    let (mut dst, layout) = destination::<T>(case);
    let mut expected = dst.clone();
    map_unary::<T, S>(
        &mut expected[layout.start..layout.start + layout.row_bytes],
        &source,
        |value| compact.embed(value),
    );
    let entries = observe_unit(|| {
        crate::ops::map::embed_into_with::<S, T>(
            &mut dst[layout.start..layout.start + layout.row_bytes],
            &prepared,
            &source,
        );
    });
    map_outcome::<S, T>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_embed_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    embed::<S, T>(case)
}

fn trace<S, T>(case: &Case) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let source = source::<T>(case, 0x51ed_2701);
    let (mut dst, layout) = destination::<S>(case);
    let mut expected = dst.clone();
    map_unary::<S, T>(
        &mut expected[layout.start..layout.start + layout.row_bytes],
        &source,
        |value| compact.trace(value),
    );
    let entries = observe_unit(|| {
        crate::ops::map::trace_into_with::<S, T>(
            &mut dst[layout.start..layout.start + layout.row_bytes],
            &prepared,
            &source,
        );
    });
    map_outcome::<S, T>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_trace_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    trace::<S, T>(case)
}

fn norm<S, T>(case: &Case) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let source = source::<T>(case, 0x51ed_2701);
    let (mut dst, layout) = destination::<S>(case);
    let mut expected = dst.clone();
    map_unary::<S, T>(
        &mut expected[layout.start..layout.start + layout.row_bytes],
        &source,
        |value| compact.norm(value),
    );
    let entries = observe_unit(|| {
        crate::ops::map::norm_into_with::<S, T>(
            &mut dst[layout.start..layout.start + layout.row_bytes],
            &prepared,
            &source,
        );
    });
    map_outcome::<S, T>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_norm_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    norm::<S, T>(case)
}

fn restrict<S, T>(case: &Case) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let source = source::<T>(case, 0x51ed_2701);
    let (mut dst, layout) = destination::<S>(case);
    let (mut valid, valid_start) = guarded_bytes(
        crate::bits::bytes_for(case.geometry.len),
        case.geometry.misalign,
    );
    let mut expected = dst.clone();
    let bytes = crate::bits::bytes_for(case.geometry.len);
    let mut expected_valid = valid.clone();
    expected_valid[valid_start..valid_start + bytes].fill(0);
    for (index, lane) in source.chunks_exact(T::BYTES).enumerate() {
        let value = compact.restrict(T::decode(lane));
        let at = layout.start + index * S::BYTES;
        let element = value.unwrap_or(Elem::<S>::ZERO);
        S::encode(&mut expected[at..at + S::BYTES], element);
        if value.is_some() {
            expected_valid[valid_start + index / 8] |= 1 << (index % 8);
        }
    }
    let entries = observe_unit(|| {
        crate::ops::map::restrict_into_with::<S, T>(
            &mut dst[layout.start..layout.start + layout.row_bytes],
            &mut valid[valid_start..valid_start + crate::bits::bytes_for(case.geometry.len)],
            &prepared,
            &source,
        );
    });
    map_outcome::<S, T>(
        vec![
            BufferCheck {
                name: "dst",
                actual: dst,
                expected,
            },
            BufferCheck {
                name: "valid",
                actual: valid,
                expected: expected_valid,
            },
        ],
        entries,
    )
}

pub(super) fn map_restrict_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    restrict::<S, T>(case)
}

fn contains<S, T>(case: &Case) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let source = source::<T>(case, 0x51ed_2701);
    let bytes = crate::bits::bytes_for(case.geometry.len);
    let (mut valid, start) = guarded_bytes(bytes, case.geometry.misalign);
    let mut expected = valid.clone();
    expected[start..start + bytes].fill(0);
    for (index, lane) in source.chunks_exact(T::BYTES).enumerate() {
        if compact.contains(T::decode(lane)) {
            expected[start + index / 8] |= 1 << (index % 8);
        }
    }
    let entries = observe_unit(|| {
        crate::ops::map::contains_into_with::<S, T>(
            &mut valid[start..start + bytes],
            &prepared,
            &source,
        );
    });
    map_outcome::<S, T>(
        vec![BufferCheck {
            name: "valid",
            actual: valid,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_contains_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    contains::<S, T>(case)
}

fn apply<S, T>(case: &Case, forward: bool) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match isomorphism::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared isomorphism: {error:?}")),
    };
    if forward {
        let source = source::<S>(case, 0x51ed_2701);
        let (mut dst, layout) = destination::<T>(case);
        let mut expected = dst.clone();
        map_unary::<T, S>(
            &mut expected[layout.start..layout.start + layout.row_bytes],
            &source,
            |value| compact.apply(value),
        );
        let entries = observe_unit(|| {
            crate::ops::map::apply_into_with::<S, T>(
                &mut dst[layout.start..layout.start + layout.row_bytes],
                &prepared,
                &source,
            );
        });
        map_outcome::<S, T>(
            vec![BufferCheck {
                name: "dst",
                actual: dst,
                expected,
            }],
            entries,
        )
    } else {
        let source = source::<T>(case, 0x51ed_2701);
        let (mut dst, layout) = destination::<S>(case);
        let mut expected = dst.clone();
        map_unary::<S, T>(
            &mut expected[layout.start..layout.start + layout.row_bytes],
            &source,
            |value| compact.unapply(value),
        );
        let entries = observe_unit(|| {
            crate::ops::map::unapply_into_with::<S, T>(
                &mut dst[layout.start..layout.start + layout.row_bytes],
                &prepared,
                &source,
            );
        });
        map_outcome::<S, T>(
            vec![BufferCheck {
                name: "dst",
                actual: dst,
                expected,
            }],
            entries,
        )
    }
}

pub(super) fn map_apply_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    apply::<S, T>(case, true)
}

pub(super) fn map_unapply_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: FieldKernels,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    apply::<S, T>(case, false)
}
