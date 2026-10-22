//! Adapters for prepared map rows over bit-packed GF(2) sources.

use super::*;
use crate::field::{
    BinaryField, Embedding, EmbeddingError, PreparedEmbedding, PrimeCharacteristic, PrimeField,
};

type EmbeddingPair<S, T> = (Embedding<S, T>, PreparedEmbedding<S, T>);

fn embedding<S, T>(_case: &Case) -> Result<EmbeddingPair<S, T>, EmbeddingError>
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new()?;
    let prepared = PreparedEmbedding::new(&compact);
    Ok((compact, prepared))
}

fn packed_source(case: &Case, salt: u64) -> (Vec<u8>, usize) {
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let (mut buffer, start) = guarded_bytes(bytes, case.geometry.misalign);
    let noise = super::super::noise(bytes, seed(case, salt) | 1);
    buffer[start..start + bytes].copy_from_slice(&noise);
    let tail = bits % 8;
    if tail != 0
        && let Some(last) = buffer[start..start + bytes].last_mut()
    {
        *last &= (1u8 << tail) - 1;
    }
    (buffer, start)
}

fn lanes<T: FieldBuffer>(case: &Case, salt: u64) -> Vec<u8> {
    elements::<T>(case, salt, case.geometry.len, false)
}

fn bit(buffer: &[u8], index: usize) -> bool {
    buffer[index / 8] & (1 << (index % 8)) != 0
}

fn write_bit(buffer: &mut [u8], index: usize, value: bool) {
    if value {
        buffer[index / 8] |= 1 << (index % 8);
    } else {
        buffer[index / 8] &= !(1 << (index % 8));
    }
}

fn map_bits_outcome<T: FieldBuffer>(checks: Vec<BufferCheck>, entries: Vec<Entry>) -> Outcome {
    let mut outcome = Outcome::new(backend(), backend()).with_trace(entries);
    outcome.source_bytes = 1;
    outcome.target_bytes = T::BYTES;
    outcome.checks = checks;
    outcome
}

fn embed<S, T>(case: &Case) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let bits = case.geometry.len;
    let (src, src_start) = packed_source(case, 0x51ed_2701);
    let bytes = crate::bits::bytes_for(bits);
    let (mut dst, layout) = guarded(case, T::BYTES);
    let mut expected = dst.clone();
    for index in 0..bits {
        let value =
            Elem::<S>::from_index(u128::from(bit(&src[src_start..src_start + bytes], index)))
                .expect("a bit is a GF(2) element");
        let at = layout.start + index * T::BYTES;
        T::encode(&mut expected[at..at + T::BYTES], compact.embed(value));
    }
    let entries = observe_unit(|| {
        crate::ops::map_bits::embed_into_with::<S, T>(
            &mut dst[layout.start..layout.start + layout.row_bytes],
            &prepared,
            &src[src_start..src_start + bytes],
            bits,
        );
    });
    map_bits_outcome::<T>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

pub(super) fn map_bits_embed_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    embed::<S, T>(case)
}

fn compress<S, T>(case: &Case, kind: Compress) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let src = lanes::<T>(case, 0x51ed_2701);
    let (mut dst, start) = guarded_bytes(bytes, case.geometry.misalign);
    let mut expected = dst.clone();
    for (index, lane) in src.chunks_exact(T::BYTES).enumerate() {
        let value = T::decode(lane);
        let set = match kind {
            Compress::Trace => !compact.trace(value).is_zero(),
            Compress::Norm => !compact.norm(value).is_zero(),
        };
        write_bit(&mut expected[start..start + bytes], index, set);
    }
    let entries = match kind {
        Compress::Trace => observe_unit(|| {
            crate::ops::map_bits::trace_into_with::<S, T>(
                &mut dst[start..start + bytes],
                &prepared,
                &src,
                bits,
            );
        }),
        Compress::Norm => observe_unit(|| {
            crate::ops::map_bits::norm_into_with::<S, T>(
                &mut dst[start..start + bytes],
                &prepared,
                &src,
                bits,
            );
        }),
    };
    map_bits_outcome::<T>(
        vec![BufferCheck {
            name: "dst",
            actual: dst,
            expected,
        }],
        entries,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Compress {
    Trace,
    Norm,
}

pub(super) fn map_bits_trace_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    compress::<S, T>(case, Compress::Trace)
}

pub(super) fn map_bits_norm_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    compress::<S, T>(case, Compress::Norm)
}

pub(super) fn map_bits_restrict_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let src = lanes::<T>(case, 0x51ed_2701);
    let (mut dst, start) = guarded_bytes(bytes, case.geometry.misalign);
    let (mut valid, valid_start) = guarded_bytes(bytes, case.geometry.misalign);
    let mut expected = dst.clone();
    let mut expected_valid = valid.clone();
    for (index, lane) in src.chunks_exact(T::BYTES).enumerate() {
        let preimage = compact.restrict(T::decode(lane));
        let set = preimage.is_some_and(|value| !value.is_zero());
        write_bit(&mut expected[start..start + bytes], index, set);
        write_bit(
            &mut expected_valid[valid_start..valid_start + bytes],
            index,
            preimage.is_some(),
        );
    }
    let entries = observe_unit(|| {
        crate::ops::map_bits::restrict_into_with::<S, T>(
            &mut dst[start..start + bytes],
            &mut valid[valid_start..valid_start + bytes],
            &prepared,
            &src,
            bits,
        );
    });
    map_bits_outcome::<T>(
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

pub(super) fn map_bits_contains_into_with<S, T>(case: &Case, _cell: &Cell) -> Outcome
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let (compact, prepared) = match embedding::<S, T>(case) {
        Ok(pair) => pair,
        Err(error) => return Outcome::skipped(format!("prepared embedding: {error:?}")),
    };
    let bits = case.geometry.len;
    let bytes = crate::bits::bytes_for(bits);
    let src = lanes::<T>(case, 0x51ed_2701);
    let (mut valid, start) = guarded_bytes(bytes, case.geometry.misalign);
    let mut expected = valid.clone();
    for (index, lane) in src.chunks_exact(T::BYTES).enumerate() {
        write_bit(
            &mut expected[start..start + bytes],
            index,
            compact.contains(T::decode(lane)),
        );
    }
    let entries = observe_unit(|| {
        crate::ops::map_bits::contains_into_with::<S, T>(
            &mut valid[start..start + bytes],
            &prepared,
            &src,
            bits,
        );
    });
    map_bits_outcome::<T>(
        vec![BufferCheck {
            name: "valid",
            actual: valid,
            expected,
        }],
        entries,
    )
}
