//! Deterministic fixtures and the scalar oracles the case checks run
//! against.
//!
//! Fixtures are built from a fixed-seed LCG and packed through
//! [`fgf::ops::pack`], so every lane is canonical before the timed region —
//! including the zero pad bits the packed presentations maintain. Each case
//! body allocates its own fixtures once, outside the timed region.
//! Verification compares kernel output against independent per-lane `Elem`
//! arithmetic (the safe scalar path is the differential oracle) on small
//! fixtures.

use std::hint::black_box;

use fgf::field::FieldElem;
use fgf::{Elem, Field, FieldBuffer, ops};

/// Step the deterministic LCG and return the raw 64-bit state.
pub fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

/// Raw byte noise from a fixed seed; fixture setup only.
pub fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            let word = lcg(&mut state);
            (word >> 33) as u8
        })
        .collect()
}

/// One deterministic canonical element, including full-width indices.
pub fn element<F: Field>(seed: &mut u64) -> Elem<F> {
    let high = lcg(seed);
    let low = lcg(seed);
    // Two LCG steps preserve the low bit. Fold upper bits into it so small
    // binary fields do not receive a constant zero or unit sequence.
    let word = (u128::from(high ^ (high >> 32)) << 64) | u128::from(low ^ (low >> 32));
    let index = word % F::ORDER.max(1);
    <Elem<F> as FieldElem>::from_index(index).expect("index below ORDER")
}

/// `count` deterministic canonical elements of `F`, from presentation
/// indices so every field family is reachable through public paths.
pub fn elements<F: Field>(count: usize, seed: &mut u64) -> Vec<Elem<F>> {
    (0..count).map(|_| element::<F>(seed)).collect()
}

/// Decode packed canonical lanes back to elements; verification only.
pub fn lanes<F: FieldBuffer>(buf: &[u8]) -> Vec<Elem<F>> {
    buf.chunks_exact(F::BYTES)
        .map(<F as FieldBuffer>::decode)
        .collect()
}

/// Encode elements into packed canonical lanes; verification only.
pub fn write_lanes<F: FieldBuffer>(dst: &mut [u8], elems: &[Elem<F>]) {
    assert_eq!(
        dst.len(),
        elems.len() * F::BYTES,
        "oracle lane count differs from the buffer geometry"
    );
    for (lane, value) in dst.chunks_exact_mut(F::BYTES).zip(elems) {
        <F as FieldBuffer>::encode(lane, *value);
    }
}

/// `lanes` canonical packed bytes of `F` from a fixed seed.
pub fn packed<F: FieldBuffer>(lane_count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let elems = elements::<F>(lane_count, &mut state);
    ops::pack_to_vec::<F>(&elems)
}

/// Panic helper naming the case and the first differing lane.
pub fn assert_bytes_equal(case: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case}: output length {} differs from oracle length {}",
        actual.len(),
        expected.len()
    );
    if actual != expected {
        let lane = actual
            .iter()
            .zip(expected)
            .position(|(a, e)| a != e)
            .unwrap_or(0);
        let end = (lane + 8).min(actual.len());
        panic!(
            "{case}: lane {lane} is {:02x?}, oracle says {:02x?}",
            &actual[lane..end],
            &expected[lane..end.min(expected.len())]
        );
    }
    black_box(actual);
}

/// Keep a result observable without letting the optimizer delete the work.
pub fn observe<T>(value: T) -> T {
    black_box(value)
}
