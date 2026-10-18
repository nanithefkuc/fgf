//! Prepared Frobenius steady-state allocation: construction outside the
//! counted window, application inside it.
//!
//! `Frobenius::new` owns its column entries by contract; `frobenius_with`
//! runs the compiled arm over stack coordinates only, so repeated
//! application allocates nothing.

#![cfg(feature = "std")]

use fgf::{Elem, Extension, FieldBuffer, FieldElem, FiniteField, Frobenius, QuadMersenne31};

#[path = "common/zero_alloc.rs"]
mod common;
use common::{TEST_LOCK, count_allocations, noise};

type Gf16 = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;

/// Steady-state application over one row: preparation outside the counted
/// window, only application inside it.
fn assert_frobenius_steady_state_zero_alloc<F: FieldBuffer>(powers: &[u32])
where
    Elem<F>: FieldElem,
{
    let maps: Vec<Frobenius<F>> = powers.iter().map(|&power| Frobenius::new(power)).collect();
    let seed = noise(64, 0xF2);
    let values: Vec<Elem<F>> = seed
        .iter()
        .enumerate()
        .map(|(i, &byte)| {
            Elem::<F>::from_index((u128::from(byte) << 32 ^ i as u128) % F::ORDER)
                .expect("sample index below the field order")
        })
        .collect();
    // Warm every arm before counting.
    for map in &maps {
        for &value in &values {
            core::hint::black_box(value.frobenius_with(map));
        }
    }
    let allocations = count_allocations(|| {
        for map in &maps {
            for &value in &values {
                core::hint::black_box(value.frobenius_with(map));
            }
        }
    });
    assert_eq!(allocations, 0, "prepared Frobenius application allocated");
}

#[test]
fn binary_prepared_frobenius_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_frobenius_steady_state_zero_alloc::<Gf16>(&[0, 1, 8, 15, 16, u32::MAX]);
}

#[test]
fn odd_prepared_frobenius_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_frobenius_steady_state_zero_alloc::<QuadMersenne31>(&[0, 1, 2, u32::MAX]);
}
