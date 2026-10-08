//! Prime-field membership and integer reduction.
//!
//! [`PrimeField`] is the sealed degree-one capability: it marks the fields
//! that are GF(P) itself, across every presentation that supplies such a
//! row. It carries no methods; the shared [`Field`] surface is the
//! arithmetic, and the mathematical residue conversions live on each
//! element family. [`reduce`] and [`reduce_wide`] are the validated
//! `const` integer reductions, independent of any element or presentation
//! type.

use super::finite_field::{FiniteField, Residue};
use super::{Field, is_prime, private};

/// A field of degree one: GF(P) itself.
///
/// Sealed marker over the degree-one rows, with no methods of its own.
/// Extensions and other higher-degree rows are excluded by construction,
/// not by a runtime check. Generic code bounds on this trait receive the
/// full [`Field`] surface plus the guarantee that every element is a
/// residue of the characteristic.
pub trait PrimeField: Field + private::PrimeFieldSeal {}

impl<const P: u64> private::PrimeFieldSeal for FiniteField<P, 1, Residue> {}

impl<const P: u64> PrimeField for FiniteField<P, 1, Residue> {}

/// Reduce a `u64` to its ordinary residue modulo the prime `P`.
///
/// `value % P`, guarded by the deterministic primality proof: an invalid
/// modulus fails const evaluation in const contexts and panics on first
/// call otherwise.
///
/// # Panics
/// Panics when `P` is not prime.
#[must_use]
pub const fn reduce<const P: u64>(value: u64) -> u64 {
    assert!(is_prime(P), "characteristic is not prime");
    value % P
}

/// Reduce a `u128` to its ordinary residue modulo the prime `P`.
///
/// `value % P` over the widened modulus, guarded by the same primality
/// proof as [`reduce`]. The residue is below every valid `P`, so it fits
/// `u64`.
///
/// # Panics
/// Panics when `P` is not prime.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub const fn reduce_wide<const P: u64>(value: u128) -> u64 {
    assert!(is_prime(P), "characteristic is not prime");
    (value % P as u128) as u64
}
