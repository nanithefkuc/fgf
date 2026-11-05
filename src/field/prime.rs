//! Prime-field membership and integer reduction.
//!
//! [`PrimeField`] is the sealed degree-one capability: it marks the fields
//! that are GF(P) itself, across every presentation that supplies such a
//! row. It carries no methods; the shared [`Field`] surface is the
//! arithmetic, and the mathematical residue conversions live on each
//! element family. [`reduce`] and [`reduce_wide`] are the validated
//! `const` integer reductions, independent of any element or presentation
//! type.

use super::binary::BinaryRepr;
use super::finite_field::{FiniteField, Residue};
use super::goldilocks::Goldilocks;
use super::mersenne31::Mersenne31;
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

impl private::PrimeFieldSeal for Mersenne31 {}

impl PrimeField for Mersenne31 {}

impl private::PrimeFieldSeal for Goldilocks {}

impl PrimeField for Goldilocks {}

impl<R: BinaryRepr<1>> private::PrimeFieldSeal for FiniteField<2, 1, R> {}

impl<R: BinaryRepr<1>> PrimeField for FiniteField<2, 1, R> {}

/// Reduce a `u64` to its ordinary residue modulo the prime `P`.
///
/// `value % P`, guarded by the deterministic primality proof: an invalid
/// modulus fails const evaluation in const contexts and panics on first
/// call otherwise.
///
/// # Panics
/// Panics when `P` is not prime.
// Re-exported through `internals`; production arithmetic reduces inline.
#[allow(dead_code)]
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
#[allow(dead_code)]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub const fn reduce_wide<const P: u64>(value: u128) -> u64 {
    assert!(is_prime(P), "characteristic is not prime");
    (value % P as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::{reduce, reduce_wide};

    /// The validated integer reductions agree with the plain remainder over
    /// the sampled characteristics.
    #[test]
    fn reductions_match_remainder() {
        const WIDE: u64 = 0xFFFF_FFFF_FFFF_FFC5;
        const WIDE2: u64 = 0xFFFF_FFFF_FFFF_FFAD;
        const WIDE3: u64 = 0x7FFF_FFFF_FFFF_FFE7;

        fn check<const P: u64>() {
            for value in [0, 1, 2, 7, P - 1, P, P + 1, 1 << 63, u64::MAX, u64::MAX - 1] {
                assert_eq!(reduce::<P>(value), value % P, "reduce {value} at {P}");
            }
            for wide in [
                0,
                1,
                7,
                u128::from(P) * u128::from(P),
                1 << 64,
                (1 << 64) + 5,
                1 << 127,
                u128::MAX,
                u128::MAX - 1,
            ] {
                assert_eq!(
                    u128::from(reduce_wide::<P>(wide)),
                    wide % u128::from(P),
                    "reduce_wide {wide} at {P}"
                );
            }
        }

        check::<2>();
        check::<3>();
        check::<251>();
        check::<0x7FFF_FFFF>();
        check::<0xFFFF_FFFF_0000_0001>();
        check::<WIDE>();
        check::<WIDE2>();
        check::<WIDE3>();
    }
}
