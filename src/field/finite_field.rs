//! The canonical generic field family `GF(P^N)`.
//!
//! [`FiniteField<P, N, R>`] names a field by its prime characteristic `P`,
//! its absolute degree `N` over GF(P), and a presentation `R`. Identity and
//! representation are type-selected: values are [`Elem<FiniteField<P, N,
//! R>>`](crate::field::Elem), and equal `P` and `N` with different
//! presentations stay distinct types. The family ships one presentation,
//! [`Residue`], which is implemented at degree one only — so
//! `FiniteField<P, 1>` is the prime field GF(P) for every `u64` prime, and
//! `FiniteField<P, N>` with `N > 1` does not compile until an extension
//! presentation names itself explicitly.
//!
//! # Capability rows
//!
//! [`Field`](crate::field::Field),
//! [`PrimeField`](crate::field::PrimeField),
//! [`FieldBuffer`](crate::field::FieldBuffer), and
//! [`FieldKernels`](crate::kernel::FieldKernels) are implemented on real
//! representation/degree rows, never as a blanket over all `P`, `N`, and
//! `R`. Metadata membership alone does not imply a degree has an arithmetic
//! body: generic code over the family supplies `R: FiniteFieldRepr<P, N>`
//! *and* the capability bound, e.g. `FiniteField<P, N, R>: Field`.
//!
//! # Validation
//!
//! The shared metadata check reuses the crate's deterministic `u64`
//! primality proof and validates, in order: the characteristic is prime, the
//! degree is positive, the binary degree-64 cap holds at characteristic two,
//! and `P^N` fits `u128` with overflow checking. Every value-producing
//! constructor, identity, and raw method on the family's rows evaluates it
//! before any remainder, exponent, or size arithmetic runs, so an invalid
//! `P` or `N` fails const evaluation in const contexts and panics at first
//! value use otherwise. Merely naming an unused type evaluates nothing.

use core::marker::PhantomData;

use super::{PrimeCharacteristic, PrimeIdentity, private};

mod residue;

pub use residue::Residue;

/// Validated `P^N` in `u128`: the shared metadata check for family rows.
///
/// Asserts the characteristic is prime, the degree is positive, the binary
/// degree cap of 64 holds at characteristic two, and the cardinality fits
/// `u128` with overflow checking, then returns the order. Central field
/// validation runs the same checks from every [`Field`] construction; this
/// is the reusable extraction the rows' own metadata items read.
#[allow(clippy::cast_lossless)]
pub(crate) const fn checked_order<const P: u64, const N: u32>() -> u128 {
    let characteristic = <PrimeCharacteristic<P> as PrimeIdentity>::CHARACTERISTIC;
    assert!(N > 0, "field degree must be positive");
    assert!(
        characteristic != 2 || N <= 64,
        "binary field degree exceeds 64"
    );
    let mut order: u128 = 1;
    let mut i = 0;
    while i < N {
        order = match order.checked_mul(characteristic as u128) {
            Some(value) => value,
            None => panic!("field order overflows u128"),
        };
        i += 1;
    }
    order
}

/// A finite field with prime characteristic `P`, absolute degree `N`, and
/// presentation `R`.
///
/// The marker carries no runtime state and imposes no traits on `R` beyond
/// the representation metadata. Arithmetic bodies live on real rows: the
/// default [`Residue`] presentation supplies the degree-one row over every
/// `u64` prime, and further presentations implement their own rows on this
/// marker.
pub struct FiniteField<const P: u64, const N: u32, R: FiniteFieldRepr<P, N> = Residue>(
    PhantomData<fn() -> R>,
);

// Manual marker impls: derives would bound the traits on `R`, which the
// representation contract does not supply. Every row is a distinct
// zero-sized identity, so equality, hashing, and ordering are constant.
impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> Clone for FiniteField<P, N, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> Copy for FiniteField<P, N, R> {}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> Default for FiniteField<P, N, R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> PartialEq for FiniteField<P, N, R> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> Eq for FiniteField<P, N, R> {}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> core::hash::Hash
    for FiniteField<P, N, R>
{
    fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {}
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> PartialOrd for FiniteField<P, N, R> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> Ord for FiniteField<P, N, R> {
    fn cmp(&self, _other: &Self) -> core::cmp::Ordering {
        core::cmp::Ordering::Equal
    }
}

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> core::fmt::Debug
    for FiniteField<P, N, R>
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "FiniteField<{}, {}, {}>", P, N, R::NAME)
    }
}

/// The presentation metadata of a canonical family row.
///
/// Sealed, like the fields and elements: representations are crate-defined,
/// and downstream code names a row with its public type rather than
/// implementing this trait. `NAME` is a human-readable presentation label,
/// distinct from the field name a [`Field`](crate::field::Field) row
/// reports. `VALID` is the use-time proof of the representation's defining
/// data; it must not read the row's field identities, and it runs before any
/// [`Field`](crate::field::Field) validation that chains it.
pub trait FiniteFieldRepr<const P: u64, const N: u32>: private::Sealed + 'static {
    /// Human-readable presentation label.
    const NAME: &'static str;
    /// Use-time validity proof for the representation's defining data.
    const VALID: ();
}
