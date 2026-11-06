//! The canonical generic field family `GF(P^N)`.
//!
//! `FiniteField<P, N, R>` names a field by its prime characteristic `P`, its
//! absolute degree `N` over GF(P), and a presentation `R`; identity and
//! representation are type-selected, so equal `P` and `N` with different
//! presentations stay distinct types. The default presentation `Residue` is
//! implemented at degree one only, so `FiniteField<P, N>` with `N > 1` needs
//! an explicit presentation. Capability traits are implemented on real
//! representation/degree rows, never as a blanket over all parameters.

use core::marker::PhantomData;

use super::{PrimeCharacteristic, PrimeIdentity, private};

mod residue;

pub use residue::Residue;

/// Validated `characteristic^degree` in `u128`: the shared metadata check.
///
/// Asserts the degree is positive, the total degree cap of 64 holds at
/// every characteristic, and the cardinality fits `u128` with overflow
/// checking, then returns the order. The caller supplies a characteristic
/// already read through the primality proof. Central field validation and
/// the family rows' metadata items both run this one body.
#[allow(clippy::cast_lossless)]
pub(crate) const fn metadata_order(characteristic: u64, degree: u32) -> u128 {
    assert!(degree > 0, "field degree must be positive");
    assert!(
        characteristic != 2 || degree <= 64,
        "binary field degree exceeds 64"
    );
    assert!(degree <= 64, "field degree exceeds 64");
    let mut order: u128 = 1;
    let mut i = 0;
    while i < degree {
        order = match order.checked_mul(characteristic as u128) {
            Some(value) => value,
            None => panic!("field order overflows u128"),
        };
        i += 1;
    }
    order
}

/// Validated `P^N` for a family row: the primality proof of `P`, then
/// [`metadata_order`].
pub(crate) const fn checked_order<const P: u64, const N: u32>() -> u128 {
    metadata_order(<PrimeCharacteristic<P> as PrimeIdentity>::CHARACTERISTIC, N)
}

/// A finite field with prime characteristic `P`, absolute degree `N`, and
/// presentation `R`.
///
/// Identity and representation are type-selected: values are
/// [`Elem<FiniteField<P, N, R>>`](crate::Elem), and equal `P` and `N` with
/// different presentations stay distinct types. The default presentation,
/// [`Residue`](crate::prime::Residue), is implemented at degree one only —
/// so `FiniteField<P, 1>` is the prime field GF(P) for every `u64` prime, and
/// `FiniteField<P, N>` with `N > 1` does not compile without an explicit
/// presentation. The other presentations are the named prime folds
/// ([`MersenneFold`](crate::prime::MersenneFold),
/// [`GoldilocksFold`](crate::prime::GoldilocksFold)), the flat binary
/// presentations at characteristic two
/// ([`Polynomial`](crate::binary::Polynomial),
/// [`Normal`](crate::binary::Normal), [`Cantor`](crate::binary::Cantor)),
/// and [`Extension`](crate::extension::Extension) over any of these rows.
///
/// The marker carries no runtime state and imposes no traits on `R` beyond
/// the representation metadata. Arithmetic bodies live on real rows: metadata
/// membership alone supplies no arithmetic body, and generic code bounds on
/// the capability trait it needs.
///
/// Every value-producing constructor, identity, and raw method validates the
/// metadata in order: the characteristic is prime, the degree is positive
/// and at most 64, and `P^N` fits `u128` with overflow checking. The field's
/// `ORDER` equals that checked power. The checks are associated constants,
/// so an invalid `P` or `N` is a compile error in any build that generates
/// code using one of those operations, whichever branch the call would
/// take at run time. Merely naming the type, or type-checking code that is
/// never generated, evaluates nothing.
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
/// Sealed, like the elements: representations are crate-defined,
/// and downstream code names a row with its public type rather than
/// implementing this trait. `NAME` is a human-readable presentation label,
/// distinct from the field name a [`Field`](crate::Field) row
/// reports. The representation's defining-data proof stays crate-private; it
/// must not read the row's field identities, and it runs before any
/// [`Field`](crate::Field) validation that chains it.
#[allow(private_bounds)]
pub trait FiniteFieldRepr<const P: u64, const N: u32>:
    private::Sealed + private::FiniteFieldReprValidation<P, N> + 'static
{
    /// Human-readable presentation label.
    const NAME: &'static str;
}
