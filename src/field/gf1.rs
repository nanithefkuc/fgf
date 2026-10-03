//! GF(2), the field with two elements — the base of every binary tower.
//!
//! Addition and subtraction are XOR, multiplication is AND, negation and
//! squaring are the identity (`x² = x`), and the multiplicative group is
//! trivial: `inv(1) = 1`, `inv(0) = 0` by the crate-wide convention, so
//! `x / 1 = x` and `x / 0 = 0`.
//!
//! The field is [`Gf1`], the degree-one polynomial basis
//! [`Poly<3>`](crate::field::poly::Poly) (`x + 1`); its element is
//! [`Elem<1, Poly<3>>`](crate::field::Elem), aliased here as
//! [`gf1::Elem`](self::Elem).
//!
//! This is the scalar oracle for the bit-packed kernels in [`crate::bits`].
//! It deliberately has no [`super::Field`] implementation: GF(2) has no
//! byte-per-element vector representation worth shipping — one element is one
//! *bit*, and the packed surface over `&[u8]` buffers is [`crate::bits`], not
//! [`crate::ops`].
//!
//! ```
//! use fgf::field::FieldElem;
//! use fgf::gf1;
//!
//! // The whole field, exhaustively.
//! for a in [gf1::Elem::from_raw(0), gf1::Elem::from_raw(1)] {
//!     for b in [gf1::Elem::from_raw(0), gf1::Elem::from_raw(1)] {
//!         assert_eq!(a.add(b).to_raw(), a.to_raw() ^ b.to_raw());
//!         assert_eq!(a.mul(b).to_raw(), a.to_raw() & b.to_raw());
//!         assert_eq!(a.sub(b), a.add(b));
//!         assert_eq!(a.neg(), a);
//!         assert_eq!(a.square(), a);
//!     }
//! }
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(gf1::Elem::from_raw(1).div(gf1::Elem::ZERO).to_raw() == 0);
//! ```

use super::poly::Poly;
use super::repr::Gf;
use super::repr::Repr;

/// Marker for GF(2): [`Gf<1, Poly<3>>`](Gf), the polynomial `x + 1`.
///
/// A zero-sized name for the field. Unlike the byte and prime fields it
/// carries no [`super::Field`] implementation: GF(2) is represented one
/// element per bit, and that packed surface lives in [`crate::bits`].
pub type Gf1 = Gf<1, Poly<3>>;

/// An element of GF(2), stored as a byte holding `0` or `1`.
///
/// The stored byte is canonical: it is always exactly `0` or `1`, never a
/// wider pattern with a meaningful low bit. Every constructor masks on the
/// way in and every operation preserves the invariant, so one field value
/// has exactly one storage form and the [`PartialEq`]/[`Hash`]/[`Ord`]
/// compare field values.
pub type Elem = super::repr::Elem<1, Poly<3>>;

/// Number of elements in the field.
pub const ORDER: u128 = 2;

/// A generator of the multiplicative group.
///
/// The group is trivial — `{1}` — so the generator is one and has order 1.
pub const GENERATOR: Elem = Elem::ONE;

impl Gf1 {
    /// Human-readable field name.
    pub const NAME: &'static str = <Poly<3> as Repr<1>>::NAME;
}
