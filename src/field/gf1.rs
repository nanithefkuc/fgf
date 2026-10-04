//! GF(2), the field with two elements — the base of every binary tower.
//!
//! Addition and subtraction are XOR, multiplication is AND, negation and
//! squaring are the identity (`x² = x`), and the multiplicative group is
//! trivial: `inv(1) = 1`, `inv(0) = 0` by the crate-wide convention, so
//! `x / 1 = x` and `x / 0 = 0`.
//!
//! The field is [`Gf1`], the degree-one polynomial basis
//! [`Poly<3>`](crate::field::poly::Poly) (`x + 1`); its element is
//! [`Elem<Gf1>`](crate::field::Elem).
//!
//! This is the scalar oracle for the bit-packed kernels in [`crate::bits`].
//! GF(2) implements [`Field`] for scalar use but no
//! [`FieldBuffer`](super::FieldBuffer): GF(2) has no byte-per-element vector
//! representation worth shipping — one element is one *bit*, and the packed
//! surface over `&[u8]` buffers is [`crate::bits`], not [`crate::ops`].
//!
//! ```
//! use fgf::field::FieldElem;
//! use fgf::{Elem, gf1};
//!
//! // The whole field, exhaustively.
//! for a in [Elem::<gf1::Gf1>::from_raw(0), Elem::<gf1::Gf1>::from_raw(1)] {
//!     for b in [Elem::<gf1::Gf1>::from_raw(0), Elem::<gf1::Gf1>::from_raw(1)] {
//!         assert_eq!(a.add(b).to_raw(), a.to_raw() ^ b.to_raw());
//!         assert_eq!(a.mul(b).to_raw(), a.to_raw() & b.to_raw());
//!         assert_eq!(a.sub(b), a.add(b));
//!         assert_eq!(a.neg(), a);
//!         assert_eq!(a.square(), a);
//!     }
//! }
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Elem::<gf1::Gf1>::from_raw(1).div(Elem::<gf1::Gf1>::ZERO).to_raw() == 0);
//! ```

use super::poly::Poly;
use super::repr::Gf;
use super::{Field, HasGenerator, PrimeCharacteristic};

/// Marker for GF(2): [`Gf<1, Poly<3>>`](Gf), the polynomial `x + 1`.
///
/// A zero-sized name for the field.
pub type Gf1 = Gf<1, Poly<3>>;

/// Number of elements in the field.
pub const ORDER: u128 = 2;

impl Field for Gf1 {
    type Raw = u8;
    type Characteristic = PrimeCharacteristic<2>;
    const NAME: &'static str = "GF(2)";
    const DEGREE: u32 = 1;
    const ORDER: u128 = 2;
    const ZERO_RAW: u8 = 0;
    const ONE_RAW: u8 = 1;
    const VALID: () = ();

    #[inline]
    fn canonical_raw(raw: u8) -> u8 {
        raw & 1
    }

    #[inline]
    fn add_raw(left: u8, right: u8) -> u8 {
        (left ^ right) & 1
    }

    #[inline]
    fn sub_raw(left: u8, right: u8) -> u8 {
        (left ^ right) & 1
    }

    #[inline]
    fn neg_raw(value: u8) -> u8 {
        value & 1
    }

    #[inline]
    fn mul_raw(left: u8, right: u8) -> u8 {
        (left & right) & 1
    }

    #[inline]
    fn inv_raw(value: u8) -> u8 {
        value & 1
    }
}

impl HasGenerator for Gf1 {
    /// The trivial group's generator is one.
    const GENERATOR_RAW: u8 = 1;
}
