//! Binary fields by total degree and representation.
//!
//! GF(2^N) under the flat representation `R` is the canonical family row
//! [`FiniteField<2, N, R>`](crate::FiniteField), and
//! [`Elem`](crate::Elem) over it is its element. [`BinaryRepr`] is the
//! sealed vocabulary of flat binary representations: it extends the
//! family's presentation metadata with the descriptor and, at degree
//! eight, the discrete-logarithm tables. [`BinaryField`] adds the canonical
//! coordinate conversions shared by every binary field, flat rows and
//! characteristic-two extension rows alike, and [`BinaryDegree`] pins the
//! exact degree generic code may build on. The shipped quadratic towers
//! ([`Rijndael16`] and the [`FanPaar`](FanPaar2) family) are
//! extension rows of the [`extension`](crate::extension) family.
//!
//! # Representation
//!
//! [`Polynomial`] is the construction `GF(2)[x] / p(x)` for an irreducible
//! degree-`N` `p`. The representation, its descriptor, and its generator
//! derive from the polynomial at compile time, so each polynomial is a
//! distinct representation with fully `const` scalar arithmetic.
//!
//! | Constant | Polynomial | Convention |
//! | --- | --- | --- |
//! | [`AES`] | `0x11B` | AES/Rijndael; the field `GF2P8MULB` multiplies natively |
//! | [`RS`] | `0x11D` | Intel ISA-L, `klauspost/reedsolomon`, QR codes |
//!
//! Any other degree-8 irreducible polynomial is spelled directly, such as
//! `Polynomial<0x12D>` for Data Matrix or `Polynomial<0x187>` for the CCSDS
//! Reed–Solomon field. Distinct polynomials are unrelated encodings: a byte
//! has different products under each, so a buffer carries one convention and
//! never two.
//!
//! [`Normal`] wraps a polynomial basis element `E` into its conjugate basis:
//! bit `i` selects `E^(2^i)`, and one is all `N` bits set. The descriptor pins
//! both coordinate matrices; the inverse columns come from bounded
//! enumeration of the forward map, never from a matrix solver.
//!
//! [`Cantor`] builds its basis from `SEED` through the relation
//! `c_(N-1) = SEED` and `c_i = c_(i+1)^2 + c_(i+1)`; bit `i` selects `c_i`.
//! Validity asserts that the chain reaches `c_0 == ONE` and checks basis rank
//! by bounded enumeration, never by a matrix solver.
//!
//! Smallest valid instances of the ordered bases: `E = 2` over `P = 0x7`,
//! `E = 8` over `P = 0x13`, and `E = 0x20` over [`AES`] and [`RS`]; the same
//! seeds are the smallest Cantor instances.
//!
//! # Layout
//!
//! Raw storage is private to the crate. Flat rows of degrees 1, 2, 4, and 8
//! store one element in a byte holding exactly the `N` low coordinate bits.
//! The stable byte encoding of an element is the little-endian encoding of
//! its raw word, and degree 64 is the crate's binary cap.
//!
//! A shipped quadratic tower names `FiniteField<2, N, Extension<S, 2>>` as
//! `a + b*t` over the previous level, under `t^2 + c1*t + c0 = 0`. The low
//! `N/2` coordinate bits hold `a` and the high `N/2` bits hold `b`; storage
//! bits at or above `N` are zero, and one is the base one in the low
//! component. Its scalar identities are:
//!
//! ```text
//! product:   (a*c + c0*b*d, a*d + b*c + c1*b*d)
//! square:    (a^2 + c0*b^2, c1*b^2)
//! conjugate: (a + c1*b, b)
//! norm:      a^2 + c1*a*b + c0*b^2
//! ```
//!
//! [`Rijndael16`], [`Rijndael32`], and [`Rijndael64`] root at the AES field
//! with `c1 = 1` and `c0` equal to `0x20`, `0x2000`, and `0x2000_0000`. The
//! Fan-Paar specs nest from GF(2) through the Wiedemann relation
//! `t^2 + alpha*t + 1`, where `alpha` is the previous level's indeterminate
//! (`1` over GF(2), then the raw `base_one << (h/2)` in its own packing).
//! [`FanPaar2`] and [`FanPaar4`] keep their coordinate word in a padded byte
//! and carry no byte buffer; every other instance fills its `u8`, `u16`,
//! `u32`, or `u64` word exactly and encodes it little-endian.
//!
//! Each shipped instance carries its frozen generator and a full `const`
//! surface on its concrete element type: constructors, byte and component
//! conversions, and arithmetic, plus `mul_alpha` on Fan-Paar GF(2^8),
//! GF(2^16), and GF(2^32).
//!
//! # Totality and canonicalization
//!
//! By crate-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
//! profile and under `const` evaluation alike.
//!
//! # Examples
//!
//! The same bytes multiply differently under the two degree-eight
//! conventions:
//!
//! ```
//! use fgf::{Elem, FiniteField, binary::{AES, Polynomial, RS}};
//!
//! let (a, b) = (0x53, 0xca);
//! assert_eq!(
//!     Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(a)
//!         .mul(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(b))
//!         .to_raw(),
//!     0x01
//! );
//! assert_eq!(
//!     Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(a)
//!         .mul(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(b))
//!         .to_raw(),
//!     0x8f
//! );
//!
//! // Any irreducible polynomial is a field; its generator is derived.
//! assert_eq!(
//!     Elem::<FiniteField<2, 8, Polynomial<0x12D>>>::GENERATOR.pow(255),
//!     Elem::<FiniteField<2, 8, Polynomial<0x12D>>>::ONE
//! );
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::FieldElem;
//!
//! // x^8 + 1 = (x + 1)^8 is reducible, so this field does not exist.
//! let _ = fgf::Elem::<fgf::FiniteField<2, 8, fgf::binary::Polynomial<0x101>>>::from_raw(1);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::FieldElem;
//!
//! // The degree-eight polynomial does not match a degree-four field.
//! let _ = fgf::Elem::<fgf::FiniteField<2, 4, fgf::binary::Polynomial<0x11B>>>::from_raw(1);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::FieldElem;
//!
//! // E = 1 has all conjugates equal to one, so the basis collapses.
//! let _ = fgf::Elem::<fgf::FiniteField<2, 8, fgf::binary::Normal<0x11B, 1>>>::from_raw(1);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::FieldElem;
//!
//! // SEED = 0 builds the zero chain, which never reaches one.
//! let _ = fgf::Elem::<fgf::FiniteField<2, 8, fgf::binary::Cantor<0x11B, 0>>>::from_raw(1);
//! ```
//!
//! A tower element round-trips through its base components:
//!
//! ```
//! use fgf::{Elem, FiniteField, binary::Rijndael16, extension::Extension};
//!
//! type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
//!
//! let x = Elem::<Gf16>::from_raw(0x1234);
//! assert_eq!(x.mul(Elem::<Gf16>::ONE), x);
//! assert_eq!(Elem::<Gf16>::GENERATOR.pow(65_535), Elem::<Gf16>::ONE);
//!
//! // Components are the base-field coefficients of `1` and `t`.
//! let [low, high] = x.to_components();
//! assert_eq!((low.to_raw(), high.to_raw()), (0x34, 0x12));
//! assert_eq!(Elem::<Gf16>::from_components([low, high]), x);
//! ```

pub use crate::field::binary::tower::{
    FanPaar2, FanPaar4, FanPaar8, FanPaar16, FanPaar32, FanPaar64, Rijndael16, Rijndael32,
    Rijndael64,
};
pub use crate::field::binary::{
    AES, BinaryDegree, BinaryField, BinaryRepr, Cantor, Normal, Polynomial, RS,
};
