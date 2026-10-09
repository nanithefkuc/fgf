//! Binary fields by total degree and representation.
//!
//! GF(2^N) under the flat representation `R` is the canonical family row
//! [`FiniteField<2, N, R>`](crate::field::FiniteField), and [`Elem`] over it
//! is its element. [`BinaryRepr`] is the sealed vocabulary of flat binary
//! representations: it extends the family's presentation metadata with the
//! descriptor and, at degree eight, the discrete-logarithm tables.
//! [`BinaryField`] adds the canonical coordinate conversions shared by every
//! binary field, flat rows and characteristic-two extension rows alike, and
//! [`BinaryDegree`] pins the exact degree generic code may build on. The
//! shipped quadratic towers are extension rows; see [`tower`].
//!
//! # Layout
//!
//! Raw storage is private to the crate. Flat rows of degrees 1, 2, 4, and 8
//! store one element in a byte holding exactly the `N` low coordinate bits.
//! The stable byte encoding of an element is the little-endian encoding of
//! its raw word, and degree 64 is the crate's binary cap.
//!
//! # Totality and canonicalization
//!
//! By crate-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
//! profile and under `const` evaluation alike.

pub mod cantor;
pub mod description;
pub mod normal;
pub mod poly;
pub mod tower;

pub use cantor::Cantor;
pub use description::{BinaryDescription, ByteLogExp};
pub use normal::Normal;
pub use poly::{AES, Polynomial, RS};

use core::fmt;

use super::finite_field::{FiniteField, FiniteFieldRepr, checked_order};
use super::{Elem, Field, FieldBuffer, PrimeCharacteristic, RangeError, Validate};

pub(crate) mod binary_private {
    pub trait Sealed {}
}

/// A sealed binary representation of GF(2^N).
///
/// The presentation label and the defining-data proof are the family's
/// [`FiniteFieldRepr::NAME`] and [`FiniteFieldRepr::VALID`]; this trait adds
/// the binary descriptor and the optional degree-eight log/exp tables.
pub trait BinaryRepr<const N: u32>: FiniteFieldRepr<2, N> {
    /// The opaque descriptor of this presentation.
    const DESCRIPTION: &'static BinaryDescription;
    /// Discrete-logarithm tables, present only for degree-eight strategies.
    const LOG_EXP: Option<&'static ByteLogExp>;
}

/// A binary field: characteristic two with a canonical coordinate word.
///
/// The descriptor names the presentation; element indices are the coordinate
/// words. Conversions between elements and their presentation indices live
/// on [`FieldElem`](super::FieldElem).
#[allow(private_bounds)]
pub trait BinaryField:
    Field<Characteristic = PrimeCharacteristic<2>> + binary_private::Sealed
{
    /// The opaque descriptor of this field's presentation.
    const DESCRIPTION: &'static BinaryDescription;
}

/// Implemented by `FiniteField<2, N, R>` at its exact supported degree.
pub trait BinaryDegree<const N: u32>: BinaryField {}

/// Emit the shared surface of one binary degree row.
///
/// `Field`, `BinaryField`, and `BinaryDegree` over every representation,
/// with the concrete raw word and coordinate mask of the row.
macro_rules! degree_row {
    ($n:literal, $raw:ty, $mask:expr) => {
        impl<R: BinaryRepr<$n>> $crate::field::binary::binary_private::Sealed
            for FiniteField<2, $n, R>
        {
        }
        impl<R: BinaryRepr<$n>> Field for FiniteField<2, $n, R> {
            type Raw = $raw;
            type Characteristic = PrimeCharacteristic<2>;
            const NAME: &'static str = <R as FiniteFieldRepr<2, $n>>::NAME;
            const DEGREE: u32 = $n;
            const ORDER: u128 = checked_order::<2, $n>();
            // The identity words evaluate the row's validity first and then
            // read row-local data, never the reverse.
            const ZERO_RAW: $raw = {
                let () = <Self as Field>::VALID;
                0
            };
            // The descriptor's identity holds at most `N` bits here.
            #[allow(clippy::cast_possible_truncation)]
            const ONE_RAW: $raw = {
                let () = <Self as Field>::VALID;
                R::DESCRIPTION.one() as $raw
            };
            // Metadata first, then the representation's defining data.
            const VALID: () = {
                checked_order::<2, $n>();
                let () = <R as FiniteFieldRepr<2, $n>>::VALID;
            };

            #[inline]
            fn canonical_raw(raw: $raw) -> $raw {
                let () = Validate::<Self>::OK;
                raw & $mask
            }

            #[inline]
            fn add_raw(left: $raw, right: $raw) -> $raw {
                let () = Validate::<Self>::OK;
                left ^ right
            }

            #[inline]
            fn sub_raw(left: $raw, right: $raw) -> $raw {
                let () = Validate::<Self>::OK;
                left ^ right
            }

            #[inline]
            fn neg_raw(value: $raw) -> $raw {
                let () = Validate::<Self>::OK;
                value
            }

            #[inline]
            fn mul_raw(left: $raw, right: $raw) -> $raw {
                Elem::<Self>::from_raw(left)
                    .mul(Elem::<Self>::from_raw(right))
                    .to_raw()
            }

            #[inline]
            fn inv_raw(value: $raw) -> $raw {
                Elem::<Self>::from_raw(value).inv().to_raw()
            }
        }

        impl<R: BinaryRepr<$n>> $crate::field::private::RowAccess for FiniteField<2, $n, R> {
            const DESCRIPTOR: &'static $crate::field::extension::Descriptor =
                R::DESCRIPTION.as_descriptor();
            const BINARY_DESCRIPTION: Option<
                &'static $crate::field::binary::description::BinaryDescription,
            > = Some(R::DESCRIPTION);
            // The row hooks leave the trait's `where Self: Field` clause
            // implicit: every row implements `Field`, and omitting the
            // clause keeps `<Self as Field>::Raw` normalizable in the
            // bodies.
            fn prime_coordinates_into(dst: &mut [u64], raw: <Self as Field>::Raw) {
                let () = Validate::<Self>::OK;
                assert_eq!(
                    dst.len(),
                    $n,
                    "prime coordinate count differs from the field degree"
                );
                // Widening the canonical word to the coordinate word is
                // exact.
                #[allow(clippy::cast_lossless)]
                let word = <Self as Field>::canonical_raw(raw) as u64;
                let mut i: u32 = 0;
                while i < $n {
                    dst[i as usize] = (word >> i) & 1;
                    i += 1;
                }
            }

            fn from_prime_coordinates(src: &[u64]) -> <Self as Field>::Raw {
                let () = Validate::<Self>::OK;
                assert_eq!(
                    src.len(),
                    $n,
                    "prime coordinate count differs from the field degree"
                );
                let mut word = 0u64;
                let mut i: u32 = 0;
                while i < $n {
                    let coordinate = src[i as usize];
                    assert!(
                        coordinate < 2,
                        "prime coordinate is not below the characteristic"
                    );
                    word |= coordinate << i;
                    i += 1;
                }
                // The occupied bits stop at the degree, which the raw word
                // holds exactly.
                #[allow(clippy::cast_possible_truncation)]
                let raw = word as $raw;
                raw
            }

            fn from_prime(value: u64) -> <Self as Field>::Raw {
                let () = Validate::<Self>::OK;
                // An odd value reduces to the identity, an even one to
                // zero.
                if value % 2 == 0 {
                    <Self as Field>::ZERO_RAW
                } else {
                    <Self as Field>::ONE_RAW
                }
            }

            fn to_prime(raw: <Self as Field>::Raw) -> Option<u64> {
                let () = Validate::<Self>::OK;
                let raw = <Self as Field>::canonical_raw(raw);
                // The prime subfield holds zero and the identity, and an
                // ordered-basis row's identity word need not be one.
                if raw == <Self as Field>::ZERO_RAW {
                    Some(0)
                } else if raw == <Self as Field>::ONE_RAW {
                    Some(1)
                } else {
                    None
                }
            }
        }
        impl<R: BinaryRepr<$n>> BinaryField for FiniteField<2, $n, R> {
            const DESCRIPTION: &'static BinaryDescription = R::DESCRIPTION;
        }
        impl<R: BinaryRepr<$n>> BinaryDegree<$n> for FiniteField<2, $n, R> {}
    };
}

/// Emit the inherent `const` scalar arithmetic of one small degree row.
///
/// Degree eight multiplies through the representation's log/exp tables;
/// degrees 1, 2, and 4 interpret the descriptor directly.
macro_rules! small_row {
    ($n:literal, $mask:expr) => {
        impl<R: BinaryRepr<$n>> Elem<FiniteField<2, $n, R>> {
            /// Wrap a raw storage word, keeping the low `N` coordinate bits.
            ///
            /// Referencing this constructor evaluates the representation's
            /// validity check, so invalid defining data fails to compile at
            /// the use site.
            #[allow(clippy::let_unit_value)]
            #[allow(clippy::ignored_unit_patterns)]
            #[inline]
            #[must_use]
            pub const fn from_raw(value: u8) -> Self {
                let () = Validate::<FiniteField<2, $n, R>>::OK;
                let () = R::VALID;
                Elem { raw: value & $mask }
            }

            /// Decode from the stable single-byte representation.
            #[inline]
            #[must_use]
            pub const fn from_bytes(bytes: [u8; 1]) -> Self {
                Self::from_raw(bytes[0])
            }

            /// Encode to the stable single-byte representation.
            #[inline]
            #[must_use]
            pub const fn to_bytes(self) -> [u8; 1] {
                [self.raw]
            }

            /// Field addition. Identical to subtraction and to bitwise XOR.
            #[inline]
            #[must_use]
            pub const fn add(self, rhs: Self) -> Self {
                Elem {
                    raw: self.raw ^ rhs.raw,
                }
            }

            /// Field subtraction. Identical to [`Elem::add`](Self::add).
            #[inline]
            #[must_use]
            pub const fn sub(self, rhs: Self) -> Self {
                self.add(rhs)
            }

            /// Additive inverse. The identity: `x + x = 0`.
            #[inline]
            #[must_use]
            pub const fn neg(self) -> Self {
                self
            }

            /// Field multiplication.
            #[inline]
            #[must_use]
            pub const fn mul(self, rhs: Self) -> Self {
                match R::LOG_EXP {
                    Some(tables) => Elem {
                        raw: tables.mul(self.raw, rhs.raw),
                    },
                    None => {
                        // Descriptor coordinates hold at most eight bits.
                        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
                        let product = R::DESCRIPTION.mul(self.raw as u64, rhs.raw as u64) as u8;
                        Elem { raw: product }
                    }
                }
            }

            /// Square.
            #[inline]
            #[must_use]
            pub const fn square(self) -> Self {
                self.mul(self)
            }

            /// Multiplicative inverse. Maps zero to zero by crate convention.
            #[inline]
            #[must_use]
            pub const fn inv(self) -> Self {
                match R::LOG_EXP {
                    Some(tables) => Elem {
                        raw: tables.inv(self.raw),
                    },
                    None => {
                        // Descriptor coordinates hold at most eight bits.
                        #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
                        let inverse = R::DESCRIPTION.inv(self.raw as u64) as u8;
                        Elem { raw: inverse }
                    }
                }
            }

            /// Field division. Returns zero when either operand is zero.
            ///
            /// `x / 0 == 0` is a definition, not an oversight: keeping
            /// division total leaves hot loops branch-free and keeps this
            /// callable from `const` context.
            #[inline]
            #[must_use]
            pub const fn div(self, rhs: Self) -> Self {
                if self.raw == 0 || rhs.raw == 0 {
                    return Self::ZERO;
                }
                self.mul(rhs.inv())
            }

            /// Raise to an unsigned integer power.
            #[must_use]
            pub const fn pow(self, mut exponent: u128) -> Self {
                let mut base = self;
                let mut result = Self::ONE;
                while exponent != 0 {
                    if exponent & 1 != 0 {
                        result = result.mul(base);
                    }
                    base = base.square();
                    exponent >>= 1;
                }
                result
            }

            /// The canonical representative of this element.
            ///
            /// Stored bytes are canonical by construction, so this is the
            /// identity. It exists because every field in the crate answers
            /// the question.
            #[inline]
            #[must_use]
            pub const fn canonical(self) -> Self {
                self
            }

            /// Whether this element is the additive identity.
            #[inline]
            #[must_use]
            pub const fn is_zero(self) -> bool {
                self.raw == 0
            }

            /// Whether this element is the multiplicative identity.
            #[inline]
            #[must_use]
            pub const fn is_one(self) -> bool {
                self.raw == Self::ONE.raw
            }
        }
    };
}

degree_row!(1, u8, 0x1);
degree_row!(2, u8, 0x3);
degree_row!(4, u8, 0xF);
degree_row!(8, u8, 0xFF);

small_row!(1, 0x1);
small_row!(2, 0x3);
small_row!(4, 0xF);
small_row!(8, 0xFF);

impl<R: BinaryRepr<1>> Elem<FiniteField<2, 1, R>> {
    /// The element with ordinary residue `value` modulo two: zero or one.
    ///
    /// # Errors
    /// Returns [`RangeError`] when `value` is two or more.
    #[inline]
    pub const fn from_residue(value: u64) -> Result<Self, RangeError> {
        match super::check_residue(value, 2) {
            Ok(0) => Ok(Self::ZERO),
            Ok(_) => Ok(Self::ONE),
            Err(error) => Err(error),
        }
    }

    /// The ordinary residue naming this element. Always `Some` at degree
    /// one.
    #[inline]
    #[must_use]
    pub const fn to_residue(self) -> Option<u64> {
        Some(if self.raw == Self::ZERO.raw { 0 } else { 1 })
    }
}

impl<R: BinaryRepr<8>> FieldBuffer for FiniteField<2, 8, R> {
    const BYTES: usize = 1;
    const STORAGE_BITS: u32 = 8;

    #[inline]
    fn decode(bytes: &[u8]) -> Elem<Self> {
        let bytes: [u8; 1] = bytes
            .try_into()
            .expect("GF(2^8) element has the wrong byte width");
        Elem::<Self>::from_raw(bytes[0])
    }

    #[inline]
    fn encode(bytes: &mut [u8], value: Elem<Self>) {
        assert_eq!(bytes.len(), 1, "GF(2^8) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl<R: BinaryRepr<8>> fmt::Display for Elem<FiniteField<2, 8, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:02x}", self.raw)
    }
}

impl<R: BinaryRepr<1>> fmt::Display for Elem<FiniteField<2, 1, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.raw)
    }
}

impl<R: BinaryRepr<2>> fmt::Display for Elem<FiniteField<2, 2, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:01x}", self.raw)
    }
}

impl<R: BinaryRepr<4>> fmt::Display for Elem<FiniteField<2, 4, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:01x}", self.raw)
    }
}
