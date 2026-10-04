//! Binary fields by total degree and representation.
//!
//! [`Gf`] names GF(2^N) under the representation `R`, and [`Elem`]
//! over it is its element. [`BinaryRepr`] is the sealed vocabulary of
//! representations: each fixes its descriptor and, at degree eight, its
//! discrete-logarithm tables. [`BinaryField`] adds the canonical coordinate
//! conversions shared by every binary presentation, and [`BinaryDegree`]
//! pins the exact degree a generic tower row may build on.
//!
//! # Layout
//!
//! Raw storage is private to the crate. Degrees 1, 2, 4, and 8 store one
//! element in a byte holding exactly the `N` low coordinate bits; degree 16
//! stores one element in two bytes. The stable byte encoding of an element
//! is the little-endian encoding of its raw word.
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
pub use poly::{AES, Poly, REED_SOLOMON};

use core::fmt;

use super::{Elem, Field, FieldBuffer, PrimeCharacteristic, Validate};

pub(crate) mod private {
    pub trait Sealed {}
}

pub(crate) mod binary_private {
    pub trait Sealed {}
}

/// A sealed representation of GF(2^N).
#[allow(private_bounds)]
pub trait BinaryRepr<const N: u8>: private::Sealed + 'static {
    /// Human-readable presentation name.
    const NAME: &'static str;
    /// The opaque descriptor of this presentation.
    const DESCRIPTION: &'static BinaryDescription;
    /// Discrete-logarithm tables, present only for degree-eight strategies.
    const LOG_EXP: Option<&'static ByteLogExp>;
    /// Use-time validity check for the representation's defining data.
    const VALID: ();
}

/// Marker for GF(2^N) under the representation `R`. Zero-sized.
pub struct Gf<const N: u8, R: BinaryRepr<N>>(core::marker::PhantomData<fn() -> R>);

impl<const N: u8, R: BinaryRepr<N>> Clone for Gf<N, R> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: u8, R: BinaryRepr<N>> Copy for Gf<N, R> {}

impl<const N: u8, R: BinaryRepr<N>> fmt::Debug for Gf<N, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(R::NAME)
    }
}

impl<const N: u8, R: BinaryRepr<N>> Default for Gf<N, R> {
    #[inline]
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<const N: u8, R: BinaryRepr<N>> PartialEq for Gf<N, R> {
    #[inline]
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<const N: u8, R: BinaryRepr<N>> Eq for Gf<N, R> {}

impl<const N: u8, R: BinaryRepr<N>> core::hash::Hash for Gf<N, R> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {}
}

impl<const N: u8, R: BinaryRepr<N>> PartialOrd for Gf<N, R> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: u8, R: BinaryRepr<N>> Ord for Gf<N, R> {
    #[inline]
    fn cmp(&self, _other: &Self) -> core::cmp::Ordering {
        core::cmp::Ordering::Equal
    }
}

/// The GF(2^8) field under any degree-eight representation `R`.
pub type Gf8<R> = Gf<8, R>;

/// Marker for GF(2): [`Gf<1, Poly<3>>`](Gf), the polynomial `x + 1`.
pub type Gf1 = Gf<1, Poly<3>>;

/// Rejected coordinate words carry this error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordinateError;

impl fmt::Display for CoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("binary coordinate has excess bits")
    }
}

impl core::error::Error for CoordinateError {}

/// A binary field: characteristic two with canonical coordinate conversions.
///
/// Coordinates are the canonical raw bits; excess bits above the degree are
/// an error rather than a mask.
#[allow(private_bounds)]
pub trait BinaryField:
    Field<Characteristic = PrimeCharacteristic<2>> + binary_private::Sealed
{
    /// The opaque descriptor of this field's presentation.
    const DESCRIPTION: &'static BinaryDescription;
    /// Unwrap to the canonical coordinate word.
    fn to_coordinates(value: Elem<Self>) -> u64;
    /// Wrap a coordinate word, rejecting excess bits above the degree.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateError`] when `value` holds bits at or above the
    /// field's degree.
    fn from_coordinates(value: u64) -> Result<Elem<Self>, CoordinateError>;
}

/// Implemented by `Gf<N, R>` at its exact supported degree.
pub trait BinaryDegree<const N: u8>: BinaryField {}

/// Whether a `u64` coordinate word fits in `bits` bits, without ever
/// shifting a `u64` by 64.
const fn coordinates_fit(value: u64, bits: u32) -> bool {
    if bits >= 64 { true } else { value >> bits == 0 }
}

/// Emit the shared surface of one binary degree row.
///
/// `Field`, `BinaryField`, and `BinaryDegree` over every representation,
/// with the concrete raw word and coordinate mask of the row.
macro_rules! degree_row {
    ($n:literal, $raw:ty, $mask:expr) => {
        impl<R: BinaryRepr<$n>> $crate::field::binary::binary_private::Sealed for Gf<$n, R> {}
        impl<R: BinaryRepr<$n>> Field for Gf<$n, R> {
            type Raw = $raw;
            type Characteristic = PrimeCharacteristic<2>;
            const NAME: &'static str = R::NAME;
            const DEGREE: u32 = $n;
            const ORDER: u128 = 1u128 << $n;
            const ZERO_RAW: $raw = 0;
            // The descriptor's identity holds at most `N` bits here.
            #[allow(clippy::cast_possible_truncation)]
            const ONE_RAW: $raw = R::DESCRIPTION.one() as $raw;
            const VALID: () = R::VALID;

            #[inline]
            fn canonical_raw(raw: $raw) -> $raw {
                raw & $mask
            }

            #[inline]
            fn add_raw(left: $raw, right: $raw) -> $raw {
                left ^ right
            }

            #[inline]
            fn sub_raw(left: $raw, right: $raw) -> $raw {
                left ^ right
            }

            #[inline]
            fn neg_raw(value: $raw) -> $raw {
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
        impl<R: BinaryRepr<$n>> BinaryField for Gf<$n, R> {
            const DESCRIPTION: &'static BinaryDescription = R::DESCRIPTION;

            #[inline]
            // Widening a stored word to the coordinate word is exact.
            #[allow(clippy::cast_lossless)]
            fn to_coordinates(value: Elem<Self>) -> u64 {
                value.to_raw() as u64
            }

            #[inline]
            fn from_coordinates(value: u64) -> Result<Elem<Self>, CoordinateError> {
                if coordinates_fit(value, $n) {
                    // The fit check proves the word fits the raw type.
                    #[allow(clippy::cast_possible_truncation)]
                    let raw = value as $raw;
                    Ok(Elem::<Self>::from_raw(raw))
                } else {
                    Err(CoordinateError)
                }
            }
        }
        impl<R: BinaryRepr<$n>> BinaryDegree<$n> for Gf<$n, R> {}
    };
}

/// Emit the inherent `const` scalar arithmetic of one small degree row.
///
/// Degree eight multiplies through the representation's log/exp tables;
/// degrees 1, 2, and 4 interpret the descriptor directly.
macro_rules! small_row {
    ($n:literal, $mask:expr) => {
        impl<R: BinaryRepr<$n>> Elem<Gf<$n, R>> {
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
                let () = Validate::<Gf<$n, R>>::OK;
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
degree_row!(16, u16, u16::MAX);
degree_row!(32, u32, u32::MAX);
degree_row!(64, u64, u64::MAX);

small_row!(1, 0x1);
small_row!(2, 0x3);
small_row!(4, 0xF);
small_row!(8, 0xFF);

/// Emit the inherent `const` scalar arithmetic of one wide degree row.
///
/// One body per operation branches on structural equality with the
/// crate-pinned descriptions: the Rijndael and Fan-Paar presentations keep
/// their specialized recurrences and every other description uses the
/// general descriptor interpreter. Custom specs cannot claim a strategy.
macro_rules! wide_row {
    (
        $n:literal, $raw:ty, $bytes:literal, $width:expr,
        $rijndael:expr, $fanpaar:expr,
        $rmul:path, $rsquare:path, $rinv:path
    ) => {
        impl<R: BinaryRepr<$n>> Elem<Gf<$n, R>> {
            /// Wrap a raw storage word.
            ///
            /// Referencing this constructor evaluates the representation's
            /// validity check, so an invalid relation fails to compile at
            /// the use site.
            #[allow(clippy::let_unit_value)]
            #[allow(clippy::ignored_unit_patterns)]
            #[inline]
            #[must_use]
            pub const fn from_raw(value: $raw) -> Self {
                let () = Validate::<Gf<$n, R>>::OK;
                let () = R::VALID;
                Elem { raw: value }
            }

            /// Decode from the stable little-endian representation.
            #[inline]
            #[must_use]
            pub const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                Elem {
                    raw: <$raw>::from_le_bytes(bytes),
                }
            }

            /// Encode to the stable little-endian representation.
            #[inline]
            #[must_use]
            pub const fn to_bytes(self) -> [u8; $bytes] {
                self.raw.to_le_bytes()
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
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
            #[inline]
            #[must_use]
            pub const fn mul(self, rhs: Self) -> Self {
                if R::DESCRIPTION.same_structure($rijndael) {
                    Elem {
                        raw: $rmul(self.raw, rhs.raw),
                    }
                } else if R::DESCRIPTION.same_structure($fanpaar) {
                    Elem {
                        raw: tower::fp_multiply(self.raw as u64, rhs.raw as u64, $n) as $raw,
                    }
                } else {
                    Elem {
                        raw: R::DESCRIPTION.mul(self.raw as u64, rhs.raw as u64) as $raw,
                    }
                }
            }

            /// Square.
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
            #[inline]
            #[must_use]
            pub const fn square(self) -> Self {
                if R::DESCRIPTION.same_structure($rijndael) {
                    Elem {
                        raw: $rsquare(self.raw),
                    }
                } else if R::DESCRIPTION.same_structure($fanpaar) {
                    Elem {
                        raw: tower::fp_square(self.raw as u64, $n) as $raw,
                    }
                } else {
                    Elem {
                        raw: R::DESCRIPTION.mul(self.raw as u64, self.raw as u64) as $raw,
                    }
                }
            }

            /// Multiplicative inverse. Maps zero to zero by crate convention.
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
            #[inline]
            #[must_use]
            pub const fn inv(self) -> Self {
                if R::DESCRIPTION.same_structure($rijndael) {
                    Elem {
                        raw: $rinv(self.raw),
                    }
                } else if R::DESCRIPTION.same_structure($fanpaar) {
                    Elem {
                        raw: tower::fp_invert(self.raw as u64, $n) as $raw,
                    }
                } else {
                    Elem {
                        raw: R::DESCRIPTION.inv(self.raw as u64) as $raw,
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

        impl<R: BinaryRepr<$n>> fmt::Display for Elem<Gf<$n, R>> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{:0width$x}", self.raw, width = $width)
            }
        }
    };
}

wide_row!(
    16,
    u16,
    2,
    4,
    tower::RIJNDAEL16_DESC,
    tower::FANPAAR16_DESC,
    tower::rijndael16_mul,
    tower::rijndael16_square,
    tower::rijndael16_inv
);
wide_row!(
    32,
    u32,
    4,
    8,
    tower::RIJNDAEL32_DESC,
    tower::FANPAAR32_DESC,
    tower::rijndael32_mul,
    tower::rijndael32_square,
    tower::rijndael32_inv
);
wide_row!(
    64,
    u64,
    8,
    16,
    tower::RIJNDAEL64_DESC,
    tower::FANPAAR64_DESC,
    tower::rijndael64_mul,
    tower::rijndael64_square,
    tower::rijndael64_inv
);

impl<R: BinaryRepr<8>> FieldBuffer for Gf<8, R> {
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

impl<R: BinaryRepr<8>> fmt::Display for Elem<Gf<8, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:02x}", self.raw)
    }
}

impl<R: BinaryRepr<1>> fmt::Display for Elem<Gf<1, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.raw)
    }
}

impl<R: BinaryRepr<2>> fmt::Display for Elem<Gf<2, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:01x}", self.raw)
    }
}

impl<R: BinaryRepr<4>> fmt::Display for Elem<Gf<4, R>> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:01x}", self.raw)
    }
}
