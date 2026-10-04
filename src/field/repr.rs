//! Binary fields parameterized by degree and representation.
//!
//! [`Gf<const N, R>`](Gf) names GF(2^N) under the representation `R` — a
//! zero-sized marker carrying no runtime data — and [`Elem`]
//! over it is its element, stored as one raw storage word of `R`'s choosing.
//! The [`Repr`] trait is the sealed vocabulary of representations this crate
//! ships: a representation fixes the storage type, the field's name, and the
//! raw encodings of one and of the canonical generator. [`ByteRepr`] narrows
//! it to the degree-eight representations whose arithmetic is shared: every
//! additive byte basis multiplies through discrete-logarithm tables, so one
//! `const` implementation in this module serves all of them.
//!
//! # Layout
//!
//! Raw storage is private to the crate. Every bit pattern of `R::Raw` is a
//! distinct canonical field value, and the stable byte encoding of an element
//! is the little-endian encoding of its raw word.
//!
//! # Totality and canonicalization
//!
//! By crate-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
//! profile and under `const` evaluation alike. The shared degree-eight
//! arithmetic keeps that convention, and so does the degree-one field
//! [`Gf1`](crate::field::gf1::Gf1).

use core::fmt;

use super::poly::Poly;
use super::{Elem, Field, FieldBuffer, HasGenerator, PrimeCharacteristic, Validate};

// Shared by the representation families in sibling modules (`tower`), so
// every sealed surface in `field` answers to one trait.
pub(crate) mod private {
    pub trait Sealed {}
}

/// A sealed representation of GF(2^N).
///
/// `Raw` is the storage type of one element and is fixed per degree: `u8`
/// for N in {1, 8}, `u16` for N = 16. Implementations are fixed inside the
/// crate.
#[allow(private_bounds)]
pub trait Repr<const N: u8>:
    Copy + core::fmt::Debug + core::hash::Hash + PartialEq + Eq + 'static + private::Sealed
{
    /// Storage of one element in this representation.
    ///
    /// Every bit pattern is a distinct canonical field value. Raw storage is
    /// private to the crate; each element exposes it through
    /// [`Elem::to_raw`](crate::field::Elem::to_raw).
    type Raw: Copy + PartialEq + Eq + core::hash::Hash + core::fmt::Debug + Default + Ord;

    /// Human-readable field name, e.g. `"GF(2^8)/0x11B"`.
    const NAME: &'static str;
    /// Raw storage of the multiplicative identity. Representation-specific:
    /// polynomial bases use one low bit; normal bases use all-ones.
    const ONE_RAW: Self::Raw;
    /// Raw storage of the canonical generator of the multiplicative group.
    const GENERATOR_RAW: Self::Raw;
}

/// A one-byte representation of GF(2^8) whose arithmetic is shared.
///
/// Every additive byte basis — polynomial, normal, Cantor — multiplies by
/// discrete logarithms, so one table pair per representation drives the one
/// `const` arithmetic implementation on [`Elem`] over
/// [`Gf8`]. The tables are built at compile time from each representation's
/// reference construction.
pub trait ByteRepr: Repr<8, Raw = u8> {
    /// `EXP[i] = GENERATOR^i` in this representation's encoding.
    const EXP: &'static [u8; 255];
    /// `LOG[EXP[i]] = i`; `LOG[0]` is undefined and callers short-circuit on
    /// zero before indexing.
    const LOG: &'static [u8; 256];
    /// Use-time validity check for the representation's defining data, so
    /// every element constructor rejects an invalid representation at
    /// monomorphization.
    const VALID: ();
}

/// Marker for GF(2^N) under the representation `R`. Zero-sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf<const N: u8, R: Repr<N>>(core::marker::PhantomData<R>);

/// The GF(2^8) field under any degree-eight representation `R`.
pub type Gf8<R> = Gf<8, R>;

/// Table multiplication over one representation's log/exp pair.
const fn table_mul(exp: &[u8; 255], log: &[u8; 256], left: u8, right: u8) -> u8 {
    if left == 0 || right == 0 {
        return 0;
    }
    exp[(log[left as usize] as usize + log[right as usize] as usize) % 255]
}

/// Table square-and-multiply over one representation's log/exp pair.
const fn table_pow(exp: &[u8; 255], log: &[u8; 256], mut base: u8, mut exponent: u64) -> u8 {
    let mut result = 1u8;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = table_mul(exp, log, result, base);
        }
        base = table_mul(exp, log, base, base);
        exponent >>= 1;
    }
    result
}

/// Whether `candidate` generates the order-255 group under the tables.
const fn is_byte_primitive(exp: &[u8; 255], log: &[u8; 256], candidate: u8) -> bool {
    if candidate == 0 || table_pow(exp, log, candidate, 255) != 1 {
        return false;
    }
    // Proper cofactor powers for 255 = 3 * 5 * 17: 85, 51, 15.
    table_pow(exp, log, candidate, 85) != 1
        && table_pow(exp, log, candidate, 51) != 1
        && table_pow(exp, log, candidate, 15) != 1
}

impl<R: ByteRepr> Field for Gf<8, R> {
    type Raw = u8;
    type Characteristic = PrimeCharacteristic<2>;
    const NAME: &'static str = R::NAME;
    const DEGREE: u32 = 8;
    const ORDER: u128 = 256;
    const ZERO_RAW: u8 = 0;
    const ONE_RAW: u8 = R::ONE_RAW;
    const VALID: () = R::VALID;

    #[inline]
    fn canonical_raw(raw: u8) -> u8 {
        raw
    }

    #[inline]
    fn add_raw(left: u8, right: u8) -> u8 {
        left ^ right
    }

    #[inline]
    fn sub_raw(left: u8, right: u8) -> u8 {
        left ^ right
    }

    #[inline]
    fn neg_raw(value: u8) -> u8 {
        value
    }

    #[inline]
    fn mul_raw(left: u8, right: u8) -> u8 {
        table_mul(R::EXP, R::LOG, left, right)
    }

    #[inline]
    fn inv_raw(value: u8) -> u8 {
        if value == 0 {
            return 0;
        }
        // The `% 255` matters only for `value == 1`, where `LOG` is 0 and
        // `255 - 0` would run off the end of the 255-entry table.
        let l = R::LOG[value as usize] as usize;
        R::EXP[(255 - l) % 255]
    }
}

impl<R: ByteRepr> HasGenerator for Gf<8, R> {
    const GENERATOR_RAW: u8 = {
        assert!(
            is_byte_primitive(R::EXP, R::LOG, R::GENERATOR_RAW),
            "byte representation generator does not have full order"
        );
        R::GENERATOR_RAW
    };
}

impl<R: ByteRepr> FieldBuffer for Gf<8, R> {
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

impl<R: ByteRepr> Elem<Gf<8, R>> {
    /// Wrap a raw storage word. Every raw word is a distinct field value.
    ///
    /// Referencing this constructor evaluates the representation's validity
    /// check, so a representation with invalid defining data fails to
    /// compile at the use site.
    // Reading `VALID` forces its evaluation; the unit value is the point.
    #[allow(clippy::let_unit_value)]
    #[allow(clippy::ignored_unit_patterns)]
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        let () = Validate::<Gf<8, R>>::OK;
        let () = R::VALID;
        Elem { raw: value }
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

    /// Field multiplication through the representation's discrete-log tables.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        // LOG has no entry for zero, so the absorbing case must
        // short-circuit.
        if self.raw == 0 || rhs.raw == 0 {
            return Self::ZERO;
        }
        let la = R::LOG[self.raw as usize] as usize;
        let lb = R::LOG[rhs.raw as usize] as usize;
        Elem {
            raw: R::EXP[(la + lb) % 255],
        }
    }

    /// Square.
    ///
    /// A flat byte field has no cheaper form than a general multiply; the
    /// tower fields do. Present for parity with them.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse. Maps zero to zero by crate convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        if self.raw == 0 {
            return Self::ZERO;
        }
        // The `% 255` matters only for `self == 1`, where `LOG` is 0 and
        // `255 - 0` would run off the end of the 255-entry table.
        let l = R::LOG[self.raw as usize] as usize;
        Elem {
            raw: R::EXP[(255 - l) % 255],
        }
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division
    /// total leaves hot loops branch-free and keeps this callable from
    /// `const` context.
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
}

impl<R: ByteRepr> fmt::Display for Elem<Gf<8, R>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}", self.raw)
    }
}

impl<const P: u32> private::Sealed for Poly<P> {}
