//! Field definitions and the scalar algebra contract.
//!
//! A [`Field`] is a zero-sized marker type carrying compile-time facts about a
//! finite field: its raw scalar word, its characteristic identity, and its
//! order. [`Elem<F>`] is the one scalar element wrapper for every field in
//! the crate. [`FieldBuffer`] adds the fixed-width byte encoding that bulk
//! operations pack into buffers, and [`HasGenerator`] selects a verified
//! multiplicative generator. Kernels in [`crate::ops`] are generic over
//! [`crate::kernel::FieldKernels`], so callers write one algorithm and get a
//! monomorphized, SIMD-dispatched implementation per field.
//!
//! # Scalar storage
//!
//! Every safely constructed element holds canonical raw coordinates, so
//! [`PartialEq`], [`Eq`], [`Hash`], and [`Ord`] compare raw words directly.
//! Ordering is deterministic coordinate order, not an order compatible with
//! field algebra. [`FieldElem::from_raw`] canonicalizes its input.
//!
//! By library-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
//! profile and under `const` evaluation alike. This is a total-function
//! convention chosen so hot loops never branch on an impossible case; it is
//! *not* a claim that zero is invertible.
//!
//! # Custom fields
//!
//! [`Field`] stays open for downstream scalar definitions. The central
//! validation block checks the characteristic identity, a positive degree,
//! `ORDER == CHARACTERISTIC^DEGREE`, and the binary degree cap of 64 before
//! evaluating `VALID`. It runs from every generic constructor, so a field
//! that misdeclares its metadata fails at value use:
//!
//! ```
//! use fgf::{Elem, Mersenne31};
//!
//! // A shipped prime field constructs its identities generically.
//! const ONE: Elem<Mersenne31> = Elem::<Mersenne31>::ONE;
//! assert_eq!(ONE.to_raw(), 1);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::{Elem, Field, PrimeCharacteristic};
//!
//! // Degree 65 exceeds the binary cap, so even the identities fail.
//! #[derive(Clone, Copy, Debug)]
//! struct Wide;
//! impl Field for Wide {
//!     type Raw = u128;
//!     type Characteristic = PrimeCharacteristic<2>;
//!     const NAME: &'static str = "wide";
//!     const DEGREE: u32 = 65;
//!     const ORDER: u128 = 0;
//!     const ZERO_RAW: u128 = 0;
//!     const ONE_RAW: u128 = 1;
//!     const VALID: () = ();
//!     fn canonical_raw(raw: u128) -> u128 { raw }
//!     fn add_raw(a: u128, b: u128) -> u128 { a ^ b }
//!     fn sub_raw(a: u128, b: u128) -> u128 { a ^ b }
//!     fn neg_raw(a: u128) -> u128 { a }
//!     fn mul_raw(a: u128, b: u128) -> u128 { a }
//!     fn inv_raw(a: u128) -> u128 { a }
//! }
//! const Z: Elem<Wide> = Elem::<Wide>::ZERO;
//! ```
//!
//! A prime identity is available at any checked `u64` prime, and a composite
//! modulus fails the same way:
//!
//! ```
//! use fgf::{PrimeCharacteristic, PrimeIdentity};
//!
//! // Seven is prime, so its identity reads back.
//! const P: u64 = <PrimeCharacteristic<7> as PrimeIdentity>::CHARACTERISTIC;
//! const _: () = assert!(P == 7);
//! ```
//!
//! ```compile_fail,E0080
//! use fgf::{PrimeCharacteristic, PrimeIdentity};
//!
//! // Ninety-one is composite, so reading the identity fails.
//! const Q: u64 = <PrimeCharacteristic<91> as PrimeIdentity>::CHARACTERISTIC;
//! ```

pub mod binary;
pub mod goldilocks;
pub mod mersenne31;
pub mod quad_mersenne31;

pub use binary::cantor;
pub use binary::description;
pub use binary::embedding;
pub use binary::normal;
pub use binary::poly;
pub use binary::tower;
pub use binary::tower::{
    FanPaar2, FanPaar2Spec, FanPaar4, FanPaar4Spec, FanPaar8, FanPaar8Spec, FanPaar16,
    FanPaar16Spec, FanPaar32, FanPaar32Spec, FanPaar64, FanPaar64Spec, Gf16, Gf32, Gf64,
    Rijndael16, Rijndael32, Rijndael64, Tower, TowerGeneratorSpec, TowerSpec,
};
pub use binary::{
    AES, BinaryDegree, BinaryDescription, BinaryField, BinaryRepr, ByteLogExp, Cantor, Embedding,
    EmbeddingError, Gf, Gf1, Gf8, Normal, Poly, REED_SOLOMON,
};
pub use goldilocks::Goldilocks;
pub use mersenne31::Mersenne31;
pub use quad_mersenne31::QuadMersenne31;

mod private {
    pub trait Sealed {}
}

mod element_private {
    pub trait Sealed {}
}

/// Modular exponentiation over `u64`, for const primality and generator checks.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub(crate) const fn powmod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1 % modulus;
    base %= modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exponent >>= 1;
    }
    result
}

/// Deterministic Miller-Rabin primality over `u64`.
///
/// Small-value guards, `u128` modular products, and the witness set below.
/// Each witness is reduced modulo the candidate and a zero residue is
/// skipped.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub const fn is_prime(candidate: u64) -> bool {
    const SMALL: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    const WITNESSES: [u64; 7] = [2, 325, 9375, 28_178, 450_775, 9_780_504, 1_795_265_022];
    if candidate < 2 {
        return false;
    }
    let mut i = 0;
    while i < SMALL.len() {
        let prime = SMALL[i];
        if candidate == prime {
            return true;
        }
        if candidate.is_multiple_of(prime) {
            return false;
        }
        i += 1;
    }
    let mut odd = candidate - 1;
    let mut shifts = 0u32;
    while odd.is_multiple_of(2) {
        odd /= 2;
        shifts += 1;
    }
    let mut i = 0;
    while i < WITNESSES.len() {
        let residue = WITNESSES[i] % candidate;
        i += 1;
        if residue == 0 {
            continue;
        }
        let mut trial = powmod(residue, odd, candidate);
        if trial == 1 || trial == candidate - 1 {
            continue;
        }
        let mut round = 1;
        let mut composite = true;
        while round < shifts {
            trial = ((trial as u128 * trial as u128) % candidate as u128) as u64;
            if trial == candidate - 1 {
                composite = false;
                break;
            }
            round += 1;
        }
        if composite {
            return false;
        }
    }
    true
}

/// The identity of a prime characteristic.
///
/// One identity type per prime, independent of coordinates or a named
/// modulus. This is not an arithmetic backend.
pub trait PrimeIdentity: private::Sealed + Copy + 'static {
    /// The prime characteristic value, checked primality on read.
    const CHARACTERISTIC: u64;
    /// Use-time primality proof for the characteristic.
    const VALID: ();
}

/// The checked identity of the prime characteristic `P`.
///
/// Both [`PrimeIdentity::CHARACTERISTIC`] and [`PrimeIdentity::VALID`] assert
/// the const deterministic Miller-Rabin check, so prime queries validate
/// themselves even if an implementation is incorrect.
#[derive(Clone, Copy, Debug)]
pub struct PrimeCharacteristic<const P: u64>;

impl<const P: u64> private::Sealed for PrimeCharacteristic<P> {}

impl<const P: u64> PrimeIdentity for PrimeCharacteristic<P> {
    const CHARACTERISTIC: u64 = {
        assert!(is_prime(P), "characteristic is not prime");
        P
    };
    const VALID: () = assert!(is_prime(P), "characteristic is not prime");
}

/// A finite field supported by this crate.
///
/// The marker carries only compile-time facts. `Raw` describes a single
/// scalar's representation, not buffer pitch, alignment, byte order, or
/// vector packing. The raw arithmetic methods are the implementation
/// contract for custom field definitions; ordinary callers work with
/// [`Elem<F>`] and its operations. Generic code reads the characteristic as
/// `<F::Characteristic as PrimeIdentity>::CHARACTERISTIC`.
pub trait Field: Copy + core::fmt::Debug + 'static {
    /// The raw word of one scalar element.
    type Raw: Copy + Eq + Ord + core::hash::Hash + core::fmt::Debug;
    /// The prime characteristic identity of the field.
    type Characteristic: PrimeIdentity;
    /// Human-readable field name, e.g. `"GF(2^8)"`.
    const NAME: &'static str;
    /// Total degree: the extension degree over GF(2) for binary fields, one
    /// for prime fields, two for the quadratic prime extension.
    const DEGREE: u32;
    /// Number of elements in the field: the characteristic raised to
    /// [`Field::DEGREE`].
    const ORDER: u128;
    /// Raw word of the additive identity, in canonical coordinates.
    const ZERO_RAW: Self::Raw;
    /// Raw word of the multiplicative identity, in canonical coordinates.
    const ONE_RAW: Self::Raw;
    /// Use-time validity check for the field's defining data.
    const VALID: ();
    /// Reduce a raw word to canonical coordinates.
    fn canonical_raw(raw: Self::Raw) -> Self::Raw;
    /// Raw field addition, over canonical inputs.
    fn add_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw;
    /// Raw field subtraction, over canonical inputs.
    fn sub_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw;
    /// Raw additive inverse, over a canonical input.
    fn neg_raw(value: Self::Raw) -> Self::Raw;
    /// Raw field multiplication, over canonical inputs.
    fn mul_raw(left: Self::Raw, right: Self::Raw) -> Self::Raw;
    /// Raw multiplicative inverse. Maps zero to zero by convention.
    fn inv_raw(value: Self::Raw) -> Self::Raw;
}

/// Central metadata validation for a field.
///
/// Reads the checked characteristic, asserts a positive degree, checks
/// `ORDER == CHARACTERISTIC^DEGREE` with overflow checking, asserts the
/// binary cap of 64 whenever the characteristic is two, then evaluates
/// `F::VALID`. Not overridable by implementors and not routable around.
pub(crate) struct Validate<F>(core::marker::PhantomData<F>);

impl<F: Field> Validate<F> {
    pub(crate) const OK: () = {
        let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
        assert!(F::DEGREE > 0, "field degree must be positive");
        assert!(
            characteristic != 2 || F::DEGREE <= 64,
            "binary field degree exceeds 64"
        );
        let mut order: u128 = 1;
        let mut i = 0;
        while i < F::DEGREE {
            order = match order.checked_mul(characteristic as u128) {
                Some(value) => value,
                None => panic!("field order overflows u128"),
            };
            i += 1;
        }
        assert!(order == F::ORDER, "ORDER is not CHARACTERISTIC^DEGREE");
        let () = F::VALID;
    };
}

/// One scalar element of the field presentation `F`, across every family.
///
/// Storage is always canonical, so comparisons are on raw words. The raw
/// field is private to the crate; the canonical word reads through the
/// generic [`Elem::to_raw`].
pub struct Elem<F: Field> {
    pub(crate) raw: F::Raw,
}

// A derive would bound the impl on `F: Clone`; the manual impl bounds
// only through `F::Raw`, which the field contract already supplies.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<F: Field> Clone for Elem<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Field> Copy for Elem<F> {}

impl<F: Field> PartialEq for Elem<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<F: Field> Eq for Elem<F> {}

impl<F: Field> core::hash::Hash for Elem<F> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<F: Field> PartialOrd for Elem<F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Field> Ord for Elem<F> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<F: Field> core::fmt::Debug for Elem<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}({:?})", F::NAME, self.raw)
    }
}

impl<F: Field> Default for Elem<F> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field> Elem<F> {
    /// The additive identity, and the absorbing element for multiplication.
    pub const ZERO: Self = {
        let () = Validate::<F>::OK;
        Elem { raw: F::ZERO_RAW }
    };
    /// The multiplicative identity.
    pub const ONE: Self = {
        let () = Validate::<F>::OK;
        Elem { raw: F::ONE_RAW }
    };

    /// Unwrap to the canonical raw storage word.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> F::Raw {
        self.raw
    }
}

/// A selected multiplicative generator of the field.
///
/// The raw provider lets downstream custom fields supply a generator without
/// a public unchecked element constructor. Raw canonicality and full
/// generator order are provider obligations; crate-owned providers verify
/// full multiplicative order inside the constant.
pub trait HasGenerator: Field {
    /// Raw word of the selected generator, of full multiplicative order.
    const GENERATOR_RAW: Self::Raw;
}

impl<F: HasGenerator> Elem<F> {
    /// The selected generator of the multiplicative group.
    pub const GENERATOR: Self = {
        let () = Validate::<F>::OK;
        Elem {
            raw: F::GENERATOR_RAW,
        }
    };
}

/// Scalar arithmetic over a finite field.
///
/// In characteristic two — every binary field in this crate, GF(2)
/// included — addition and subtraction are the same operation (XOR) and
/// negation is the identity. Prime fields reduce modulo their
/// characteristic instead. Both `add` and `sub` are provided because
/// algorithms read more clearly when they say what they mean.
///
/// Elements compare by field value through their canonical storage. The
/// trait is sealed: only the blanket implementation for [`Elem<F>`]
/// implements it. Generic callers use the operators or this trait; concrete
/// families additionally expose inherent `const` arithmetic.
///
/// By library-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
/// profile and under `const` evaluation alike. This is a total-function
/// convention chosen so hot loops never branch on an impossible case; it is
/// *not* a claim that zero is invertible.
#[allow(private_bounds)]
pub trait FieldElem:
    element_private::Sealed + Copy + Eq + core::fmt::Debug + core::hash::Hash + Default + 'static
{
    /// The field this is an element of.
    type Field: Field;
    /// Wrap a raw word, canonicalizing it into storage.
    ///
    /// Runs the central field validation, then
    /// [`Field::canonical_raw`].
    fn from_raw(raw: <Self::Field as Field>::Raw) -> Self;
    /// The additive identity, and the absorbing element for multiplication.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Field addition. XOR in characteristic two.
    #[must_use]
    fn add(self, rhs: Self) -> Self;
    /// Field subtraction. Identical to [`FieldElem::add`] in characteristic
    /// two.
    #[must_use]
    fn sub(self, rhs: Self) -> Self;
    /// Additive inverse: the unique `y` with `self.add(y) == ZERO`.
    #[must_use]
    fn neg(self) -> Self;
    /// Field multiplication.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;
    /// Multiplicative inverse. Maps zero to zero by convention.
    #[must_use]
    fn inv(self) -> Self;
    /// Field division. Returns zero when the divisor is zero.
    #[must_use]
    fn div(self, rhs: Self) -> Self;
    /// Square.
    #[must_use]
    fn square(self) -> Self;
    /// Raise to an unsigned integer power by square-and-multiply.
    ///
    /// Present on the trait so generic code can exponentiate without knowing
    /// the field's group order. `pow(0)` is [`FieldElem::ONE`] for every
    /// element, zero included. The exponent accepts `u128` so it also
    /// serves the odd-characteristic cardinality range.
    #[must_use]
    fn pow(self, exponent: u128) -> Self;
    /// Whether this element is the additive identity.
    fn is_zero(self) -> bool;
    /// Whether this element is the multiplicative identity.
    fn is_one(self) -> bool;
}

impl<F: Field> element_private::Sealed for Elem<F> {}

impl<F: Field> FieldElem for Elem<F> {
    type Field = F;

    #[inline]
    fn from_raw(raw: F::Raw) -> Self {
        let () = Validate::<F>::OK;
        Elem {
            raw: F::canonical_raw(raw),
        }
    }

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem {
            raw: F::add_raw(self.raw, rhs.raw),
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem {
            raw: F::sub_raw(self.raw, rhs.raw),
        }
    }

    #[inline]
    fn neg(self) -> Self {
        Elem {
            raw: F::neg_raw(self.raw),
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: F::mul_raw(self.raw, rhs.raw),
        }
    }

    #[inline]
    fn inv(self) -> Self {
        Elem {
            raw: F::inv_raw(self.raw),
        }
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.raw == F::ZERO_RAW {
            Self::ZERO
        } else {
            Elem {
                raw: F::mul_raw(self.raw, F::inv_raw(rhs.raw)),
            }
        }
    }

    #[inline]
    fn square(self) -> Self {
        Elem {
            raw: F::mul_raw(self.raw, self.raw),
        }
    }

    #[inline]
    fn pow(self, mut exponent: u128) -> Self {
        let mut base = self.raw;
        let mut result = F::ONE_RAW;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = F::mul_raw(result, base);
            }
            base = F::mul_raw(base, base);
            exponent >>= 1;
        }
        Elem { raw: result }
    }

    #[inline]
    fn is_zero(self) -> bool {
        self.raw == F::ZERO_RAW
    }

    #[inline]
    fn is_one(self) -> bool {
        self.raw == F::ONE_RAW
    }
}

impl<F: Field> core::ops::Add for Elem<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem {
            raw: F::add_raw(self.raw, rhs.raw),
        }
    }
}

impl<F: Field> core::ops::Sub for Elem<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem {
            raw: F::sub_raw(self.raw, rhs.raw),
        }
    }
}

impl<F: Field> core::ops::Neg for Elem<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Elem {
            raw: F::neg_raw(self.raw),
        }
    }
}

impl<F: Field> core::ops::Mul for Elem<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem {
            raw: F::mul_raw(self.raw, rhs.raw),
        }
    }
}

impl<F: Field> core::ops::Div for Elem<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        <Self as FieldElem>::div(self, rhs)
    }
}

impl<F: Field> core::ops::AddAssign for Elem<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> core::ops::SubAssign for Elem<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> core::ops::MulAssign for Elem<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> core::ops::DivAssign for Elem<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Field> core::iter::Sum for Elem<F> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |left, right| left + right)
    }
}

impl<'a, F: Field> core::iter::Sum<&'a Elem<F>> for Elem<F> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem<F>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |left, &right| left + right)
    }
}

impl<F: Field> core::iter::Product for Elem<F> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |left, right| left * right)
    }
}

impl<'a, F: Field> core::iter::Product<&'a Elem<F>> for Elem<F> {
    #[inline]
    fn product<I: Iterator<Item = &'a Elem<F>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |left, &right| left * right)
    }
}

/// Fixed-width byte encoding of a field, suitable for field buffers.
///
/// `BYTES` is the width of one element lane and `STORAGE_BITS` is `8 *
/// BYTES`, never confused with the extension degree. Encoders emit canonical
/// little-endian coordinates; decoders canonicalize accepted aliases.
/// Buffers hold fixed-width coordinates, not prepared coefficients or native
/// scratch.
pub trait FieldBuffer: Field {
    /// Width of the stable byte representation of one element.
    const BYTES: usize;
    /// Storage bit width of one element: `8 * BYTES`.
    const STORAGE_BITS: u32;
    /// Decode one element from its stable little-endian byte representation.
    ///
    /// # Panics
    /// Panics if `bytes.len() != Self::BYTES`.
    fn decode(bytes: &[u8]) -> Elem<Self>;
    /// Encode one element into its stable little-endian byte representation.
    ///
    /// # Panics
    /// Panics if `bytes.len() != Self::BYTES`.
    fn encode(bytes: &mut [u8], value: Elem<Self>);
}
