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
//! The canonical generic family [`FiniteField`] names a field by its prime
//! characteristic, its degree, and a presentation, with [`Residue`] as the
//! ordinary-residue presentation at degree one over every `u64` prime; see
//! [`finite_field`] and [`prime`].
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
//! # Row vocabulary
//!
//! [`Field`] is sealed: the crate implements it for the canonical family
//! rows and the named rows only, so a presentation's defining facts are
//! always crate-owned. A custom prime field is the family's degree-one row
//! [`FiniteField<P, 1>`](FiniteField) over any checked `u64` prime. The
//! central validation block checks the characteristic identity, a positive
//! degree, `ORDER == CHARACTERISTIC^DEGREE`, and the total degree cap of 64
//! before evaluating `VALID`. It runs from every generic constructor, so a
//! row that misdeclares its metadata fails at value use:
//!
//! ```
//! use fgf::{Elem, Mersenne31};
//!
//! // A shipped prime field constructs its identities generically.
//! const ONE: Elem<Mersenne31> = Elem::<Mersenne31>::ONE;
//! assert_eq!(ONE.to_raw(), 1);
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
pub mod embedding;
pub mod extension;
pub mod finite_field;
pub mod goldilocks;
pub mod mersenne31;
pub mod prime;
pub mod quad_mersenne31;

pub use binary::cantor;
pub use binary::description;
pub use binary::normal;
pub use binary::poly;
pub use binary::tower;
pub use binary::tower::{
    FanPaar2, FanPaar4, FanPaar8, FanPaar16, FanPaar32, FanPaar64, Rijndael16, Rijndael32,
    Rijndael64,
};
pub use binary::{
    AES, BinaryDegree, BinaryDescription, BinaryField, BinaryRepr, ByteLogExp, Cantor, Normal,
    Polynomial, RS,
};
pub use embedding::{Embedding, EmbeddingError, Isomorphism, Postcompose};
#[cfg(feature = "alloc")]
pub use embedding::{Frobenius, PreparedEmbedding, PreparedIsomorphism};
pub use extension::{
    Components, Extension, ExtensionField, ExtensionSpec, ExtensionStorage, Packed, PackedPadded,
    PackedWord, Relation,
};
pub use finite_field::{FiniteField, FiniteFieldRepr, Residue};
pub use goldilocks::{Goldilocks, GoldilocksFold};
pub use mersenne31::{Mersenne31, MersenneFold};
pub use prime::PrimeField;
pub use quad_mersenne31::{ComplexMersenne31, QuadMersenne31};

mod private {
    use super::Field;
    use super::extension::Descriptor;

    pub trait Sealed {}

    /// Seals the capabilities a field row carries beyond [`Field`]:
    /// every implementor is a crate-owned row.
    pub trait FieldSeal {}

    /// Byte-level add/sub dispatch for buffer composition.
    ///
    /// Crate-private seam: every [`super::FieldBuffer`] provides byte add
    /// and subtract hooks that component storage and recursive extension
    /// encoding decompose into. The methods take raw byte slices — no
    /// prepared coefficients, no backend tokens — so they compose without
    /// allocation or unsafe.
    pub(crate) trait BufferDispatch {
        /// `dst += src` on raw field-element bytes.
        fn add_assign_bytes(proof: crate::kernel::RawDispatch, dst: &mut [u8], src: &[u8]);
        /// `dst -= src` on raw field-element bytes.
        fn sub_assign_bytes(proof: crate::kernel::RawDispatch, dst: &mut [u8], src: &[u8]);
    }

    /// Marks the degree-one rows eligible for prime-field membership.
    pub trait PrimeFieldSeal {}

    /// The ordinary-residue row of a prime characteristic.
    ///
    /// The private counterpart of the public characteristic query: each
    /// prime identity names the degree-one row its compiled odd maps
    /// accumulate through. `PrimeCharacteristic<P>` is the sole
    /// implementor.
    pub trait ResidueRowOf: Sealed {
        /// The degree-one ordinary-residue row of this characteristic.
        type Row: crate::kernel::FieldKernels + super::Field<Characteristic = Self>;
    }

    /// The row hooks behind the `Field` seal: the presentation access the
    /// crate's relation algorithms share.
    ///
    /// Every hook evaluates the row's central validation first, as the raw
    /// methods do, and canonicalizes `raw` inputs through
    /// `Field::canonical_raw` before reading them.
    pub(crate) trait RowAccess: 'static {
        /// Descriptor of the row's presentation.
        const DESCRIPTOR: &'static Descriptor;
        /// Binary presentation description, when the row is a binary row.
        ///
        /// `Some` exactly for the rows implementing `BinaryField`: flat
        /// binary rows and characteristic-two extension rows over binary
        /// bases. The generic embedding reads it for the frozen
        /// characteristic-two image rule; every other row reports `None`.
        const BINARY_DESCRIPTION: Option<&'static super::binary::description::BinaryDescription>;
        /// Writes the flattened prime-subfield coordinates of `raw` into
        /// `dst`, in the descriptor's coordinate order.
        ///
        /// A prime row writes its ordinary residue; a binary row writes the
        /// bits of its coordinate word, least significant first; a quadratic
        /// prime extension writes its limbs in component order.
        ///
        /// # Panics
        /// Panics when `dst.len()` differs from `Field::DEGREE`.
        fn prime_coordinates_into(dst: &mut [u64], raw: <Self as Field>::Raw)
        where
            Self: Field;
        /// The raw element named by the given flattened prime-subfield
        /// coordinates, in the descriptor's coordinate order.
        ///
        /// # Panics
        /// Panics when `src.len()` differs from `Field::DEGREE` or a
        /// coordinate is at or above the characteristic.
        fn from_prime_coordinates(src: &[u64]) -> <Self as Field>::Raw
        where
            Self: Field;
        /// `value · 1`, with `value` reduced modulo the characteristic.
        fn from_prime(value: u64) -> <Self as Field>::Raw
        where
            Self: Field;
        /// `Some(c)` exactly when `raw == c · 1` with `c` below the
        /// characteristic; `None` otherwise.
        fn to_prime(raw: <Self as Field>::Raw) -> Option<u64>
        where
            Self: Field;
        /// The packed coordinate word of `raw`, in characteristic two.
        ///
        /// Bit `i` of the word is the `i`th prime-subfield coordinate in
        /// the descriptor's coordinate order. Flat rows cast their word;
        /// packed rows read the pinned word; component rows convert
        /// recursively through `RowAccess`, including residue-two leaves.
        /// The engine behind `Frobenius` is the only production caller;
        /// builds without it still run these hooks through the tests.
        ///
        /// # Panics
        /// Panics on any row of odd characteristic.
        #[cfg_attr(not(feature = "alloc"), allow(dead_code))]
        fn coordinate_word(raw: <Self as Field>::Raw) -> u64
        where
            Self: Field;
        /// The raw element named by a packed coordinate word.
        ///
        /// The inverse of `coordinate_word`: bit `i` of `word` is the
        /// `i`th prime-subfield coordinate in the descriptor's
        /// coordinate order.
        ///
        /// # Panics
        /// Panics on any row of odd characteristic.
        #[cfg_attr(not(feature = "alloc"), allow(dead_code))]
        fn from_coordinate_word(word: u64) -> <Self as Field>::Raw
        where
            Self: Field;
    }
}
/// Checks a residue against the characteristic before narrowing.
///
/// Returns `value` when it names an element of the prime subfield and
/// [`RangeError`] otherwise. The inherent `const` residue constructors and
/// the [`FieldElem::from_residue`] blanket implementation share this range
/// check; construction and readout share each row's conversion body.
pub(crate) const fn check_residue(value: u64, characteristic: u64) -> Result<u64, RangeError> {
    if value < characteristic {
        Ok(value)
    } else {
        Err(RangeError)
    }
}

/// The presentation index of flattened prime-subfield coordinates.
///
/// `Σ_i coordinates[i] · characteristic^i`, evaluated widening so the full
/// `u128` order range never overflows. Shared by the [`FieldElem::to_index`]
/// blanket implementation.
pub(crate) fn coordinates_index(coordinates: &[u64], characteristic: u64) -> u128 {
    let mut index = 0u128;
    let mut place = 1u128;
    let mut i = 0;
    while i < coordinates.len() {
        index += u128::from(coordinates[i]) * place;
        place *= u128::from(characteristic);
        i += 1;
    }
    index
}

/// Splits a presentation index into flattened prime-subfield coordinates.
///
/// The inverse of [`coordinates_index`]: `degree` base-`characteristic`
/// digits, least significant first. The buffer is the shared coordinate
/// capacity, so every degree at or under the total cap fits. Shared by the
/// [`FieldElem::from_index`] blanket implementation.
pub(crate) fn index_coordinates(
    index: u128,
    degree: u32,
    characteristic: u64,
) -> [u64; super::extension::descriptor::COORDINATES] {
    use super::extension::descriptor::COORDINATES;
    let mut coordinates = [0u64; COORDINATES];
    let mut rest = index;
    let mut i = 0;
    while i < degree {
        let modulus = u128::from(characteristic);
        // The remainder sits below the modulus, so it fits the lane.
        #[allow(clippy::cast_possible_truncation)]
        let digit = (rest % modulus) as u64;
        coordinates[i as usize] = digit;
        rest /= modulus;
        i += 1;
    }
    coordinates
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
#[allow(private_bounds)]
pub trait PrimeIdentity: private::Sealed + private::ResidueRowOf + Copy + 'static {
    /// The prime characteristic value, checked primality on read.
    const CHARACTERISTIC: u64;
    /// Use-time primality proof for the characteristic.
    const VALID: ();
    /// Bit width of the characteristic: `64 - P.leading_zeros()`. Two at
    /// `P = 2`, sixty-four at the widest `u64` primes.
    const BITS: u32;
    /// The power of two dividing `P - 1`: `(P - 1).trailing_zeros()`. Zero
    /// at `P = 2`, where `P - 1` is odd.
    const TWO_ADICITY: u32;
    /// The characteristic modulo eight: `(P % 8) as u8`. Two at `P = 2`.
    const RESIDUE_MOD_8: u8;
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
    #[allow(clippy::cast_possible_truncation)]
    const BITS: u32 = {
        assert!(is_prime(P), "characteristic is not prime");
        64 - P.leading_zeros()
    };
    const TWO_ADICITY: u32 = {
        assert!(is_prime(P), "characteristic is not prime");
        (P - 1).trailing_zeros()
    };
    #[allow(clippy::cast_possible_truncation)]
    const RESIDUE_MOD_8: u8 = {
        assert!(is_prime(P), "characteristic is not prime");
        (P % 8) as u8
    };
}

impl<const P: u64> private::ResidueRowOf for PrimeCharacteristic<P> {
    type Row = FiniteField<P, 1, Residue>;
}

/// The ordinary-residue row of a field's characteristic.
///
/// The degree-one row the compiled odd maps of `F` accumulate through.
#[cfg(feature = "alloc")]
pub(crate) type ResidueRow<F> = <<F as Field>::Characteristic as private::ResidueRowOf>::Row;

/// A finite field supported by this crate.
///
/// The marker carries only compile-time facts. `Raw` describes a single
/// scalar's representation, not buffer pitch, alignment, byte order, or
/// vector packing. The trait is sealed to the crate's own rows: the raw
/// arithmetic methods are a row's scalar body, and ordinary callers work
/// with [`Elem<F>`] and its operations. Generic code reads the characteristic
/// as `<F::Characteristic as PrimeIdentity>::CHARACTERISTIC`.
#[allow(private_bounds)]
pub trait Field: private::RowAccess + Copy + core::fmt::Debug + 'static {
    /// The raw word of one scalar element.
    type Raw: Copy + Eq + Ord + core::hash::Hash + core::fmt::Debug + 'static;
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
        let order = finite_field::metadata_order(characteristic, F::DEGREE);
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
/// The trait is sealed: only the crate's shipped rows implement it, each
/// with its frozen generator. Raw canonicality and full multiplicative order
/// hold inside the constant: each provider verifies the full order against
/// the factorization of the group order.
pub trait HasGenerator: Field + private::FieldSeal {
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
///
/// Residues are mathematical values; indices are presentation coordinates.
///
/// [`FieldElem::from_residue`] maps an ordinary residue to its multiple of
/// one whatever the basis, while [`FieldElem::from_index`] decodes the
/// flattened prime-subfield coordinates. The two agree only where one is
/// coordinate word one:
///
/// ```
/// use fgf::{Elem, FieldElem, FiniteField, Normal};
///
/// type Row = FiniteField<2, 2, Normal<7, 2>>;
/// let one = Elem::<Row>::ONE;
/// assert_eq!(one.to_residue(), Some(1));
/// assert_eq!(one.to_index(), 3);
/// ```
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
    /// The element named by an ordinary residue of the characteristic.
    ///
    /// Returns `value · 1` when `value` is below the characteristic and
    /// [`RangeError`] otherwise, before any narrowing: `256` on a
    /// byte-backed binary row and `1 << 32` on Mersenne31 are rejected even
    /// though their narrowed low bits look valid. Mathematical and
    /// independent of presentation: a basis row whose one is not coordinate
    /// word one still maps one to its one.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError`] when `value` is at or above the
    /// characteristic.
    fn from_residue(value: u64) -> Result<Self, RangeError>;
    /// The ordinary residue naming this element, if it lies in the prime
    /// subfield.
    ///
    /// Returns `Some(c)` exactly when `self` equals `c · 1` for `c` below
    /// the characteristic, and `None` otherwise. At degree one always
    /// `Some`.
    #[must_use]
    fn to_residue(self) -> Option<u64>;
    /// The element named by a presentation index.
    ///
    /// The inverse of [`FieldElem::to_index`]: the flattened
    /// prime-subfield coordinates are the base-characteristic digits of
    /// `index`, least significant first. Residues and indices are distinct:
    /// the index of one is the presentation's coordinate encoding of one.
    ///
    /// # Errors
    ///
    /// Returns [`RangeError`] when `index` is at or above the field order.
    fn from_index(index: u128) -> Result<Self, RangeError>;
    /// The presentation index of this element.
    ///
    /// `Σ_i d_i · P^i` over the flattened prime-subfield coordinates in
    /// descriptor order, a bijection between the field and `0..ORDER`. A
    /// prime row's index is its ordinary residue; a binary row's index is
    /// its coordinate word.
    #[must_use]
    fn to_index(self) -> u128;
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

    #[inline]
    fn from_residue(value: u64) -> Result<Self, RangeError> {
        let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
        let () = <F::Characteristic as PrimeIdentity>::VALID;
        check_residue(value, characteristic)?;
        Ok(Elem {
            raw: F::canonical_raw(F::from_prime(value)),
        })
    }

    #[inline]
    fn to_residue(self) -> Option<u64> {
        F::to_prime(self.raw)
    }

    #[inline]
    fn from_index(index: u128) -> Result<Self, RangeError> {
        if index >= F::ORDER {
            return Err(RangeError);
        }
        let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
        let coordinates = index_coordinates(index, F::DEGREE, characteristic);
        Ok(Elem {
            raw: F::canonical_raw(F::from_prime_coordinates(
                &coordinates[..F::DEGREE as usize],
            )),
        })
    }

    #[inline]
    fn to_index(self) -> u128 {
        use crate::field::extension::descriptor::COORDINATES;
        let characteristic = <F::Characteristic as PrimeIdentity>::CHARACTERISTIC;
        let mut coordinates = [0u64; COORDINATES];
        F::prime_coordinates_into(&mut coordinates[..F::DEGREE as usize], self.raw);
        coordinates_index(&coordinates[..F::DEGREE as usize], characteristic)
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
/// scratch. The trait is sealed: the crate's rows define every byte layout.
#[allow(private_bounds)]
pub trait FieldBuffer: Field + private::FieldSeal + private::BufferDispatch {
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

impl<const P: u64, const N: u32, R: FiniteFieldRepr<P, N>> private::FieldSeal
    for FiniteField<P, N, R>
{
}

/// An integer outside the range a conversion accepts.
///
/// Returned when a value does not name an element of the target field: a
/// residue at or above the characteristic, or an index at or above the field
/// order. The unit carries no payload; tests compare by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeError;

impl core::fmt::Display for RangeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "value is outside the field's range")
    }
}

impl core::error::Error for RangeError {}

#[cfg(test)]
mod tests {
    //! Cross-row checks of the row hooks, at the common parent of the rows:
    //! descriptor agreement against independently built shipped-instance
    //! descriptors, flattened prime-subfield coordinates, and the
    //! prime-value embedding, against expectations derived by remainder.

    use super::binary::tower::{
        FanPaar2Field, FanPaar4Field, FanPaar8Field, FanPaar16Field, FanPaar32Field,
        FanPaar64Field, Rijndael16Field, Rijndael32Field, Rijndael64Field,
    };
    use super::extension::Descriptor;
    use super::extension::pinned;
    use super::private::RowAccess as _;
    use super::*;

    /// The largest `u64` prime, for full-width residue samples.
    const WIDEST_PRIME: u64 = 0xFFFF_FFFF_FFFF_FFC5;

    /// Exercises the coordinate hooks on one raw word — canonical or an
    /// alias of it — against an independently computed coordinate vector
    /// and canonical expectation: the write canonicalizes, and the read
    /// rebuilds the canonical word exactly, with no canonicalization of
    /// the output.
    fn coordinates_round_trip<F: Field>(raw: F::Raw, expected: &[u64], canonical: F::Raw) {
        let degree = F::DEGREE as usize;
        assert_eq!(expected.len(), degree, "expectation arity");
        let mut buffer = [u64::MAX; 64];
        F::prime_coordinates_into(&mut buffer[..degree], raw);
        assert_eq!(&buffer[..degree], expected, "coordinates of {raw:?}");
        let rebuilt = F::from_prime_coordinates(&buffer[..degree]);
        assert_eq!(rebuilt, canonical, "round trip of {raw:?}");
    }

    /// Every element of a residue row, canonical and as the alias one
    /// modulus up: the residue is the coordinate, the prime readout covers
    /// the whole field, and `from_prime` reduces.
    macro_rules! residue_row_exhaustive {
        ($p:literal) => {{
            type Row = FiniteField<$p, 1>;
            for value in 0..$p {
                coordinates_round_trip::<Row>(value, &[value], value);
                assert_eq!(Row::to_prime(value), Some(value), "prime readout");
                let alias = value + $p;
                coordinates_round_trip::<Row>(alias, &[value], value);
                assert_eq!(Row::to_prime(alias), Some(value), "alias readout");
            }
            let widest = u64::MAX % $p;
            coordinates_round_trip::<Row>(u64::MAX, &[widest], widest);
            assert_eq!(Row::to_prime(u64::MAX), Some(widest), "widest readout");
            let large = $p + 5 * $p;
            assert_eq!(Row::from_prime(large), large % $p, "from_prime reduces");
            assert_eq!(Row::from_prime(u64::MAX), widest, "from_prime widest");
        }};
    }

    #[test]
    fn residue_rows_round_trip_every_element() {
        residue_row_exhaustive!(2);
        residue_row_exhaustive!(3);
        residue_row_exhaustive!(5);
        residue_row_exhaustive!(7);
        residue_row_exhaustive!(251);
    }

    /// Every element of one enumerable binary row: the coordinate word's
    /// bits are the coordinates, and the prime readout covers exactly
    /// `{0, 1}`.
    macro_rules! binary_row_exhaustive {
        ($row:ty, $n:literal) => {{
            type Row = $row;
            let one = Elem::<Row>::ONE.to_raw();
            let mut expected = [0u64; 64];
            for word in 0..(1u64 << $n) {
                let element =
                    Elem::<Row>::from_index(u128::from(word)).expect("word fits the degree");
                let raw = element.to_raw();
                for (i, coordinate) in expected.iter_mut().enumerate().take($n) {
                    *coordinate = (word >> i) & 1;
                }
                coordinates_round_trip::<Row>(raw, &expected[..$n], raw);
                let readout = Row::to_prime(raw);
                let prime_member = if element == Elem::<Row>::ZERO {
                    Some(0)
                } else if element == Elem::<Row>::ONE {
                    Some(1)
                } else {
                    None
                };
                assert_eq!(readout, prime_member, "prime readout of word {word}");
                if let Some(c) = prime_member {
                    assert_eq!(Elem::<Row>::from_raw(Row::from_prime(c)), element);
                }
            }
            assert_eq!(Row::from_prime(0), Elem::<Row>::ZERO.to_raw());
            assert_eq!(Row::from_prime(2), Elem::<Row>::ZERO.to_raw());
            assert_eq!(Row::from_prime(1), one);
            assert_eq!(Row::from_prime(u64::MAX), one, "odd values reduce to one");
        }};
    }

    #[test]
    fn binary_rows_round_trip_every_element() {
        binary_row_exhaustive!(FiniteField<2, 1, Polynomial<3>>, 1);
        binary_row_exhaustive!(FiniteField<2, 2, Polynomial<7>>, 2);
        binary_row_exhaustive!(FiniteField<2, 2, Normal<7, 2>>, 2);
        binary_row_exhaustive!(FiniteField<2, 2, Cantor<7, 2>>, 2);
        binary_row_exhaustive!(FiniteField<2, 4, Polynomial<0x13>>, 4);
        binary_row_exhaustive!(FiniteField<2, 4, Normal<0x13, 8>>, 4);
        binary_row_exhaustive!(FiniteField<2, 4, Cantor<0x13, 8>>, 4);
        binary_row_exhaustive!(FiniteField<2, 8, Polynomial<AES>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Polynomial<RS>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Polynomial<0x12D>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Polynomial<0x187>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Normal<0x11B, 0x20>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Normal<0x11D, 0x20>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Cantor<0x11B, 0x20>>, 8);
        binary_row_exhaustive!(FiniteField<2, 8, Cantor<0x11D, 0x20>>, 8);
    }

    /// Every word of a sub-byte binary row with every bit above the degree
    /// set: the hooks read the canonical word's coordinates and readout.
    fn sub_byte_aliases<F: Field<Raw = u8>>() {
        let degree = F::DEGREE;
        let high = !((1u8 << degree) - 1);
        let mut expected = [0u64; 8];
        for word in 0..(1u8 << degree) {
            for (i, coordinate) in expected.iter_mut().enumerate().take(degree as usize) {
                *coordinate = u64::from(word >> i) & 1;
            }
            let alias = word | high;
            coordinates_round_trip::<F>(alias, &expected[..degree as usize], word);
            let member = if word == 0 {
                Some(0)
            } else if word == F::ONE_RAW {
                Some(1)
            } else {
                None
            };
            assert_eq!(F::to_prime(alias), member, "readout of alias {alias:#x}");
        }
    }

    #[test]
    fn sub_byte_binary_rows_read_aliases_canonically() {
        sub_byte_aliases::<FiniteField<2, 1, Polynomial<3>>>();
        sub_byte_aliases::<FiniteField<2, 2, Polynomial<7>>>();
        sub_byte_aliases::<FiniteField<2, 2, Normal<7, 2>>>();
        sub_byte_aliases::<FiniteField<2, 2, Cantor<7, 2>>>();
        sub_byte_aliases::<FiniteField<2, 4, Polynomial<0x13>>>();
        sub_byte_aliases::<FiniteField<2, 4, Normal<0x13, 8>>>();
        sub_byte_aliases::<FiniteField<2, 4, Cantor<0x13, 8>>>();
        sub_byte_aliases::<FanPaar2Field>();
        sub_byte_aliases::<FanPaar4Field>();
    }

    #[test]
    fn tower_rows_round_trip_every_element() {
        binary_row_exhaustive!(FanPaar2Field, 2);
        binary_row_exhaustive!(FanPaar4Field, 4);
        binary_row_exhaustive!(FanPaar8Field, 8);
        // Sixty-five thousand elements each.
        binary_row_exhaustive!(Rijndael16Field, 16);
        binary_row_exhaustive!(FanPaar16Field, 16);
    }

    /// A spread of full-width binary words: boundaries and a deterministic
    /// spray, checked through the same expectations as the enumerable rows.
    macro_rules! binary_row_samples {
        ($row:ty, $n:literal, [$($word:expr),+]) => {{
            type Row = $row;
            let one = Elem::<Row>::ONE.to_raw();
            let mut expected = [0u64; 64];
            $(
                let word: u64 = $word;
                let element = Elem::<Row>::from_index(u128::from(word)).expect("sample fits the degree");
                let raw = element.to_raw();
                for (i, coordinate) in expected.iter_mut().enumerate().take($n) {
                    *coordinate = (word >> i) & 1;
                }
                coordinates_round_trip::<Row>(raw, &expected[..$n], raw);
                let prime_member = if element == Elem::<Row>::ZERO {
                    Some(0)
                } else if element == Elem::<Row>::ONE {
                    Some(1)
                } else {
                    None
                };
                assert_eq!(Row::to_prime(raw), prime_member, "word {word}");
            )+
            assert_eq!(Row::from_prime(u64::MAX), one, "odd values reduce to one");
        }};
    }

    #[test]
    fn wide_binary_rows_round_trip_samples() {
        binary_row_samples!(
            Rijndael32Field,
            32,
            [
                0,
                1,
                2,
                0x0001_0002,
                0xa55a_1234,
                0x7fff_ffff,
                u64::from(u32::MAX)
            ]
        );
        binary_row_samples!(
            Rijndael64Field,
            64,
            [
                0,
                1,
                2,
                0x0000_0001_0000_0004,
                0xa55a_1234_dead_beef,
                u64::MAX
            ]
        );
        binary_row_samples!(FanPaar32Field, 32, [0, 1, 0x0100, 0x03e2_1cea, 0xffff_ffff]);
        binary_row_samples!(
            FanPaar64Field,
            64,
            [0, 1, 0x0001_0000, 0x070f_870d_cd9c_1d88, u64::MAX]
        );
    }

    #[test]
    fn odd_rows_read_unreduced_lanes_by_remainder() {
        type M31 = Mersenne31;
        type Gld = Goldilocks;
        type Qm = QuadMersenne31;
        const M31_MODULUS: u64 = 0x7FFF_FFFF;
        const GLD_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

        // The plain remainder is the independent reduction: every raw
        // lane below is handed to the hooks unreduced.
        for sample in [0u32, 1, 2, 0x5555_5555, 0x7FFF_FFFE, 0x7FFF_FFFF, u32::MAX] {
            let expected = u64::from(sample) % M31_MODULUS;
            // The remainder sits below the modulus, so it fits the lane.
            #[allow(clippy::cast_possible_truncation)]
            let canonical = expected as u32;
            coordinates_round_trip::<M31>(sample, &[expected], canonical);
            assert_eq!(M31::to_prime(sample), Some(expected), "lane {sample:#x}");
        }
        #[allow(clippy::cast_lossless)]
        let widest_m31 = M31::from_prime(u64::MAX) as u64;
        assert_eq!(widest_m31, u64::MAX % M31_MODULUS);

        for sample in [
            0u64,
            1,
            2,
            GLD_MODULUS - 1,
            GLD_MODULUS,
            GLD_MODULUS + 1,
            u64::MAX,
        ] {
            let expected = sample % GLD_MODULUS;
            coordinates_round_trip::<Gld>(sample, &[expected], expected);
            assert_eq!(Gld::to_prime(sample), Some(expected), "lane {sample:#x}");
        }
        assert_eq!(Gld::from_prime(u64::MAX), u64::MAX % GLD_MODULUS);

        // An unreduced limb aliases its remainder: a limb at the modulus
        // reads as zero, so the element stays in the prime subfield.
        for [re, im] in [
            [0u32, 0],
            [1, 0],
            [0, 1],
            [0x7FFF_FFFE, 0],
            [0x7FFF_FFFF, 0x7FFF_FFFF],
            [0x5555_5555, 0x5555_5555],
            [u32::MAX, 0],
            [u32::MAX, u32::MAX],
        ] {
            let expected = [u64::from(re) % M31_MODULUS, u64::from(im) % M31_MODULUS];
            // Both remainders sit below the modulus, so they fit the
            // lanes.
            #[allow(clippy::cast_possible_truncation)]
            let canonical = [expected[0] as u32, expected[1] as u32];
            coordinates_round_trip::<Qm>([re, im], &expected, canonical);
            let readout = if expected[1] == 0 {
                Some(expected[0])
            } else {
                None
            };
            assert_eq!(Qm::to_prime([re, im]), readout, "lane [{re:#x}, {im:#x}]");
        }
        let widest_qm = Qm::from_prime(u64::MAX);
        assert_eq!(u64::from(widest_qm[0]), u64::MAX % M31_MODULUS);
        assert_eq!(widest_qm[1], 0);
    }

    #[test]
    fn extension_rows_round_trip_components_and_prime_readout() {
        struct Spec;
        impl ExtensionSpec<2> for Spec {
            type Base = FiniteField<7, 1>;
            type Storage = Components;
            const NAME: &'static str = "GF(49) hook row";
            const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
        }
        type Row = FiniteField<7, 2, Extension<Spec, 2>>;
        // Every element: the components are the coordinates in component
        // order, and the prime readout covers exactly the base row.
        for a0 in 0..7u64 {
            for a1 in 0..7u64 {
                let raw = [a0, a1];
                coordinates_round_trip::<Row>(raw, &[a0, a1], raw);
                let readout = (a1 == 0).then_some(a0);
                assert_eq!(Row::to_prime(raw), readout, "components ({a0}, {a1})");
            }
        }
        assert_eq!(Row::from_prime(9), [2, 0]);
        assert_eq!(Row::from_prime(u64::MAX), [u64::MAX % 7, 0]);
    }

    #[test]
    fn extension_row_hooks_canonicalize_the_prime_readout() {
        struct Gf49Spec;
        impl ExtensionSpec<2> for Gf49Spec {
            type Base = FiniteField<7, 1>;
            type Storage = Components;
            const NAME: &'static str = "GF(49) hook row";
            const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
        }
        type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

        struct Gf4Spec;
        impl ExtensionSpec<2> for Gf4Spec {
            type Base = FiniteField<2, 1, Polynomial<3>>;
            type Storage = PackedPadded<u8>;
            const NAME: &'static str = "GF(4) hook row";
            const RELATION: Relation =
                Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[1, 1]);
        }
        type Gf4Row = FiniteField<2, 2, Extension<Gf4Spec, 2>>;

        struct Gf8Spec;
        impl ExtensionSpec<3> for Gf8Spec {
            type Base = FiniteField<2, 1, Polynomial<3>>;
            type Storage = PackedPadded<u8>;
            const NAME: &'static str = "GF(8) hook row";
            const RELATION: Relation =
                Relation::from_binary_words::<FiniteField<2, 1, Polynomial<3>>>(&[1, 1, 0]);
        }
        type Gf8Row = FiniteField<2, 3, Extension<Gf8Spec, 3>>;
        struct Gf64Spec;
        impl ExtensionSpec<2> for Gf64Spec {
            type Base = Gf8Row;
            type Storage = PackedPadded<u8>;
            const NAME: &'static str = "GF(64) hook row";
            const RELATION: Relation = Relation::from_binary_words::<Gf8Row>(&[1, 2]);
        }
        type Gf64Row = FiniteField<2, 6, Extension<Gf64Spec, 2>>;

        // Component storage: an aliased base component reduces before the
        // readout.
        assert_eq!(Gf49::to_prime([3, 7]), Some(3), "aliased base component");
        assert_eq!(Gf49::to_prime([7, 0]), Some(0), "aliased whole word");
        assert_eq!(Gf49::to_prime([3, 1]), None, "off the prime subfield");

        // Packed storage: the padding bits of a padded word clear before
        // the readout.
        assert_eq!(Gf4Row::to_prime(0b101), Some(1), "padding above the degree");
        assert_eq!(
            Gf4Row::to_prime(0b10),
            None,
            "generator off the prime subfield"
        );

        // A nested extension row: the outer word canonicalizes through the
        // base row's own canonicalization.
        assert_eq!(
            Gf64Row::to_prime(0b1100_0001),
            Some(1),
            "padding above six bits"
        );
        assert_eq!(Gf64Row::to_prime(0b000_010), None, "base generator readout");
        assert_eq!(
            Gf64Row::to_prime(0b001_001),
            None,
            "nonzero upper component"
        );
    }

    #[test]
    fn widest_residue_row_round_trip_boundaries() {
        type Row = FiniteField<WIDEST_PRIME, 1>;
        for sample in [
            0u64,
            1,
            2,
            WIDEST_PRIME - 1,
            WIDEST_PRIME,
            WIDEST_PRIME + 1,
            u64::MAX,
        ] {
            let residue = sample % WIDEST_PRIME;
            coordinates_round_trip::<Row>(sample, &[residue], residue);
            assert_eq!(Row::to_prime(sample), Some(residue));
        }
        assert_eq!(Row::from_prime(u64::MAX), u64::MAX % WIDEST_PRIME);
    }

    /// Descriptor agreement for one row: its own facts, and its expected
    /// descriptor built from the shipped-instance table where the table
    /// holds an entry for the row.
    macro_rules! descriptor_agreement {
        ($row:ty $(, expected $expected:expr)?) => {{
            type Row = $row;
            assert_eq!(Row::DESCRIPTOR.degree(), Row::DEGREE);
            assert_eq!(
                Row::DESCRIPTOR.characteristic(),
                <Row as Field>::Characteristic::CHARACTERISTIC
            );
            $(
                let expected: Descriptor = $expected;
                assert!(
                    Row::DESCRIPTOR.same_structure(&expected),
                    "descriptor is not the expected shipped instance"
                );
            )?
        }};
    }

    #[test]
    fn row_descriptors_agree_with_their_rows_and_the_instance_table() {
        descriptor_agreement!(FiniteField<7, 1>);
        descriptor_agreement!(FiniteField<WIDEST_PRIME, 1>);
        descriptor_agreement!(Mersenne31);
        descriptor_agreement!(Goldilocks);
        descriptor_agreement!(
            QuadMersenne31,
            expected pinned::shipped_instances::quad_mersenne31()
        );
        descriptor_agreement!(FiniteField<2, 1, Polynomial<3>>);
        descriptor_agreement!(FiniteField<2, 2, Polynomial<7>>);
        descriptor_agreement!(FiniteField<2, 2, Normal<7, 2>>);
        descriptor_agreement!(FiniteField<2, 2, Cantor<7, 2>>);
        descriptor_agreement!(FiniteField<2, 4, Polynomial<0x13>>);
        descriptor_agreement!(FiniteField<2, 4, Normal<0x13, 8>>);
        descriptor_agreement!(FiniteField<2, 4, Cantor<0x13, 8>>);
        descriptor_agreement!(FiniteField<2, 8, Polynomial<AES>>);
        descriptor_agreement!(FiniteField<2, 8, Polynomial<RS>>);
        descriptor_agreement!(
            Rijndael16Field,
            expected pinned::shipped_instances::rijndael16()
        );
        descriptor_agreement!(
            Rijndael32Field,
            expected pinned::shipped_instances::rijndael32()
        );
        descriptor_agreement!(
            Rijndael64Field,
            expected pinned::shipped_instances::rijndael64()
        );
        descriptor_agreement!(
            FanPaar2Field,
            expected pinned::shipped_instances::fanpaar2()
        );
        descriptor_agreement!(
            FanPaar4Field,
            expected pinned::shipped_instances::fanpaar4()
        );
        descriptor_agreement!(
            FanPaar8Field,
            expected pinned::shipped_instances::fanpaar8()
        );
        descriptor_agreement!(
            FanPaar16Field,
            expected pinned::shipped_instances::fanpaar16()
        );
        descriptor_agreement!(
            FanPaar32Field,
            expected pinned::shipped_instances::fanpaar32()
        );
        descriptor_agreement!(
            FanPaar64Field,
            expected pinned::shipped_instances::fanpaar64()
        );

        // One modulus under two presentations is two different rows.
        assert!(!FiniteField::<0x7FFF_FFFF, 1>::DESCRIPTOR.same_structure(Mersenne31::DESCRIPTOR));
        assert!(
            FiniteField::<0x7FFF_FFFF, 1>::DESCRIPTOR
                .same_structure(FiniteField::<0x7FFF_FFFF, 1>::DESCRIPTOR)
        );
    }

    #[test]
    #[should_panic(expected = "prime coordinate count differs from the field degree")]
    fn coordinate_write_rejects_a_wrong_buffer_length() {
        FiniteField::<7, 1>::prime_coordinates_into(&mut [0u64; 2], 3);
    }

    #[test]
    #[should_panic(expected = "prime coordinate count differs from the field degree")]
    fn coordinate_read_rejects_a_wrong_buffer_length() {
        QuadMersenne31::from_prime_coordinates(&[0u64; 1]);
    }

    #[test]
    #[should_panic(expected = "prime coordinate is not below the characteristic")]
    fn coordinate_read_rejects_a_coordinate_at_the_characteristic() {
        FiniteField::<7, 1>::from_prime_coordinates(&[7]);
    }

    #[test]
    #[should_panic(expected = "prime coordinate is not below the characteristic")]
    fn coordinate_read_rejects_a_binary_coordinate_above_one() {
        FiniteField::<2, 4, Polynomial<0x13>>::from_prime_coordinates(&[0, 2, 0, 0]);
    }

    /// The coordinate word of one element equals its prime-coordinate
    /// bits: bit `i` is coordinate `i` in descriptor order.
    fn word_matches_coordinates<F: Field>(raw: F::Raw) {
        let degree = F::DEGREE as usize;
        let canonical = F::canonical_raw(raw);
        let word = F::coordinate_word(raw);
        let mut coordinates = [0u64; 64];
        F::prime_coordinates_into(&mut coordinates[..degree], canonical);
        let mut expected = 0u64;
        for (i, coordinate) in coordinates[..degree].iter().enumerate() {
            assert!(*coordinate < 2, "binary coordinate above one");
            expected |= coordinate << i;
        }
        assert_eq!(word, expected, "coordinate word of {raw:?}");
        assert_eq!(
            F::from_coordinate_word(word),
            canonical,
            "word round trip of {raw:?}"
        );
    }

    /// Every element of one enumerable binary row: the word hook agrees
    /// with the coordinate hooks exactly.
    macro_rules! word_row_exhaustive {
        ($row:ty, $n:literal) => {{
            type Row = $row;
            for index in 0..(1u128 << $n) {
                let element = Elem::<Row>::from_index(index).expect("index fits the degree");
                word_matches_coordinates::<Row>(element.to_raw());
            }
        }};
    }

    #[test]
    fn coordinate_words_match_prime_coordinates() {
        word_row_exhaustive!(FiniteField<2, 1, Polynomial<3>>, 1);
        word_row_exhaustive!(FiniteField<2, 2, Polynomial<0x7>>, 2);
        word_row_exhaustive!(FiniteField<2, 2, Normal<7, 2>>, 2);
        word_row_exhaustive!(FiniteField<2, 2, Cantor<7, 2>>, 2);
        word_row_exhaustive!(FiniteField<2, 4, Polynomial<0x13>>, 4);
        word_row_exhaustive!(FiniteField<2, 4, Normal<0x13, 8>>, 4);
        word_row_exhaustive!(FiniteField<2, 8, Polynomial<AES>>, 8);
        word_row_exhaustive!(FiniteField<2, 8, Polynomial<RS>>, 8);
        word_row_exhaustive!(FiniteField<2, 8, Normal<0x11B, 0x20>>, 8);
        word_row_exhaustive!(FanPaar2Field, 2);
        word_row_exhaustive!(FanPaar4Field, 4);
        word_row_exhaustive!(FanPaar8Field, 8);
        word_row_exhaustive!(FiniteField<2, 1, Residue>, 1);
        word_row_exhaustive!(Rijndael16Field, 16);
        // Sampled words at the wider tower degrees, plus edge words.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut sample = |mask: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) & mask
        };
        for _ in 0..256 {
            // The mask keeps only the storage width, so the narrowing
            // conversion holds exactly the sampled bits.
            let narrow32 = u32::try_from(sample(u64::from(u32::MAX)))
                .expect("masked sample fits the storage width");
            word_matches_coordinates::<Rijndael32Field>(narrow32);
            word_matches_coordinates::<Rijndael64Field>(
                sample(u64::MAX) ^ (sample(u64::MAX) << 32),
            );
            let narrow16 = u16::try_from(sample(u64::from(u16::MAX)))
                .expect("masked sample fits the storage width");
            word_matches_coordinates::<FanPaar16Field>(narrow16);
            word_matches_coordinates::<FanPaar32Field>(narrow32);
            word_matches_coordinates::<FanPaar64Field>(sample(u64::MAX) ^ (sample(u64::MAX) << 32));
        }
        word_matches_coordinates::<Rijndael32Field>(0);
        word_matches_coordinates::<Rijndael32Field>(u32::MAX);
        word_matches_coordinates::<Rijndael64Field>(0);
        word_matches_coordinates::<Rijndael64Field>(u64::MAX);
        word_matches_coordinates::<FanPaar64Field>(0);
        word_matches_coordinates::<FanPaar64Field>(u64::MAX);
    }

    #[test]
    #[should_panic(expected = "coordinate word on a row of odd characteristic")]
    fn coordinate_word_rejects_an_odd_row() {
        let _ = FiniteField::<7, 1>::coordinate_word(3);
    }

    #[test]
    #[should_panic(expected = "coordinate word on a row of odd characteristic")]
    fn from_coordinate_word_rejects_an_odd_row() {
        let _ = FiniteField::<7, 1>::from_coordinate_word(1);
    }

    #[test]
    #[should_panic(expected = "coordinate word holds bits at or above the field degree")]
    fn from_coordinate_word_rejects_excess_bits() {
        let _ = FiniteField::<2, 4, Polynomial<0x13>>::from_coordinate_word(1 << 4);
    }
}
