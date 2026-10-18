//! General field extensions: one presentation over every canonical row.
//!
//! [`FiniteField<P, N, Extension<S, D>>`](super::FiniteField) names
//! GF(P^N) as a relative degree-`D` extension of the base row `S::Base`.
//! [`ExtensionSpec`] is open and data-only: a base row, a storage
//! selection, a name, and the relation. The defining relation is
//! validated in const at first value use — the field metadata first
//! (prime characteristic, binary degree cap, cardinality bound), then
//! the base's validity, the characteristic agreement, the degree
//! identity, the relation's length and coordinate range, storage
//! admissibility, the descriptor capacity, and finally the
//! irreducibility of the relation over its base — so an invalid row fails
//! to compile in const contexts and panics at first use otherwise.
//!
//! # Storage and layout
//!
//! [`Components`] keeps one base element per component; over a byte-backed
//! base the stable byte form concatenates the base encodings.
//! [`Packed<W>`](Packed) and [`PackedPadded<W>`](PackedPadded) keep the
//! row's coordinate word in one of the sealed [`PackedWord`] integers,
//! component `i` at the bits `[i*B, (i+1)*B)` of the base degree `B`, and
//! require a binary base. `Packed<W>` fills its word exactly and encodes
//! little-endian; `PackedPadded<W>` keeps the coordinate word below its
//! word width with every padding bit zero and has no byte form. Every
//! row's raw word, bytes, and coordinate word are canonical: constructors
//! canonicalize components through the base and clear padding bits.
//!
//! # Arithmetic and capabilities
//!
//! Every row shares one generic body over the base's arithmetic: component
//! addition, subtraction, and negation; schoolbook products reduced by
//! `t^D = -(c_{D-1} t^{D-1} + ... + c_0)`; inversion by the conjugate-norm
//! formula at relative degree two and by extended Euclid above it. Zero
//! maps to zero under inversion and division, and `0^0 = 1`. Elements are
//! [`Elem`] values with the crate-wide operators;
//! [`ExtensionField::from_components`] and
//! [`ExtensionField::to_components`] convert against the base. Generic and
//! custom rows carry const identities and `to_raw`, and no const element
//! arithmetic. Rows with a byte form implement
//! [`FieldBuffer`](super::FieldBuffer) and the packed operation surface
//! over [`crate::kernel::FieldKernels`], reporting the scalar backend.
//!
//! The shipped instances — the Rijndael and Fan-Paar towers in
//! [`tower`](super::tower) and
//! [`QuadMersenne31`](super::quad_mersenne31::QuadMersenne31) — are rows of
//! this family with specialized scalar bodies, a full `const` element
//! surface, a checked generator, and their vector kernels. A row whose
//! descriptor equals a shipped instance's — the same base presentations,
//! relations, and storages at every level — runs that instance's body and
//! kernels, so equal presentations produce equal bytes.
//!
//! # Capacity
//!
//! Validation is exact, and a relation whose validation exceeds rustc's
//! `long_running_const_eval` limit fails to compile as a capacity limit
//! rather than a reducibility verdict; fgf does not `allow` that lint on
//! its own items, so an over-budget relation stays a compile error. Deep
//! towers also cost compiler time within the limit.
//!
//! # Examples
//!
//! GF(49) as `t^2 - 3` over GF(7):
//!
//! ```
//! use fgf::{
//!     Components, Elem, Extension, ExtensionField, ExtensionSpec, FieldElem, FiniteField,
//!     Relation,
//! };
//!
//! struct Gf49Spec;
//! impl ExtensionSpec<2> for Gf49Spec {
//!     type Base = FiniteField<7, 1>;
//!     type Storage = Components;
//!     const NAME: &'static str = "GF(49)";
//!     // `t^2 - 3`: constant coefficient `4 = -3` first, then the linear
//!     // coefficient.
//!     const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
//! }
//!
//! type Gf49 = FiniteField<7, 2, Extension<Gf49Spec, 2>>;
//!
//! // The identities are const, over the base identities.
//! const ONE: Elem<Gf49> = Elem::<Gf49>::ONE;
//!
//! // `t^2 = 3` in components: `(0, 1)^2 = (3, 0)`.
//! let t = <Gf49 as ExtensionField<2>>::from_components([
//!     Elem::<FiniteField<7, 1>>::ZERO,
//!     Elem::<FiniteField<7, 1>>::ONE,
//! ]);
//! let square = t * t;
//! let components = <Gf49 as ExtensionField<2>>::to_components(square);
//! assert_eq!(components[0].to_raw(), 3);
//! assert_eq!(components[1].to_raw(), 0);
//!
//! // Every nonzero element inverts, and the defining relation holds.
//! let inverse = (t + ONE).inv();
//! assert_eq!((t + ONE) * inverse, ONE);
//! ```

use core::marker::PhantomData;

use super::{Elem, Field};

pub(crate) mod arithmetic;
pub(crate) mod descriptor;
pub(crate) mod irreducibility;
pub(crate) mod pinned;
pub(crate) mod relation;
pub(crate) mod row;
pub(crate) mod storage;

pub use relation::Relation;
pub use storage::{Components, ExtensionStorage, Packed, PackedPadded, PackedWord};

pub(crate) use descriptor::{Descriptor, Node};

mod private {
    pub trait Sealed {}
}

/// The data-only provider of one extension row.
///
/// A spec names its base row, its storage, its field name, and its
/// defining relation; it carries no arithmetic, kernel, or generator data.
/// The base is any [`Field`] row fgf implements, and
/// the relation is the monic polynomial
/// `t^D + c_{D-1} t^{D-1} + ... + c_0` over it.
#[allow(private_bounds)]
pub trait ExtensionSpec<const D: usize>: 'static {
    /// The base row this field extends.
    type Base: Field;
    /// The element storage of the row.
    type Storage: ExtensionStorage<Self::Base, D> + storage::StorageRepr<Self::Base, D>;
    /// Human-readable field name; the row's `Field::NAME`.
    const NAME: &'static str;
    /// The defining relation's non-leading coefficients as flattened
    /// prime-subfield coordinates, constant term first, each coefficient's
    /// coordinates in the base's own order.
    const RELATION: Relation;
}

/// The extension presentation over a spec at relative degree `D`.
///
/// A field is the canonical row
/// `FiniteField<P, N, Extension<S, D>>` with `D` at least two and `N` the
/// product of the base's absolute degree and `D`; both are checked by the
/// row's validation. The marker whose `D` matches no
/// [`ExtensionSpec<D>`](ExtensionSpec) implementation fails the trait
/// bound.
pub struct Extension<S: ExtensionSpec<D>, const D: usize>(PhantomData<fn() -> S>);

impl<const D: usize, S: ExtensionSpec<D>> super::private::Sealed for Extension<S, D> {}

// Manual marker impls: derives would bound the traits on `S`, which the
// spec contract does not supply. The markers implement `Clone`, `Copy`,
// `Default`, and `Debug` only.
impl<S: ExtensionSpec<D>, const D: usize> Clone for Extension<S, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: ExtensionSpec<D>, const D: usize> Copy for Extension<S, D> {}

impl<S: ExtensionSpec<D>, const D: usize> Default for Extension<S, D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S: ExtensionSpec<D>, const D: usize> core::fmt::Debug for Extension<S, D> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(S::NAME)
    }
}

/// The component capability of an extension row.
///
/// Component `i` is the coefficient of `t^i` over the row's base. The
/// conversions are exact inverses over canonical elements and canonicalize
/// every component on the way in.
///
/// At relative degree two the quadratic conveniences live on
/// [`QuadraticExtension`]: the closed-form conjugate, trace, and norm of
/// the row's defining relation, each equal to the corresponding
/// [`Embedding`](super::embedding::Embedding) map of the base-into-row
/// embedding.
pub trait ExtensionField<const D: usize>: Field + private::Sealed {
    /// The base row this field extends.
    type Base: Field;

    /// The element with the given components, canonicalized on the way
    /// in.
    fn from_components(components: [Elem<Self::Base>; D]) -> Elem<Self>;

    /// The components of an element, in component order.
    fn to_components(value: Elem<Self>) -> [Elem<Self::Base>; D];
}

/// The closed-form quadratic conveniences of a relative-degree-2 row.
///
/// The defining relation reads `t^2 + c1*t + c0` over the base, with `c0`
/// the constant coefficient and `c1` the linear one. For `x = a + b*t`
/// the Vieta formulas give the conjugate `(a - c1*b) - b*t`, the trace
/// `2*a - c1*b`, and the norm `a^2 - c1*a*b + c0*b^2`.
///
/// Each equals the corresponding
/// [`Embedding`](super::embedding::Embedding) map of the base-into-row
/// embedding: [`conjugate`](QuadraticExtension::conjugate) is its
/// Frobenius `x^|Base|`, [`trace`](QuadraticExtension::trace) is its
/// trace, and [`norm`](QuadraticExtension::norm) is its norm. In
/// characteristic two subtraction is addition, so the conjugate reads
/// `(a + c1*b) + b*t` and the trace `c1*b`, since `2*a = 0` and `-b = b`.
#[allow(private_bounds)]
pub trait QuadraticExtension: ExtensionField<2> {
    /// The nontrivial automorphism `x^|Base|`: `(a - c1*b) - b*t` for
    /// `x = a + b*t`.
    ///
    /// Equals the Frobenius of the base-into-row embedding.
    fn conjugate(value: Elem<Self>) -> Elem<Self>;

    /// The sum `x + conjugate(x) = 2*a - c1*b` of the Galois conjugates,
    /// in the base row.
    ///
    /// Equals the trace of the base-into-row embedding.
    fn trace(value: Elem<Self>) -> Elem<Self::Base>;

    /// The product `x * conjugate(x) = a^2 - c1*a*b + c0*b^2` of the
    /// Galois conjugates, in the base row.
    ///
    /// Equals the norm of the base-into-row embedding.
    fn norm(value: Elem<Self>) -> Elem<Self::Base>;
}
