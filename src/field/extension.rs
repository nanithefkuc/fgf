//! The extension row implementation.
//!
//! The open `ExtensionSpec` provider contract, the storage representations,
//! the relation, and the generic arithmetic every extension row shares. The
//! public surface ships through the crate's `extension` facade; the
//! submodules hold the descriptor machinery, irreducibility validation,
//! pinned shipped instances, and the row hooks the kernel dispatch consumes.

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
/// At relative degree two the closed-form conjugate, trace, and norm of
/// the row's defining relation back the corresponding
/// [`Embedding`](super::embedding::Embedding) map of the base-into-row
/// embedding; see its `frobenius`, `trace`, and `norm` methods.
pub trait ExtensionField<const D: usize>: Field + private::Sealed {
    /// The base row this field extends.
    type Base: Field;

    /// The element with the given components, canonicalized on the way
    /// in.
    fn from_components(components: [Elem<Self::Base>; D]) -> Elem<Self>;

    /// The components of an element, in component order.
    fn to_components(value: Elem<Self>) -> [Elem<Self::Base>; D];
}
