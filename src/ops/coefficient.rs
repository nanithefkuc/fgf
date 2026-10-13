//! Prepared coefficients and the storage types that reuse them.
//!
//! [`Coeff`] resolves one coefficient into the host backend's preferred
//! form; [`CoeffVec`] and [`CoeffMatrix`] collect many, source-major for the
//! matrix. The operation modules in [`crate::ops`] consume these through the
//! borrowed [`CoeffRef`] and [`CoeffVecRef`] views, so a preparation is
//! built once and applied without copying backend tables.

use crate::field::Elem;
use crate::kernel::{FieldKernels, KernelDispatch, RawDispatch};

use super::{PreparedCoefficient, PreparedRepr, private};

/// A coefficient already resolved into the host backend's preferred form.
///
/// Build once, use many times. Construction is not free on every field — see
/// the [`ops`](crate::ops) module docs — but it is idempotent, allocates
/// nothing (the resolved form is a plain value), and the result is immutable,
/// so a `Coeff` can be cached alongside a coding matrix for the life of a
/// codec.
///
/// The prepared representation is backend-defined and deliberately opaque:
/// the element it multiplies by is the whole public surface ([`Coeff::value`]).
///
/// Bound to the process's backend at construction. That is not a hazard in
/// practice (the backend is fixed after first use), but it does mean a
/// `Coeff` is not meaningful to serialize — rebuild it from the element.
pub struct Coeff<F: FieldKernels> {
    prepared: <F as KernelDispatch>::Prepared,
}

impl<F: FieldKernels> Coeff<F> {
    /// Resolve `coeff` for this host.
    #[inline]
    #[must_use]
    pub fn new(coeff: Elem<F>) -> Self {
        Self {
            prepared: F::prepare(RawDispatch, coeff),
        }
    }

    /// The field element this was built from.
    #[inline]
    #[must_use]
    pub fn value(&self) -> Elem<F> {
        F::prepared_coeff(RawDispatch, &self.prepared)
    }
}

impl<F: FieldKernels> Clone for Coeff<F> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
        }
    }
}

impl<F: FieldKernels> core::fmt::Debug for Coeff<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Coeff").field(&self.value()).finish()
    }
}

impl<F: FieldKernels> private::Sealed for Coeff<F> {}

impl<F: FieldKernels> PreparedCoefficient<F> for Coeff<F> {
    #[inline]
    fn value(&self) -> Elem<F> {
        Coeff::value(self)
    }
}

impl<F: FieldKernels> PreparedRepr<F> for Coeff<F> {
    #[inline]
    fn repr(&self, _proof: RawDispatch) -> &<F as KernelDispatch>::Prepared {
        &self.prepared
    }
}

/// A borrowed prepared coefficient inside a [`CoeffVec`] or [`CoeffMatrix`].
///
/// Unlike cloning a [`Coeff`], this copies only a pointer even when the
/// backend representation is a full GF(2^16) nibble-table set.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy)]
pub struct CoeffRef<'a, F: FieldKernels> {
    prepared: &'a <F as KernelDispatch>::Prepared,
    field: core::marker::PhantomData<F>,
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> CoeffRef<'_, F> {
    /// The field element this prepared representation multiplies by.
    #[inline]
    #[must_use]
    pub fn value(self) -> Elem<F> {
        F::prepared_coeff(RawDispatch, self.prepared)
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> private::Sealed for CoeffRef<'_, F> {}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> PreparedCoefficient<F> for CoeffRef<'_, F> {
    #[inline]
    fn value(&self) -> Elem<F> {
        F::prepared_coeff(RawDispatch, self.prepared)
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> PreparedRepr<F> for CoeffRef<'_, F> {
    #[inline]
    fn repr(&self, _proof: RawDispatch) -> &<F as KernelDispatch>::Prepared {
        self.prepared
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> core::fmt::Debug for CoeffRef<'_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CoeffRef").field(&self.value()).finish()
    }
}

/// A reusable vector of prepared coefficients: one coefficient per position,
/// with the position's meaning supplied by the consuming operation.
///
/// Positions are output rows for
/// [`mul_add_scatter_with`](crate::ops::mul_add_scatter_with) and sources for
/// the gather operations. A vector has a length and nothing else — it carries
/// no source or output axis, which is exactly why it is a separate type from
/// [`CoeffMatrix`].
///
/// Preparation happens once in [`CoeffVec::new`], which allocates the
/// coefficient storage; [`CoeffVec::get`] then borrows an entry without
/// rebuilding or copying its backend tables, and the operations that consume
/// the view allocate nothing. Store one beside the coding column it
/// represents.
///
/// Requires `alloc`: the collection owns dynamically sized storage. The
/// operations take the borrowed [`CoeffVecRef`] view, which
/// [`CoeffVec::as_ref`] produces for free.
#[cfg(feature = "alloc")]
pub struct CoeffVec<F: FieldKernels> {
    prepared: alloc::boxed::Box<[<F as KernelDispatch>::Prepared]>,
    values: alloc::boxed::Box<[Elem<F>]>,
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> CoeffVec<F> {
    /// Resolve every coefficient for this host.
    #[must_use]
    pub fn new(coeffs: &[Elem<F>]) -> Self {
        Self {
            prepared: prepare_all::<F>(coeffs),
            values: alloc::boxed::Box::from(coeffs),
        }
    }

    /// Borrow the whole vector as the view the operations consume.
    #[inline]
    #[must_use]
    pub fn as_ref(&self) -> CoeffVecRef<'_, F> {
        CoeffVecRef {
            prepared: &self.prepared,
            values: &self.values,
            field: core::marker::PhantomData,
        }
    }

    /// Number of prepared coefficients.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.prepared.len()
    }

    /// Whether the vector contains no coefficients.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Borrow coefficient `index`, or return `None` when out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<CoeffRef<'_, F>> {
        self.as_ref().get(index)
    }

    /// Iterate over prepared coefficients without copying their backend form.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn coeffs(&self) -> impl ExactSizeIterator<Item = CoeffRef<'_, F>> {
        self.as_ref().into_coeffs()
    }

    /// Iterate over the original field elements.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn values(&self) -> impl ExactSizeIterator<Item = Elem<F>> + '_ {
        self.values.iter().copied()
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> Clone for CoeffVec<F> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
            values: self.values.clone(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> core::fmt::Debug for CoeffVec<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CoeffVec")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// A borrowed prepared coefficient vector.
///
/// Every vector-shaped prepared operation takes this, so one type serves both
/// a standalone [`CoeffVec`] and a single source's coefficients inside a
/// [`CoeffMatrix`] — [`CoeffMatrix::source`] hands a matrix row straight to
/// [`mul_add_scatter_with`](crate::ops::mul_add_scatter_with) without copying
/// or restaging anything.
///
/// `Copy`: it is two slice references.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy)]
pub struct CoeffVecRef<'a, F: FieldKernels> {
    pub(super) prepared: &'a [<F as KernelDispatch>::Prepared],
    pub(super) values: &'a [Elem<F>],
    field: core::marker::PhantomData<F>,
}

#[cfg(feature = "alloc")]
impl<'a, F: FieldKernels> CoeffVecRef<'a, F> {
    /// Number of prepared coefficients.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.prepared.len()
    }

    /// Whether the view covers no coefficients.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.prepared.is_empty()
    }

    /// Borrow coefficient `index`, or return `None` when out of bounds.
    #[inline]
    #[must_use]
    pub fn get(self, index: usize) -> Option<CoeffRef<'a, F>> {
        self.prepared.get(index).map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        })
    }

    /// Iterate over prepared coefficients without copying their backend form.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn into_coeffs(self) -> impl ExactSizeIterator<Item = CoeffRef<'a, F>> {
        self.prepared.iter().map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        })
    }

    /// Iterate over the original field elements.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn values(self) -> impl ExactSizeIterator<Item = Elem<F>> + 'a {
        self.values.iter().copied()
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> core::fmt::Debug for CoeffVecRef<'_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CoeffVecRef")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// A reusable source-major matrix of prepared coefficients.
///
/// The matrix is **source-major**: the coefficient pairing source `s` with
/// output row `o` is `coeffs[s * outputs + o]` — `coeff[source][output]` in
/// two-dimensional notation. The constructor is named
/// [`CoeffMatrix::from_source_major`] rather than `matrix` because a square
/// output-major array passed to a source-major constructor is a silent
/// transpose, and the axis has to be stated where the caller writes it.
///
/// Requires `alloc`: construction allocates the coefficient storage, while
/// the operations that consume a [`CoeffMatrix`] borrow it and allocate
/// nothing. Store one beside the coding matrix it represents.
#[cfg(feature = "alloc")]
pub struct CoeffMatrix<F: FieldKernels> {
    pub(super) prepared: alloc::boxed::Box<[<F as KernelDispatch>::Prepared]>,
    pub(super) values: alloc::boxed::Box<[Elem<F>]>,
    sources: usize,
    outputs: usize,
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> CoeffMatrix<F> {
    /// Resolve a `sources × outputs` matrix held in source-major order.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != sources * outputs` or the product overflows.
    #[must_use]
    pub fn from_source_major(sources: usize, outputs: usize, coeffs: &[Elem<F>]) -> Self {
        let len = sources
            .checked_mul(outputs)
            .expect("CoeffMatrix::from_source_major: dimensions overflow");
        assert_eq!(
            coeffs.len(),
            len,
            "CoeffMatrix::from_source_major: {} coefficients for {sources}x{outputs} source-major matrix",
            coeffs.len()
        );
        Self {
            prepared: prepare_all::<F>(coeffs),
            values: alloc::boxed::Box::from(coeffs),
            sources,
            outputs,
        }
    }

    /// Number of source terms the coefficients are grouped by.
    #[inline]
    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.sources
    }

    /// Number of output rows each source's coefficients address.
    #[inline]
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.outputs
    }

    /// Total number of prepared coefficients, `source_count * output_count`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.prepared.len()
    }

    /// Whether the matrix holds no coefficients.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Borrow the coefficient pairing `source` with `output`, or return
    /// `None` when either is out of bounds.
    #[inline]
    #[must_use]
    pub fn get_at(&self, source: usize, output: usize) -> Option<CoeffRef<'_, F>> {
        self.source(source)?.get(output)
    }

    /// Borrow the `output_count()` coefficients pairing `source` with every
    /// output row, or return `None` when `source` is out of bounds.
    ///
    /// The result is the same view the vector-shaped operations take, so a
    /// matrix column drives
    /// [`mul_add_scatter_with`](crate::ops::mul_add_scatter_with) directly.
    ///
    /// Total: every `usize` answers without overflow, `usize::MAX` included,
    /// and a zero-output matrix yields an empty view for each of its sources.
    #[must_use]
    pub fn source(&self, source: usize) -> Option<CoeffVecRef<'_, F>> {
        if source >= self.sources {
            return None;
        }
        // `source < sources` and `sources * outputs == len`, so both bounds
        // are within `len` and cannot overflow.
        let start = source * self.outputs;
        let end = start + self.outputs;
        Some(CoeffVecRef {
            prepared: self.prepared.get(start..end)?,
            values: self.values.get(start..end)?,
            field: core::marker::PhantomData,
        })
    }

    /// Iterate over the original field elements in source-major order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn values(&self) -> impl ExactSizeIterator<Item = Elem<F>> + '_ {
        self.values.iter().copied()
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> Clone for CoeffMatrix<F> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
            values: self.values.clone(),
            sources: self.sources,
            outputs: self.outputs,
        }
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> core::fmt::Debug for CoeffMatrix<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CoeffMatrix")
            .field("sources", &self.sources)
            .field("outputs", &self.outputs)
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
fn prepare_all<F: FieldKernels>(
    coeffs: &[Elem<F>],
) -> alloc::boxed::Box<[<F as KernelDispatch>::Prepared]> {
    coeffs
        .iter()
        .copied()
        .map(|coeff| F::prepare(RawDispatch, coeff))
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::poly::AES;
    use crate::field::{Elem as E8, FiniteField, Polynomial};
    use alloc::vec::Vec;

    #[test]
    fn matrix_source_lookup_is_total() {
        let matrix = CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(
            3,
            2,
            &[E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1); 6],
        );
        assert!(matrix.source(0).is_some());
        assert!(matrix.source(2).is_some());
        assert!(matrix.source(3).is_none());
        assert!(matrix.source(usize::MAX).is_none());
        assert_eq!(matrix.source(1).map(CoeffVecRef::len), Some(2));

        // Zero-output matrices: every source exists and is empty.
        let empty = CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(3, 0, &[]);
        assert_eq!(empty.source_count(), 3);
        assert_eq!(empty.output_count(), 0);
        assert_eq!(empty.source(0).map(CoeffVecRef::len), Some(0));
        assert_eq!(empty.source(2).map(CoeffVecRef::len), Some(0));
        assert!(empty.source(3).is_none());
        assert!(empty.source(usize::MAX).is_none());

        // Coordinate lookup stays total at the extremes.
        assert!(matrix.get_at(0, usize::MAX).is_none());
        assert!(matrix.get_at(usize::MAX, 0).is_none());
        assert!(matrix.get_at(usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn coeff_matrix_is_source_major() {
        let values: Vec<E8<FiniteField<2, 8, Polynomial<AES>>>> = (0u8..6)
            .map(|i| E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(i * 37))
            .collect();
        let matrix =
            CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(2, 3, &values);
        for source in 0..2 {
            for output in 0..3 {
                let at = matrix
                    .get_at(source, output)
                    .map(CoeffRef::value)
                    .expect("in-bounds coordinate");
                assert_eq!(at, values[source * 3 + output]);
            }
        }
        let row_one: Vec<E8<FiniteField<2, 8, Polynomial<AES>>>> = matrix
            .source(1)
            .unwrap()
            .into_coeffs()
            .map(CoeffRef::value)
            .collect();
        assert_eq!(row_one, values[3..6]);
    }
}
