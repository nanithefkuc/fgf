//! The public vector operation surface.
//!
//! Every function here is generic over [`Field`] and monomorphizes to that
//! field's dispatched SIMD kernel. Buffers are plain `&[u8]` holding packed
//! elements in the field's stable little-endian representation. Canonically
//! encoded payloads need no repacking; prime-field inputs must meet the
//! canonical-lane contract below.
//!
//! # Naming
//!
//! `mul_add` is the fused `dst += coeff * src` (an AXPY; XOR in
//! characteristic two). `_scatter` fans one source out to many rows,
//! `_gather` folds many sources into one row, and `_matrix` does
//! many-to-many with the destination held in registers across sources.
//! These three shapes cover systematic encoding, symbol reconstruction,
//! and erasure decoding respectively.
//!
//! # Reusing a coefficient
//!
//! Each `mul_*` call resolves its coefficient into the form the host backend
//! wants. For GF(2^8) that is an array index; for GF(2^16) on a shuffle
//! backend it is four nibble tables. Amortized over a large buffer that cost
//! vanishes, but across many short buffers with the same coefficient it does
//! not. [`Coeff`] hoists it:
//!
//! ```
//! use fgf::{Gf16, gf16, ops};
//!
//! let coeff = ops::Coeff::<Gf16>::new(gf16::Elem::from_raw(0x0108));
//! let src = [0u8; 64];
//! for _ in 0..3 {
//!     let mut symbol = [0u8; 64];
//!     ops::mul_add_with(&mut symbol, &coeff, &src);   // prepared once
//! }
//! ```
//!
//! # Preconditions
//!
//! Buffer lengths must be whole multiples of `F::BYTES`, paired buffers must
//! be equal in length, and `dst`/`src` must not alias. Length constraints are
//! checked and violations panic; Rust's borrowing rules enforce disjointness.
//!
//! # Prime fields: canonical lanes in, canonical lanes out
//!
//! The packed operations over the prime fields ([`Mersenne31`], [`Goldilocks`],
//! [`QuadMersenne31`]) are defined on **canonical lanes** — every input lane
//! below the modulus — and produce canonical lanes on output. That is the
//! invariant a codec maintains over its buffers, and it is what these kernels
//! preserve; no normalization pass runs inside them. Feeding a packed buffer
//! whose lanes hold arbitrary raw bit patterns computes unspecified field
//! values without memory-unsafety. Scalar element arithmetic is different and
//! stronger: the [`Elem`] operations are total over any
//! raw lane value and always reduce canonically.
//!
//! [`Mersenne31`]: crate::Mersenne31
//! [`Goldilocks`]: crate::Goldilocks
//! [`QuadMersenne31`]: crate::QuadMersenne31

use crate::field::{Elem, Field};
use crate::kernel::{FieldKernels, KernelDispatch, RawDispatch};

/// A coefficient already resolved into the host backend's preferred form.
///
/// Build once, use many times. Construction is not free on every field — see
/// the module docs — but it is idempotent and the result is immutable, so a
/// `Coeff` can be cached alongside a coding matrix for the life of a codec.
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
    pub fn new(coeff: F::Elem) -> Self {
        Self {
            prepared: F::prepare(RawDispatch, coeff),
        }
    }

    /// The field element this was built from.
    #[inline]
    #[must_use]
    pub fn value(&self) -> F::Elem {
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

mod private {
    pub trait Sealed {}
}

/// A borrowed or owned backend-prepared coefficient.
///
/// This trait is sealed: values come from [`Coeff`] or, with `alloc`, a
/// `CoeffVec` or `CoeffMatrix`.
/// It exists so the `_with` operations can consume either without cloning the
/// prepared tables. It exposes the coefficient's field value only — the
/// backend representation stays crate-private, so no consumer can extract or
/// depend on a preparation format.
// The private supertrait seals access to backend preparation.
#[allow(private_bounds)]
pub trait PreparedCoefficient<F: FieldKernels>: private::Sealed + PreparedRepr<F> {
    /// The field element this prepared representation multiplies by.
    fn value(&self) -> F::Elem;
}

/// Crate-private access to the backend representation behind
/// [`PreparedCoefficient`].
///
/// A supertrait with an unnameable proof argument: external generic code
/// bound on `PreparedCoefficient` can see the method through bound
/// elaboration but cannot construct the [`RawDispatch`] it takes, so the
/// representation is unreachable outside the crate.
pub(crate) trait PreparedRepr<F: FieldKernels>: private::Sealed {
    /// Borrow the backend-private representation.
    fn repr(&self, _proof: RawDispatch) -> &<F as KernelDispatch>::Prepared;
}

impl<F: FieldKernels> private::Sealed for Coeff<F> {}

impl<F: FieldKernels> PreparedCoefficient<F> for Coeff<F> {
    #[inline]
    fn value(&self) -> F::Elem {
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
    pub fn value(self) -> F::Elem {
        F::prepared_coeff(RawDispatch, self.prepared)
    }
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> private::Sealed for CoeffRef<'_, F> {}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> PreparedCoefficient<F> for CoeffRef<'_, F> {
    #[inline]
    fn value(&self) -> F::Elem {
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
/// Positions are output rows for [`mul_add_scatter_with`] and sources for the
/// gather operations. A vector has a length and nothing else — it carries no
/// source or output axis, which is exactly why it is a separate type from
/// [`CoeffMatrix`].
///
/// Preparation happens once in [`CoeffVec::new`]; [`CoeffVec::get`] then
/// borrows an entry without rebuilding or copying its backend tables. Store
/// one beside the coding column it represents.
///
/// Requires `alloc`: the collection owns dynamically sized storage. The
/// operations take the borrowed [`CoeffVecRef`] view, which
/// [`CoeffVec::as_ref`] produces for free.
#[cfg(feature = "alloc")]
pub struct CoeffVec<F: FieldKernels> {
    prepared: alloc::boxed::Box<[<F as KernelDispatch>::Prepared]>,
    values: alloc::boxed::Box<[F::Elem]>,
}

#[cfg(feature = "alloc")]
impl<F: FieldKernels> CoeffVec<F> {
    /// Resolve every coefficient for this host.
    #[must_use]
    pub fn new(coeffs: &[F::Elem]) -> Self {
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
    #[must_use]
    pub fn coeffs(&self) -> impl ExactSizeIterator<Item = CoeffRef<'_, F>> {
        self.as_ref().into_coeffs()
    }

    /// Iterate over the original field elements.
    #[must_use]
    pub fn values(&self) -> impl ExactSizeIterator<Item = F::Elem> + '_ {
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
/// [`mul_add_scatter_with`] without copying or restaging anything.
///
/// `Copy`: it is two slice references.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy)]
pub struct CoeffVecRef<'a, F: FieldKernels> {
    prepared: &'a [<F as KernelDispatch>::Prepared],
    values: &'a [F::Elem],
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
    #[must_use]
    pub fn into_coeffs(self) -> impl ExactSizeIterator<Item = CoeffRef<'a, F>> {
        self.prepared.iter().map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        })
    }

    /// Iterate over the original field elements.
    #[must_use]
    pub fn values(self) -> impl ExactSizeIterator<Item = F::Elem> + 'a {
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
/// Requires `alloc`. Store one beside the coding matrix it represents.
#[cfg(feature = "alloc")]
pub struct CoeffMatrix<F: FieldKernels> {
    prepared: alloc::boxed::Box<[<F as KernelDispatch>::Prepared]>,
    values: alloc::boxed::Box<[F::Elem]>,
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
    pub fn from_source_major(sources: usize, outputs: usize, coeffs: &[F::Elem]) -> Self {
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
    /// matrix column drives [`mul_add_scatter_with`] directly.
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
    #[must_use]
    pub fn values(&self) -> impl ExactSizeIterator<Item = F::Elem> + '_ {
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
    coeffs: &[F::Elem],
) -> alloc::boxed::Box<[<F as KernelDispatch>::Prepared]> {
    coeffs
        .iter()
        .copied()
        .map(|coeff| F::prepare(RawDispatch, coeff))
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

#[inline]
fn check_width<F: Field>(name: &str, len: usize) {
    assert!(
        len.is_multiple_of(F::BYTES),
        "{name}: buffer of {len} bytes is not a whole number of {} elements",
        F::NAME
    );
}

#[inline]
fn check_pair<F: Field>(name: &str, left_name: &str, left: usize, right_name: &str, right: usize) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
    check_width::<F>(name, left);
}

/// `dst += src`, field addition.
///
/// XOR in characteristic two; a canonical modular fold in the prime fields.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn add_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("add_assign", "dst", dst.len(), "src", src.len());
    F::add_assign(RawDispatch, dst, src);
}

/// Add pairwise rows: `dst_row[j] += src_row[j]` for every row.
///
/// Both buffers are interpreted as the same number of contiguous
/// `row_len`-byte rows. Semantically identical to [`add_assign`] — row
/// boundaries cannot change elementwise addition — and currently implemented
/// exactly that way on every backend: a four-stream row-interleaved XOR
/// candidate built for this shape measured at parity with the flat kernel
/// from L1 to DRAM on the reference host, so no override is wired
/// (BENCHMARKS.md, "Row-interleaved XOR"). Call it anyway when the row
/// geometry is known: backends may interleave independent row streams once
/// any platform measures a repeatable win, at which point existing callers
/// pick it up without code changes.
///
/// ```
/// use fgf::{Gf8B, ops};
///
/// let src = [0x01u8, 0x02, 0x03, 0x04];
/// let mut dst = [0x10u8, 0x20, 0x30, 0x40];
///
/// // Two rows of two bytes each, added pairwise.
/// ops::add_assign_rows::<Gf8B>(&mut dst, 2, &src);
/// assert_eq!(dst, [0x11, 0x22, 0x33, 0x44]);
/// ```
///
/// # Panics
/// Panics if `row_len == 0`, if `row_len` is not a whole number of field
/// elements, if the buffers differ in length, or if their length is not a
/// whole number of rows.
#[inline]
pub fn add_assign_rows<F: FieldKernels>(dst: &mut [u8], row_len: usize, src: &[u8]) {
    assert_ne!(row_len, 0, "add_assign_rows: row length must be nonzero");
    check_width::<F>("add_assign_rows", row_len);
    assert_eq!(
        dst.len(),
        src.len(),
        "add_assign_rows: dst is {} bytes but src is {} bytes",
        dst.len(),
        src.len(),
    );
    assert_eq!(
        dst.len() % row_len,
        0,
        "add_assign_rows: partial trailing row",
    );
    F::add_assign_rows(RawDispatch, dst, src, row_len);
}

/// Fold byte-offset sources out of one backing region into one row, every
/// coefficient implicitly one: `dst += sum(region[off..off + dst.len()])`
/// (field addition; byte XOR only in characteristic two).
///
/// The unit-coefficient gather: sources are rows of one buffer — a solved
/// solution block, a table of packed elements — addressed by their start
/// offsets, so a caller folds an index table without staging a slice per
/// source. Fields whose addition is XOR hold the destination in registers
/// across the whole source list; the rest fold one [`add_assign`] per
/// source.
/// The `_offsets` in the name: the *sources* are the offset-addressed rows
/// of one backing region; the destination is the single row that comes first.
///
/// ```
/// use fgf::{Gf8B, ops};
///
/// // Two 2-byte rows in one backing region, folded into `dst`.
/// let region = [0x01u8, 0x02, 0x10, 0x20];
/// let mut dst = [0x40u8, 0x80];
/// ops::add_gather_offsets::<Gf8B>(&mut dst, &region, &[0, 2]);
/// assert_eq!(dst, [0x51, 0xa2]);
/// ```
///
/// # Panics
/// Panics if `dst` is not a whole number of field elements, or if any
/// offset plus `dst.len()` falls outside `region`. `region` and `dst`
/// borrow disjoint memory by construction, as with every two-slice
/// operation in this crate.
#[inline]
pub fn add_gather_offsets<F: FieldKernels>(dst: &mut [u8], region: &[u8], offsets: &[u32]) {
    check_width::<F>("add_gather_offsets", dst.len());
    let live = dst.len();
    for (index, &start) in offsets.iter().enumerate() {
        let start = start as usize;
        assert!(
            start + live <= region.len(),
            "add_gather_offsets: offset {index} ({start}) + {live} exceeds region of {} bytes",
            region.len(),
        );
    }
    F::add_gather_offsets(RawDispatch, region, dst, offsets);
}

/// `dst -= src`. Identical to [`add_assign`] in characteristic two.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn sub_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("sub_assign", "dst", dst.len(), "src", src.len());
    F::sub_assign(RawDispatch, dst, src);
}

/// `dst += coeff * src`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    check_pair::<F>("mul_add", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        F::add_assign(RawDispatch, dst, src);
        return;
    }
    F::mul_add(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
}

/// `dst += coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        return;
    }
    if value.is_one() {
        F::add_assign(RawDispatch, dst, src);
        return;
    }
    F::mul_add(RawDispatch, dst, coeff.repr(RawDispatch), src);
}

/// `dst = coeff * src`, overwriting `dst`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    check_pair::<F>("mul_into", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    if coeff.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(RawDispatch, dst, &F::prepare(RawDispatch, coeff), src);
}

/// `dst = coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_into_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    if value.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(RawDispatch, dst, coeff.repr(RawDispatch), src);
}

/// `dst *= coeff`, in place.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem) {
    check_width::<F>("mul_assign", dst.len());
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(RawDispatch, dst, &F::prepare(RawDispatch, coeff));
}

/// `dst *= coeff`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign_with<F: FieldKernels>(dst: &mut [u8], coeff: &impl PreparedCoefficient<F>) {
    check_width::<F>("mul_assign_with", dst.len());
    let value = coeff.value();
    if value.is_one() {
        return;
    }
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(RawDispatch, dst, coeff.repr(RawDispatch));
}

/// Fan one source out to many rows: `rows[j] += coeffs[j] * src`.
///
/// `rows` is a flat buffer of `coeffs.len()` contiguous rows, each `row_len`
/// bytes and the same length as `src`. This is the systematic-encode shape:
/// one arriving data symbol updates every parity row.
///
/// A zero-length row with a zero-length source is a valid zero-work shape
/// and a no-op; the structural checks above still apply.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
pub fn mul_add_scatter<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[F::Elem],
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_scatter(RawDispatch, &mut rows[..used], row_len, coeffs, src);
}

/// Fan one source out to many rows using prepared coefficients.
///
/// Unlike [`mul_add_scatter`], repeated calls do not rebuild backend-specific
/// coefficient tables. The view comes from [`CoeffVec::as_ref`] or, for one
/// column of a coding matrix, [`CoeffMatrix::source`].
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_add_scatter_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: CoeffVecRef<'_, F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter_with", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter_with: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if row_len == 0 || coeffs.values().all(Elem::is_zero) {
        return;
    }
    F::mul_add_scatter_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        coeffs.values,
        coeffs.prepared,
        src,
    );
}

/// Fold many sources into one row: `dst += sum(coeffs[i] * srcs[i])`.
///
/// The transpose of [`mul_add_scatter`], and the shape that rebuilds a single
/// lost symbol from its survivors. Blocked backends hold the destination tile
/// in registers while every source is folded in, so `dst` is read and written
/// once per tile rather than once per source — prefer this over a loop of
/// [`mul_add`] whenever the sources share a destination.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length, or on a partial trailing element.
pub fn mul_add_gather<F: FieldKernels>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    check_width::<F>("mul_add_gather", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_add_gather: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len()
    );
    for &src in srcs {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather: source is {} bytes, expected {}",
            src.len(),
            dst.len()
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_gather(RawDispatch, dst, coeffs, srcs);
}

/// Fold many sources into one row using prepared coefficients.
///
/// One coefficient per source. The view comes from [`CoeffVec::as_ref`].
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_add_gather_with<F: FieldKernels>(
    dst: &mut [u8],
    coeffs: CoeffVecRef<'_, F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_add_gather_with", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_add_gather_with: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather_with: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.values().all(Elem::is_zero) {
        return;
    }
    F::mul_add_gather_plan(RawDispatch, dst, coeffs.values, coeffs.prepared, srcs);
}

/// Overwrite one row with a field dot product:
/// `dst = sum(coeffs[i] * srcs[i])`.
///
/// The overwrite counterpart of [`mul_add_gather`]: the previous destination
/// is ignored. Zero sources or all-zero coefficients fill `dst` with zero;
/// one source is the same operation as [`mul_into`].
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length, or on a partial trailing element.
pub fn mul_into_gather<F: FieldKernels>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    check_width::<F>("mul_into_gather", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_into_gather: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len()
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_into_gather: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len()
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.iter().all(|coeff| coeff.is_zero()) {
        dst.fill(0);
        return;
    }
    if let [coeff] = coeffs {
        mul_into::<F>(dst, *coeff, srcs[0]);
        return;
    }
    F::mul_into_gather(RawDispatch, dst, coeffs, srcs);
}

/// Overwrite one row with a field dot product using prepared coefficients.
///
/// One coefficient per source. The view comes from [`CoeffVec::as_ref`].
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length, or on a partial trailing element.
#[cfg(feature = "alloc")]
pub fn mul_into_gather_with<F: FieldKernels>(
    dst: &mut [u8],
    coeffs: CoeffVecRef<'_, F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_into_gather_with", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_into_gather_with: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_into_gather_with: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() || coeffs.values().all(Elem::is_zero) {
        dst.fill(0);
        return;
    }
    if coeffs.len() == 1 {
        let coeff = coeffs
            .get(0)
            .expect("single-entry view has one coefficient");
        mul_into_with::<F>(dst, &coeff, srcs[0]);
        return;
    }
    F::mul_into_gather_plan(RawDispatch, dst, coeffs.values, coeffs.prepared, srcs);
}

/// Apply many sources to many rows: for each `(coeffs, src)` term,
/// `rows[j] += coeffs[j] * src` for every `j` in `0..nrows`.
///
/// Semantically a [`mul_add_scatter`] per term. The difference is memory
/// traffic: blocked backends keep a destination tile in registers while every
/// source is folded in, so the destination is read and written once per tile
/// rather than once per term. That is the whole game once
/// `nrows * row_len` exceeds L1 — which is the normal case for erasure
/// reconstruction.
///
/// A zero-length row with zero-length sources is a valid zero-work shape and
/// a no-op; the structural checks above still apply.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes, every
/// term supplies `nrows` coefficients, and every source is `row_len` bytes.
pub fn mul_add_matrix<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Elem], &[u8])],
) {
    check_width::<F>("mul_add_matrix", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 || terms.is_empty() {
        return;
    }
    F::mul_add_matrix(RawDispatch, &mut rows[..used], row_len, nrows, terms);
}

/// Apply many sources to many rows using a prepared source-major coefficient
/// matrix.
///
/// The matrix must have dimensions `(srcs.len(), n)` in `(sources, outputs)`:
/// matrix row `s` holds the coefficients pairing source `s` with every output
/// row. The output-row count is
/// [`coeffs.output_count()`](CoeffMatrix::output_count) — it is not a
/// parameter.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.output_count()` rows of
/// `row_len` bytes, `coeffs.source_count() == srcs.len()`, and every source
/// is `row_len` bytes.
#[cfg(feature = "alloc")]
pub fn mul_add_matrix_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_add_matrix_with", row_len);
    let nrows = coeffs.output_count();
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_add_matrix_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if nrows == 0 || row_len == 0 || srcs.is_empty() {
        return;
    }
    F::mul_add_matrix_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        nrows,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Overwrite many rows with field dot products: for each output row `j`,
/// `rows[j] = sum_t coeffs[t][j] * src[t]`.
///
/// The overwrite counterpart of [`mul_add_matrix`] and the erasure-encode
/// shape: the previous destination is ignored, so callers need not zero it.
/// A register-blocked backend seeds accumulators from zero and writes each row
/// once, with no destination read and no separate `fill(0)`.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes, every
/// term supplies `nrows` coefficients, and every source is `row_len` bytes.
pub fn mul_into_matrix<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Elem], &[u8])],
) {
    check_width::<F>("mul_into_matrix", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_into_matrix: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_into_matrix: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_into_matrix: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 {
        return;
    }
    if terms.is_empty() {
        rows[..used].fill(0);
        return;
    }
    F::mul_into_matrix(RawDispatch, &mut rows[..used], row_len, nrows, terms);
}

/// Overwrite many rows with field dot products using a prepared source-major
/// coefficient matrix: `rows[j] = sum_t coeffs[t][j] * src[t]`.
///
/// The overwrite counterpart of [`mul_add_matrix_with`]. The matrix must have
/// `srcs.len()` sources; the output-row count is
/// [`coeffs.output_count()`](CoeffMatrix::output_count).
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.output_count()` rows of
/// `row_len` bytes, `coeffs.source_count() == srcs.len()`, and every source
/// is `row_len` bytes.
#[cfg(feature = "alloc")]
pub fn mul_into_matrix_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &CoeffMatrix<F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_into_matrix_with", row_len);
    let nrows = coeffs.output_count();
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_into_matrix_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_into_matrix_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    assert_eq!(
        coeffs.source_count(),
        srcs.len(),
        "mul_into_matrix_with: matrix holds coefficients for {} sources but there are {}",
        coeffs.source_count(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_into_matrix_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if nrows == 0 || row_len == 0 {
        return;
    }
    if srcs.is_empty() {
        rows[..used].fill(0);
        return;
    }
    F::mul_into_matrix_plan(
        RawDispatch,
        &mut rows[..used],
        row_len,
        nrows,
        coeffs.values.as_ref(),
        coeffs.prepared.as_ref(),
        srcs,
    );
}

/// Apply many sources to many disjoint rows scattered through `dst`: for each
/// `(coeffs, src)` term, `dst[row_starts[j]..][..row_len] += coeffs[j] * src`
/// for every `j` in `0..row_starts.len()`.
///
/// The reconstruction shape when recovered rows land at scattered positions in
/// a larger buffer — their natural slots in a codeword. Unlike
/// [`mul_add_matrix`], no contiguous staging buffer and copy-out is required:
/// a blocked backend writes each recovered row straight to its final location.
///
/// `row_starts` must address pairwise-disjoint, in-bounds rows of `row_len`
/// bytes, each starting on an element boundary. Disjointness is the safety
/// contract the kernel relies on and is checked here, before dispatch.
///
/// A zero-length row is a valid zero-work shape and a no-op; the structural
/// checks above (including disjointness of the named rows) still apply.
/// The `_at` in the name: the destination rows are addressed by explicit
/// offsets into `dst`, not held contiguously from its start.
///
/// # Panics
/// Panics unless every `row_starts[j] + row_len <= dst.len()`, every row start
/// is a whole number of elements, the rows are pairwise disjoint, every term
/// supplies `row_starts.len()` coefficients, and every source is `row_len`
/// bytes; or on a partial trailing element.
pub fn mul_add_matrix_at<F: FieldKernels>(
    dst: &mut [u8],
    row_len: usize,
    row_starts: &[usize],
    terms: &[(&[F::Elem], &[u8])],
) {
    check_width::<F>("mul_add_matrix_at", row_len);
    for (j, &start) in row_starts.iter().enumerate() {
        let end = start
            .checked_add(row_len)
            .expect("mul_add_matrix_at: row offset + length overflows");
        assert!(
            end <= dst.len(),
            "mul_add_matrix_at: row {j} spans {start}..{end} but dst is {} bytes",
            dst.len(),
        );
        assert!(
            start.is_multiple_of(F::BYTES),
            "mul_add_matrix_at: row {j} offset {start} is not a whole number of {} elements",
            F::NAME,
        );
    }
    // Pairwise disjointness. `O(r^2)` in the erasure count `r`, but this is a
    // per-solve setup cost paid once per pattern, not per byte, and it
    // allocates nothing. Overlap is a caller error: the kernel writes each row
    // through a raw pointer and cannot tolerate aliasing.
    for (a, &sa) in row_starts.iter().enumerate() {
        for &sb in &row_starts[a + 1..] {
            let (lo, hi) = if sa <= sb { (sa, sb) } else { (sb, sa) };
            assert!(
                hi - lo >= row_len,
                "mul_add_matrix_at: rows at {sa} and {sb} overlap for {row_len}-byte rows",
            );
        }
    }
    let nrows = row_starts.len();
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix_at: term supplies {} coefficients for {nrows} rows",
            coeffs.len(),
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_at: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || row_len == 0 || terms.is_empty() {
        return;
    }
    F::mul_add_matrix_at(RawDispatch, dst, row_len, row_starts, terms);
}

/// Elementwise product: `dst[i] = a[i] * b[i]`.
///
/// Both operands vary per lane, so there is no coefficient to broadcast and
/// no nibble table to index. GFNI multiplies the vectors directly; `AArch64`
/// uses PMULL when available and a lane-parallel shift/reduce sequence
/// otherwise; Wasm `simd128` uses the same shift/reduce construction.
/// Shuffle-only x86 backends fall back to the scalar reference path.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
pub fn mul_elementwise<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "a", a.len());
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "b", b.len());
    F::mul_elementwise(RawDispatch, dst, a, b);
}

/// In-place elementwise product: `dst[i] *= src[i]`.
///
/// The two-slice spelling of [`mul_elementwise`]: the destination is also the
/// first operand, so consumers composing lane-parallel Horner steps update
/// their accumulator without a second output buffer.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_elementwise_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("mul_elementwise_assign", "dst", dst.len(), "src", src.len());
    F::mul_elementwise_assign(RawDispatch, dst, src);
}

/// `dst[i] += value`, one field element broadcast across every lane.
///
/// The value is canonicalized once before dispatch, matching
/// [`mul_add`]'s treatment of its coefficient. Destination lanes holding
/// non-canonical prime-field words are canonicalized on load, exactly as in
/// [`add_assign`].
///
/// # Panics
/// Panics if `dst` holds a partial trailing element.
#[inline]
pub fn add_assign_scalar<F: FieldKernels>(dst: &mut [u8], value: F::Elem) {
    check_width::<F>("add_assign_scalar", dst.len());
    F::add_assign_scalar(RawDispatch, dst, &F::prepare(RawDispatch, value));
}

/// `dst[i] -= value`, one field element broadcast across every lane.
///
/// The value is canonicalized once before dispatch. Destination lanes
/// holding non-canonical prime-field words are canonicalized on load,
/// exactly as in [`sub_assign`].
///
/// # Panics
/// Panics if `dst` holds a partial trailing element.
#[inline]
pub fn sub_assign_scalar<F: FieldKernels>(dst: &mut [u8], value: F::Elem) {
    check_width::<F>("sub_assign_scalar", dst.len());
    F::sub_assign_scalar(RawDispatch, dst, &F::prepare(RawDispatch, value));
}

/// Pack field elements into their stable little-endian byte representation.
///
/// Prime-field representations are preserved, not normalized. Canonicalize
/// raw prime elements before packing inputs for the arithmetic operations.
///
/// # Panics
/// Panics unless `dst.len() == elems.len() * F::BYTES`.
pub fn pack<F: Field>(dst: &mut [u8], elems: &[F::Elem]) {
    let expected = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack: buffer geometry overflows");
    assert_eq!(
        dst.len(),
        expected,
        "pack: dst is {} bytes but {} {} elements need {expected}",
        dst.len(),
        elems.len(),
        F::NAME,
    );
    for (bytes, &elem) in dst.chunks_exact_mut(F::BYTES).zip(elems) {
        F::encode(bytes, elem);
    }
}

/// Decode packed little-endian bytes into field elements.
///
/// Prime-field lanes retain their raw representation, including
/// noncanonical values; this conversion does not validate field canonicality.
///
/// # Panics
/// Panics unless `src.len() == dst.len() * F::BYTES`.
pub fn unpack<F: Field>(dst: &mut [F::Elem], src: &[u8]) {
    let expected = dst
        .len()
        .checked_mul(F::BYTES)
        .expect("unpack: buffer geometry overflows");
    assert_eq!(
        src.len(),
        expected,
        "unpack: src is {} bytes but dst holds {} {} elements ({expected} bytes)",
        src.len(),
        dst.len(),
        F::NAME,
    );
    for (elem, bytes) in dst.iter_mut().zip(src.chunks_exact(F::BYTES)) {
        *elem = F::decode(bytes);
    }
}

/// Pack field elements into a newly allocated byte vector.
///
/// Like [`pack`], this preserves prime-field raw representations.
///
/// # Panics
/// Panics if the required byte length overflows `usize`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn pack_to_vec<F: Field>(elems: &[F::Elem]) -> alloc::vec::Vec<u8> {
    let len = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack_to_vec: buffer geometry overflows");
    let mut bytes = alloc::vec![0; len];
    pack::<F>(&mut bytes, elems);
    bytes
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::{Gf8B, Gf16, gf8b, gf16};
    use alloc::vec::Vec;

    /// `row_len == 0` with otherwise valid geometry is a no-op everywhere —
    /// including the scalar backend, whose `chunks_exact_mut(0)` used to
    /// panic where GFNI silently succeeded.
    #[test]
    fn zero_row_length_is_a_no_op() {
        let coeffs = [gf8b::Elem::from_raw(0x07); 3];
        let mut rows = [0xA5u8; 8];
        mul_add_scatter::<Gf8B>(&mut rows, 0, &coeffs, &[]);
        assert_eq!(rows, [0xA5; 8]);

        let coeffs_vec = CoeffVec::<Gf8B>::new(&coeffs);
        mul_add_scatter_with::<Gf8B>(&mut rows, 0, coeffs_vec.as_ref(), &[]);
        assert_eq!(rows, [0xA5; 8]);

        let terms: &[(&[gf8b::Elem], &[u8])] = &[(&coeffs, &[])];
        mul_add_matrix::<Gf8B>(&mut rows, 0, 3, terms);
        assert_eq!(rows, [0xA5; 8]);
        mul_into_matrix::<Gf8B>(&mut rows, 0, 3, terms);
        assert_eq!(rows, [0xA5; 8]);
        mul_add_matrix_at::<Gf8B>(&mut rows, 0, &[1, 4], &[(&coeffs[..2], &[])]);
        assert_eq!(rows, [0xA5; 8]);

        let matrix = CoeffMatrix::<Gf8B>::from_source_major(1, 3, &coeffs);
        mul_add_matrix_with::<Gf8B>(&mut rows, 0, &matrix, &[&[]]);
        assert_eq!(rows, [0xA5; 8]);
        mul_into_matrix_with::<Gf8B>(&mut rows, 0, &matrix, &[&[]]);
        assert_eq!(rows, [0xA5; 8]);

        // GF(2^16): the field whose odd-length buffers make the element
        // check matter.
        let wide = [gf16::Elem::from_raw(0x0103); 2];
        let mut rows16 = [0x5Au8; 4];
        mul_add_scatter::<Gf16>(&mut rows16, 0, &wide, &[]);
        assert_eq!(rows16, [0x5A; 4]);
    }

    /// Zero-work shapes still validate structure: a non-empty source with a
    /// zero row length is a geometry error, not a silent skip.
    #[test]
    #[should_panic(expected = "mul_add_scatter")]
    fn zero_row_length_still_checks_pairing() {
        let coeffs = [gf8b::Elem::from_raw(0x07); 3];
        let mut rows = [0u8; 8];
        mul_add_scatter::<Gf8B>(&mut rows, 0, &coeffs, &[1, 2, 3]);
    }

    /// Term coefficient counts are still validated at zero row length.
    #[test]
    #[should_panic(expected = "term supplies")]
    fn zero_row_length_still_checks_term_counts() {
        let coeffs = [gf8b::Elem::from_raw(0x07); 2];
        let mut rows = [0u8; 8];
        mul_add_matrix::<Gf8B>(&mut rows, 0, 3, &[(&coeffs, &[])]);
    }

    #[test]
    fn matrix_source_lookup_is_total() {
        let matrix = CoeffMatrix::<Gf8B>::from_source_major(3, 2, &[gf8b::Elem::from_raw(1); 6]);
        assert!(matrix.source(0).is_some());
        assert!(matrix.source(2).is_some());
        assert!(matrix.source(3).is_none());
        assert!(matrix.source(usize::MAX).is_none());
        assert_eq!(matrix.source(1).map(CoeffVecRef::len), Some(2));

        // Zero-output matrices: every source exists and is empty.
        let empty = CoeffMatrix::<Gf8B>::from_source_major(3, 0, &[]);
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
        let values: Vec<gf8b::Elem> = (0u8..6).map(|i| gf8b::Elem::from_raw(i * 37)).collect();
        let matrix = CoeffMatrix::<Gf8B>::from_source_major(2, 3, &values);
        for source in 0..2 {
            for output in 0..3 {
                let at = matrix
                    .get_at(source, output)
                    .map(CoeffRef::value)
                    .expect("in-bounds coordinate");
                assert_eq!(at, values[source * 3 + output]);
            }
        }
        let row_one: Vec<gf8b::Elem> = matrix
            .source(1)
            .unwrap()
            .into_coeffs()
            .map(CoeffRef::value)
            .collect();
        assert_eq!(row_one, values[3..6]);
    }

    #[test]
    fn prepared_matrix_ops_match_one_shot() {
        // source-major matrix: source s contributes coeffs[s * outputs + o].
        let values: Vec<gf16::Elem> = (0u16..3 * 4)
            .map(|i| gf16::Elem::from_raw(i * 511 + 3))
            .collect();
        let matrix = CoeffMatrix::<Gf16>::from_source_major(3, 4, &values);
        let srcs: Vec<Vec<u8>> = (0u8..3).map(|s| alloc::vec![s + 1; 8]).collect();
        let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();

        let terms: Vec<(&[gf16::Elem], &[u8])> = (0..3)
            .map(|s| (&values[s * 4..s * 4 + 4], src_refs[s]))
            .collect();

        let mut one_shot = [0x11u8; 32];
        mul_add_matrix::<Gf16>(&mut one_shot, 8, 4, &terms);
        let mut prepared = [0x11u8; 32];
        mul_add_matrix_with::<Gf16>(&mut prepared, 8, &matrix, &src_refs);
        assert_eq!(one_shot, prepared);

        let mut one_shot = [0x22u8; 32];
        mul_into_matrix::<Gf16>(&mut one_shot, 8, 4, &terms);
        let mut prepared = [0x22u8; 32];
        mul_into_matrix_with::<Gf16>(&mut prepared, 8, &matrix, &src_refs);
        assert_eq!(one_shot, prepared);
    }
}
