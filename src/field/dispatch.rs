//! The dispatch contract behind byte-row operations.
//!
//! [`KernelDispatch`] is the field layer's declaration of what an engine
//! provides for one row: raw entry points, one prepared coefficient form,
//! and the backend-policy queries. The contract is declarations only —
//! every body, default composition included, lives in the engine's
//! [`crate::kernel`] subtree, and the engine implements the whole contract
//! per row family. [`FieldBuffer`] inherits the contract, so `F:
//! FieldBuffer` alone reaches a row's dispatched kernels through the
//! validated [`crate::ops`] facade; the contract extends nothing back over
//! [`FieldBuffer`], so no reverse supertrait edge exists.
//!
//! Every entry point takes [`RawDispatch`], the zero-sized call proof.
//! Outside the crate the proof type cannot be named, constructed, or
//! accepted as a parameter, so no consumer code can invoke an entry
//! directly, whatever bounds it holds; inside the crate it costs nothing.

use super::Field;
use crate::field::Elem;

/// The raw per-field kernel entry points.
///
/// Implementations own runtime dispatch for their field. Every method's
/// preconditions are checked by the [`crate::ops`] wrappers, not here, and
/// every entry takes the [`RawDispatch`] proof. The trait is crate-private
/// and a supertrait of [`FieldBuffer`]: the byte surface below `Field`'s
/// scalar contract is reachable only through the checked facade or, under
/// the unstable `internals` feature, through the token-gated architecture
/// modules — never as bare unchecked dispatch.
///
/// # Preconditions
///
/// All slice lengths are in **bytes** and must be whole multiples of the
/// row's element width. `dst` and `src` must have equal length where the
/// operation pairs them.
pub(crate) trait KernelDispatch: Field {
    /// The backend-ready form of one coefficient.
    ///
    /// Different backends want different things from a coefficient: GFNI
    /// wants a broadcast word, the shuffle backends want nibble tables, the
    /// scalar path wants the element itself. [`KernelDispatch::prepare`]
    /// resolves that once — which is why the single-coefficient entries
    /// below take a `Prepared` and not an [`Elem`]. The backend is fixed
    /// for the life of the process, so this moves the backend decision
    /// *out* of the hot call rather than adding one.
    type Prepared: Clone + Send + Sync + core::fmt::Debug;

    /// Resolves a coefficient into the form this host's backend wants.
    fn prepare(_proof: RawDispatch, coeff: Elem<Self>) -> Self::Prepared;

    /// Recover the coefficient a [`KernelDispatch::Prepared`] was built from.
    fn prepared_coeff(_proof: RawDispatch, prepared: &Self::Prepared) -> Elem<Self>;

    /// Backend used by this field's kernels.
    fn backend(_proof: RawDispatch) -> simdispatch::Backend;

    /// Whether elementwise multiplication is vectorized for this field on
    /// this host.
    fn has_vector_elementwise(_proof: RawDispatch) -> bool;

    /// Minimum row length in bytes at which the vector elementwise kernels
    /// take over on this host; zero when no such threshold applies.
    fn vector_elementwise_min_bytes(_proof: RawDispatch) -> usize;

    /// `dst += src`, elementwise field addition.
    ///
    /// Required, with no fallback: addition is XOR for the binary fields and
    /// a characteristic-`p` fold for the prime fields, so a wrong shared
    /// body would be a silent defect. Binary rows route to
    /// [`crate::kernel::xor`].
    fn add_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// `dst += sum(srcs[i])`: fold byte-offset sources out of one backing
    /// region into one row, every coefficient implicitly one.
    ///
    /// The gather whose coefficients are all the multiplicative identity —
    /// the shape of back-substitution over solved rows. The engine fallback
    /// folds sources one at a time through
    /// [`KernelDispatch::add_assign`], which is exact for every field;
    /// fields whose addition is XOR override it with the crate's blocked
    /// XOR gather, which holds the destination in registers across the
    /// whole source list.
    ///
    /// Callers pass offsets that satisfy `offset + dst.len() <=
    /// region.len()`; [`crate::ops::add_gather_offsets`] validates that.
    fn add_gather_offsets(_proof: RawDispatch, region: &[u8], dst: &mut [u8], offsets: &[u32]);

    /// `dst -= src`, elementwise field subtraction.
    ///
    /// Identical to [`KernelDispatch::add_assign`] in characteristic two; prime
    /// fields subtract with a canonicalizing fold. Required, no shared
    /// body, for the same reason.
    fn sub_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// Pairwise row addition over two equal flat row buffers.
    ///
    /// Both buffers hold the same whole number of contiguous `row_len`-byte
    /// rows, and every row pair adds fieldwise: `dst_row[j] += src_row[j]`.
    /// Row boundaries do not change elementwise addition, so running
    /// [`KernelDispatch::add_assign`] over the whole buffers is exact for every
    /// field; that is the engine fallback. A backend overrides it only with
    /// a kernel that exploits the row geometry.
    fn add_assign_rows(_proof: RawDispatch, dst: &mut [u8], src: &[u8], _row_len: usize);

    /// Pairwise row subtraction over two equal flat row buffers.
    ///
    /// The row-geometry counterpart of [`KernelDispatch::sub_assign`],
    /// mirroring [`KernelDispatch::add_assign_rows`]: row boundaries do not
    /// change elementwise subtraction, so running
    /// [`KernelDispatch::sub_assign`] over the whole buffers is exact for every
    /// field; that is the engine fallback.
    fn sub_assign_rows(_proof: RawDispatch, dst: &mut [u8], src: &[u8], _row_len: usize);

    /// `dst += coeff * src`. The workhorse AXPY.
    fn mul_add(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// `dst = coeff * src`, out of place.
    ///
    /// The engine fallback copies `src` into `dst` and scales in place —
    /// two passes over the destination. Backends with a fused single-pass
    /// kernel override this to write the destination in one pass.
    fn mul_into(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// `dst *= coeff`, in place.
    fn mul_assign(_proof: RawDispatch, dst: &mut [u8], coeff: &Self::Prepared);

    /// One source into many rows: `rows[j] += coeffs[j] * src` for each `j`.
    ///
    /// `rows` is a flat buffer of `coeffs.len()` contiguous rows of
    /// `row_len` bytes. This is the systematic-encode shape; blocked
    /// backends load `src` once per tile and update several rows from it.
    ///
    /// The coefficients are raw elements, so every backend has to resolve
    /// each one into its own form before it can use it. Callers holding the
    /// resolved form call [`KernelDispatch::mul_add_scatter_with`].
    fn mul_add_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    );

    /// One source overwrites many rows: `rows[j] = coeffs[j] * src` for each `j`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_scatter`]: the
    /// previous contents of the named rows are ignored, so callers need not
    /// zero them. The engine fallback applies the fused single-row
    /// [`KernelDispatch::mul_into`] per row without allocation;
    /// register-blocked backends override it to seed accumulators from the
    /// source with no destination read.
    fn mul_into_scatter(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Elem<Self>],
        src: &[u8],
    );

    /// Many sources into one row: `dst += sum(coeffs[i] * srcs[i])`.
    ///
    /// The transpose of [`KernelDispatch::mul_add_scatter`], and the shape that
    /// rebuilds a single lost symbol. Blocked backends hold the destination
    /// tile in registers while every source is folded in, so the
    /// destination is read and written once per tile rather than once per
    /// source.
    ///
    /// See [`KernelDispatch::mul_add_gather_with`] for the prepared form.
    fn mul_add_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]);

    /// Many sources overwrite one row: `dst = sum(coeffs[i] * srcs[i])`.
    ///
    /// Unlike [`KernelDispatch::mul_add_gather`], the previous destination is
    /// ignored. The engine fallback starts with a fused single-source
    /// [`KernelDispatch::mul_into`], then accumulates the remaining prepared
    /// terms without allocation.
    fn mul_into_gather(_proof: RawDispatch, dst: &mut [u8], coeffs: &[Elem<Self>], srcs: &[&[u8]]);

    /// Many sources into many rows: for each `(coeffs, src)` term,
    /// `rows[j] += coeffs[j] * src` for every `j` in `0..nrows`.
    ///
    /// Equivalent to a [`KernelDispatch::mul_add_scatter`] per term, but
    /// blocked backends hold a destination tile in registers across all
    /// terms, so destination memory traffic is independent of the term
    /// count. This is the decode/reconstruction shape.
    ///
    /// See [`KernelDispatch::mul_add_matrix_with`] for the prepared form.
    fn mul_add_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    );

    /// [`KernelDispatch::mul_add_scatter`] over already-prepared coefficients.
    ///
    /// The engine fallback keeps preparation out of the row loop by applying
    /// the single-coefficient kernel to each row. Fields override this when
    /// a backend can retain several prepared coefficients in registers.
    #[cfg(feature = "alloc")]
    fn mul_add_scatter_with(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Self::Prepared],
        src: &[u8],
    );

    /// Prepared-plan scatter with access to both original and resolved
    /// coefficients.
    ///
    /// The engine fallback uses the prepared single-row path. Blocked
    /// backends use `values` for representations whose preparation is
    /// already free.
    #[cfg(feature = "alloc")]
    fn mul_add_scatter_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        _values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        src: &[u8],
    );

    /// [`KernelDispatch::mul_add_gather`] over already-prepared coefficients.
    ///
    /// The engine fallback applies the single-coefficient kernel once per
    /// source.
    #[cfg(feature = "alloc")]
    fn mul_add_gather_with(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// Prepared-plan gather with access to both original and resolved
    /// coefficients.
    ///
    /// The engine fallback applies prepared AXPY once per source.
    #[cfg(feature = "alloc")]
    fn mul_add_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// [`KernelDispatch::mul_into_gather`] over already-prepared coefficients.
    #[cfg(any(feature = "alloc", test))]
    fn mul_into_gather_with(
        _proof: RawDispatch,
        dst: &mut [u8],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// Prepared-plan dot product with original and resolved coefficients.
    #[cfg(feature = "alloc")]
    fn mul_into_gather_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        _values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// [`KernelDispatch::mul_add_matrix`] over already-prepared coefficients.
    ///
    /// The engine fallback applies each term row by row. Fields override
    /// this with a blocked implementation that retains destination tiles in
    /// registers.
    #[cfg(test)]
    fn mul_add_matrix_with(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Prepared], &[u8])],
    );

    /// Prepared-plan matrix using flat row-major coefficients and source rows.
    ///
    /// The engine fallback is allocation-free repeated prepared AXPY.
    /// Register-blocked backends override it and consume the same flat
    /// geometry directly.
    #[cfg(feature = "alloc")]
    fn mul_add_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        _values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// Many sources overwrite many rows: `rows[j] = sum_t coeffs[t][j] * src[t]`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_matrix`] and the
    /// erasure-encode shape: the previous destination is ignored. The
    /// engine fallback zeroes the first `nrows` rows and accumulates;
    /// register-blocked backends override it to seed accumulators from zero
    /// in registers, writing each row once with no destination read and no
    /// separate fill.
    fn mul_into_matrix(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Elem<Self>], &[u8])],
    );

    /// [`KernelDispatch::mul_add_matrix_plan`].
    #[cfg(feature = "alloc")]
    fn mul_into_matrix_plan(
        _proof: RawDispatch,
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// Many sources into many disjoint rows scattered through `dst`: for
    /// each `(coeffs, src)` term, `dst[row_starts[j]..][..row_len] +=
    /// coeffs[j] * src` for every `j`.
    ///
    /// Like [`KernelDispatch::mul_add_matrix`], but the destination rows are not
    /// contiguous — row `j` occupies `dst[row_starts[j] .. row_starts[j] +
    /// row_len]`. [`crate::ops::mul_add_matrix_at`] validates the offsets
    /// are in-bounds and pairwise disjoint before dispatch, so a blocked
    /// backend can write each recovered row to its final scattered slot and
    /// skip the staging copy a contiguous kernel forces on scattered decode
    /// outputs.
    ///
    /// The engine fallback composes
    /// [`KernelDispatch::mul_add_matrix_at_rows`]. Register-blocked backends
    /// override it to retain destination tiles in registers across terms.
    fn mul_add_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    );

    /// [`KernelDispatch::mul_add_matrix_at`] as one dispatched
    /// [`KernelDispatch::mul_add`] per row and term, so it inherits the field's
    /// own arithmetic and single-row vector kernel. The engine fallback for
    /// every backend without a blocked scattered kernel.
    fn mul_add_matrix_at_rows(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    );

    /// Prepared-plan scattered matrix using flat source-major coefficients
    /// and source rows.
    ///
    /// The scattered counterpart of [`KernelDispatch::mul_add_matrix_plan`].
    /// The engine fallback is allocation-free repeated prepared AXPY over
    /// the named rows. Register-blocked backends override it and consume
    /// the same flat geometry directly.
    #[cfg(feature = "alloc")]
    fn mul_add_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        _values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// Many sources overwrite many disjoint rows scattered through `dst`:
    /// for each output row `j`,
    /// `dst[row_starts[j]..][..row_len] = sum_t coeffs[t][j] * src[t]`.
    ///
    /// The overwrite counterpart of [`KernelDispatch::mul_add_matrix_at`].
    /// The engine fallback zeroes the named rows and accumulates;
    /// register-blocked backends override it to seed accumulators from zero
    /// in registers, writing each named row once with no destination read
    /// and no separate fill.
    fn mul_into_matrix_at(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        terms: &[(&[Elem<Self>], &[u8])],
    );

    /// [`KernelDispatch::mul_add_matrix_at_plan`].
    #[cfg(feature = "alloc")]
    fn mul_into_matrix_at_plan(
        _proof: RawDispatch,
        dst: &mut [u8],
        row_len: usize,
        row_starts: &[usize],
        values: &[Elem<Self>],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    );

    /// `dst[i] = a[i] * b[i]`, elementwise over two full vectors.
    ///
    /// Both operands vary per lane, so there is no coefficient to broadcast
    /// and no table to index. GFNI multiplies vectors directly; shuffle-only
    /// `x86_64` backends use a branchless shift/reduce vector multiply.
    /// Other targets run the portable reference path.
    fn mul_elementwise(_proof: RawDispatch, dst: &mut [u8], a: &[u8], b: &[u8]);

    /// `dst[i] *= src[i]`, elementwise, in place.
    fn mul_elementwise_assign(_proof: RawDispatch, dst: &mut [u8], src: &[u8]);

    /// `dst[i] += value`, one field element broadcast across every lane.
    fn add_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared);

    /// `dst[i] -= value`, one field element broadcast across every lane.
    fn sub_assign_scalar(_proof: RawDispatch, dst: &mut [u8], value: &Self::Prepared);
}

/// Zero-sized proof of the right to call [`KernelDispatch`] entry points.
///
/// Every [`KernelDispatch`] entry takes this argument. The type is
/// `pub(crate)`: outside the crate it cannot be named, constructed, or
/// accepted as a parameter, so consumer code cannot supply it to a dispatch
/// entry whatever bounds it holds. Inside the crate it is a zero-sized
/// unit value.
pub(crate) struct RawDispatch;
