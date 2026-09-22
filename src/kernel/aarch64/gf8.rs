//! GF(2^8) NEON kernels.
//!
//! NEON has no byte-wide field multiply, so every product here is the
//! split-nibble form: `vqtbl1q_u8` performs a 16-entry lookup on all 16 lanes
//! at once, and `c * x` becomes `lo[x & 0xf] ^ hi[x >> 4]` against the
//! coefficient's precomputed [`ScaleTable`].
//!
//! Three shapes, in increasing arithmetic intensity:
//!
//! - [`mul_add_neon`] and [`mul_assign_neon`] — one buffer, two 16-byte lanes
//!   per iteration.
//! - [`scatter_neon`] — one source load feeds four contiguous rows, so the
//!   source is read once per group of four instead of once per row.
//! - [`matrix_neon`] — a four-row destination tile is loaded into
//!   accumulators once, every `(coeffs, src)` term is folded in, and the tile
//!   is stored once. Destination traffic is therefore independent of
//!   `terms.len()`, which is the entire reason the shape exists.
//!
//! Every entry is a safe [`archmage`] capability-token function taking the
//! exact token its instructions require, validating its own geometry, and
//! handing its sub-lane remainder to the scalar nibble kernels in
//! [`crate::kernel::gf8`]; buffer lengths are arbitrary and are never assumed
//! to be lane-aligned. Inner loops are `#[rite]` helpers inside the
//! token-proven region. Unsafe remains only where no safe primitive expresses
//! the structure — the offset-addressed rows of the scatter and matrix bodies
//! — each carrying a per-item `#[allow(unsafe_code)]` and local SINCE–THUS
//! proofs.
//!
//! [`archmage`]: https://docs.rs/archmage

use core::arch::aarch64::*;

use crate::field::gf8b::Elem;
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::proven_checks::{check_equal, check_row_span, check_terms};
use crate::kernel::tables::{ScaleTable, scale_table};

/// Load a coefficient's nibble tables into two vector registers.
#[inline]
#[archmage::rite(neon, import_intrinsics)]
fn load_tables(table: &ScaleTable) -> (uint8x16_t, uint8x16_t) {
    (vld1q_u8(&table.lo), vld1q_u8(&table.hi))
}

/// `coeff * x` for one 16-byte lane, given that coefficient's nibble tables.
///
/// `vshrq_n_u8` is a logical shift, so the high-nibble indices already land in
/// `0..16` and need no extra mask before the lookup.
#[inline]
#[archmage::rite(neon, import_intrinsics)]
fn scale(lo: uint8x16_t, hi: uint8x16_t, x: uint8x16_t) -> uint8x16_t {
    let low = vandq_u8(x, vdupq_n_u8(0x0f));
    let high = vshrq_n_u8(x, 4);
    veorq_u8(vqtbl1q_u8(lo, low), vqtbl1q_u8(hi, high))
}

/// Which of the three cases a row's coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: the row contributes nothing.
    Skip,
    /// Coefficient one: XOR the source in unscaled, no lookup needed.
    Identity,
    /// Any other coefficient: two nibble lookups and an XOR.
    Table,
}

/// One coefficient resolved into the form the vector loops consume.
///
/// Branching on [`Kind::Skip`] and [`Kind::Identity`] pays for itself: the
/// coefficient arrays handed to [`scatter_neon`] and [`matrix_neon`] are full
/// of zeros and ones, and each case removes two `TBL`s per lane — a skip also
/// removes the destination load and store entirely.
#[derive(Clone, Copy)]
struct Scaling {
    /// Low-nibble lookup table. Meaningless unless `kind` is [`Kind::Table`].
    lo: uint8x16_t,
    /// High-nibble lookup table. Meaningless unless `kind` is [`Kind::Table`].
    hi: uint8x16_t,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff` against the shared table bank.
    #[inline]
    #[archmage::rite(neon, import_intrinsics)]
    fn new(coeff: Elem) -> Self {
        let zero = vdupq_n_u8(0);
        if coeff == Elem::ZERO {
            return Self {
                lo: zero,
                hi: zero,
                kind: Kind::Skip,
            };
        }
        if coeff == Elem::ONE {
            return Self {
                lo: zero,
                hi: zero,
                kind: Kind::Identity,
            };
        }
        let (lo, hi) = load_tables(scale_table(coeff));
        Self {
            lo,
            hi,
            kind: Kind::Table,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[archmage::rite(neon, import_intrinsics)]
    fn fold(self, acc: uint8x16_t, x: uint8x16_t) -> uint8x16_t {
        match self.kind {
            Kind::Skip => acc,
            Kind::Identity => veorq_u8(acc, x),
            Kind::Table => veorq_u8(acc, scale(self.lo, self.hi, x)),
        }
    }
}

/// One destination row of a [`scatter_neon`] group, resolved once per group.
#[derive(Clone, Copy)]
struct PreparedRow {
    /// First byte of the row.
    ptr: *mut u8,
    /// Nibble tables for this row's coefficient, used by the scalar tail.
    ///
    /// Correct for all three [`Kind`]s: the tables of `0` are all zero and
    /// the tables of `1` are the identity map, so the tail needs no case
    /// analysis of its own.
    table: &'static ScaleTable,
    /// Vector strategy for this row's coefficient.
    scaling: Scaling,
}

impl PreparedRow {
    /// Resolve the row starting at `ptr` for coefficient `coeff`.
    ///
    /// `ptr` is only recorded here; the loops that dereference it carry the
    /// in-bounds and disjointness argument.
    #[inline]
    #[archmage::rite(neon, import_intrinsics)]
    fn new(ptr: *mut u8, coeff: Elem) -> Self {
        Self {
            ptr,
            table: scale_table(coeff),
            scaling: Scaling::new(coeff),
        }
    }
}

/// `dst ^= coeff * src`, two 16-byte NEON lanes per iteration.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_add_neon(_token: archmage::NeonToken, dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    check_equal("gf8::mul_add_neon", "dst", dst.len(), "src", src.len());
    mul_add_impl(dst, table, src);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_add_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let (lo, hi) = load_tables(table);

    // Two independent lanes per iteration: the second lane's loads and nibble
    // splits hide behind the first lane's XOR latency.
    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    let (src_tiles, _) = src[..pair_len].as_chunks::<32>();
    for (d_tile, s_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d0, d1) = d_tile.split_at_mut(16);
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) =
            (d0.try_into().unwrap(), d1.try_into().unwrap());
        let (s0, s1): (&[u8; 16], &[u8; 16]) = {
            let (s0, s1) = s_tile.split_at(16);
            (s0.try_into().unwrap(), s1.try_into().unwrap())
        };
        let x0 = vld1q_u8(s0);
        let x1 = vld1q_u8(s1);
        let d0v = vld1q_u8(&*d0);
        let d1v = vld1q_u8(&*d1);
        vst1q_u8(d0, veorq_u8(d0v, scale(lo, hi, x0)));
        vst1q_u8(d1, veorq_u8(d1v, scale(lo, hi, x1)));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = vld1q_u8(s);
        let d0v = vld1q_u8(&*d);
        vst1q_u8(d, veorq_u8(d0v, scale(lo, hi, x)));
    }

    mul_add_nibble(&mut dst[vector_len..span], table, &src[vector_len..span]);
}

/// `dst = coeff * dst`, 16 bytes per iteration.
///
/// Deliberately unblocked: scaling a row in place is a setup step, not a hot
/// loop, and the simple shape is the one that is obviously right. Accepts
/// buffers of any byte length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_assign_neon(_token: archmage::NeonToken, dst: &mut [u8], table: &ScaleTable) {
    mul_assign_impl(dst, table);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_assign_impl(dst: &mut [u8], table: &ScaleTable) {
    let (lanes, tail) = dst.as_chunks_mut::<16>();
    let (lo, hi) = load_tables(table);

    for lane in lanes {
        let d = vld1q_u8(&*lane);
        vst1q_u8(lane, scale(lo, hi, d));
    }

    mul_assign_nibble(tail, table);
}

/// `dst = coeff * src`, out of place, over 16-byte NEON lanes.
///
/// Fuses what would otherwise be a copy followed by an in-place scale: one
/// pass, and `dst` is never read.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_into_neon(_token: archmage::NeonToken, dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    check_equal("gf8::mul_into_neon", "dst", dst.len(), "src", src.len());
    mul_into_impl(dst, table, src);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_into_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let (lo, hi) = load_tables(table);

    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    let (src_tiles, _) = src[..pair_len].as_chunks::<32>();
    for (d_tile, s_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d0, d1) = d_tile.split_at_mut(16);
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) =
            (d0.try_into().unwrap(), d1.try_into().unwrap());
        let (s0, s1): (&[u8; 16], &[u8; 16]) = {
            let (s0, s1) = s_tile.split_at(16);
            (s0.try_into().unwrap(), s1.try_into().unwrap())
        };
        vst1q_u8(d0, scale(lo, hi, vld1q_u8(s0)));
        vst1q_u8(d1, scale(lo, hi, vld1q_u8(s1)));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        vst1q_u8(d, scale(lo, hi, vld1q_u8(s)));
    }

    mul_into_nibble(&mut dst[vector_len..span], table, &src[vector_len..span]);
}

/// `rows[j] ^= coeffs[j] * src` for every row, four rows per source load.
///
/// `rows` is `coeffs.len()` contiguous rows of `row_len` bytes, and
/// `row_len == src.len()`.
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
/// bytes and `row_len == src.len()`.
#[allow(unsafe_code)]
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn scatter_neon(
    _token: archmage::NeonToken,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
    src: &[u8],
) {
    check_equal("gf8::scatter_neon", "row_len", row_len, "src", src.len());
    check_row_span("gf8::scatter_neon", rows.len(), row_len, coeffs.len());
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checks above prove `coeffs.len() * row_len <= rows.len()`.
    // THUS: the unsafe body below only touches in-bounds rows and the
    //        `span`-byte source windows it clamps itself.
    unsafe { mul_add_scatter_impl(rows, row_len, coeffs, src) };
}

/// # Safety
///
/// `rows` is uniquely borrowed and holds whole `row_len`-byte rows; `span` is
/// clamped to the shorter of `row_len` and `src.len()`.
///
/// MEMORY VALIDITY
/// SINCE: the entry proved `coeffs.len() * row_len <= rows.len()` and the body
///        clamps the row count to the whole rows `rows` actually holds.
/// THUS: every `base.add(k * row_len)` below starts an in-bounds row span,
///        and every source read stays inside `src`.
///
/// ALIASING
/// SINCE: distinct row indices name disjoint spans of the uniquely borrowed
///        `rows`, and `src` is an independent read-only borrow.
/// THUS: no store below aliases another row's load or the source.
#[allow(unsafe_code)]
#[archmage::rite(neon)]
unsafe fn mul_add_scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let span = row_len.min(src.len());
    let nrows = coeffs.len().min(rows.len() / row_len);
    let base = rows.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= nrows {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `j + 4 <= nrows` and `nrows * row_len <= rows.len()`.
        // THUS: all four pointers start in-bounds `row_len`-byte spans.
        // ALIASING
        // SINCE: distinct row indices name disjoint spans.
        // THUS: no store below aliases another row's load.
        let quad = unsafe {
            [
                PreparedRow::new(base.add(j * row_len), coeffs[j]),
                PreparedRow::new(base.add((j + 1) * row_len), coeffs[j + 1]),
                PreparedRow::new(base.add((j + 2) * row_len), coeffs[j + 2]),
                PreparedRow::new(base.add((j + 3) * row_len), coeffs[j + 3]),
            ]
        };
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above, plus `span <= src.len()` bounds the source reads.
        // THUS: every access in `scatter_quad` stays inside its span.
        // ALIASING
        // SINCE: the four spans are pairwise disjoint and disjoint from `src`.
        // THUS: the stores below cannot alias the source or each other.
        unsafe { scatter_quad(&quad, src, span) };
        j += 4;
    }
    while j < nrows {
        let coeff = coeffs[j];
        if coeff != Elem::ZERO {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `(j + 1) * row_len <= rows.len()` and `span <= row_len`.
            // THUS: this window is in bounds for both buffers.
            // ALIASING
            // SINCE: rows are disjoint by construction.
            // THUS: the `&mut` is unique and does not alias `src`.
            let row = unsafe { core::slice::from_raw_parts_mut(base.add(j * row_len), span) };
            mul_add_impl(row, scale_table(coeff), &src[..span]);
        }
        j += 1;
    }
}

/// Fold one source into four rows: one source load, four destination updates.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: every plan pointer starts a writable span of at least `span` bytes
///        and `src` holds at least `span` bytes.
/// THUS: every load and store below stays inside its span.
///
/// ALIASING
/// SINCE: the four spans are pairwise disjoint and disjoint from `src`.
/// THUS: no store aliases another row's load or the source.
#[allow(unsafe_code)]
#[archmage::rite(neon)]
unsafe fn scatter_quad(plans: &[PreparedRow; 4], src: &[u8], span: usize) {
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset + 16 <= span {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 16 <= span <= src.len()`.
        // THUS: the source load stays inside `src`.
        let x = unsafe { vld1q_u8(src_ptr.add(offset)) };
        for plan in plans {
            if plan.scaling.kind == Kind::Skip {
                continue;
            }
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 16 <= span` bounds this row.
            // THUS: the load and store stay inside the row's span.
            // ALIASING
            // SINCE: the four rows are disjoint.
            // THUS: this store cannot alias another row or the source.
            unsafe {
                let dp = plan.ptr.add(offset);
                vst1q_u8(dp, plan.scaling.fold(vld1q_u8(dp), x));
            }
        }
        offset += 16;
    }

    if offset == span {
        return;
    }
    let tail = span - offset;
    for plan in plans {
        if plan.scaling.kind == Kind::Skip {
            continue;
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + tail == span` bytes remain in this row.
        // THUS: the tail window is in bounds.
        // ALIASING
        // SINCE: the rows are disjoint.
        // THUS: this `&mut` window is unique.
        let row = unsafe { core::slice::from_raw_parts_mut(plan.ptr.add(offset), tail) };
        mul_add_nibble(row, plan.table, &src[offset..span]);
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked: each group of four rows loads a 32-byte-per-row
/// destination tile into eight accumulators once, folds in every term, and
/// stores once. Destination memory traffic is therefore independent of
/// `terms.len()`, which is what separates this from a `scatter_neon` per term.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
/// every term supplies `nrows` coefficients, and every source is `row_len`
/// bytes.
#[allow(unsafe_code)]
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn matrix_neon(
    _token: archmage::NeonToken,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    check_row_span("gf8::matrix_neon", rows.len(), row_len, nrows);
    check_terms("gf8::matrix_neon", row_len, nrows, terms);
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checks above bound `nrows * row_len <= rows.len()` and pin
    //        every term's coefficient count and source length.
    // THUS: the unsafe body below only touches in-bounds, disjoint rows.
    unsafe { mul_add_matrix_impl(rows, row_len, nrows, terms) };
}

/// # Safety
///
/// `rows` is uniquely borrowed; every term supplies whole `row_len`-byte
/// sources and `nrows`-long coefficient slices.
///
/// MEMORY VALIDITY
/// SINCE: the entry proved `nrows * row_len <= rows.len()` and the body clamps
///        the row count and span to what the buffers actually hold.
/// THUS: every `base.add(k * row_len)` below addresses a distinct in-bounds
///        row and every source read stays inside its slice.
///
/// ALIASING
/// SINCE: distinct row indices name disjoint spans and the sources are
///        independent read-only borrows.
/// THUS: the accumulator stores never alias a source or another row.
#[allow(unsafe_code)]
#[archmage::rite(neon)]
unsafe fn mul_add_matrix_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    // One pass over `terms` — outside every hot loop — establishes the bounds
    // the raw-pointer loops rely on, so a caller that violates the documented
    // geometry gets a short update rather than out-of-bounds reads.
    let mut span = row_len;
    let mut count = nrows.min(rows.len() / row_len);
    for &(coeffs, src) in terms {
        debug_assert_eq!(coeffs.len(), nrows);
        debug_assert_eq!(src.len(), row_len);
        span = span.min(src.len());
        count = count.min(coeffs.len());
    }
    let base = rows.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= count {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `count * row_len <= rows.len()`, so each of rows `j..j + 4`
        //        starts an in-bounds `row_len`-byte span.
        // THUS: the four recorded pointers address valid memory.
        // ALIASING
        // SINCE: distinct indices give disjoint spans.
        // THUS: the four accumulator streams never alias.
        let quad = unsafe {
            [
                base.add(j * row_len),
                base.add((j + 1) * row_len),
                base.add((j + 2) * row_len),
                base.add((j + 3) * row_len),
            ]
        };
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: as above; `span` bounds every source and every row, and
        //        `j + 3 < count <= coeffs.len()` for every term.
        // THUS: `matrix_quad` reads and writes only inside those spans.
        // ALIASING
        // SINCE: the four spans are pairwise disjoint from every source.
        // THUS: no store aliases a live read.
        unsafe { matrix_quad(&quad, span, j, terms) };
        j += 4;
    }
    while j < count {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: row `j` is an in-bounds `row_len`-byte span of `rows`, and
        //        `j < count <= coeffs.len()` for every term.
        // THUS: `matrix_single` reads and writes only inside that span.
        // ALIASING
        // SINCE: rows are disjoint and the sources are read-only.
        // THUS: no store aliases another row or a source.
        unsafe { matrix_single(base.add(j * row_len), span, j, terms) };
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold every term, store once.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: each pointer in `rows` starts a writable span of at least `span`
///        bytes, every term's source holds at least `span` bytes, and every
///        term's coefficient slice has more than `first + 3` entries.
/// THUS: every load, store, and coefficient index below stays in bounds.
///
/// ALIASING
/// SINCE: the four spans are pairwise disjoint and disjoint from every
///        source.
/// THUS: the accumulator stores cannot alias a live read.
#[allow(unsafe_code)]
#[archmage::rite(neon)]
unsafe fn matrix_quad(rows: &[*mut u8; 4], span: usize, first: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 32 <= span {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 32 <= span` bounds all four rows.
        // THUS: the eight accumulator loads stay inside their spans.
        // ALIASING
        // SINCE: the rows are disjoint.
        // THUS: the loads never overlap.
        let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = unsafe {
            (
                vld1q_u8(rows[0].add(tile)),
                vld1q_u8(rows[0].add(tile + 16)),
                vld1q_u8(rows[1].add(tile)),
                vld1q_u8(rows[1].add(tile + 16)),
                vld1q_u8(rows[2].add(tile)),
                vld1q_u8(rows[2].add(tile + 16)),
                vld1q_u8(rows[3].add(tile)),
                vld1q_u8(rows[3].add(tile + 16)),
            )
        };
        for &(coeffs, src) in terms {
            // Each row's tables are resolved once per term and reused for both
            // vectors of the tile; only one row's pair is live at a time,
            // which keeps the eight accumulators off the stack.
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 32 <= span <= src.len()` bounds both source
            //        loads.
            // THUS: both source loads stay inside the term's source slice.
            let (x0, x1) = unsafe {
                let sp = src.as_ptr().add(tile);
                (vld1q_u8(sp), vld1q_u8(sp.add(16)))
            };

            let s = Scaling::new(coeffs[first]);
            a00 = s.fold(a00, x0);
            a01 = s.fold(a01, x1);

            let s = Scaling::new(coeffs[first + 1]);
            a10 = s.fold(a10, x0);
            a11 = s.fold(a11, x1);

            let s = Scaling::new(coeffs[first + 2]);
            a20 = s.fold(a20, x0);
            a21 = s.fold(a21, x1);

            let s = Scaling::new(coeffs[first + 3]);
            a30 = s.fold(a30, x0);
            a31 = s.fold(a31, x1);
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: same bounds as the loads above.
        // THUS: the stores stay inside their spans.
        // ALIASING
        // SINCE: the rows are disjoint.
        // THUS: the stores cannot alias a live read.
        unsafe {
            vst1q_u8(rows[0].add(tile), a00);
            vst1q_u8(rows[0].add(tile + 16), a01);
            vst1q_u8(rows[1].add(tile), a10);
            vst1q_u8(rows[1].add(tile + 16), a11);
            vst1q_u8(rows[2].add(tile), a20);
            vst1q_u8(rows[2].add(tile + 16), a21);
            vst1q_u8(rows[3].add(tile), a30);
            vst1q_u8(rows[3].add(tile + 16), a31);
        }
        tile += 32;
    }

    if tile + 16 <= span {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 16 <= span` bounds all four rows.
        // THUS: the four accumulator loads stay inside their spans.
        // ALIASING
        // SINCE: the rows are disjoint.
        // THUS: the loads never overlap.
        let (mut a0, mut a1, mut a2, mut a3) = unsafe {
            (
                vld1q_u8(rows[0].add(tile)),
                vld1q_u8(rows[1].add(tile)),
                vld1q_u8(rows[2].add(tile)),
                vld1q_u8(rows[3].add(tile)),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 16 <= span <= src.len()`.
            // THUS: the source load stays inside the term's source slice.
            let x = unsafe { vld1q_u8(src.as_ptr().add(tile)) };
            a0 = Scaling::new(coeffs[first]).fold(a0, x);
            a1 = Scaling::new(coeffs[first + 1]).fold(a1, x);
            a2 = Scaling::new(coeffs[first + 2]).fold(a2, x);
            a3 = Scaling::new(coeffs[first + 3]).fold(a3, x);
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: same bounds and disjointness as the loads above.
        // THUS: the stores stay inside their spans.
        // ALIASING
        // SINCE: the rows are disjoint.
        // THUS: the stores cannot alias a live read.
        unsafe {
            vst1q_u8(rows[0].add(tile), a0);
            vst1q_u8(rows[1].add(tile), a1);
            vst1q_u8(rows[2].add(tile), a2);
            vst1q_u8(rows[3].add(tile), a3);
        }
        tile += 16;
    }

    if tile < span {
        let tail = span - tile;
        for (slot, &ptr) in rows.iter().enumerate() {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tail` bytes remain in this row.
            // THUS: the tail window is in bounds.
            // ALIASING
            // SINCE: the rows are disjoint.
            // THUS: this `&mut` window is unique.
            let row = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), tail) };
            for &(coeffs, src) in terms {
                let coeff = coeffs[first + slot];
                if coeff != Elem::ZERO {
                    mul_add_nibble(row, scale_table(coeff), &src[tile..span]);
                }
            }
        }
    }
}

/// Register-blocked single row, for the 1-3 rows a group of four leaves over.
///
/// Still loads its destination once and folds in every term, so the blocking
/// property holds for the remainder as well.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: `ptr` starts a writable span of at least `span` bytes that no other
///        row and no source aliases, every term's source holds at least
///        `span` bytes, and every term's coefficient slice has more than
///        `index` entries.
/// THUS: every load, store, and coefficient index below stays in bounds.
///
/// ALIASING
/// SINCE: the row is uniquely borrowed and disjoint from every source.
/// THUS: no store aliases a live read.
#[allow(unsafe_code)]
#[archmage::rite(neon)]
unsafe fn matrix_single(ptr: *mut u8, span: usize, index: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 32 <= span {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 32 <= span` bounds the row.
        // THUS: both accumulator loads stay inside the span.
        let (mut a0, mut a1) = unsafe { (vld1q_u8(ptr.add(tile)), vld1q_u8(ptr.add(tile + 16))) };
        for &(coeffs, src) in terms {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 32 <= span <= src.len()`.
            // THUS: both source loads stay inside the term's source slice.
            let (x0, x1) = unsafe {
                let sp = src.as_ptr().add(tile);
                (vld1q_u8(sp), vld1q_u8(sp.add(16)))
            };
            let s = Scaling::new(coeffs[index]);
            a0 = s.fold(a0, x0);
            a1 = s.fold(a1, x1);
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: same bounds as the loads above.
        // THUS: the stores stay inside the span.
        unsafe {
            vst1q_u8(ptr.add(tile), a0);
            vst1q_u8(ptr.add(tile + 16), a1);
        }
        tile += 32;
    }

    if tile + 16 <= span {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tile + 16 <= span` bounds the row.
        // THUS: the accumulator load stays inside the span.
        let mut a = unsafe { vld1q_u8(ptr.add(tile)) };
        for &(coeffs, src) in terms {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `tile + 16 <= span <= src.len()`.
            // THUS: the source load stays inside the term's source slice.
            let x = unsafe { vld1q_u8(src.as_ptr().add(tile)) };
            a = Scaling::new(coeffs[index]).fold(a, x);
        }
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: same bounds as the load above.
        // THUS: the store stays inside the span.
        unsafe { vst1q_u8(ptr.add(tile), a) };
        tile += 16;
    }

    if tile < span {
        let tail = span - tile;
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `tail` bytes remain in this row.
        // THUS: the tail window is in bounds.
        // ALIASING
        // SINCE: nothing else aliases this row.
        // THUS: the `&mut` window is unique.
        let row = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), tail) };
        for &(coeffs, src) in terms {
            let coeff = coeffs[index];
            if coeff != Elem::ZERO {
                mul_add_nibble(row, scale_table(coeff), &src[tile..span]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1: gather and varying-operand multiply.
// ---------------------------------------------------------------------------

/// Fold many sources into one destination while keeping each 32-byte
/// destination tile in registers across every source.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn gather_neon(_token: archmage::NeonToken, dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    check_equal(
        "gf8::gather_neon",
        "coefficients",
        coeffs.len(),
        "sources",
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        check_equal(
            "gf8::gather_neon",
            "dst",
            dst.len(),
            format_args!("source {index}"),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    mul_add_gather_impl(dst, coeffs, srcs);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_add_gather_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let count = coeffs.len().min(srcs.len());
    let mut span = dst.len();
    for &src in &srcs[..count] {
        span = span.min(src.len());
    }
    let pair_len = span & !31;
    let vector_len = span & !15;
    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    for (t, d_tile) in dst_tiles.iter_mut().enumerate() {
        let (d0, d1) = d_tile.split_at_mut(16);
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) =
            (d0.try_into().unwrap(), d1.try_into().unwrap());
        let mut acc0 = vld1q_u8(&*d0);
        let mut acc1 = vld1q_u8(&*d1);
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            // `pair_len <= span <= srcs[k].len()` bounds this 32-byte source
            // tile.
            let s_tile: &[u8; 32] = srcs[k][t * 32..t * 32 + 32].try_into().unwrap();
            let (s0, s1): (&[u8; 16], &[u8; 16]) = {
                let (s0, s1) = s_tile.split_at(16);
                (s0.try_into().unwrap(), s1.try_into().unwrap())
            };
            acc0 = plan.fold(acc0, vld1q_u8(s0));
            acc1 = plan.fold(acc1, vld1q_u8(s1));
        }
        vst1q_u8(d0, acc0);
        vst1q_u8(d1, acc1);
    }
    if vector_len > pair_len {
        let mut acc = vld1q_u8(dst[pair_len..vector_len].first_chunk().unwrap());
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            acc = plan.fold(
                acc,
                vld1q_u8(srcs[k][pair_len..vector_len].first_chunk().unwrap()),
            );
        }
        vst1q_u8(dst[pair_len..vector_len].first_chunk_mut().unwrap(), acc);
    }

    for k in 0..count {
        let coeff = coeffs[k];
        if coeff != Elem::ZERO {
            mul_add_nibble(
                &mut dst[vector_len..span],
                scale_table(coeff),
                &srcs[k][vector_len..span],
            );
        }
    }
}

/// Lane-parallel GF(2^8) multiplication for two varying byte vectors.
///
/// `PMULL` is in `AArch64`'s optional crypto extension, not baseline NEON, so
/// the portable NEON backend uses eight branchless shift/reduce rounds. Every
/// round processes all 16 lanes and the scalar tail is shorter than one lane.
#[inline]
#[archmage::rite(neon, import_intrinsics)]
fn multiply_vectors(mut a: uint8x16_t, mut b: uint8x16_t) -> uint8x16_t {
    let one = vdupq_n_u8(1);
    let high_bit = vdupq_n_u8(0x80);
    let reduction = vdupq_n_u8(0x1b);
    let mut product = vdupq_n_u8(0);
    for _ in 0..8 {
        let active = vceqq_u8(vandq_u8(b, one), one);
        product = veorq_u8(product, vandq_u8(a, active));
        let reduce = vceqq_u8(vandq_u8(a, high_bit), high_bit);
        a = veorq_u8(vshlq_n_u8(a, 1), vandq_u8(reduction, reduce));
        b = vshrq_n_u8(b, 1);
    }
    product
}

/// Eight lane-independent carry-less byte products reduced by the AES
/// polynomial. `PMULL` widens each byte lane to 16 bits; two fixed folds of
/// the high half suffice because an 8-by-8 polynomial product has degree 14.
#[inline]
#[archmage::rite(neon_aes, import_intrinsics)]
fn multiply8_pmull(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let product = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(a), vreinterpret_p8_u8(b)));
    let mask = vdupq_n_u16(0x00ff);
    let high = vshrq_n_u16(product, 8);
    let mut reduced = veorq_u16(vandq_u16(product, mask), high);
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 1));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 3));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 4));
    let high = vshrq_n_u16(reduced, 8);
    reduced = veorq_u16(vandq_u16(reduced, mask), high);
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 1));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 3));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 4));
    vmovn_u16(reduced)
}

/// Sixteen lane-independent base-field products using two `PMULL`s.
#[inline]
#[archmage::rite(neon_aes, import_intrinsics)]
pub(super) fn multiply_vectors_pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vcombine_u8(
        multiply8_pmull(vget_low_u8(a), vget_low_u8(b)),
        multiply8_pmull(vget_high_u8(a), vget_high_u8(b)),
    )
}

// Why there is no fixed-coefficient `PMULL` kernel here.
//
// A broadcast coefficient makes PMULL table-free, which looks attractive next
// to a `ScaleTable`. It was written, measured, and lost by a wide margin at
// every size, for both GF(2^8) and the GF(2^16) tower form (BENCHMARKS.md).
// The arithmetic explains it — two `vmull_p8`s plus a twenty-instruction
// reduction network per 16 bytes against `vqtbl1q_u8`'s five — and no core
// makes PMULL fast enough to close a gap that large.
// The only shape where that reduction is cheaper than the alternative is a
// *varying* operand pair, where the alternative is eight bit-serial rounds:
// hence `elementwise_pmull` below, and nothing else.

/// `dst[i] = a[i] * b[i]` using the optional `AArch64` crypto extension.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn elementwise_pmull(_token: archmage::NeonAesToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("gf8::elementwise_pmull", "dst", dst.len(), "a", a.len());
    check_equal("gf8::elementwise_pmull", "dst", dst.len(), "b", b.len());
    elementwise_pmull_impl(dst, a, b);
}

#[archmage::rite(neon_aes, import_intrinsics)]
fn elementwise_pmull_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (a_lanes, a_tail) = a.as_chunks::<16>();
    let (b_lanes, b_tail) = b.as_chunks::<16>();
    for ((d, x), y) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        vst1q_u8(d, multiply_vectors_pmull(vld1q_u8(x), vld1q_u8(y)));
    }
    for ((d, &x), &y) in dst_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}

/// `dst[i] = a[i] * b[i]` over byte field elements.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn elementwise_neon(_token: archmage::NeonToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("gf8::elementwise_neon", "dst", dst.len(), "a", a.len());
    check_equal("gf8::elementwise_neon", "dst", dst.len(), "b", b.len());
    elementwise_impl(dst, a, b);
}

#[archmage::rite(neon, import_intrinsics)]
fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let (dst_lanes, dst_tail) = dst.as_chunks_mut::<16>();
    let (a_lanes, a_tail) = a.as_chunks::<16>();
    let (b_lanes, b_tail) = b.as_chunks::<16>();
    for ((d, x), y) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        vst1q_u8(d, multiply_vectors(vld1q_u8(x), vld1q_u8(y)));
    }
    for ((d, &x), &y) in dst_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}
