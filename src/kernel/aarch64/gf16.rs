//! GF(2^16) NEON kernels.
//!
//! Every multiply here is the tower identity documented on
//! [`TowerCoeff`](crate::kernel::tables::TowerCoeff). With interleaved bytes
//! `[a, b]` meaning `a + b*u`, multiplying by `c0 + c1*u` gives
//!
//! ```text
//! even lane:  c0*a       ^  (DELTA*c1)*b
//! odd  lane:  (c0+c1)*b  ^  c1*a
//! ```
//!
//! `b` is `a`'s adjacent-byte neighbour, so the crossed half is the same
//! alternating byte multiply applied to `vrev16q_u8(source)`, and the four
//! base coefficients are exactly
//! [`TowerTables::factors`].
//!
//! NEON has no byte-wide field multiply, so each base coefficient is a
//! split-nibble `vqtbl1q_u8` pair — two lookups and an XOR. A whole 16-byte
//! block therefore costs twenty instructions: five to split the source and
//! its swap into nibble indices, eight lookups, and seven to recombine. The
//! bit-serial alternative a table-free port would need is eight rounds of
//! shift-and-reduce per base multiply, an order of magnitude worse.
//!
//! Every entry is a safe [`archmage`] capability-token function taking the
//! exact token its instructions require and validating its own geometry;
//! inner loops are `#[rite]` helpers inside the token-proven region. Unsafe
//! remains only where no safe primitive expresses the structure — the
//! offset-addressed rows of the scatter and matrix bodies — each carrying a
//! per-item `#[allow(unsafe_code)]` and local SINCE–THUS proofs.
//!
//! [`archmage`]: https://docs.rs/archmage

use core::arch::aarch64::*;

use crate::field::gf16::Elem;
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::proven_checks::{check_elem_multiple, check_equal, check_row_span, check_terms};
use crate::kernel::tables::TowerTables;

/// Terms folded into a single register-resident destination pass.
///
/// [`matrix_neon`] holds its destination tile in registers while it folds in
/// every term of a block, so destination traffic is one load/store pass per
/// block rather than one per term. The block is bounded only because each
/// `(term, row)` pair needs its own eight lookup vectors and the kernel
/// caches them in a fixed-size stack array rather than allocating.
const TERM_BLOCK: usize = 8;

/// The eight lookup vectors one GF(2^16) coefficient needs.
///
/// `lo[i]` and `hi[i]` are the nibble tables of
/// [`TowerTables::factors`]`[i]`,
/// that is of `[c0, c0+c1, DELTA*c1, c1]`.
#[derive(Clone, Copy)]
struct Factors {
    /// Low-nibble tables, one per base-field factor.
    lo: [uint8x16_t; 4],
    /// High-nibble tables, one per base-field factor.
    hi: [uint8x16_t; 4],
}

/// A `Factors` whose every product is zero, used to fill unwritten cache
/// slots.
#[inline]
#[archmage::rite(neon)]
fn empty_factors() -> Factors {
    let zero = vdupq_n_u8(0);
    Factors {
        lo: [zero; 4],
        hi: [zero; 4],
    }
}

/// Hoist a coefficient's four nibble table pairs into registers.
#[inline]
#[archmage::rite(neon, import_intrinsics)]
fn load_factors(tables: &TowerTables) -> Factors {
    let f = &tables.factors;
    Factors {
        lo: [
            vld1q_u8(&f[0].lo),
            vld1q_u8(&f[1].lo),
            vld1q_u8(&f[2].lo),
            vld1q_u8(&f[3].lo),
        ],
        hi: [
            vld1q_u8(&f[0].hi),
            vld1q_u8(&f[1].hi),
            vld1q_u8(&f[2].hi),
            vld1q_u8(&f[3].hi),
        ],
    }
}

/// The nibble indices one source block feeds to every lookup.
///
/// Split once per block and reused by every row of a register-blocked group:
/// the five instructions it costs are then amortized across the group.
#[derive(Clone, Copy)]
struct Nibbles {
    /// Low nibbles of the block.
    source_lo: uint8x16_t,
    /// High nibbles of the block.
    source_hi: uint8x16_t,
    /// Low nibbles of the adjacent-byte-swapped block.
    swapped_lo: uint8x16_t,
    /// High nibbles of the adjacent-byte-swapped block.
    swapped_hi: uint8x16_t,
}

/// Split a source block and its adjacent-byte swap into nibble indices.
#[inline]
#[archmage::rite(neon)]
fn split(source: uint8x16_t) -> Nibbles {
    let mask = vdupq_n_u8(0x0f);
    // `vrev16q_u8` exchanges the two bytes of every element, pairing each
    // component with the one the crossed factors must multiply.
    let swapped = vrev16q_u8(source);
    Nibbles {
        source_lo: vandq_u8(source, mask),
        source_hi: vshrq_n_u8(source, 4),
        swapped_lo: vandq_u8(swapped, mask),
        swapped_hi: vshrq_n_u8(swapped, 4),
    }
}

/// `coeff * source` for one block, from pre-split nibbles.
#[inline]
#[archmage::rite(neon)]
fn scaled(nibbles: Nibbles, factors: &Factors) -> uint8x16_t {
    let Nibbles {
        source_lo,
        source_hi,
        swapped_lo,
        swapped_hi,
    } = nibbles;
    let direct_even = veorq_u8(
        vqtbl1q_u8(factors.lo[0], source_lo),
        vqtbl1q_u8(factors.hi[0], source_hi),
    );
    let direct_odd = veorq_u8(
        vqtbl1q_u8(factors.lo[1], source_lo),
        vqtbl1q_u8(factors.hi[1], source_hi),
    );
    let cross_even = veorq_u8(
        vqtbl1q_u8(factors.lo[2], swapped_lo),
        vqtbl1q_u8(factors.hi[2], swapped_hi),
    );
    let cross_odd = veorq_u8(
        vqtbl1q_u8(factors.lo[3], swapped_lo),
        vqtbl1q_u8(factors.hi[3], swapped_hi),
    );
    // Even byte lanes carry the constant component and odd lanes the
    // extension component, so `0x00ff` per 16-bit element selects the
    // even-lane result and rejects the odd-lane one.
    let even_lanes = vreinterpretq_u8_u16(vdupq_n_u16(0x00ff));
    vbslq_u8(
        even_lanes,
        veorq_u8(direct_even, cross_even),
        veorq_u8(direct_odd, cross_odd),
    )
}

/// `coeff * source` for a block that no other row shares.
#[inline]
#[archmage::rite(neon)]
fn scaled_vector(source: uint8x16_t, factors: &Factors) -> uint8x16_t {
    scaled(split(source), factors)
}

/// `dst ^= coeff * src` over interleaved GF(2^16) elements, two 16-byte lanes
/// at a time.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_add_neon(
    _token: archmage::NeonToken,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_equal("gf16::mul_add_neon", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gf16::mul_add_neon", dst.len(), 2);
    mul_add_impl(dst, tables, src);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_add_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

    // Two independent lanes per iteration: eight table lookups per lane have
    // enough latency to hide the other lane's loads and nibble splits behind
    // (BENCHMARKS.md).
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
        let p0 = scaled_vector(vld1q_u8(s0), &factors);
        let p1 = scaled_vector(vld1q_u8(s1), &factors);
        let d0v = vld1q_u8(&*d0);
        let d1v = vld1q_u8(&*d1);
        vst1q_u8(d0, veorq_u8(d0v, p0));
        vst1q_u8(d1, veorq_u8(d1v, p1));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let p = scaled_vector(vld1q_u8(s), &factors);
        let cur = vld1q_u8(&*d);
        vst1q_u8(d, veorq_u8(cur, p));
    }

    mul_add_scalar(&mut dst[vector_len..span], tables.coeff, &src[vector_len..span]);
}

/// `dst = coeff * dst` over interleaved GF(2^16) elements.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_assign_neon(_token: archmage::NeonToken, dst: &mut [u8], tables: &TowerTables) {
    check_elem_multiple("gf16::mul_assign_neon", dst.len(), 2);
    mul_assign_impl(dst, tables);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_assign_impl(dst: &mut [u8], tables: &TowerTables) {
    let vector_len = dst.len() & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    for d_tile in dst_tiles {
        let (d0, d1) = d_tile.split_at_mut(16);
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) =
            (d0.try_into().unwrap(), d1.try_into().unwrap());
        let p0 = scaled_vector(vld1q_u8(&*d0), &factors);
        let p1 = scaled_vector(vld1q_u8(&*d1), &factors);
        vst1q_u8(d0, p0);
        vst1q_u8(d1, p1);
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    for d in dst_lanes {
        vst1q_u8(d, scaled_vector(vld1q_u8(&*d), &factors));
    }

    mul_assign_scalar(&mut dst[vector_len..], tables.coeff);
}

/// `dst = coeff * src` out of place, two 16-byte lanes at a time.
///
/// Fuses what would otherwise be a copy followed by an in-place scale: one
/// pass over the destination, which is never read.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_into_neon(
    _token: archmage::NeonToken,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_equal("gf16::mul_into_neon", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gf16::mul_into_neon", dst.len(), 2);
    mul_into_impl(dst, tables, src);
}

#[archmage::rite(neon, import_intrinsics)]
fn mul_into_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

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
        vst1q_u8(d0, scaled_vector(vld1q_u8(s0), &factors));
        vst1q_u8(d1, scaled_vector(vld1q_u8(s1), &factors));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        vst1q_u8(d, scaled_vector(vld1q_u8(s), &factors));
    }

    mul_into_scalar(&mut dst[vector_len..span], tables.coeff, &src[vector_len..span]);
}

/// `row ^= src`, the whole job when a scattered coefficient is one.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: the caller proves `row..row + span` is a valid, uniquely borrowed
///        range and `span` does not exceed `src.len()`.
/// THUS: every load and store below, including the tail window, stays inside
///        that range and inside `src`.
///
/// ALIASING
/// SINCE: the row range is uniquely borrowed.
/// THUS: the tail `&mut` window is unique.
#[allow(unsafe_code)]
#[archmage::rite(neon, import_intrinsics)]
unsafe fn xor_row(row: *mut u8, span: usize, src: &[u8]) {
    let len = span & !15;
    let src_ptr = src.as_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 16 <= len <= span <= src.len()`.
        // THUS: both loads and the store stay inside the row and the source.
        unsafe {
            let dst_ptr = row.add(offset);
            let current = vld1q_u8(dst_ptr);
            let source = vld1q_u8(src_ptr.add(offset));
            vst1q_u8(dst_ptr, veorq_u8(current, source));
        }
        offset += 16;
    }
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: `len..span` is the in-bounds tail of the proven row range.
    // THUS: the tail window is valid for writes and reads.
    // ALIASING
    // SINCE: the row is uniquely borrowed.
    // THUS: the tail `&mut` is unique.
    let tail = unsafe { core::slice::from_raw_parts_mut(row.add(len), span - len) };
    for (d, &s) in tail.iter_mut().zip(&src[len..span]) {
        *d ^= s;
    }
}

/// Fold `src` into `N` rows at once.
///
/// The source block is loaded and split once per 16-byte tile and reused by
/// every row of the group, so the split amortizes `N` ways and the source
/// is read once instead of `N` times.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: for every `k`, `base + starts[k]..+ span` is a valid range inside
///        one allocation and `span` does not exceed `src.len()`.
/// THUS: every row window below and every source load stays in bounds.
///
/// ALIASING
/// SINCE: the `N` ranges are pairwise disjoint and uniquely borrowed.
/// THUS: no store below aliases another row's load.
#[allow(unsafe_code)]
#[archmage::rite(neon, import_intrinsics)]
unsafe fn scatter_group<const N: usize>(
    base: *mut u8,
    span: usize,
    starts: [usize; N],
    factors: [Factors; N],
    coeffs: [Elem; N],
    src: &[u8],
) {
    let len = span & !15;
    let src_ptr = src.as_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `offset + 16 <= len <= span`, so the source window and every
        //        row window are in bounds.
        // THUS: all loads and stores below stay inside their ranges.
        // ALIASING
        // SINCE: the rows are disjoint by contract.
        // THUS: the stores cannot alias each other or the source.
        unsafe {
            let nibbles = split(vld1q_u8(src_ptr.add(offset)));
            for (&start, factor) in starts.iter().zip(&factors) {
                let row = base.add(start + offset);
                vst1q_u8(row, veorq_u8(vld1q_u8(row), scaled(nibbles, factor)));
            }
        }
        offset += 16;
    }
    for (&start, &coeff) in starts.iter().zip(&coeffs) {
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `start + len..start + span` is the in-bounds tail of one
        //        disjoint row.
        // THUS: the tail window is valid.
        // ALIASING
        // SINCE: the rows are disjoint and the borrow ends before the next
        //        iteration.
        // THUS: the tail `&mut` is unique.
        let tail = unsafe { core::slice::from_raw_parts_mut(base.add(start + len), span - len) };
        mul_add_scalar(tail, coeff, &src[len..span]);
    }
}

/// `rows[j] ^= coeffs[j] * src` for every row, four rows per source load.
///
/// A zero-length row with a zero-length source is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len`
/// bytes (whole elements) and `row_len == src.len()`.
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
    check_equal("gf16::scatter_neon", "row_len", row_len, "src", src.len());
    check_elem_multiple("gf16::scatter_neon", row_len, 2);
    check_row_span("gf16::scatter_neon", rows.len(), row_len, coeffs.len());
    if row_len == 0 || coeffs.is_empty() || src.is_empty() {
        return;
    }
    // SAFETY:
    // MEMORY VALIDITY
    // SINCE: the checks above bound `coeffs.len() * row_len <= rows.len()`
    //        and the body clamps the row count to the whole rows `rows`
    //        holds.
    // THUS: every `base.add(j * row_len)` below addresses an in-bounds row.
    unsafe { mul_add_scatter_impl(rows, row_len, coeffs, src) };
}

/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: the entry proved `coeffs.len() * row_len <= rows.len()` and the
///        body clamps the row count to what `rows` holds and the span to
///        what `src` provides.
/// THUS: every `base.add(j * row_len)` below addresses a distinct in-bounds
///        row and every source read stays inside `src`.
///
/// ALIASING
/// SINCE: distinct row indices name disjoint spans of the uniquely borrowed
///        `rows`.
/// THUS: the row groups below never alias each other.
#[allow(unsafe_code)]
#[archmage::rite(neon, import_intrinsics)]
unsafe fn mul_add_scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let nrows = coeffs.len().min(rows.len() / row_len);
    let span = row_len.min(src.len());
    let base = rows.as_mut_ptr();

    // Rows with a general coefficient accumulate into groups of four; zero
    // rows drop out entirely and unit rows degenerate to a plain XOR, both
    // of which are common in coding matrices.
    let mut starts = [0usize; 4];
    let mut factors = [empty_factors(); 4];
    let mut group = [Elem::ZERO; 4];
    let mut count = 0usize;

    for (j, &coeff) in coeffs.iter().take(nrows).enumerate() {
        if coeff == Elem::ZERO {
            continue;
        }
        let start = j * row_len;
        if coeff == Elem::ONE {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `j < nrows <= rows.len() / row_len`, so this row spans
            //        `start..start + row_len` inside `rows`, and
            //        `span <= row_len`.
            // THUS: `xor_row` touches only this in-bounds row and the source.
            unsafe { xor_row(base.add(start), span, src) };
            continue;
        }
        starts[count] = start;
        factors[count] = load_factors(&TowerTables::new(coeff));
        group[count] = coeff;
        count += 1;
        if count == 4 {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the four starts are distinct multiples of `row_len`
            //        below `nrows * row_len <= rows.len()`.
            // THUS: every row window is in bounds.
            // ALIASING
            // SINCE: distinct multiples of `row_len` name disjoint spans.
            // THUS: the groups never alias.
            unsafe { scatter_group::<4>(base, span, starts, factors, group, src) };
            count = 0;
        }
    }

    match count {
        // SAFETY (every arm): same bounds and disjointness argument as the
        // full group above, over the leading `count` slots.
        1 => unsafe {
            scatter_group::<1>(base, span, [starts[0]], [factors[0]], [group[0]], src);
        },
        2 => unsafe {
            scatter_group::<2>(
                base,
                span,
                [starts[0], starts[1]],
                [factors[0], factors[1]],
                [group[0], group[1]],
                src,
            );
        },
        3 => unsafe {
            scatter_group::<3>(
                base,
                span,
                [starts[0], starts[1], starts[2]],
                [factors[0], factors[1], factors[2]],
                [group[0], group[1], group[2]],
                src,
            );
        },
        _ => {}
    }
}

/// What one `(term, row)` pair contributes to a destination tile.
#[derive(Clone, Copy)]
enum Mode {
    /// Coefficient zero: nothing to fold in.
    Skip,
    /// Coefficient one: a plain XOR, no lookup.
    Xor,
    /// Anything else: the full tower multiply.
    Mul,
}

/// Apply every term to `N` rows, holding the destination tile in registers.
///
/// Terms are folded [`TERM_BLOCK`] at a time: a pass loads the tile once,
/// XORs in every term of the block, and stores once, so destination traffic
/// is `terms.len() / TERM_BLOCK` passes instead of one per term. The lookup
/// vectors for the block's `(term, row)` pairs are derived once per pass
/// into a stack cache and never rebuilt inside the byte loop.
///
/// # Safety
///
/// MEMORY VALIDITY
/// SINCE: for every `k`, `base + starts[k]..+ span` is a valid range inside
///        one allocation, and every term supplies at least `span` source
///        bytes and at least `g + N` coefficients.
/// THUS: every row window, source load, and coefficient index below stays in
///        bounds.
///
/// ALIASING
/// SINCE: the `N` ranges are pairwise disjoint and uniquely borrowed.
/// THUS: no store below aliases another row's load.
#[allow(unsafe_code)]
#[archmage::rite(neon, import_intrinsics)]
unsafe fn matrix_group<const N: usize>(
    base: *mut u8,
    span: usize,
    starts: [usize; N],
    g: usize,
    terms: &[(&[Elem], &[u8])],
) {
    let len = span & !15;

    for block in terms.chunks(TERM_BLOCK) {
        let mut cache = [[empty_factors(); N]; TERM_BLOCK];
        let mut modes = [[Mode::Skip; N]; TERM_BLOCK];
        for ((&(coeffs, _), slots), kinds) in block.iter().zip(&mut cache).zip(&mut modes) {
            let group = &coeffs[g..g + N];
            for ((slot, kind), &coeff) in slots.iter_mut().zip(kinds.iter_mut()).zip(group) {
                *kind = if coeff == Elem::ZERO {
                    Mode::Skip
                } else if coeff == Elem::ONE {
                    Mode::Xor
                } else {
                    *slot = load_factors(&TowerTables::new(coeff));
                    Mode::Mul
                };
            }
        }

        let mut offset = 0;
        while offset < len {
            let mut tile = [vdupq_n_u8(0); N];
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: `offset + 16 <= len <= span`, so every row window is in
            //        bounds.
            // THUS: the tile loads stay inside their rows.
            // ALIASING
            // SINCE: the rows are disjoint by contract.
            // THUS: the loads never overlap.
            unsafe {
                for (acc, &start) in tile.iter_mut().zip(&starts) {
                    *acc = vld1q_u8(base.add(start + offset));
                }
            }
            for ((&(_, src), slots), kinds) in block.iter().zip(&cache).zip(&modes) {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `offset + 16 <= len <= span <= src.len()`.
                // THUS: the source load stays inside the term's source.
                let source = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
                let nibbles = split(source);
                for ((acc, factor), kind) in tile.iter_mut().zip(slots).zip(kinds) {
                    match kind {
                        Mode::Skip => {}
                        Mode::Xor => *acc = veorq_u8(*acc, source),
                        Mode::Mul => *acc = veorq_u8(*acc, scaled(nibbles, factor)),
                    }
                }
            }
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the same windows that were just read.
            // THUS: the tile stores stay inside their rows.
            // ALIASING
            // SINCE: the rows are disjoint.
            // THUS: the stores cannot alias a live read.
            unsafe {
                for (acc, &start) in tile.iter().zip(&starts) {
                    vst1q_u8(base.add(start + offset), *acc);
                }
            }
            offset += 16;
        }

        if len < span {
            for (k, &start) in starts.iter().enumerate() {
                // SAFETY:
                // MEMORY VALIDITY
                // SINCE: `start + len..start + span` is the in-bounds tail of
                //        one disjoint row.
                // THUS: the tail window is valid.
                // ALIASING
                // SINCE: the rows are disjoint and the borrow ends with this
                //        iteration.
                // THUS: the tail `&mut` is unique.
                let tail =
                    unsafe { core::slice::from_raw_parts_mut(base.add(start + len), span - len) };
                for &(coeffs, src) in block {
                    mul_add_scalar(tail, coeffs[g + k], &src[len..span]);
                }
            }
        }
    }
}

/// Apply every `(coeffs, src)` term to the leading `nrows` rows.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
/// (whole elements), every term supplies `nrows` coefficients, and every
/// source is `row_len` bytes.
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
    check_elem_multiple("gf16::matrix_neon", row_len, 2);
    check_row_span("gf16::matrix_neon", rows.len(), row_len, nrows);
    check_terms("gf16::matrix_neon", row_len, nrows, terms);
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
/// MEMORY VALIDITY
/// SINCE: the entry proved `nrows * row_len <= rows.len()` and the body
///        clamps the row count and span to what the buffers hold.
/// THUS: every `base.add(k * row_len)` below addresses a distinct in-bounds
///        row and every source read stays inside its slice.
///
/// ALIASING
/// SINCE: distinct row indices name disjoint spans and the sources are
///        independent read-only borrows.
/// THUS: the accumulator stores never alias a source or another row.
#[allow(unsafe_code)]
#[archmage::rite(neon, import_intrinsics)]
unsafe fn mul_add_matrix_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    let mut count = nrows.min(rows.len() / row_len);
    let mut span = row_len;
    for &(coeffs, src) in terms {
        count = count.min(coeffs.len());
        span = span.min(src.len());
    }
    if count == 0 || span == 0 {
        return;
    }

    let base = rows.as_mut_ptr();
    let mut g = 0;
    while g + 4 <= count {
        let starts = [
            g * row_len,
            (g + 1) * row_len,
            (g + 2) * row_len,
            (g + 3) * row_len,
        ];
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the four starts are distinct multiples of `row_len` below
        //        `count * row_len <= rows.len()`.
        // THUS: every row window is in bounds.
        // ALIASING
        // SINCE: distinct multiples of `row_len` name disjoint spans.
        // THUS: the groups never alias.
        unsafe { matrix_group::<4>(base, span, starts, g, terms) };
        g += 4;
    }

    match count - g {
        // SAFETY (every arm): same bounds and disjointness argument as the
        // full group above, over the `count - g` rows that remain.
        1 => unsafe { matrix_group::<1>(base, span, [g * row_len], g, terms) },
        2 => unsafe {
            matrix_group::<2>(base, span, [g * row_len, (g + 1) * row_len], g, terms);
        },
        3 => unsafe {
            matrix_group::<3>(
                base,
                span,
                [g * row_len, (g + 1) * row_len, (g + 2) * row_len],
                g,
                terms,
            );
        },
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tier 1: gather and varying-operand multiply.
// ---------------------------------------------------------------------------

/// Fold many tower-field sources into one destination.
///
/// Coefficients are prepared in blocks of eight, matching [`matrix_neon`].
/// Each block then sweeps the destination once; the bounded block avoids an
/// allocation while keeping the expensive four-table preparation out of the
/// byte loop.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches `dst`
/// in length (whole elements).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn gather_neon(
    _token: archmage::NeonToken,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_equal(
        "gf16::gather_neon",
        "coefficients",
        coeffs.len(),
        "sources",
        srcs.len(),
    );
    check_elem_multiple("gf16::gather_neon", dst.len(), 2);
    for (index, &src) in srcs.iter().enumerate() {
        check_equal(
            "gf16::gather_neon",
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
    let vector_len = dst.len() & !15;
    let (dst_lanes, dst_tail) = dst[..vector_len].as_chunks_mut::<16>();
    for block in (0..coeffs.len()).step_by(TERM_BLOCK) {
        let count = (coeffs.len() - block).min(TERM_BLOCK);
        let mut factors = [empty_factors(); TERM_BLOCK];
        for i in 0..count {
            factors[i] = load_factors(&TowerTables::new(coeffs[block + i]));
        }
        for (t, d) in dst_lanes.iter_mut().enumerate() {
            let mut acc = vld1q_u8(&*d);
            for i in 0..count {
                // Every source has exactly `dst.len()` bytes, so this
                // 16-byte source window is in bounds.
                let source: &[u8; 16] = srcs[block + i][t * 16..t * 16 + 16]
                    .try_into()
                    .unwrap();
                acc = veorq_u8(acc, scaled_vector(vld1q_u8(source), &factors[i]));
            }
            vst1q_u8(d, acc);
        }
        for i in 0..count {
            mul_add_scalar(
                dst_tail,
                coeffs[block + i],
                &srcs[block + i][vector_len..],
            );
        }
    }
}

/// Lane-parallel base-field multiply for two varying byte vectors.
///
/// `AArch64` guarantees NEON but not the optional crypto extension containing
/// `PMULL`; eight branchless shift/reduce rounds therefore form the portable
/// vector primitive used by both supported fields.
#[inline]
#[archmage::rite(neon)]
fn multiply_base_vectors(mut a: uint8x16_t, mut b: uint8x16_t) -> uint8x16_t {
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

// The tower form of the same identity over `PMULL` — two period-2 broadcasts
// (`[c0, c0+c1]` on the block, `[DELTA*c1, c1]` on its adjacent-byte swap),
// no nibble tables at all — was written and measured far behind the
// four-shuffle kernels above at every row length. Cheap preparation only paid
// on rows below one vector, and only for one-shot calls, which is not worth
// carrying a second prepared form for. See the note in `super::gf8` for the
// instruction-count reason, and BENCHMARKS.md for the numbers.

/// `dst[i] = a[i] * b[i]` over interleaved tower elements.
///
/// # Panics
/// Panics unless all three buffers match in length (whole elements).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn elementwise_neon(_token: archmage::NeonToken, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("gf16::elementwise_neon", "dst", dst.len(), "a", a.len());
    check_equal("gf16::elementwise_neon", "dst", dst.len(), "b", b.len());
    check_elem_multiple("gf16::elementwise_neon", dst.len(), 2);
    elementwise_impl(dst, a, b);
}

#[archmage::rite(neon, import_intrinsics)]
fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let even = vreinterpretq_u8_u16(vdupq_n_u16(0x00ff));
    let delta_even = vreinterpretq_u8_u16(vdupq_n_u16(u16::from_le_bytes([
        crate::field::gf16::DELTA.0,
        0,
    ])));
    let (dst_lanes, dst_tail) = dst[..len].as_chunks_mut::<16>();
    let (a_lanes, a_tail) = a[..len].as_chunks::<16>();
    let (b_lanes, b_tail) = b[..len].as_chunks::<16>();
    for ((d, x), y) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        // For x=[a,b], y=[c,d]:
        // constant = ac ^ DELTA*bd
        // extension = ad ^ bc ^ bd.
        let xv = vld1q_u8(x);
        let yv = vld1q_u8(y);
        let direct = multiply_base_vectors(xv, yv);
        let crossed = multiply_base_vectors(xv, vrev16q_u8(yv));
        let delta_bd = multiply_base_vectors(vrev16q_u8(direct), delta_even);
        let constant = veorq_u8(direct, delta_bd);
        let extension = veorq_u8(veorq_u8(crossed, vrev16q_u8(crossed)), direct);
        vst1q_u8(d, vbslq_u8(even, constant, extension));
    }
    for ((d, x), y) in dst_tail
        .chunks_exact_mut(2)
        .zip(a_tail.chunks_exact(2))
        .zip(b_tail.chunks_exact(2))
    {
        d.copy_from_slice(&Elem::from_bytes([x[0], x[1]]).mul(Elem::from_bytes([y[0], y[1]])).to_bytes());
    }
}