//! GF(2^16) WebAssembly `simd128` tower kernels.
//!
//! Every multiply here is the tower identity: with interleaved bytes
//! `[a, b]` meaning `a + b*u`, multiplying by `c0 + c1*u` gives
//!
//! ```text
//! even lane:  c0*a       ^  (DELTA*c1)*b
//! odd  lane:  (c0+c1)*b  ^  c1*a
//! ```
//!
//! `b` is `a`'s adjacent-byte neighbour, so the crossed half is the same
//! alternating byte multiply applied to `swap_adjacent(source)`, and the four
//! base coefficients are exactly [`TowerTables::factors`]. Each base
//! coefficient is a split-nibble `u8x16_swizzle` pair — two lookups and an
//! XOR — so a whole 16-byte block costs twenty instructions: five to split
//! the source and its swap into nibble indices, eight lookups, and seven to
//! recombine.
//!
//! Every entry is a safe [`archmage`] capability-token function taking
//! [`archmage::Wasm128Token`] and validating its own geometry; inner loops
//! are `#[rite]` helpers inside the token-proven region. The whole subtree
//! is safe: reference-based `v128_load`/`v128_store` over 16-byte chunk
//! arrays and `split_at_mut` row groups express every shape here, so no
//! unsafe remains.
//!
//! [`archmage`]: https://docs.rs/archmage

use core::arch::wasm32::*;

use crate::field::gf16::{Elem, Gf16};
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::proven_checks::{check_elem_multiple, check_equal, check_row_span, check_terms};
use crate::kernel::tables::TowerTables;
use crate::kernel::wasm32::gf8::multiply_vectors;

/// How many terms the multi-row kernels prepare before sweeping bytes.
///
/// A GF(2^16) coefficient costs four derived nibble tables, so preparation
/// must never happen inside a byte loop. Terms are therefore handled in
/// bounded blocks whose lookup vectors live in a stack array — no allocation,
/// and one preparation per `(term, row)` pair per kernel call.
const TERM_BLOCK: usize = 8;

#[derive(Clone, Copy)]
struct Factors {
    lo: [v128; 4],
    hi: [v128; 4],
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn load_factors(tables: &TowerTables) -> Factors {
    let factors = &tables.factors;
    Factors {
        lo: core::array::from_fn(|i| v128_load(&factors[i].lo)),
        hi: core::array::from_fn(|i| v128_load(&factors[i].hi)),
    }
}

/// A `Factors` whose every product is zero, for slots no lookup will read.
#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn empty_factors() -> Factors {
    let zero = u8x16_splat(0);
    Factors {
        lo: [zero; 4],
        hi: [zero; 4],
    }
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn swap_adjacent(value: v128) -> v128 {
    const SWAP: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];
    let mask = v128_load(&SWAP);
    u8x16_swizzle(value, mask)
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn lookup(lo: v128, hi: v128, low: v128, high: v128) -> v128 {
    v128_xor(u8x16_swizzle(lo, low), u8x16_swizzle(hi, high))
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn scaled(value: v128, factors: Factors) -> v128 {
    let nibble = u8x16_splat(0x0f);
    let swapped = swap_adjacent(value);
    let low = v128_and(value, nibble);
    let high = u8x16_shr(value, 4);
    let swapped_low = v128_and(swapped, nibble);
    let swapped_high = u8x16_shr(swapped, 4);
    let direct_even = lookup(factors.lo[0], factors.hi[0], low, high);
    let direct_odd = lookup(factors.lo[1], factors.hi[1], low, high);
    let cross_even = lookup(factors.lo[2], factors.hi[2], swapped_low, swapped_high);
    let cross_odd = lookup(factors.lo[3], factors.hi[3], swapped_low, swapped_high);
    let even_lanes = u16x8_splat(0x00ff);
    v128_bitselect(
        v128_xor(direct_even, cross_even),
        v128_xor(direct_odd, cross_odd),
        even_lanes,
    )
}

/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_add_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_equal("gf16::mul_add_simd128", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gf16::mul_add_simd128", dst.len(), 2);
    mul_add_impl(dst, tables, src)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

    // Eight swizzles per lane leave plenty of room for a second independent
    // lane's loads and splits, so the block is 32 bytes wide.
    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    let (src_tiles, _) = src[..pair_len].as_chunks::<32>();
    for (d_tile, s_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) = {
            let (d0, d1) = d_tile.split_at_mut(16);
            (d0.try_into().unwrap(), d1.try_into().unwrap())
        };
        let (s0, s1): (&[u8; 16], &[u8; 16]) = {
            let (s0, s1) = s_tile.split_at(16);
            (s0.try_into().unwrap(), s1.try_into().unwrap())
        };
        let d0v = v128_load(&*d0);
        let d1v = v128_load(&*d1);
        v128_store(d0, v128_xor(d0v, scaled(v128_load(s0), factors)));
        v128_store(d1, v128_xor(d1v, scaled(v128_load(s1), factors)));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let cur = v128_load(&*d);
        v128_store(d, v128_xor(cur, scaled(v128_load(s), factors)));
    }

    mul_add_scalar(
        &mut dst[vector_len..span],
        tables.coeff,
        &src[vector_len..span],
    );
}

/// `dst = coeff * dst` over interleaved tower elements.
///
/// # Panics
/// Panics on a partial trailing element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_assign_simd128(_token: archmage::Wasm128Token, dst: &mut [u8], tables: &TowerTables) {
    check_elem_multiple("gf16::mul_assign_simd128", dst.len(), 2);
    mul_assign_impl(dst, tables)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_assign_impl(dst: &mut [u8], tables: &TowerTables) {
    let vector_len = dst.len() & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    for d_tile in dst_tiles {
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) = {
            let (d0, d1) = d_tile.split_at_mut(16);
            (d0.try_into().unwrap(), d1.try_into().unwrap())
        };
        v128_store(d0, scaled(v128_load(&*d0), factors));
        v128_store(d1, scaled(v128_load(&*d1), factors));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    for d in dst_lanes {
        v128_store(d, scaled(v128_load(&*d), factors));
    }

    mul_assign_scalar(&mut dst[vector_len..], tables.coeff);
}

/// `dst = coeff * src` out of place over interleaved tower elements.
///
/// Fuses what would otherwise be a copy followed by an in-place scale: one
/// pass over the destination, which is never read.
///
/// # Panics
/// Panics if the slices differ in length or hold a partial element.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_into_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    tables: &TowerTables,
    src: &[u8],
) {
    check_equal("gf16::mul_into_simd128", "dst", dst.len(), "src", src.len());
    check_elem_multiple("gf16::mul_into_simd128", dst.len(), 2);
    mul_into_impl(dst, tables, src)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_into_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let pair_len = vector_len & !31;
    let factors = load_factors(tables);

    let (dst_tiles, _) = dst[..pair_len].as_chunks_mut::<32>();
    let (src_tiles, _) = src[..pair_len].as_chunks::<32>();
    for (d_tile, s_tile) in dst_tiles.iter_mut().zip(src_tiles) {
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) = {
            let (d0, d1) = d_tile.split_at_mut(16);
            (d0.try_into().unwrap(), d1.try_into().unwrap())
        };
        let (s0, s1): (&[u8; 16], &[u8; 16]) = {
            let (s0, s1) = s_tile.split_at(16);
            (s0.try_into().unwrap(), s1.try_into().unwrap())
        };
        v128_store(d0, scaled(v128_load(s0), factors));
        v128_store(d1, scaled(v128_load(s1), factors));
    }
    let (dst_lanes, _) = dst[pair_len..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[pair_len..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        v128_store(d, scaled(v128_load(s), factors));
    }

    mul_into_scalar(&mut dst[vector_len..span], tables.coeff, &src[vector_len..span]);
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements.
///
/// # Panics
/// Panics unless all three buffers match in length (whole elements).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn elementwise_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
) {
    check_equal("gf16::elementwise_simd128", "dst", dst.len(), "a", a.len());
    check_equal("gf16::elementwise_simd128", "dst", dst.len(), "b", b.len());
    check_elem_multiple("gf16::elementwise_simd128", dst.len(), 2);
    elementwise_impl(dst, a, b)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let span = dst.len().min(a.len()).min(b.len());
    let vector_len = span & !15;
    let even = u16x8_splat(0x00ff);
    let delta_even = u16x8_splat(u16::from_le_bytes([crate::field::gf16::DELTA.0, 0]));

    let (dst_lanes, _) = dst[..vector_len].as_chunks_mut::<16>();
    let (a_lanes, _) = a[..vector_len].as_chunks::<16>();
    let (b_lanes, _) = b[..vector_len].as_chunks::<16>();
    for ((d, x), y) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        let x = v128_load(x);
        let y = v128_load(y);
        let direct = multiply_vectors(x, y);
        let crossed = multiply_vectors(x, swap_adjacent(y));
        let delta_bd = multiply_vectors(swap_adjacent(direct), delta_even);
        let constant = v128_xor(direct, delta_bd);
        let extension = v128_xor(v128_xor(crossed, swap_adjacent(crossed)), direct);
        v128_store(d, v128_bitselect(constant, extension, even));
    }

    crate::kernel::scalar::mul_elementwise::<Gf16>(
        &mut dst[vector_len..span],
        &a[vector_len..span],
        &b[vector_len..span],
    );
}

/// Which of the three cases a coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: nothing to fold in.
    Skip,
    /// Coefficient one: a plain XOR, no swizzle and no tables.
    Identity,
    /// Anything else: the full eight-swizzle tower multiply.
    Table,
}

/// One coefficient resolved into the form the multi-row loops consume.
///
/// Resolving is what the naive per-row loops kept redoing: `Kind::Skip` and
/// `Kind::Identity` — both common in coding matrices — skip the four-table
/// derivation entirely, and `Kind::Table` pays for it once per kernel call
/// instead of once per `(term, row, block)`.
#[derive(Clone, Copy)]
struct Scaling {
    /// Lookup vectors. Meaningless unless `kind` is [`Kind::Table`].
    factors: Factors,
    /// The coefficient itself, for the element-aligned tail.
    coeff: Elem,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff`, deriving nibble tables only when they will be used.
    #[inline]
    #[archmage::rite(wasm128, import_intrinsics)]
    fn new(coeff: Elem) -> Self {
        if coeff == Elem::ZERO {
            return Self {
                factors: empty_factors(),
                coeff,
                kind: Kind::Skip,
            };
        }
        if coeff == Elem::ONE {
            return Self {
                factors: empty_factors(),
                coeff,
                kind: Kind::Identity,
            };
        }
        Self {
            factors: load_factors(&TowerTables::new(coeff)),
            coeff,
            kind: Kind::Table,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[archmage::rite(wasm128, import_intrinsics)]
    fn fold(&self, acc: v128, x: v128) -> v128 {
        match self.kind {
            Kind::Skip => acc,
            Kind::Identity => v128_xor(acc, x),
            Kind::Table => v128_xor(acc, scaled(x, self.factors)),
        }
    }
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
/// bytes (whole elements) and `row_len == src.len()`.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn scatter_simd128(
    _token: archmage::Wasm128Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
    src: &[u8],
) {
    check_equal(
        "gf16::scatter_simd128",
        "row_len",
        row_len,
        "src",
        src.len(),
    );
    check_elem_multiple("gf16::scatter_simd128", row_len, 2);
    check_row_span("gf16::scatter_simd128", rows.len(), row_len, coeffs.len());
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    mul_add_scatter_impl(rows, row_len, coeffs, src)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let span = row_len.min(src.len());
    let count = coeffs.len().min(rows.len() / row_len);
    let src = &src[..span];
    let mut rest = &mut rows[..count * row_len];

    let mut j = 0;
    while j + 4 <= count {
        let (block, tail) = rest.split_at_mut(4 * row_len);
        rest = tail;
        let (r0, block) = block.split_at_mut(row_len);
        let (r1, block) = block.split_at_mut(row_len);
        let (r2, r3) = block.split_at_mut(row_len);
        let plans = [
            Scaling::new(coeffs[j]),
            Scaling::new(coeffs[j + 1]),
            Scaling::new(coeffs[j + 2]),
            Scaling::new(coeffs[j + 3]),
        ];
        // `split_at_mut` gives four disjoint rows, each truncated to the
        // `span` bytes the source also holds.
        scatter_quad(
            [
                &mut r0[..span],
                &mut r1[..span],
                &mut r2[..span],
                &mut r3[..span],
            ],
            &plans,
            src,
        );
        j += 4;
    }
    while j < count {
        let (row, tail) = rest.split_at_mut(row_len);
        rest = tail;
        let plan = Scaling::new(coeffs[j]);
        match plan.kind {
            Kind::Skip => {}
            Kind::Identity => xor_row(&mut row[..span], src),
            Kind::Table => mul_add_impl(&mut row[..span], &TowerTables::new(plan.coeff), src),
        }
        j += 1;
    }
}

/// `dst ^= src`, the whole job when a coefficient is one.
#[archmage::rite(wasm128, import_intrinsics)]
fn xor_row(dst: &mut [u8], src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;

    let (dst_lanes, _) = dst[..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        v128_store(d, v128_xor(v128_load(&*d), v128_load(s)));
    }

    for (d, &s) in dst[vector_len..span].iter_mut().zip(&src[vector_len..span]) {
        *d ^= s;
    }
}

/// Fold one source into four rows: one source load, four destination updates.
#[archmage::rite(wasm128, import_intrinsics)]
fn scatter_quad(rows: [&mut [u8]; 4], plans: &[Scaling; 4], src: &[u8]) {
    let span = src.len();
    let vector_len = span & !15;
    let (src_lanes, _) = src[..vector_len].as_chunks::<16>();

    // Split each row once into 16-byte lanes plus the scalar tail; the four
    // rows come from `split_at_mut`, so the lane slices are pairwise
    // disjoint and disjoint from the source.
    let [r0, r1, r2, r3] = rows;
    let (l0, t0) = r0.as_chunks_mut::<16>();
    let (l1, t1) = r1.as_chunks_mut::<16>();
    let (l2, t2) = r2.as_chunks_mut::<16>();
    let (l3, t3) = r3.as_chunks_mut::<16>();
    let mut lanes: [&mut [[u8; 16]]; 4] = [l0, l1, l2, l3];
    let tails: [&mut [u8]; 4] = [t0, t1, t2, t3];

    for (i, s) in src_lanes.iter().enumerate() {
        let x = v128_load(s);
        for (lane, plan) in lanes.iter_mut().zip(plans) {
            if plan.kind == Kind::Skip {
                continue;
            }
            let d = &mut lane[i];
            v128_store(d, plan.fold(v128_load(&*d), x));
        }
    }

    for (tail, plan) in tails.into_iter().zip(plans) {
        mul_add_scalar(tail, plan.coeff, &src[vector_len..]);
    }
}

/// `dst ^= sum(coeffs[k] * srcs[k])`, holding a 32-byte destination tile in
/// registers across a block of sources so `dst` is touched once per block.
///
/// Sources are handled [`TERM_BLOCK`] at a time: their coefficients are
/// resolved once per block, before the byte loop, so the four-table
/// derivation never repeats per tile.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length (whole elements).
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn gather_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_equal(
        "gf16::gather_simd128",
        "coefficients",
        coeffs.len(),
        "sources",
        srcs.len(),
    );
    check_elem_multiple("gf16::gather_simd128", dst.len(), 2);
    for (index, &src) in srcs.iter().enumerate() {
        check_equal(
            "gf16::gather_simd128",
            "dst",
            dst.len(),
            format_args!("source {index}"),
            src.len(),
        );
    }
    if dst.is_empty() || srcs.is_empty() {
        return;
    }
    mul_add_gather_impl(dst, coeffs, srcs)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_gather_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let count = coeffs.len().min(srcs.len());
    let mut span = dst.len();
    for &src in &srcs[..count] {
        span = span.min(src.len());
    }

    for first in (0..count).step_by(TERM_BLOCK) {
        let block = (count - first).min(TERM_BLOCK);
        let mut plans = [Scaling::new(Elem::ZERO); TERM_BLOCK];
        for (plan, &coeff) in plans[..block].iter_mut().zip(&coeffs[first..first + block]) {
            *plan = Scaling::new(coeff);
        }

        let mut offset = 0;
        while offset + 32 <= span {
            let d = &mut dst[offset..offset + 32];
            let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) = {
                let (d0, d1) = d.split_at_mut(16);
                (d0.try_into().unwrap(), d1.try_into().unwrap())
            };
            let (mut a0, mut a1) = (v128_load(d0), v128_load(d1));
            for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
                if plan.kind == Kind::Skip {
                    continue;
                }
                let (s0, s1): (&[u8; 16], &[u8; 16]) = {
                    let (s0, s1) = src[offset..offset + 32].split_at(16);
                    (s0.try_into().unwrap(), s1.try_into().unwrap())
                };
                a0 = plan.fold(a0, v128_load(s0));
                a1 = plan.fold(a1, v128_load(s1));
            }
            v128_store(d0, a0);
            v128_store(d1, a1);
            offset += 32;
        }
        if offset + 16 <= span {
            let d: &mut [u8; 16] = (&mut dst[offset..offset + 16]).try_into().unwrap();
            let mut acc = v128_load(d);
            for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
                if plan.kind == Kind::Skip {
                    continue;
                }
                let s: &[u8; 16] = src[offset..offset + 16].try_into().unwrap();
                acc = plan.fold(acc, v128_load(s));
            }
            v128_store(d, acc);
            offset += 16;
        }

        for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
            mul_add_scalar(&mut dst[offset..span], plan.coeff, &src[offset..span]);
        }
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked in both directions: each group of four rows loads a
/// 16-byte-per-row destination tile once, folds in a block of terms, and
/// stores once. Every `(term, row)` coefficient in the block is resolved
/// before the byte loop, so nothing is derived per tile.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes
/// (whole elements), every term supplies `nrows` coefficients, and every
/// source is `row_len` bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn matrix_simd128(
    _token: archmage::Wasm128Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    check_elem_multiple("gf16::matrix_simd128", row_len, 2);
    check_row_span("gf16::matrix_simd128", rows.len(), row_len, nrows);
    check_terms("gf16::matrix_simd128", row_len, nrows, terms);
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_impl(rows, row_len, nrows, terms)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_matrix_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    // One pass over `terms` — outside every hot loop — establishes the bounds
    // the vector loops rely on, so a caller that violates the documented
    // geometry gets a short update rather than out-of-bounds reads.
    let mut span = row_len;
    let mut count = nrows.min(rows.len() / row_len);
    for &(coeffs, src) in terms {
        debug_assert_eq!(coeffs.len(), nrows);
        debug_assert_eq!(src.len(), row_len);
        span = span.min(src.len());
        count = count.min(coeffs.len());
    }
    let mut rest = &mut rows[..count * row_len];

    let mut j = 0;
    while j + 4 <= count {
        let (block, tail) = rest.split_at_mut(4 * row_len);
        rest = tail;
        let (r0, block) = block.split_at_mut(row_len);
        let (r1, block) = block.split_at_mut(row_len);
        let (r2, r3) = block.split_at_mut(row_len);
        // `split_at_mut` gives four disjoint rows, each truncated to the
        // `span` bytes every source also holds, and `j + 3 < count` indexes
        // every term's coefficients.
        matrix_quad(
            [
                &mut r0[..span],
                &mut r1[..span],
                &mut r2[..span],
                &mut r3[..span],
            ],
            j,
            terms,
        );
        j += 4;
    }
    while j < count {
        let (row, tail) = rest.split_at_mut(row_len);
        rest = tail;
        for &(coeffs, src) in terms {
            let plan = Scaling::new(coeffs[j]);
            match plan.kind {
                Kind::Skip => {}
                Kind::Identity => xor_row(&mut row[..span], &src[..span]),
                Kind::Table => mul_add_impl(
                    &mut row[..span],
                    &TowerTables::new(plan.coeff),
                    &src[..span],
                ),
            }
        }
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold a block of terms, store
/// once.
#[archmage::rite(wasm128, import_intrinsics)]
fn matrix_quad(rows: [&mut [u8]; 4], first: usize, terms: &[(&[Elem], &[u8])]) {
    let span = rows[0].len();
    let vector_len = span & !15;

    // Split each row once into 16-byte lanes plus the scalar tail; the four
    // rows come from `split_at_mut`, so the lane slices are pairwise
    // disjoint and disjoint from every source.
    let [r0, r1, r2, r3] = rows;
    let (l0, t0) = r0.as_chunks_mut::<16>();
    let (l1, t1) = r1.as_chunks_mut::<16>();
    let (l2, t2) = r2.as_chunks_mut::<16>();
    let (l3, t3) = r3.as_chunks_mut::<16>();
    let mut lanes: [&mut [[u8; 16]]; 4] = [l0, l1, l2, l3];
    let mut tails: [&mut [u8]; 4] = [t0, t1, t2, t3];

    for block in terms.chunks(TERM_BLOCK) {
        let mut plans = [[Scaling::new(Elem::ZERO); 4]; TERM_BLOCK];
        for (&(coeffs, _), slots) in block.iter().zip(&mut plans) {
            for (slot, &coeff) in slots.iter_mut().zip(&coeffs[first..first + 4]) {
                *slot = Scaling::new(coeff);
            }
        }

        let mut offset = 0;
        while offset + 16 <= vector_len {
            let i = offset / 16;
            let mut acc = [
                v128_load(&lanes[0][i]),
                v128_load(&lanes[1][i]),
                v128_load(&lanes[2][i]),
                v128_load(&lanes[3][i]),
            ];
            for (&(_, src), slots) in block.iter().zip(&plans) {
                let x: &[u8; 16] = src[offset..offset + 16].try_into().unwrap();
                let x = v128_load(x);
                for (a, plan) in acc.iter_mut().zip(slots) {
                    *a = plan.fold(*a, x);
                }
            }
            for (lane, &a) in lanes.iter_mut().zip(&acc) {
                v128_store(&mut lane[i], a);
            }
            offset += 16;
        }

        for i in 0..4 {
            for &(coeffs, src) in block {
                mul_add_scalar(&mut tails[i], coeffs[first + i], &src[offset..span]);
            }
        }
    }
}
