//! GF(2^8) WebAssembly `simd128` kernels.
//!
//! Wasm's lane-local `i8x16.swizzle` has the same 16-entry lookup shape as
//! SSSE3 `PSHUFB` and NEON `TBL`, so every product here is the split-nibble
//! form: `u8x16_swizzle` performs a 16-entry lookup on all 16 lanes at once,
//! and `c * x` becomes `lo[x & 0xf] ^ hi[x >> 4]` against the coefficient's
//! precomputed [`ScaleTable`].
//!
//! Every entry is a safe [`archmage`] capability-token function taking
//! [`archmage::Wasm128Token`], validating its own geometry, and handing its
//! sub-lane remainder to the scalar nibble kernels in [`crate::kernel::gf8`];
//! buffer lengths are arbitrary and are never assumed to be lane-aligned.
//! Inner loops are `#[rite]` helpers inside the token-proven region. The
//! whole subtree is safe: reference-based `v128_load`/`v128_store` over
//! 16-byte chunk arrays and `split_at_mut` row groups express every shape
//! here, so no unsafe remains.
//!
//! [`archmage`]: https://docs.rs/archmage

use core::arch::wasm32::*;

use crate::field::gf8b::{Elem, Gf8B};
use crate::kernel::gf8::mul_add_nibble;
use crate::kernel::proven_checks::{check_equal, check_row_span, check_terms};
use crate::kernel::tables::{ScaleTable, scale_table};

#[derive(Clone, Copy)]
struct Factors {
    lo: v128,
    hi: v128,
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn load_factors(table: &ScaleTable) -> Factors {
    Factors {
        lo: v128_load(&table.lo),
        hi: v128_load(&table.hi),
    }
}

#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
fn scaled(value: v128, factors: Factors) -> v128 {
    let low = v128_and(value, u8x16_splat(0x0f));
    let high = u8x16_shr(value, 4);
    v128_xor(
        u8x16_swizzle(factors.lo, low),
        u8x16_swizzle(factors.hi, high),
    )
}

/// `dst ^= coeff * src` over 16-byte SIMD lanes.
///
/// Deliberately one lane per iteration. A two-lane unroll — the shape the
/// GF(2^16) kernels gain from on the same runtime — measured as no change
/// here (BENCHMARKS.md): two swizzles per lane is too little work to have any
/// latency left to hide.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_add_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    table: &ScaleTable,
    src: &[u8],
) {
    check_equal("gf8::mul_add_simd128", "dst", dst.len(), "src", src.len());
    mul_add_impl(dst, table, src)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let factors = load_factors(table);

    let (dst_lanes, _) = dst[..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        let x = v128_load(s);
        let d0v = v128_load(&*d);
        v128_store(d, v128_xor(d0v, scaled(x, factors)));
    }

    mul_add_nibble(&mut dst[vector_len..span], table, &src[vector_len..span]);
}

/// `dst = coeff * dst` over 16-byte SIMD lanes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_assign_simd128(_token: archmage::Wasm128Token, dst: &mut [u8], table: &ScaleTable) {
    mul_assign_impl(dst, table)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_assign_impl(dst: &mut [u8], table: &ScaleTable) {
    let (lanes, tail) = dst.as_chunks_mut::<16>();
    let factors = load_factors(table);

    for lane in lanes {
        v128_store(lane, scaled(v128_load(&*lane), factors));
    }

    crate::kernel::gf8::mul_assign_nibble(tail, table);
}

/// `dst = coeff * src`, out of place, over 16-byte SIMD lanes.
///
/// Fuses what would otherwise be a copy then an in-place scale: one pass, no `dst` read.
///
/// # Panics
/// Panics if the slices differ in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn mul_into_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    table: &ScaleTable,
    src: &[u8],
) {
    check_equal("gf8::mul_into_simd128", "dst", dst.len(), "src", src.len());
    mul_into_impl(dst, table, src)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_into_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let span = dst.len().min(src.len());
    let vector_len = span & !15;
    let factors = load_factors(table);

    let (dst_lanes, _) = dst[..vector_len].as_chunks_mut::<16>();
    let (src_lanes, _) = src[..vector_len].as_chunks::<16>();
    for (d, s) in dst_lanes.iter_mut().zip(src_lanes) {
        v128_store(d, scaled(v128_load(s), factors));
    }

    crate::kernel::gf8::mul_into_nibble(&mut dst[vector_len..span], table, &src[vector_len..span]);
}

/// Lane-parallel multiply for two varying base-field vectors.
#[inline]
#[archmage::rite(wasm128, import_intrinsics)]
pub(super) fn multiply_vectors(mut a: v128, mut b: v128) -> v128 {
    let one = u8x16_splat(1);
    let reduction = u8x16_splat(0x1b);
    let mut product = u8x16_splat(0);
    for _ in 0..8 {
        let selected = u8x16_eq(v128_and(b, one), one);
        product = v128_xor(product, v128_and(a, selected));
        let high = u8x16_eq(u8x16_shr(a, 7), one);
        a = v128_xor(u8x16_shl(a, 1), v128_and(reduction, high));
        b = u8x16_shr(b, 1);
    }
    product
}

/// `dst[i] = a[i] * b[i]` over 16-byte SIMD lanes.
///
/// # Panics
/// Panics unless all three buffers match in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn elementwise_simd128(_token: archmage::Wasm128Token, dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_equal("gf8::elementwise_simd128", "dst", dst.len(), "a", a.len());
    check_equal("gf8::elementwise_simd128", "dst", dst.len(), "b", b.len());
    elementwise_impl(dst, a, b)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let span = dst.len().min(a.len()).min(b.len());
    let vector_len = span & !15;

    let (dst_lanes, _) = dst[..vector_len].as_chunks_mut::<16>();
    let (a_lanes, _) = a[..vector_len].as_chunks::<16>();
    let (b_lanes, _) = b[..vector_len].as_chunks::<16>();
    for ((d, x), y) in dst_lanes.iter_mut().zip(a_lanes).zip(b_lanes) {
        v128_store(d, multiply_vectors(v128_load(x), v128_load(y)));
    }

    crate::kernel::scalar::mul_elementwise::<Gf8B>(
        &mut dst[vector_len..span],
        &a[vector_len..span],
        &b[vector_len..span],
    );
}

/// Which of the three cases a row's coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: the row contributes nothing.
    Skip,
    /// Coefficient one: XOR the source in unscaled, no swizzle needed.
    Identity,
    /// Any other coefficient: two nibble swizzles and an XOR.
    Table,
}

/// One coefficient resolved into the form the multi-row loops consume.
///
/// Branching on [`Kind::Skip`] and [`Kind::Identity`] pays for itself: the
/// coefficient arrays handed to [`scatter_simd128`] and [`matrix_simd128`]
/// are full of zeros and ones, and each case removes two `i8x16.swizzle`s
/// per lane.
#[derive(Clone, Copy)]
struct Scaling {
    /// Nibble tables in registers. Meaningless unless `kind` is [`Kind::Table`].
    factors: Factors,
    /// The same tables in memory, for the element-aligned tail.
    table: &'static ScaleTable,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff` against the shared table bank.
    #[inline]
    #[archmage::rite(wasm128, import_intrinsics)]
    fn new(coeff: Elem) -> Self {
        let table = scale_table(coeff);
        let kind = if coeff == Elem::ZERO {
            Kind::Skip
        } else if coeff == Elem::ONE {
            Kind::Identity
        } else {
            Kind::Table
        };
        Self {
            factors: load_factors(table),
            table,
            kind,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[archmage::rite(wasm128, import_intrinsics)]
    fn fold(self, acc: v128, x: v128) -> v128 {
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
/// bytes and `row_len == src.len()`.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn scatter_simd128(
    _token: archmage::Wasm128Token,
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[Elem],
    src: &[u8],
) {
    check_equal("gf8::scatter_simd128", "row_len", row_len, "src", src.len());
    check_row_span("gf8::scatter_simd128", rows.len(), row_len, coeffs.len());
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
        let coeff = coeffs[j];
        if coeff != Elem::ZERO {
            mul_add_impl(&mut row[..span], scale_table(coeff), src);
        }
        j += 1;
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
        if plan.kind == Kind::Skip {
            continue;
        }
        mul_add_nibble(tail, plan.table, &src[vector_len..]);
    }
}

/// `dst ^= sum(coeffs[k] * srcs[k])`, holding a 32-byte destination tile in
/// registers across every source so `dst` is read and written once.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn gather_simd128(
    _token: archmage::Wasm128Token,
    dst: &mut [u8],
    coeffs: &[Elem],
    srcs: &[&[u8]],
) {
    check_equal(
        "gf8::gather_simd128",
        "coefficients",
        coeffs.len(),
        "sources",
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        check_equal(
            "gf8::gather_simd128",
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

    let mut offset = 0;
    while offset + 32 <= span {
        let d = &mut dst[offset..offset + 32];
        let (d0, d1): (&mut [u8; 16], &mut [u8; 16]) = {
            let (d0, d1) = d.split_at_mut(16);
            (d0.try_into().unwrap(), d1.try_into().unwrap())
        };
        let (mut a0, mut a1) = (v128_load(d0), v128_load(d1));
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            let (s0, s1): (&[u8; 16], &[u8; 16]) = {
                let (s0, s1) = srcs[k][offset..offset + 32].split_at(16);
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
        let mut a = v128_load(d);
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            let s: &[u8; 16] = srcs[k][offset..offset + 16].try_into().unwrap();
            a = plan.fold(a, v128_load(s));
        }
        v128_store(d, a);
        offset += 16;
    }

    for k in 0..count {
        let coeff = coeffs[k];
        if coeff != Elem::ZERO {
            mul_add_nibble(
                &mut dst[offset..span],
                scale_table(coeff),
                &srcs[k][offset..span],
            );
        }
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked: each group of four rows loads a 16-byte-per-row
/// destination tile into four accumulators once, folds in every term, and
/// stores once, so destination traffic is independent of `terms.len()`.
///
/// A zero-length row with zero-length sources is a no-op.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
/// every term supplies `nrows` coefficients, and every source is `row_len`
/// bytes.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane]
pub fn matrix_simd128(
    _token: archmage::Wasm128Token,
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    check_row_span("gf8::matrix_simd128", rows.len(), row_len, nrows);
    check_terms("gf8::matrix_simd128", row_len, nrows, terms);
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    mul_add_matrix_impl(rows, row_len, nrows, terms)
}

#[archmage::rite(wasm128, import_intrinsics)]
fn mul_add_matrix_impl(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
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
            let coeff = coeffs[j];
            if coeff != Elem::ZERO {
                mul_add_impl(&mut row[..span], scale_table(coeff), &src[..span]);
            }
        }
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold every term, store once.
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
    let tails: [&mut [u8]; 4] = [t0, t1, t2, t3];

    let mut offset = 0;
    while offset + 16 <= vector_len {
        let i = offset / 16;
        let mut acc = [
            v128_load(&lanes[0][i]),
            v128_load(&lanes[1][i]),
            v128_load(&lanes[2][i]),
            v128_load(&lanes[3][i]),
        ];
        for &(coeffs, src) in terms {
            let x: &[u8; 16] = src[offset..offset + 16].try_into().unwrap();
            let x = v128_load(x);
            for (a, &coeff) in acc.iter_mut().zip(&coeffs[first..first + 4]) {
                *a = Scaling::new(coeff).fold(*a, x);
            }
        }
        for (lane, &a) in lanes.iter_mut().zip(&acc) {
            v128_store(&mut lane[i], a);
        }
        offset += 16;
    }

    for (i, tail) in tails.into_iter().enumerate() {
        for &(coeffs, src) in terms {
            let coeff = coeffs[first + i];
            if coeff != Elem::ZERO {
                mul_add_nibble(tail, scale_table(coeff), &src[offset..span]);
            }
        }
    }
}
