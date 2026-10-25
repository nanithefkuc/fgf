//! Packed acceptance for the residue rows: the full public operation
//! surface against a scalar-field oracle, on whatever backend the host
//! selected.
//!
//! Covers empty and minimal geometries, invalid geometries, rectangular and
//! offset-addressed row shapes, overwrite versus accumulation, prepared
//! reuse, zero/one/mixed/max coefficients, noncanonical raw lanes, and the
//! canonical characteristic-two broadcast output.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
use fgf::field::{Field, FieldBuffer, FieldElem};
use fgf::{Elem, FieldKernels, FiniteField, Residue, ops};

type P2 = FiniteField<2, 1, Residue>;
type P7 = FiniteField<7, 1, Residue>;
type PByte = FiniteField<101, 1, Residue>;
type PShort = FiniteField<65_521, 1, Residue>;
type PPrime = FiniteField<1_000_000_007, 1, Residue>;
type PFull = FiniteField<4_294_967_291, 1, Residue>;
type PWide = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1, Residue>;

/// Byte lengths crossing every lane width of the rows under test plus
/// longer rows; every packed call requires whole lanes.
const LENS: [usize; 8] = [0, 8, 16, 24, 40, 64, 72, 256];
/// Row lengths in elements for the multi-row shapes.
const ROW_ELEMS: [usize; 5] = [1, 2, 3, 8, 64];

/// Deterministic pseudo-random bytes. No dependency, reproducible failures.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    // `2 * seed + 1` keeps the LCG state odd without collapsing neighbouring
    // seeds onto one stream.
    let mut state = seed.wrapping_mul(2).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

fn oracle_add_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).add(F::decode(s)));
    }
}

fn oracle_sub_assign<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).sub(F::decode(s)));
    }
}

fn oracle_mul_add<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).add(F::decode(s).mul(coeff)));
    }
}

fn oracle_mul_assign<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>) {
    for d in dst.chunks_exact_mut(F::BYTES) {
        F::encode(d, F::decode(d).mul(coeff));
    }
}

fn oracle_mul_elementwise<F: FieldBuffer>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(x).mul(F::decode(y)));
    }
}

fn oracle_add_scalar<F: FieldBuffer>(dst: &mut [u8], value: Elem<F>) {
    for d in dst.chunks_exact_mut(F::BYTES) {
        F::encode(d, F::decode(d).add(value));
    }
}

fn oracle_sub_scalar<F: FieldBuffer>(dst: &mut [u8], value: Elem<F>) {
    for d in dst.chunks_exact_mut(F::BYTES) {
        F::encode(d, F::decode(d).sub(value));
    }
}

/// Canonicalize every lane in place through the scalar field, giving the
/// canonical bytes the packed kernels must produce.
fn canon<F: FieldBuffer>(buf: &mut [u8]) {
    let zero = vec![0u8; buf.len()];
    oracle_add_assign::<F>(buf, &zero);
}

/// Exercise the elementwise operation surface against the oracle.
fn check_elementwise<F: FieldKernels>(coeffs: &[Elem<F>]) {
    for &len in &LENS {
        let mut src = noise(len, 0xa1);
        canon::<F>(&mut src);
        let dst0 = {
            let mut b = noise(len, 0xb2);
            canon::<F>(&mut b);
            b
        };

        {
            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::add_assign::<F>(&mut got, &src);
            oracle_add_assign::<F>(&mut want, &src);
            assert_eq!(got, want, "add_assign len {len}");
            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::sub_assign::<F>(&mut got, &src);
            oracle_sub_assign::<F>(&mut want, &src);
            assert_eq!(got, want, "sub_assign len {len}");
            // add then sub restores the addend.
            let mut round_trip = dst0.clone();
            ops::add_assign::<F>(&mut round_trip, &src);
            ops::sub_assign::<F>(&mut round_trip, &src);
            assert_eq!(round_trip, dst0, "round trip len {len}");
        }

        {
            let mut b = noise(len, 0xc3);
            canon::<F>(&mut b);
            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            ops::mul_elementwise::<F>(&mut got, &src, &b);
            oracle_mul_elementwise::<F>(&mut want, &src, &b);
            assert_eq!(got, want, "mul_elementwise len {len}");

            let mut got = dst0.clone();
            ops::mul_elementwise_assign::<F>(&mut got, &src);
            let mut want = vec![0u8; len];
            oracle_mul_elementwise::<F>(&mut want, &dst0, &src);
            assert_eq!(got, want, "mul_elementwise_assign len {len}");
        }

        for &coeff in coeffs {
            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::mul_add::<F>(&mut got, coeff, &src);
            oracle_mul_add::<F>(&mut want, coeff, &src);
            assert_eq!(got, want, "mul_add len {len} coeff {:?}", coeff.to_raw());

            let mut got = dst0.clone();
            ops::mul_assign::<F>(&mut got, coeff);
            let mut want = dst0.clone();
            oracle_mul_assign::<F>(&mut want, coeff);
            assert_eq!(got, want, "mul_assign len {len}");

            // Overwrite ignores prior destination bytes.
            let mut got = noise(len, 0xd4);
            canon::<F>(&mut got);
            let mut want = src.clone();
            oracle_mul_assign::<F>(&mut want, coeff);
            ops::mul_into::<F>(&mut got, coeff, &src);
            assert_eq!(got, want, "mul_into len {len}");

            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::add_assign_scalar::<F>(&mut got, coeff);
            oracle_add_scalar::<F>(&mut want, coeff);
            assert_eq!(got, want, "add_assign_scalar len {len}");

            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::sub_assign_scalar::<F>(&mut got, coeff);
            oracle_sub_scalar::<F>(&mut want, coeff);
            assert_eq!(got, want, "sub_assign_scalar len {len}");
        }
    }
}

/// Gather, scatter, matrix, offset-addressed, and row-interleaved shapes,
/// each against repeated single-row accumulation.
fn check_shapes<F: FieldKernels>(coeffs: &[Elem<F>]) {
    for &row_elems in &ROW_ELEMS {
        let row_len = row_elems * F::BYTES;
        let nsrc = coeffs.len();
        let srcs: Vec<Vec<u8>> = (0..nsrc)
            .map(|i| {
                let mut s = noise(row_len, 0x100 + i as u64);
                canon::<F>(&mut s);
                s
            })
            .collect();
        let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();

        // gather accumulates every term.
        {
            let mut got = noise(row_len, 0x55);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::mul_add_gather::<F>(&mut got, coeffs, &src_refs);
            for (&c, s) in coeffs.iter().zip(&src_refs) {
                oracle_mul_add::<F>(&mut want, c, s);
            }
            assert_eq!(got, want, "gather row {row_len}");
        }

        // mul_into_gather overwrites: it must equal accumulating from zero.
        {
            let mut got = noise(row_len, 0x55);
            canon::<F>(&mut got);
            let mut want = vec![0u8; row_len];
            ops::mul_into_gather::<F>(&mut got, coeffs, &src_refs);
            for (&c, s) in coeffs.iter().zip(&src_refs) {
                oracle_mul_add::<F>(&mut want, c, s);
            }
            assert_eq!(got, want, "mul_into_gather row {row_len}");
        }

        // scatter fans one source into many rows.
        {
            let mut got = noise(row_len * nsrc, 0x66);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::mul_add_scatter::<F>(&mut got, row_len, coeffs, &srcs[0]);
            for (row, &c) in want.chunks_exact_mut(row_len).zip(coeffs) {
                oracle_mul_add::<F>(row, c, &srcs[0]);
            }
            assert_eq!(got, want, "scatter row {row_len}");
        }

        // matrix: many sources into many rows, plus its overwrite form,
        // which must ignore prior bytes and equal accumulation from zero.
        {
            let nrows = nsrc;
            let terms: Vec<(&[Elem<F>], &[u8])> = src_refs.iter().map(|s| (coeffs, *s)).collect();
            let mut got = noise(row_len * nrows, 0x77);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::mul_add_matrix::<F>(&mut got, row_len, nrows, &terms);
            for &(cs, s) in &terms {
                for (row, &c) in want.chunks_exact_mut(row_len).take(nrows).zip(cs) {
                    oracle_mul_add::<F>(row, c, s);
                }
            }
            assert_eq!(got, want, "matrix row {row_len}");

            let mut overwrote = noise(row_len * nrows, 0x88);
            canon::<F>(&mut overwrote);
            let mut want_overwrite = vec![0u8; row_len * nrows];
            for &(cs, s) in &terms {
                for (row, &c) in want_overwrite.chunks_exact_mut(row_len).take(nrows).zip(cs) {
                    oracle_mul_add::<F>(row, c, s);
                }
            }
            ops::mul_into_matrix::<F>(&mut overwrote, row_len, nrows, &terms);
            assert_eq!(overwrote, want_overwrite, "mul_into_matrix row {row_len}");
        }

        // Offset-addressed destination rows, rectangular with gaps.
        {
            let nrows = nsrc;
            let gap = F::BYTES;
            let stride = row_len + gap;
            let mut got = noise(stride * nrows, 0x99);
            canon::<F>(&mut got);
            let starts: Vec<usize> = (0..nrows).map(|r| r * stride).collect();
            let terms: Vec<(&[Elem<F>], &[u8])> = src_refs.iter().map(|s| (coeffs, *s)).collect();
            let rows_before: Vec<Vec<u8>> = starts
                .iter()
                .map(|&s| got[s..s + row_len].to_vec())
                .collect();
            let gaps_before: Vec<Vec<u8>> = starts
                .iter()
                .map(|&s| got[s + row_len..s + row_len + gap].to_vec())
                .collect();
            ops::mul_add_matrix_at::<F>(&mut got, row_len, &starts, &terms);
            for (row, &start) in starts.iter().enumerate() {
                let mut want = rows_before[row].clone();
                for &(cs, s) in &terms {
                    oracle_mul_add::<F>(&mut want, cs[row], s);
                }
                assert_eq!(
                    &got[start..start + row_len],
                    &want[..],
                    "matrix_at row {row} row_len {row_len} field {} order {}",
                    F::NAME,
                    <F as Field>::ORDER
                );
            }
            for (&start, gap) in starts.iter().zip(&gaps_before) {
                let gap_start = start + row_len;
                assert_eq!(
                    &got[gap_start..gap_start + gap.len()],
                    &gap[..],
                    "gap after row untouched"
                );
            }
        }

        // Row-interleaved addition stays elementwise over whole rows.
        {
            let rows = nsrc.max(2);
            let len = row_len * rows;
            let mut src = noise(len, 0x1a1);
            canon::<F>(&mut src);
            let mut got = noise(len, 0x1a2);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::add_assign_rows::<F>(&mut got, row_len, &src);
            oracle_add_assign::<F>(&mut want, &src);
            assert_eq!(got, want, "add_assign_rows row {row_len}");
        }

        // Offset gather with every coefficient one.
        {
            let live = row_len;
            let region = noise(live * 4, 0x2a1);
            let offsets: Vec<u32> = (0..4u32).map(|i| i * live as u32).collect();
            let mut got = noise(live, 0x2a2);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::add_gather_offsets::<F>(&mut got, &region, &offsets);
            for &start in &offsets {
                oracle_add_assign::<F>(&mut want, &region[start as usize..start as usize + live]);
            }
            assert_eq!(got, want, "add_gather_offsets row {row_len}");
        }
    }
}

/// Prepared forms drive the same kernels with the coefficient resolved once.
#[cfg(feature = "alloc")]
fn check_prepared_reuse<F: FieldKernels>(coeffs: &[Elem<F>]) {
    let row_len = 8 * F::BYTES;
    let nsrc = coeffs.len();
    let srcs: Vec<Vec<u8>> = (0..nsrc)
        .map(|i| {
            let mut s = noise(row_len, 0x300 + i as u64);
            canon::<F>(&mut s);
            s
        })
        .collect();
    let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    assert_ne!(srcs[0], srcs[1], "matrix sources must be distinguishable");

    // Every coefficient, identity or not, is prepared once and reused over
    // distinct sources.
    for &coeff in coeffs {
        let single = ops::Coeff::<F>::new(coeff);
        for src in &src_refs {
            let mut got = noise(row_len, 0x301);
            canon::<F>(&mut got);
            let mut want = got.clone();
            ops::mul_add_with::<F>(&mut got, &single, src);
            oracle_mul_add::<F>(&mut want, coeff, src);
            assert_eq!(got, want, "prepared mul_add reuse");
            let mut got = noise(row_len, 0x302);
            ops::mul_into_with::<F>(&mut got, &single, src);
            let mut want = src.to_vec();
            oracle_mul_assign::<F>(&mut want, coeff);
            assert_eq!(got, want, "prepared mul_into reuse");
        }
        assert_eq!(single.value(), coeff, "prepared keeps its coefficient");
    }

    let vector = ops::CoeffVec::<F>::new(coeffs);
    {
        let mut got = noise(row_len * nsrc, 0x310);
        canon::<F>(&mut got);
        let mut want = got.clone();
        ops::mul_add_scatter_with::<F>(&mut got, row_len, vector.as_ref(), &srcs[0]);
        for (row, &c) in want.chunks_exact_mut(row_len).zip(coeffs) {
            oracle_mul_add::<F>(row, c, &srcs[0]);
        }
        assert_eq!(got, want, "prepared scatter");
    }
    {
        let mut got = noise(row_len, 0x320);
        canon::<F>(&mut got);
        let mut want = got.clone();
        ops::mul_add_gather_with::<F>(&mut got, vector.as_ref(), &src_refs);
        for (&c, s) in coeffs.iter().zip(&src_refs) {
            oracle_mul_add::<F>(&mut want, c, s);
        }
        assert_eq!(got, want, "prepared gather");
    }
    {
        let mut got = noise(row_len, 0x330);
        canon::<F>(&mut got);
        let mut want = vec![0u8; row_len];
        ops::mul_into_gather_with::<F>(&mut got, vector.as_ref(), &src_refs);
        for (&c, s) in coeffs.iter().zip(&src_refs) {
            oracle_mul_add::<F>(&mut want, c, s);
        }
        assert_eq!(got, want, "prepared overwrite gather");
    }

    // Two distinct sources with distinct coefficient columns, so a swapped
    // source index or column stride changes the result.
    let flat: Vec<Elem<F>> = coeffs
        .iter()
        .copied()
        .chain(coeffs.iter().rev().map(|c| c.add(Elem::<F>::ONE)))
        .collect();
    let matrix = ops::CoeffMatrix::<F>::from_source_major(2, nsrc, &flat);
    let matrix_refs: Vec<&[u8]> = vec![src_refs[0], src_refs[1]];
    {
        let nrows = nsrc;
        let mut got = noise(row_len * nrows, 0x340);
        canon::<F>(&mut got);
        let mut want = got.clone();
        ops::mul_add_matrix_with::<F>(&mut got, row_len, &matrix, &matrix_refs);
        for row in 0..nrows {
            // Source-major matrix: source s pairs with coefficient row s of
            // the flat layout, and every output row sees every source.
            for (source, &src) in matrix_refs.iter().enumerate() {
                let coeff = flat[source * nrows + row];
                oracle_mul_add::<F>(&mut want[row * row_len..(row + 1) * row_len], coeff, src);
            }
        }
        assert_eq!(got, want, "prepared matrix");

        let mut overwrote = noise(row_len * nrows, 0x350);
        canon::<F>(&mut overwrote);
        let mut want_overwrite = vec![0u8; row_len * nrows];
        for row in 0..nrows {
            for (source, &src) in matrix_refs.iter().enumerate() {
                let coeff = flat[source * nrows + row];
                let seg = &mut want_overwrite[row * row_len..(row + 1) * row_len];
                oracle_mul_add::<F>(seg, coeff, src);
            }
        }
        ops::mul_into_matrix_with::<F>(&mut overwrote, row_len, &matrix, &matrix_refs);
        assert_eq!(overwrote, want_overwrite, "prepared overwrite matrix");
    }
}

/// Noncanonical lanes through the packed dispatch: arithmetic paths emit
/// canonical lanes equal to the oracle over the reduced values, and the
/// documented shortcuts preserve the alias bytes exactly.
fn check_alias_lanes<F: FieldKernels>(modulus: u64, aliases: &[u64], coeff: Elem<F>) {
    let alias_buf: Vec<u8> = aliases.iter().flat_map(|a| a.to_le_bytes()).collect();
    let len = alias_buf.len();
    let mut other = noise(len, 0x500);
    canon::<F>(&mut other);
    let assert_canonical = |buf: &[u8], what: &str| {
        for lane in buf.chunks_exact(F::BYTES) {
            // Zero-extend the lane's little-endian word to `u64`.
            let mut word = 0u64;
            for (shift, &byte) in lane.iter().enumerate() {
                word |= u64::from(byte) << (8 * shift);
            }
            assert!(word < modulus, "{what}: lane {word} not canonical");
        }
    };

    // Aliases as the destination of accumulation and in-place scaling.
    let mut got = alias_buf.clone();
    let mut want = alias_buf.clone();
    ops::add_assign::<F>(&mut got, &other);
    oracle_add_assign::<F>(&mut want, &other);
    assert_eq!(got, want, "add_assign over aliases");
    assert_canonical(&got, "add_assign");

    let mut got = alias_buf.clone();
    let mut want = alias_buf.clone();
    ops::mul_add::<F>(&mut got, coeff, &other);
    oracle_mul_add::<F>(&mut want, coeff, &other);
    assert_eq!(got, want, "mul_add into aliases");
    assert_canonical(&got, "mul_add");

    // A unit coefficient takes the byte-preserving shortcut checked below.
    if !coeff.is_one() {
        let mut got = alias_buf.clone();
        let mut want = alias_buf.clone();
        ops::mul_assign::<F>(&mut got, coeff);
        oracle_mul_assign::<F>(&mut want, coeff);
        assert_eq!(got, want, "mul_assign over aliases");
        assert_canonical(&got, "mul_assign");
    }

    let mut got = alias_buf.clone();
    let mut want = alias_buf.clone();
    ops::sub_assign_scalar::<F>(&mut got, coeff);
    oracle_sub_scalar::<F>(&mut want, coeff);
    assert_eq!(got, want, "sub_assign_scalar over aliases");
    assert_canonical(&got, "sub_assign_scalar");

    // Aliases as the source of accumulation, overwrite, and products.
    let mut got = other.clone();
    let mut want = other.clone();
    ops::sub_assign::<F>(&mut got, &alias_buf);
    oracle_sub_assign::<F>(&mut want, &alias_buf);
    assert_eq!(got, want, "sub_assign from aliases");
    assert_canonical(&got, "sub_assign");

    if !coeff.is_one() {
        let mut got = noise(len, 0x501);
        ops::mul_into::<F>(&mut got, coeff, &alias_buf);
        let mut want = alias_buf.clone();
        oracle_mul_assign::<F>(&mut want, coeff);
        assert_eq!(got, want, "mul_into from aliases");
        assert_canonical(&got, "mul_into");
    }

    let mut got = vec![0u8; len];
    let mut want = vec![0u8; len];
    ops::mul_elementwise::<F>(&mut got, &alias_buf, &other);
    oracle_mul_elementwise::<F>(&mut want, &alias_buf, &other);
    assert_eq!(got, want, "mul_elementwise of aliases");
    assert_canonical(&got, "mul_elementwise");

    // Shortcuts preserve bytes, aliases included.
    let mut got = alias_buf.clone();
    ops::mul_add::<F>(&mut got, Elem::<F>::ZERO, &other);
    assert_eq!(got, alias_buf, "zero accumulation keeps alias bytes");
    let mut got = alias_buf.clone();
    ops::mul_assign::<F>(&mut got, Elem::<F>::ONE);
    assert_eq!(got, alias_buf, "one scaling keeps alias bytes");
    let mut got = noise(len, 0x502);
    ops::mul_into::<F>(&mut got, Elem::<F>::ONE, &alias_buf);
    assert_eq!(got, alias_buf, "one overwrite copies alias bytes");
}

#[test]
fn residue_rows_match_oracle() {
    let coeffs7 = [
        Elem::<P7>::ZERO,
        Elem::<P7>::ONE,
        Elem::<P7>::from_residue(3).expect("in range"),
        Elem::<P7>::from_residue(6).expect("in range"),
        Elem::<P7>::from_raw(7),        // alias of zero
        Elem::<P7>::from_raw(8),        // alias of one
        Elem::<P7>::from_raw(u64::MAX), // alias of 3
    ];
    check_elementwise::<P7>(&coeffs7);
    check_shapes::<P7>(&coeffs7[1..6]);

    let coeffs_prime = [
        Elem::<PPrime>::ZERO,
        Elem::<PPrime>::ONE,
        Elem::<PPrime>::from_raw(123_456_789),
        Elem::<PPrime>::from_raw(999_999_999),
        Elem::<PPrime>::from_raw(1_000_000_007), // alias of zero
        Elem::<PPrime>::from_raw(u64::MAX),
    ];
    let coeffs_byte = [
        Elem::<PByte>::ZERO,
        Elem::<PByte>::ONE,
        Elem::<PByte>::from_residue(2).expect("in range"),
        Elem::<PByte>::from_residue(100).expect("in range"),
        Elem::<PByte>::from_raw(101), // alias of zero
        Elem::<PByte>::from_raw(u64::MAX),
    ];
    check_elementwise::<PByte>(&coeffs_byte);
    check_shapes::<PByte>(&coeffs_byte[1..5]);

    let coeffs_short = [
        Elem::<PShort>::ZERO,
        Elem::<PShort>::ONE,
        Elem::<PShort>::from_residue(2).expect("in range"),
        Elem::<PShort>::from_residue(65_520).expect("in range"),
        Elem::<PShort>::from_raw(65_521), // alias of zero
        Elem::<PShort>::from_raw(u64::MAX),
    ];
    check_elementwise::<PShort>(&coeffs_short);
    check_shapes::<PShort>(&coeffs_short[1..5]);

    check_elementwise::<PPrime>(&coeffs_prime);
    check_shapes::<PPrime>(&coeffs_prime[1..5]);

    let coeffs_full = [
        Elem::<PFull>::ZERO,
        Elem::<PFull>::ONE,
        Elem::<PFull>::from_raw(2),
        Elem::<PFull>::from_raw(4_294_967_290),
        Elem::<PFull>::from_raw(4_294_967_291), // alias of zero
        Elem::<PFull>::from_raw(u64::MAX),
    ];
    check_elementwise::<PFull>(&coeffs_full);
    check_shapes::<PFull>(&coeffs_full[1..5]);

    let coeffs_wide = [
        Elem::<PWide>::ZERO,
        Elem::<PWide>::ONE,
        Elem::<PWide>::from_raw(1 << 63),
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC4), // P - 1
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC5), // alias of zero
        Elem::<PWide>::from_raw(u64::MAX),
    ];
    check_elementwise::<PWide>(&coeffs_wide);
    check_shapes::<PWide>(&coeffs_wide[1..5]);

    // The characteristic-two row: modular broadcast, never XOR.
    let coeffs2 = [
        Elem::<P2>::ZERO,
        Elem::<P2>::ONE,
        Elem::<P2>::from_raw(2), // alias of zero
        Elem::<P2>::from_raw(3), // alias of one
        Elem::<P2>::from_raw(u64::MAX),
    ];
    check_elementwise::<P2>(&coeffs2);
    check_shapes::<P2>(&coeffs2[1..4]);
}

/// Prepared reuse needs the allocation feature for the owned collections.
#[test]
#[cfg(feature = "alloc")]
fn residue_prepared_reuse_matches_one_shot() {
    let coeffs7 = [
        Elem::<P7>::ONE,
        Elem::<P7>::from_residue(3).expect("in range"),
        Elem::<P7>::from_residue(6).expect("in range"),
        Elem::<P7>::from_raw(7), // alias of zero
        Elem::<P7>::from_raw(8), // alias of one
    ];
    check_prepared_reuse::<P7>(&coeffs7);

    let coeffs_prime = [
        Elem::<PPrime>::ONE,
        Elem::<PPrime>::from_raw(123_456_789),
        Elem::<PPrime>::from_raw(999_999_999),
        Elem::<PPrime>::from_raw(1_000_000_007), // alias of zero
    ];
    let coeffs_byte = [
        Elem::<PByte>::ONE,
        Elem::<PByte>::from_residue(2).expect("in range"),
        Elem::<PByte>::from_residue(100).expect("in range"),
        Elem::<PByte>::from_raw(101), // alias of zero
    ];
    check_prepared_reuse::<PByte>(&coeffs_byte);

    let coeffs_short = [
        Elem::<PShort>::ONE,
        Elem::<PShort>::from_residue(2).expect("in range"),
        Elem::<PShort>::from_residue(65_520).expect("in range"),
        Elem::<PShort>::from_raw(65_521), // alias of zero
    ];
    check_prepared_reuse::<PShort>(&coeffs_short);

    check_prepared_reuse::<PPrime>(&coeffs_prime);

    let coeffs_full = [
        Elem::<PFull>::ONE,
        Elem::<PFull>::from_raw(2),
        Elem::<PFull>::from_raw(4_294_967_290),
        Elem::<PFull>::from_raw(4_294_967_291), // alias of zero
    ];
    check_prepared_reuse::<PFull>(&coeffs_full);

    let coeffs_wide = [
        Elem::<PWide>::ONE,
        Elem::<PWide>::from_raw(1 << 63),
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC4), // P - 1
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC5), // alias of zero
    ];
    check_prepared_reuse::<PWide>(&coeffs_wide);

    let coeffs2 = [
        Elem::<P2>::ONE,
        Elem::<P2>::from_raw(2), // alias of zero
        Elem::<P2>::from_raw(3), // alias of one
    ];
    check_prepared_reuse::<P2>(&coeffs2);
}

#[test]
fn alias_lanes_canonicalize_through_packed_ops() {
    check_alias_lanes::<P7>(7, &[7, 8, 14, u64::MAX], Elem::<P7>::from_raw(3));
    check_alias_lanes::<PPrime>(
        1_000_000_007,
        &[1_000_000_007, 1_000_000_008, u64::MAX],
        Elem::<PPrime>::from_raw(123_456_789),
    );
    check_alias_lanes::<PWide>(
        0xFFFF_FFFF_FFFF_FFC5,
        &[0xFFFF_FFFF_FFFF_FFC5, 0xFFFF_FFFF_FFFF_FFC6, u64::MAX],
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC4),
    );
    check_alias_lanes::<P2>(2, &[2, 3, 4, u64::MAX], Elem::<P2>::ONE);
    type MersenneResidue = FiniteField<0x7fff_ffff, 1, Residue>;
    let aliases = [0x7fff_ffff, 0x8000_0000, u64::MAX].repeat(65);
    check_alias_lanes::<MersenneResidue>(
        0x7fff_ffff,
        &aliases,
        Elem::<MersenneResidue>::from_raw(2),
    );
    type GoldilocksResidue = FiniteField<0xffff_ffff_0000_0001, 1, Residue>;
    let aliases = [0xffff_ffff_0000_0001, 0xffff_ffff_0000_0002, u64::MAX].repeat(65);
    check_alias_lanes::<GoldilocksResidue>(
        0xffff_ffff_0000_0001,
        &aliases,
        Elem::<GoldilocksResidue>::from_raw(2),
    );
}

/// Every packed kernel must leave the characteristic-two row's lanes
/// canonical: the XOR broadcast default would leave alias words behind.
#[test]
fn characteristic_two_broadcast_output_is_canonical() {
    for &lane in &[2u64, 3, 4, 5, u64::MAX] {
        let mut buf = lane.to_le_bytes().repeat(3);
        // Broadcast add and subtract of one must read canonical residues.
        ops::add_assign_scalar::<P2>(&mut buf, Elem::<P2>::ONE);
        for (i, lane_bytes) in buf.chunks_exact(8).enumerate() {
            let word = u64::from_le_bytes(lane_bytes.try_into().expect("lane width"));
            let want = (lane % 2 + 1) % 2;
            assert_eq!(word, want, "broadcast add lane {i} from {lane}");
        }
        ops::sub_assign_scalar::<P2>(&mut buf, Elem::<P2>::ONE);
        for (i, lane_bytes) in buf.chunks_exact(8).enumerate() {
            let word = u64::from_le_bytes(lane_bytes.try_into().expect("lane width"));
            assert_eq!(word, lane % 2, "broadcast sub lane {i} from {lane}");
        }
    }

    // Elementwise add/sub of alias lanes lands on canonical residues too.
    let mut a = 3u64.to_le_bytes().repeat(2);
    let mut b = 6u64.to_le_bytes().repeat(2);
    canon::<P2>(&mut a);
    canon::<P2>(&mut b);
    ops::add_assign::<P2>(&mut a, &b);
    for lane_bytes in a.chunks_exact(8) {
        assert_eq!(u64::from_le_bytes(lane_bytes.try_into().expect("w")), 1);
    }
}

#[test]
fn zero_and_one_coefficients_preserve_contracts() {
    let row_len = 16;
    let len = row_len * 3;
    let mut src = noise(len, 0x410);
    canon::<PPrime>(&mut src);
    let dst0 = noise(len, 0x420);

    // Zero accumulation leaves bytes unchanged; in-place scaling by one
    // leaves bytes unchanged, and overwrite by one copies bytes.
    let mut got = dst0.clone();
    canon::<PPrime>(&mut got);
    let want = got.clone();
    ops::mul_add::<PPrime>(&mut got, Elem::<PPrime>::ZERO, &src);
    assert_eq!(got, want, "zero accumulation");

    let mut got = src.clone();
    ops::mul_assign::<PPrime>(&mut got, Elem::<PPrime>::ONE);
    assert_eq!(got, src, "one scaling");

    let mut got = vec![0u8; len];
    ops::mul_into::<PPrime>(&mut got, Elem::<PPrime>::ONE, &src);
    assert_eq!(got, src, "one overwrite copies");

    // Zero scaling zeroes; packed packing round-trips through aliases.
    let mut got = src.clone();
    ops::mul_assign::<PPrime>(&mut got, Elem::<PPrime>::ZERO);
    assert!(got.iter().all(|&b| b == 0), "zero scaling");
}

#[test]
#[cfg(feature = "alloc")]
fn packing_round_trips_and_rejects_bad_geometry() {
    let elems = [
        Elem::<PPrime>::ZERO,
        Elem::<PPrime>::ONE,
        Elem::<PPrime>::from_raw(1_000_000_007), // alias of zero
        Elem::<PPrime>::from_raw(515),
    ];
    let packed = ops::pack_to_vec::<PPrime>(&elems);
    assert_eq!(packed.len(), elems.len() * <PPrime as FieldBuffer>::BYTES);
    let mut out = [Elem::<PPrime>::ZERO; 4];
    ops::unpack::<PPrime>(&mut out, &packed);
    assert_eq!(out, elems);
    // Packing emits canonical lanes: the third element is the alias of zero.
    assert_eq!(
        &packed[8..12],
        &0u32.to_le_bytes(),
        "alias packs canonically"
    );
}

#[test]
#[should_panic(expected = "bytes but")]
fn mul_add_rejects_length_mismatch() {
    let mut dst = vec![0u8; 16];
    let src = vec![0u8; 24];
    ops::mul_add::<P7>(&mut dst, Elem::<P7>::ONE, &src);
}

#[test]
#[should_panic(expected = "not a whole number")]
fn mul_add_rejects_partial_lane() {
    // Four-byte lanes: a matching length that is not a whole lane.
    let mut dst = vec![0u8; 14];
    let src = vec![0u8; 14];
    ops::mul_add::<PPrime>(&mut dst, Elem::<PPrime>::ONE, &src);
}

#[test]
#[should_panic(expected = "mul_add_scatter: rows is")]
fn scatter_rejects_wrong_row_count() {
    let mut rows = vec![0u8; 8 * 2];
    let src = vec![0u8; 8];
    let coeffs = [Elem::<P7>::ONE, Elem::<P7>::ONE, Elem::<P7>::ONE];
    ops::mul_add_scatter::<P7>(&mut rows, 8, &coeffs, &src);
}

#[test]
#[should_panic(expected = "term supplies")]
fn matrix_rejects_wrong_coefficient_count() {
    let mut rows = vec![0u8; 8 * 2];
    let srcs = [vec![0u8; 8], vec![0u8; 8]];
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let coeffs = [Elem::<P7>::ONE];
    let terms: Vec<(&[Elem<P7>], &[u8])> = refs.iter().map(|s| (&coeffs[..], *s)).collect();
    ops::mul_add_matrix::<P7>(&mut rows, 8, 2, &terms);
}

#[test]
#[should_panic(expected = "mul_into_gather:")]
fn mul_into_gather_rejects_mismatched_source_count() {
    let srcs = [[0u8; 8]];
    let refs: Vec<&[u8]> = srcs.iter().map(|s| s.as_slice()).collect();
    let coeffs = [Elem::<P7>::ONE, Elem::<P7>::ONE];
    let dst = &mut [0u8; 8];
    ops::mul_into_gather::<P7>(dst, &coeffs, &refs);
}

#[test]
#[cfg(feature = "alloc")]
#[should_panic(expected = "mul_into_gather_with:")]
fn prepared_zero_gather_rejects_mismatched_source_count() {
    let coeffs = ops::CoeffVec::<PPrime>::new(&[Elem::<PPrime>::ZERO]);
    let src = [0u8; 8];
    let mut dst = [0u8; 8];
    ops::mul_into_gather_with(&mut dst, coeffs.as_ref(), &[&src, &src]);
}

#[test]
#[cfg(feature = "alloc")]
#[should_panic(expected = "mul_into_gather_with: source")]
fn prepared_zero_gather_rejects_mismatched_source_length() {
    let coeffs = ops::CoeffVec::<PPrime>::new(&[Elem::<PPrime>::ZERO]);
    let src = [0u8; 4];
    let mut dst = [0u8; 8];
    ops::mul_into_gather_with(&mut dst, coeffs.as_ref(), &[&src]);
}

#[test]
fn empty_buffers_are_no_ops() {
    let mut dst: Vec<u8> = vec![];
    let src: Vec<u8> = vec![];
    ops::mul_add::<P7>(&mut dst, Elem::<P7>::ONE, &src);
    ops::add_assign::<P7>(&mut dst, &src);
    ops::sub_assign::<P7>(&mut dst, &src);
    ops::mul_elementwise::<P7>(&mut dst, &src, &src);
    ops::mul_assign::<P7>(&mut dst, Elem::<P7>::ZERO);
    assert!(dst.is_empty());
    // Empty gather writes nothing and must not panic.
    ops::mul_into_gather::<P7>(&mut dst, &[], &[]);
    assert!(dst.is_empty());
}

/// Four-byte row elementwise products against the independent `u128` oracle,
/// with raw noncanonical lanes in and canonical lanes out. Runs on every
/// backend tier the host walk selects.
#[test]
fn four_byte_elementwise_matches_u128_oracle() {
    type Row = FiniteField<4_294_967_291, 1, Residue>;
    const P: u64 = 4_294_967_291;
    const EDGES: [u64; 8] = [0, 1, P - 1, P, P + 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    for lanes in [0usize, 1, 2, 7, 8, 9, 16, 64] {
        let mut a = noise(lanes * 4, 0x5eed + lanes as u64);
        let mut b = noise(lanes * 4, 0xb00b + lanes as u64);
        for (i, lane) in a.chunks_exact_mut(4).enumerate() {
            lane.copy_from_slice(&(EDGES[i % EDGES.len()] as u32).to_le_bytes());
        }
        for (i, lane) in b.chunks_exact_mut(4).enumerate() {
            lane.copy_from_slice(&(EDGES[(i + 3) % EDGES.len()] as u32).to_le_bytes());
        }
        let mut want = vec![0u8; lanes * 4];
        for ((w, x), y) in want
            .chunks_exact_mut(4)
            .zip(a.chunks_exact(4))
            .zip(b.chunks_exact(4))
        {
            let x = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % P;
            let y = u64::from(u32::from_le_bytes(y.try_into().unwrap())) % P;
            w.copy_from_slice(&((x as u128 * y as u128 % P as u128) as u32).to_le_bytes());
        }
        let mut got = vec![0u8; lanes * 4];
        ops::mul_elementwise::<Row>(&mut got, &a, &b);
        assert_eq!(got, want, "mul_elementwise lanes {lanes}");
        for lane in got.chunks_exact(4) {
            assert!(
                u64::from(u32::from_le_bytes(lane.try_into().unwrap())) < P,
                "canonical output lanes {lanes}"
            );
        }
        let mut got_assign = a.clone();
        ops::mul_elementwise_assign::<Row>(&mut got_assign, &b);
        assert_eq!(got_assign, want, "mul_elementwise_assign lanes {lanes}");
    }
}

/// Delayed-reduction prepared matrices through `ops`: the routed and
/// fallback geometries against the independent block-final `u128` oracle
/// `sum_i (x_i mod P) * c_i mod P`, then overwrite or modular add of the
/// raw destination. Covers every listed prime (including the `K = 1` row,
/// which stays on the fallback path), source counts around each `K`,
/// output counts 1..=6, lane counts 0..=17 plus one long row, all-ones
/// sources, coefficients {0, 1, P-1}, and destination offsets.
#[cfg(feature = "alloc")]
#[test]
fn delayed_prepared_matrix_matches_u128_oracle() {
    fn check<const P: u64>() {
        let k = ((1u128 << 32) / P as u128) as usize;
        let mut counts = vec![0usize, 1, 16, 17];
        if k >= 2 {
            counts.extend([k - 1, k, k + 1]);
        }
        counts.sort_unstable();
        counts.dedup();
        let mut lane_counts: Vec<usize> = (0..=17).collect();
        lane_counts.push(260);
        for &nterms in &counts {
            // The oracle is linear in sources; wide counts run at two
            // lane counts.
            let lanes: Vec<usize> = if nterms > 32 {
                vec![1, 9]
            } else {
                lane_counts.clone()
            };
            for &lanes in &lanes {
                let row_len = lanes * 4;
                for nrows in 1..=6usize {
                    for &offset in &[0usize, 4] {
                        let coeffs: Vec<Elem<FiniteField<P, 1, Residue>>> = (0..nterms * nrows)
                            .map(|i| {
                                Elem::<FiniteField<P, 1, Residue>>::from_raw([0, 1, P - 1][i % 3])
                            })
                            .collect();
                        let matrix =
                            ops::CoeffMatrix::<FiniteField<P, 1, Residue>>::from_source_major(
                                nterms, nrows, &coeffs,
                            );
                        for &ones in &[false, true] {
                            let sources: Vec<Vec<u8>> = (0..nterms)
                                .map(|t| {
                                    if ones {
                                        vec![0xFFu8; row_len]
                                    } else {
                                        noise(row_len, 0xe1 + P + t as u64 + lanes as u64)
                                    }
                                })
                                .collect();
                            let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
                            let mut want = vec![0u8; nrows * row_len];
                            for j in 0..nrows {
                                for (l, w) in want[j * row_len..(j + 1) * row_len]
                                    .chunks_exact_mut(4)
                                    .enumerate()
                                {
                                    let mut acc: u128 = 0;
                                    for (t, src) in sources.iter().enumerate() {
                                        let x = u64::from(u32::from_le_bytes(
                                            src[l * 4..l * 4 + 4].try_into().unwrap(),
                                        )) % P;
                                        acc += x as u128 * coeffs[t * nrows + j].to_raw() as u128;
                                    }
                                    w.copy_from_slice(&((acc % P as u128) as u32).to_le_bytes());
                                }
                            }
                            // Overwrite through dispatch, with guards.
                            let pad = offset + nrows * row_len + offset;
                            let mut backing = noise(pad, 0xe2 + P + nterms as u64);
                            let before = backing.clone();
                            {
                                let rows = &mut backing[offset..offset + nrows * row_len];
                                for b in rows.iter_mut() {
                                    *b = b.wrapping_mul(3).wrapping_add(0x5a);
                                }
                                ops::mul_into_matrix_with::<FiniteField<P, 1, Residue>>(
                                    rows, row_len, &matrix, &refs,
                                );
                                assert_eq!(
                                    rows, &want,
                                    "into P {P} terms {nterms} rows {nrows} lanes {lanes} ones {ones} off {offset}"
                                );
                            }
                            assert_eq!(&backing[..offset], &before[..offset], "into head");
                            assert_eq!(
                                &backing[offset + nrows * row_len..],
                                &before[offset + nrows * row_len..],
                                "into tail"
                            );
                            // Accumulation through dispatch, with guards. The
                            // portable composition applies one `mul_add` per
                            // (term, row) pair, and a zero coefficient leaves
                            // the row's bytes untouched — aliases included —
                            // while every other coefficient stores canonical
                            // lanes. Rows whose coefficients are all zero
                            // therefore keep their incoming bytes exactly.
                            let mut backing = noise(pad, 0xe3 + P + nterms as u64);
                            let mut want_add = backing[offset..offset + nrows * row_len].to_vec();
                            for j in 0..nrows {
                                if (0..nterms).all(|t| coeffs[t * nrows + j].is_zero()) {
                                    continue;
                                }
                                for (l, w) in want_add[j * row_len..(j + 1) * row_len]
                                    .chunks_exact_mut(4)
                                    .enumerate()
                                {
                                    let d = u64::from(u32::from_le_bytes(
                                        backing[offset + j * row_len + l * 4
                                            ..offset + j * row_len + l * 4 + 4]
                                            .try_into()
                                            .unwrap(),
                                    )) % P;
                                    let r = u64::from(u32::from_le_bytes(
                                        want[j * row_len + l * 4..j * row_len + l * 4 + 4]
                                            .try_into()
                                            .unwrap(),
                                    ));
                                    w.copy_from_slice(
                                        &(((d as u128 + r as u128) % P as u128) as u32)
                                            .to_le_bytes(),
                                    );
                                }
                            }
                            let before = backing.clone();
                            {
                                let rows = &mut backing[offset..offset + nrows * row_len];
                                ops::mul_add_matrix_with::<FiniteField<P, 1, Residue>>(
                                    rows, row_len, &matrix, &refs,
                                );
                                assert_eq!(
                                    rows, &want_add,
                                    "add P {P} terms {nterms} rows {nrows} lanes {lanes} ones {ones} off {offset}"
                                );
                            }
                            assert_eq!(&backing[..offset], &before[..offset], "add head");
                            assert_eq!(
                                &backing[offset + nrows * row_len..],
                                &before[offset + nrows * row_len..],
                                "add tail"
                            );
                            // Canonical outputs everywhere.
                            for lane in want.chunks_exact(4) {
                                assert!(
                                    u64::from(u32::from_le_bytes(lane.try_into().unwrap())) < P,
                                    "canonical overwrite output"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    check::<65_537>();
    check::<1_000_000_007>();
    check::<1_073_741_827>();
    check::<1_280_000_003>();
    check::<1_431_655_751>();
    check::<2_013_265_921>();
    check::<2_147_483_647>();
    check::<4_294_967_291>();
}

/// Routed residue results equal the portable per-pair composition byte for
/// byte on every tier: single-row `_into`/`_add`/`_assign`, their `_with`
/// prepared forms, and prepared matrix `_into`/`_add` — over raw
/// all-ones sources and destinations, zero/unit coefficients, and
/// coefficient matrices with 0/1 entries. The portable expectation is the
/// independent shortcut-encoding oracle below, not the crate's own
/// dispatch, so a tier that canonicalizes where the portable path
/// preserves bytes fails here.
#[cfg(feature = "alloc")]
#[test]
fn routed_residue_matches_portable_composition_byte_for_byte() {
    /// Portable `mul_into`: zero fills zeros, one copies source bytes
    /// unchanged, other coefficients decode/multiply/encode canonical
    /// lanes.
    fn portable_into<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        if coeff.is_zero() {
            dst.fill(0);
            return;
        }
        if coeff.is_one() {
            dst.copy_from_slice(src);
            return;
        }
        for (d, s) in dst
            .chunks_exact_mut(F::BYTES)
            .zip(src.chunks_exact(F::BYTES))
        {
            F::encode(d, F::decode(s).mul(coeff));
        }
    }

    /// Portable `mul_add`: zero leaves destination bytes untouched, one
    /// adds, other coefficients decode/multiply/add/encode.
    fn portable_add<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        if coeff.is_zero() {
            return;
        }
        if coeff.is_one() {
            oracle_add_assign::<F>(dst, src);
            return;
        }
        for (d, s) in dst
            .chunks_exact_mut(F::BYTES)
            .zip(src.chunks_exact(F::BYTES))
        {
            F::encode(d, F::decode(d).add(F::decode(s).mul(coeff)));
        }
    }

    /// Portable `mul_assign`: one leaves bytes untouched, zero fills
    /// zeros, other coefficients decode/multiply/encode.
    fn portable_assign<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>) {
        if coeff.is_one() {
            return;
        }
        if coeff.is_zero() {
            dst.fill(0);
            return;
        }
        for d in dst.chunks_exact_mut(F::BYTES) {
            F::encode(d, F::decode(d).mul(coeff));
        }
    }

    fn check<F: FieldKernels>(coeffs: &[Elem<F>]) {
        assert!(F::BYTES == 4, "consistency test covers four-byte rows");
        // One lane plus rows below, across, and above the vector body:
        // the single-row floor sits at eight lanes while the elementwise
        // overwrite floor sits at sixteen.
        for &lanes in &[1usize, 7, 8, 9, 15, 16, 17] {
            let row_len = lanes * 4;
            let raw_srcs: Vec<Vec<u8>> =
                [noise(row_len, 0x600 + lanes as u64), vec![0xFFu8; row_len]]
                    .into_iter()
                    .collect();
            let raw_dsts: Vec<Vec<u8>> =
                [noise(row_len, 0x610 + lanes as u64), vec![0xFFu8; row_len]]
                    .into_iter()
                    .collect();
            for src in &raw_srcs {
                for dst0 in &raw_dsts {
                    for &coeff in coeffs {
                        let single = ops::Coeff::<F>::new(coeff);
                        let raw = coeff.to_raw();
                        let mut got = dst0.clone();
                        ops::mul_into::<F>(&mut got, coeff, src);
                        let mut want = dst0.clone();
                        portable_into::<F>(&mut want, coeff, src);
                        assert_eq!(got, want, "mul_into lanes {lanes} coeff {raw:?}");
                        let mut got = dst0.clone();
                        ops::mul_into_with::<F>(&mut got, &single, src);
                        assert_eq!(got, want, "mul_into_with lanes {lanes} coeff {raw:?}");
                        let mut got = dst0.clone();
                        ops::mul_add::<F>(&mut got, coeff, src);
                        let mut want = dst0.clone();
                        portable_add::<F>(&mut want, coeff, src);
                        assert_eq!(got, want, "mul_add lanes {lanes} coeff {raw:?}");
                        let mut got = dst0.clone();
                        ops::mul_add_with::<F>(&mut got, &single, src);
                        assert_eq!(got, want, "mul_add_with lanes {lanes} coeff {raw:?}");
                        let mut got = dst0.clone();
                        ops::mul_assign::<F>(&mut got, coeff);
                        let mut want = dst0.clone();
                        portable_assign::<F>(&mut want, coeff);
                        assert_eq!(got, want, "mul_assign lanes {lanes} coeff {raw:?}");
                        let mut got = dst0.clone();
                        ops::mul_assign_with::<F>(&mut got, &single);
                        assert_eq!(got, want, "mul_assign_with lanes {lanes} coeff {raw:?}");
                    }
                }
            }
            // Prepared matrices over 0/1-heavy coefficient patterns,
            // including an all-zero row and a single-unit row, against the
            // repeated portable per-pair composition.
            let patterns: Vec<Vec<u64>> = vec![
                vec![0, 0, 0, 0],
                vec![1, 0, 0, 1],
                vec![0, 1, 1, 0],
                vec![1, 1, 1, 1],
            ];
            for pattern in &patterns {
                let flat: Vec<Elem<F>> = pattern
                    .iter()
                    .map(|&c| Elem::<F>::from_residue(c).expect("0/1 below the characteristic"))
                    .collect();
                let matrix = ops::CoeffMatrix::<F>::from_source_major(1, 4, &flat);
                let src_refs: Vec<&[u8]> = raw_srcs.iter().map(Vec::as_slice).collect();
                let rows: Vec<u8> = (0..4)
                    .flat_map(|j| raw_dsts[j % 2].iter().copied())
                    .collect();
                let mut got = rows.clone();
                ops::mul_into_matrix_with::<F>(&mut got, row_len, &matrix, &src_refs[..1]);
                let mut want = vec![0u8; 4 * row_len];
                // Overwrite composes as zero-fill plus per-pair `mul_add`
                // (the default `mul_into_matrix_plan`), never per-pair
                // `mul_into`: a unit coefficient adds the source onto
                // zeros — canonicalizing it — rather than copying raw
                // bytes.
                for (j, seg) in want.chunks_exact_mut(row_len).enumerate() {
                    portable_add::<F>(seg, flat[j], &raw_srcs[0]);
                }
                assert_eq!(got, want, "matrix into lanes {lanes} pattern {pattern:?}");
                let mut got = rows.clone();
                ops::mul_add_matrix_with::<F>(&mut got, row_len, &matrix, &src_refs[..1]);
                let mut want = rows.clone();
                for (j, seg) in want.chunks_exact_mut(row_len).enumerate() {
                    portable_add::<F>(seg, flat[j], &raw_srcs[0]);
                }
                assert_eq!(got, want, "matrix add lanes {lanes} pattern {pattern:?}");
            }
        }
    }

    type P65537 = FiniteField<65_537, 1, Residue>;
    check::<P65537>(&[
        Elem::<P65537>::ZERO,
        Elem::<P65537>::ONE,
        Elem::<P65537>::from_raw(2),
        Elem::<P65537>::from_raw(65_536),
    ]);
    check::<PPrime>(&[
        Elem::<PPrime>::ZERO,
        Elem::<PPrime>::ONE,
        Elem::<PPrime>::from_raw(2),
        Elem::<PPrime>::from_raw(1_000_000_006),
    ]);
}

/// Four-output prepared matrices on both sides of the delayed route,
/// against the `u128` oracle: the `K == 1` row (never routed, depth below
/// two), rows below the routed depth floor, and deeper rows, at source
/// counts below, at, and above the routed source minimum, for overwrite
/// and accumulation over raw destinations.
#[cfg(feature = "alloc")]
#[test]
fn delayed_route_boundaries_match_u128_oracle() {
    fn oracle<const P: u64>(
        sources: &[Vec<u8>],
        coeffs: &[Elem<FiniteField<P, 1, Residue>>],
        row: usize,
        nrows: usize,
    ) -> Vec<u8> {
        let row_len = sources[0].len();
        let mut out = vec![0u8; row_len];
        for (l, w) in out.chunks_exact_mut(4).enumerate() {
            let mut acc: u128 = 0;
            for (t, src) in sources.iter().enumerate() {
                let x = u64::from(u32::from_le_bytes(
                    src[l * 4..l * 4 + 4].try_into().unwrap(),
                )) % P;
                acc += x as u128 * coeffs[t * nrows + row].to_raw() as u128;
            }
            w.copy_from_slice(&((acc % P as u128) as u32).to_le_bytes());
        }
        out
    }
    fn check<const P: u64>() {
        type R<const Q: u64> = FiniteField<Q, 1, Residue>;
        let row_len = 64 * 4;
        for &nterms in &[7usize, 8, 9] {
            let coeffs: Vec<Elem<R<P>>> = (0..nterms * 4)
                .map(|i| Elem::<R<P>>::from_raw([2, 3, 5][i % 3]))
                .collect();
            let matrix = ops::CoeffMatrix::<R<P>>::from_source_major(nterms, 4, &coeffs);
            let sources: Vec<Vec<u8>> = (0..nterms)
                .map(|t| noise(row_len, 0xf1 + P + t as u64))
                .collect();
            let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let mut want = vec![0u8; 4 * row_len];
            for j in 0..4 {
                want[j * row_len..(j + 1) * row_len]
                    .copy_from_slice(&oracle::<P>(&sources, &coeffs, j, 4));
            }
            let mut got = noise(4 * row_len, 0xf2 + P + nterms as u64);
            ops::mul_into_matrix_with::<R<P>>(&mut got, row_len, &matrix, &refs);
            assert_eq!(got, want, "into P {P} terms {nterms}");
            let mut got = noise(4 * row_len, 0xf3 + P + nterms as u64);
            let before = got.clone();
            ops::mul_add_matrix_with::<R<P>>(&mut got, row_len, &matrix, &refs);
            let mut want_add = before;
            for j in 0..4 {
                for (l, w) in want_add[j * row_len..(j + 1) * row_len]
                    .chunks_exact_mut(4)
                    .enumerate()
                {
                    let d = u64::from(u32::from_le_bytes(w.try_into().unwrap())) % P;
                    let r = u64::from(u32::from_le_bytes(
                        want[j * row_len + l * 4..j * row_len + l * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ));
                    w.copy_from_slice(
                        &(((d as u128 + r as u128) % P as u128) as u32).to_le_bytes(),
                    );
                }
            }
            assert_eq!(got, want_add, "add P {P} terms {nterms}");
        }
    }
    check::<1_073_741_827>();
    check::<1_431_655_751>();
    check::<1_000_000_007>();
    check::<858_993_433>();
    check::<536_870_909>();
    check::<65_537>();
    check::<1_280_000_003>();
    check::<4_294_967_291>();
}

/// A four-output accumulate with one all-zero output row, over a
/// destination holding noncanonical lanes, on a routed row and the
/// unrouted (`K == 1`, depth below two) row: both agree byte-for-byte with
/// the portable composition, leaving that row untouched and storing
/// canonical sums elsewhere.
#[cfg(feature = "alloc")]
#[test]
fn matrix_accumulate_all_zero_row_matches_portable() {
    fn check<const P: u64>() {
        type Row<const Q: u64> = FiniteField<Q, 1, Residue>;
        let row_len = 64 * 4;
        let nterms = 16;
        // Row 2 has all-zero coefficients; the other rows use nonzero values.
        let coeffs: Vec<Elem<Row<P>>> = (0..nterms * 4)
            .map(|i| {
                if i % 4 == 2 {
                    Elem::<Row<P>>::ZERO
                } else {
                    Elem::<Row<P>>::from_raw(2 + (i as u64 % 7))
                }
            })
            .collect();
        let matrix = ops::CoeffMatrix::<Row<P>>::from_source_major(nterms, 4, &coeffs);
        let sources: Vec<Vec<u8>> = (0..nterms)
            .map(|t| noise(row_len, 0xf8 + t as u64))
            .collect();
        let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        // Noncanonical destination lanes.
        let mut dst: Vec<u8> = (0..4 * row_len)
            .map(|i| (P + (i as u64 % (u64::from(u32::MAX) - P + 1))) as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let before = dst.clone();
        ops::mul_add_matrix_with::<Row<P>>(&mut dst, row_len, &matrix, &refs);
        // Portable composition: one `mul_add` per (term, row) pair.
        let mut want = before.clone();
        for j in 0..4 {
            for t in 0..nterms {
                let c = coeffs[t * 4 + j];
                if c.is_zero() {
                    continue;
                }
                for (d, x) in want[j * row_len..(j + 1) * row_len]
                    .chunks_exact_mut(4)
                    .zip(sources[t].chunks_exact(4))
                {
                    let a = u64::from(u32::from_le_bytes(d.try_into().unwrap())) % P;
                    let xv = u64::from(u32::from_le_bytes(x.try_into().unwrap())) % P;
                    d.copy_from_slice(
                        &(((a as u128 + xv as u128 * c.to_raw() as u128) % P as u128) as u32)
                            .to_le_bytes(),
                    );
                }
            }
        }
        assert_eq!(dst, want, "all-zero row accumulate P {P}");
        assert_eq!(
            &dst[2 * row_len..3 * row_len],
            &before[2 * row_len..3 * row_len],
            "all-zero row untouched P {P}"
        );
    }
    check::<1_000_000_007>();
    check::<858_993_433>();
    check::<4_294_967_291>();
}

#[cfg(feature = "alloc")]
#[test]
fn delayed_unit_row_matches_dispatched_composition_over_aliases() {
    type F = FiniteField<0x7fff_ffff, 1, Residue>;
    let modulus = u128::from(0x7fff_ffff_u64);
    let nrows = 4;
    let nterms = 8;
    let coefficients: Vec<_> = (0..nterms)
        .flat_map(|term| {
            [
                Elem::<F>::ONE,
                Elem::<F>::from_raw(0x7fff_fffe),
                Elem::<F>::from_raw(if term % 2 == 0 { 1 } else { 3 }),
                Elem::<F>::from_raw(5 * (u64::try_from(term).unwrap() + 2)),
            ]
        })
        .collect();
    let plan = ops::CoeffMatrix::<F>::from_source_major(nterms, nrows, &coefficients);
    for row_len in [256, 260, 512] {
        let sources: Vec<_> = (0..nterms)
            .map(|term| {
                let mut source = noise(row_len, 0xa00 + u64::try_from(term).unwrap());
                source[..4].fill(0xff);
                source
            })
            .collect();
        let references: Vec<_> = sources.iter().map(Vec::as_slice).collect();
        let terms: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(term, source)| {
                (
                    &coefficients[term * nrows..(term + 1) * nrows],
                    source.as_slice(),
                )
            })
            .collect();
        let mut prepared = vec![0xff; nrows * row_len];
        let mut composed = prepared.clone();
        let mut expected = prepared.clone();
        for (row, destination) in expected.chunks_exact_mut(row_len).enumerate() {
            for (lane, word) in destination.chunks_exact_mut(4).enumerate() {
                let mut sum = u128::from(u32::MAX) % modulus;
                for (term, source) in sources.iter().enumerate() {
                    let value =
                        u32::from_le_bytes(source[4 * lane..4 * lane + 4].try_into().unwrap());
                    let coefficient = coefficients[term * nrows + row].to_raw();
                    sum = (sum + u128::from(value) * u128::from(coefficient)) % modulus;
                }
                word.copy_from_slice(&u32::try_from(sum).unwrap().to_le_bytes());
            }
        }
        ops::mul_add_matrix_with::<F>(&mut prepared, row_len, &plan, &references);
        ops::mul_add_matrix::<F>(&mut composed, row_len, nrows, &terms);
        assert_eq!(prepared, expected, "prepared unit row at {row_len}");
        assert_eq!(composed, expected, "dispatched unit row at {row_len}");
    }
}

/// Overwrite gathers ignore the prior destination and treat their
/// degenerate coefficient lists - no sources, all zero, one source, and a
/// zero, unit, or zero-alias leading term - as accumulation from zero, in
/// both the one-shot and prepared forms on every residue width class.
#[cfg(feature = "alloc")]
#[test]
fn overwrite_gather_degenerate_coefficients_match_oracle() {
    fn check<F: FieldKernels>(general: Elem<F>, alias: Elem<F>) {
        let row_len = 64 * F::BYTES;
        let srcs: Vec<Vec<u8>> = (0..3)
            .map(|i| {
                let mut s = noise(row_len, 0x700 + i);
                canon::<F>(&mut s);
                s
            })
            .collect();
        let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        let (zero, one) = (Elem::<F>::ZERO, Elem::<F>::ONE);
        let shapes: [&[Elem<F>]; 7] = [
            &[],
            &[zero, alias],
            &[general],
            &[zero],
            &[zero, general, one],
            &[one, general, zero],
            &[alias, general],
        ];
        for (shape, coeffs) in shapes.iter().enumerate() {
            let sources = &refs[..coeffs.len()];
            let mut want = vec![0u8; row_len];
            for (&c, s) in coeffs.iter().zip(sources) {
                oracle_mul_add::<F>(&mut want, c, s);
            }

            let mut got = noise(row_len, 0x7FF);
            ops::mul_into_gather::<F>(&mut got, coeffs, sources);
            assert_eq!(got, want, "{} mul_into_gather shape {shape}", F::NAME);

            let prepared = ops::CoeffVec::<F>::new(coeffs);
            let mut got = noise(row_len, 0x7FF);
            ops::mul_into_gather_with::<F>(&mut got, prepared.as_ref(), sources);
            assert_eq!(got, want, "{} mul_into_gather_with shape {shape}", F::NAME);
        }
    }
    // Each row's raw modulus is its zero alias.
    check::<P2>(Elem::<P2>::ONE, Elem::<P2>::from_raw(2));
    check::<P7>(Elem::<P7>::from_raw(3), Elem::<P7>::from_raw(7));
    check::<PByte>(Elem::<PByte>::from_raw(57), Elem::<PByte>::from_raw(101));
    check::<PShort>(
        Elem::<PShort>::from_raw(40_000),
        Elem::<PShort>::from_raw(65_521),
    );
    check::<PPrime>(
        Elem::<PPrime>::from_raw(123_456_789),
        Elem::<PPrime>::from_raw(1_000_000_007),
    );
    check::<PFull>(
        Elem::<PFull>::from_raw(4_000_000_000),
        Elem::<PFull>::from_raw(4_294_967_291),
    );
    check::<PWide>(
        Elem::<PWide>::from_raw(u64::MAX - 1),
        Elem::<PWide>::from_raw(0xFFFF_FFFF_FFFF_FFC5),
    );
}

/// Elementwise vectorization is available only for u32 rows on supported x86 hosts.
#[test]
fn vector_elementwise_report_matches_the_four_byte_class() {
    fn check<F: FieldKernels>() {
        assert_eq!(
            fgf::has_vector_elementwise::<F>(),
            F::BYTES == 4 && is_vector_host()
        );
    }
    check::<FiniteField<2, 1>>();
    check::<FiniteField<251, 1>>();
    check::<FiniteField<65_521, 1>>();
    check::<FiniteField<1_000_000_007, 1>>();
    check::<FiniteField<4_294_967_291, 1>>();
    check::<FiniteField<18_446_744_073_709_551_557, 1>>();
}

/// `backend_for` reports the vector tier exactly for the rows whose
/// prepared fixed-coefficient operations vectorize: byte rows at
/// `P <= 127`, every four-byte row, eight-byte rows at odd `P`. Byte
/// primes above 127, two-byte rows, and `P = 2` stay scalar on every host.
#[test]
fn backend_report_matches_the_vectorized_width_classes() {
    use fgf::Backend;
    fn check<const P: u64>(expected: Backend) {
        assert_eq!(
            fgf::backend_for::<FiniteField<P, 1>>(),
            expected,
            "backend for characteristic {P}"
        );
    }
    let vector = match fgf::backend() {
        Backend::V4x => Backend::V4x,
        Backend::V3GfniCrypto | Backend::V3 => Backend::V3,
        _ => Backend::Scalar,
    };
    check::<3>(vector);
    check::<127>(vector);
    check::<131>(Backend::Scalar);
    check::<251>(Backend::Scalar);
    check::<257>(Backend::Scalar);
    check::<65_521>(Backend::Scalar);
    check::<65_537>(vector);
    check::<1_000_000_007>(vector);
    check::<4_294_967_291>(vector);
    check::<18_446_744_073_709_551_557>(vector);
    check::<2>(Backend::Scalar);
}

/// Whether this host can serve the x86 vector tier at all.
fn is_vector_host() -> bool {
    matches!(
        fgf::backend(),
        fgf::Backend::V4x | fgf::Backend::V3GfniCrypto | fgf::Backend::V3
    )
}

/// Byte primes above the lookup boundary stay on the portable path: the
/// packed operations match the shortcut-modeling integer oracle on raw
/// lanes at 131 and 251.
#[test]
fn byte_primes_above_lookup_match_oracle() {
    fn check<const P: u64>(coeffs: &[u64]) {
        for &len in &LENS {
            let src = noise(len, 0x900 + P + len as u64);
            let dst_seed = noise(len, 0x910 + P + len as u64);
            for &c in coeffs {
                let coeff = Elem::<FiniteField<P, 1>>::from_raw(c);
                // The oracle models the byte-preserving shortcuts: zero
                // leaves accumulation untouched (and fills overwrite),
                // one copies/adds raw bytes without canonicalizing.
                let mut got = dst_seed.clone();
                ops::mul_add::<FiniteField<P, 1>>(&mut got, coeff, &src);
                let mut want = dst_seed.clone();
                if c != 0 {
                    if c == 1 {
                        oracle_add_assign::<FiniteField<P, 1>>(&mut want, &src);
                    } else {
                        oracle_mul_add::<FiniteField<P, 1>>(&mut want, coeff, &src);
                    }
                }
                assert_eq!(got, want, "P {P} c {c} len {len} add");
                let mut got = dst_seed.clone();
                ops::mul_assign::<FiniteField<P, 1>>(&mut got, coeff);
                let mut want = dst_seed.clone();
                if c != 1 {
                    if c == 0 {
                        want.fill(0);
                    } else {
                        oracle_mul_assign::<FiniteField<P, 1>>(&mut want, coeff);
                    }
                }
                assert_eq!(got, want, "P {P} c {c} len {len} assign");
                let mut got = dst_seed.clone();
                ops::mul_into::<FiniteField<P, 1>>(&mut got, coeff, &src);
                let mut want = dst_seed.clone();
                if c == 0 {
                    want.fill(0);
                } else if c == 1 {
                    want.copy_from_slice(&src);
                } else {
                    for (d, s) in want.chunks_exact_mut(1).zip(src.chunks_exact(1)) {
                        FiniteField::<P, 1>::encode(d, FiniteField::<P, 1>::decode(s).mul(coeff));
                    }
                }
                assert_eq!(got, want, "P {P} c {c} len {len} into");
            }
        }
    }
    check::<131>(&[0, 1, 2, 65, 130]);
    check::<251>(&[0, 1, 2, 125, 250]);
}
