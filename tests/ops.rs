//! Vector kernel semantics, against an independent elementwise oracle.
//!
//! These run on whatever backend the host selected, so on a GFNI machine they
//! exercise GFNI. `SIMD_BACKEND=scalar cargo test` reruns the same assertions
//! against the portable path; the per-backend differential sweep that covers
//! every backend in one process lives in the crate's unit tests.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
use fgf::field::{Elem as _, Field};
use fgf::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, FieldKernels, Gf8B, Gf8D, Gf16, Gf32, Gf64,
    Goldilocks, Mersenne31, QuadMersenne31, fan_paar, gf8b, gf8d, gf16, gf32, gf64, goldilocks,
    mersenne31, ops, quad_mersenne31,
};

/// Deterministic pseudo-random bytes. No dependency, reproducible failures.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Elementwise `dst ^= coeff * src`, straight from the field definition.
fn oracle_mul_add<F: Field>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::read(d).add(F::read(s).mul(coeff));
        F::write(d, value);
    }
}

/// Lengths chosen to straddle every lane boundary in the crate: below one
/// lane, exactly one, one plus a byte, and several unroll tiles plus an odd
/// tail. GF(2^16) needs even lengths, so all of these are even.
///
/// Under Miri the sweep truncates to the boundary cases: the interpreter
/// executes every scalar op itself, so full-length rows only multiply the
/// runtime; aliasing and overflow bugs appear at length 2 just as at 1024.
#[cfg(not(miri))]
const LENGTHS: [usize; 12] = [0, 2, 8, 16, 18, 32, 34, 64, 66, 128, 254, 1024];
#[cfg(miri)]
const LENGTHS: [usize; 7] = [0, 2, 8, 16, 18, 32, 34];

/// Row lengths in elements for the dot-product/matrix sweeps: one element,
/// whole lanes, an unaligned tile, and a long row. Truncated under Miri to
/// the same boundary cases (see `LENGTHS`).
#[cfg(not(miri))]
const RL_ELEMS: [usize; 6] = [1, 16, 33, 64, 100, 512];
#[cfg(miri)]
const RL_ELEMS: [usize; 3] = [1, 16, 33];

// ---------------------------------------------------------------------------
// mul_add
// ---------------------------------------------------------------------------

#[test]
fn gf8_mul_add_matches_oracle() {
    for len in LENGTHS {
        let src = noise(len, 0xa1);
        for coeff in (0..=u8::MAX).map(gf8b::Elem) {
            let mut got = noise(len, 0xb2);
            let mut want = got.clone();
            ops::mul_add::<Gf8B>(&mut got, coeff, &src);
            oracle_mul_add::<Gf8B>(&mut want, coeff, &src);
            assert_eq!(got, want, "len {len}, coeff {coeff:?}");
        }
    }
}

#[test]
fn gf16_mul_add_matches_oracle() {
    // Sweep both component planes independently plus a spray of mixed values.
    let coeffs: Vec<_> = (0..256u16)
        .map(gf16::Elem)
        .chain((0..256u16).map(|i| gf16::Elem(i << 8)))
        .chain([0x0108, 0x1234, 0xbeef, 0xffff].map(gf16::Elem))
        .collect();

    for len in LENGTHS {
        let src = noise(len, 0xc3);
        for &coeff in &coeffs {
            let mut got = noise(len, 0xd4);
            let mut want = got.clone();
            ops::mul_add::<Gf16>(&mut got, coeff, &src);
            oracle_mul_add::<Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "len {len}, coeff {coeff:?}");
        }
    }
}

#[test]
fn mul_add_is_its_own_inverse() {
    // Characteristic two: applying the same term twice cancels it.
    let src = noise(300, 0xe5);
    let original = noise(300, 0xf6);

    let mut buffer = original.clone();
    ops::mul_add::<Gf8B>(&mut buffer, gf8b::Elem(0x8d), &src);
    assert_ne!(buffer, original, "coefficient had no effect");
    ops::mul_add::<Gf8B>(&mut buffer, gf8b::Elem(0x8d), &src);
    assert_eq!(buffer, original);

    let mut buffer = original.clone();
    ops::mul_add::<Gf16>(&mut buffer, gf16::Elem(0x9ace), &src);
    assert_ne!(buffer, original, "coefficient had no effect");
    ops::mul_add::<Gf16>(&mut buffer, gf16::Elem(0x9ace), &src);
    assert_eq!(buffer, original);
}

// ---------------------------------------------------------------------------
// mul_into / mul_assign
// ---------------------------------------------------------------------------

#[test]
fn gf8_mul_into_and_mul_assign_agree() {
    for len in LENGTHS {
        let src = noise(len, 0x11);
        for coeff in (0..=u8::MAX).map(gf8b::Elem) {
            let mut into = vec![0xaa; len];
            ops::mul_into::<Gf8B>(&mut into, coeff, &src);

            let mut assign = src.clone();
            ops::mul_assign::<Gf8B>(&mut assign, coeff);

            let mut want = vec![0u8; len];
            oracle_mul_add::<Gf8B>(&mut want, coeff, &src);

            assert_eq!(into, want, "mul_into len {len} coeff {coeff:?}");
            assert_eq!(assign, want, "mul_assign len {len} coeff {coeff:?}");
        }
    }
}

#[test]
fn gf16_mul_into_and_mul_assign_agree() {
    let coeffs = [0u16, 1, 0x0100, 0x0108, 0x00ff, 0xff00, 0x1234, 0xffff].map(gf16::Elem);
    for len in LENGTHS {
        let src = noise(len, 0x22);
        for coeff in coeffs {
            let mut into = vec![0xaa; len];
            ops::mul_into::<Gf16>(&mut into, coeff, &src);

            let mut assign = src.clone();
            ops::mul_assign::<Gf16>(&mut assign, coeff);

            let mut want = vec![0u8; len];
            oracle_mul_add::<Gf16>(&mut want, coeff, &src);

            assert_eq!(into, want, "mul_into len {len} coeff {coeff:?}");
            assert_eq!(assign, want, "mul_assign len {len} coeff {coeff:?}");
        }
    }
}

#[test]
fn scaling_by_a_coefficient_then_its_inverse_is_identity() {
    let original = noise(512, 0x33);

    let mut buffer = original.clone();
    let c = gf8b::Elem(0x57);
    ops::mul_assign::<Gf8B>(&mut buffer, c);
    ops::mul_assign::<Gf8B>(&mut buffer, c.inv());
    assert_eq!(buffer, original);

    let mut buffer = original.clone();
    let c = gf16::Elem(0x57a3);
    ops::mul_assign::<Gf16>(&mut buffer, c);
    ops::mul_assign::<Gf16>(&mut buffer, c.inv());
    assert_eq!(buffer, original);
}

// ---------------------------------------------------------------------------
// add_assign
// ---------------------------------------------------------------------------

#[test]
fn add_assign_is_xor_and_self_cancels() {
    for len in LENGTHS {
        let src = noise(len, 0x44);
        let original = noise(len, 0x55);

        let mut buffer = original.clone();
        ops::add_assign::<Gf8B>(&mut buffer, &src);
        let want: Vec<u8> = original.iter().zip(&src).map(|(a, b)| a ^ b).collect();
        assert_eq!(buffer, want, "len {len}");

        ops::sub_assign::<Gf8B>(&mut buffer, &src);
        assert_eq!(buffer, original, "len {len}");
    }
}

// ---------------------------------------------------------------------------
// scatter / gather / matrix
// ---------------------------------------------------------------------------

/// Every multi-row shape must agree with repeated single-row `mul_add`.
/// That is the whole contract: blocking is an optimization, not a semantic.
#[test]
fn gf8_scatter_matches_repeated_mul_add() {
    for row_len in [1usize, 15, 16, 31, 32, 33, 64, 129, 512] {
        for nrows in [1usize, 2, 3, 4, 5, 7, 8, 9] {
            let src = noise(row_len, 0x66);
            let coeffs: Vec<_> = (0..nrows)
                .map(|j| gf8b::Elem((j as u8).wrapping_mul(37)))
                .collect();

            let mut got = noise(row_len * nrows, 0x77);
            let mut want = got.clone();

            ops::mul_add_scatter::<Gf8B>(&mut got, row_len, &coeffs, &src);
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                oracle_mul_add::<Gf8B>(row, coeff, &src);
            }
            assert_eq!(got, want, "row_len {row_len}, nrows {nrows}");
        }
    }
}

#[test]
fn gf16_scatter_matches_repeated_mul_add() {
    for row_len in [2usize, 16, 30, 32, 34, 64, 130, 512] {
        for nrows in [1usize, 2, 3, 4, 5, 7, 8, 9] {
            let src = noise(row_len, 0x88);
            let coeffs: Vec<_> = (0..nrows)
                .map(|j| gf16::Elem((j as u16).wrapping_mul(9871)))
                .collect();

            let mut got = noise(row_len * nrows, 0x99);
            let mut want = got.clone();

            ops::mul_add_scatter::<Gf16>(&mut got, row_len, &coeffs, &src);
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                oracle_mul_add::<Gf16>(row, coeff, &src);
            }
            assert_eq!(got, want, "row_len {row_len}, nrows {nrows}");
        }
    }
}

#[test]
fn gf8_matrix_matches_repeated_scatter() {
    for row_len in RL_ELEMS {
        for nrows in [1usize, 2, 3, 4, 6, 8] {
            for nterms in [1usize, 2, 5] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x100 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<gf8b::Elem>> = (0..nterms)
                    .map(|t| {
                        (0..nrows)
                            .map(|j| gf8b::Elem(((t * 31 + j * 17) as u8).wrapping_add(1)))
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[gf8b::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                let mut got = noise(row_len * nrows, 0xaa);
                let mut want = got.clone();

                ops::mul_add_matrix::<Gf8B>(&mut got, row_len, nrows, &terms);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        oracle_mul_add::<Gf8B>(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

#[test]
fn gf16_matrix_matches_repeated_scatter() {
    for row_len in [2usize, 16, 34, 64, 100, 512] {
        for nrows in [1usize, 2, 3, 4, 6, 8] {
            for nterms in [1usize, 2, 5] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x200 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<gf16::Elem>> = (0..nterms)
                    .map(|t| {
                        (0..nrows)
                            .map(|j| gf16::Elem(((t * 7919 + j * 613) as u16).wrapping_add(1)))
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[gf16::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                let mut got = noise(row_len * nrows, 0xbb);
                let mut want = got.clone();

                ops::mul_add_matrix::<Gf16>(&mut got, row_len, nrows, &terms);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        oracle_mul_add::<Gf16>(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

/// The overwrite matrix ignores prior destination bytes, matches the zeroed
/// accumulate over the same terms, leaves surplus rows untouched, and agrees
/// with its prepared plan form.
fn check_dot_product_matrix<F: fgf::FieldKernels>(tag: &str, seed: u64) {
    let b = F::BYTES;
    for &rl_elems in &RL_ELEMS {
        let row_len = rl_elems * b;
        for &nrows in &[1usize, 2, 3, 4, 5, 6, 8] {
            for &nterms in &[0usize, 1, 2, 5] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, seed + 0x100 + t as u64))
                    .collect();
                #[cfg(feature = "std")]
                let src_refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
                let coeff_sets: Vec<Vec<F::Elem>> = (0..nterms)
                    .map(|t| {
                        noise(nrows * b, seed + 0x200 + t as u64)
                            .chunks_exact(b)
                            .map(F::read)
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[F::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                // Oracle: zero, then accumulate every term.
                let mut want = vec![0u8; row_len * nrows];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        oracle_mul_add::<F>(row, coeff, src);
                    }
                }

                // One extra row of nonzero noise proves overwrite ignores the
                // prior destination and never touches surplus rows.
                let mut got = noise(row_len * (nrows + 1), seed + 0x300);
                let surplus = got[row_len * nrows..].to_vec();
                ops::dot_product_matrix::<F>(&mut got, row_len, nrows, &terms);
                assert_eq!(
                    &got[..row_len * nrows],
                    want.as_slice(),
                    "{tag}: overwrite rl={row_len} nrows={nrows} nterms={nterms}"
                );
                assert_eq!(
                    &got[row_len * nrows..],
                    surplus.as_slice(),
                    "{tag}: overwrite touched surplus rl={row_len} nrows={nrows} nterms={nterms}"
                );

                // The prepared plan form must match the one-shot overwrite.
                #[cfg(feature = "std")]
                if nterms > 0 {
                    let flat: Vec<F::Elem> =
                        coeff_sets.iter().flat_map(|c| c.iter().copied()).collect();
                    let plan = ops::Plan::<F>::matrix(nterms, nrows, &flat);
                    let mut prepared = noise(row_len * nrows, seed + 0x500);
                    ops::dot_product_matrix_with::<F>(
                        &mut prepared,
                        row_len,
                        nrows,
                        &plan,
                        &src_refs,
                    );
                    assert_eq!(
                        prepared, want,
                        "{tag}: prepared rl={row_len} nrows={nrows} nterms={nterms}"
                    );
                }
            }
        }
    }
}

#[test]
fn gf8_dot_product_matrix_overwrites() {
    check_dot_product_matrix::<Gf8B>("gf8b", 0x7a1);
}

#[test]
fn gf8d_dot_product_matrix_overwrites() {
    check_dot_product_matrix::<Gf8D>("gf8d", 0x7a2);
}

#[test]
fn gf16_dot_product_matrix_overwrites() {
    check_dot_product_matrix::<Gf16>("gf16", 0x7a3);
}

// ---------------------------------------------------------------------------
// mul_add_matrix_scattered
// ---------------------------------------------------------------------------

/// Scattered reconstruction must equal the contiguous matrix kernel gathered
/// back into place, and must not touch the gaps between rows. The contiguous
/// kernel is independently validated above, so this pins the scattered path to
/// it across lane/tile boundaries, row-group sizes, and term counts.
fn check_matrix_scattered<F: fgf::FieldKernels>(tag: &str, seed: u64) {
    let b = F::BYTES;
    for &rl_elems in &RL_ELEMS {
        let row_len = rl_elems * b;
        for &nrows in &[1usize, 2, 3, 4, 6, 8] {
            for &nterms in &[1usize, 2, 5] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, seed + 0x100 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<F::Elem>> = (0..nterms)
                    .map(|t| {
                        noise(nrows * b, seed + 0x200 + t as u64)
                            .chunks_exact(b)
                            .map(F::read)
                            .collect()
                    })
                    .collect();
                let terms: Vec<(&[F::Elem], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                // Contiguous reference over the same initial row bytes.
                let init = noise(row_len * nrows, seed + 0x300);
                let mut want = init.clone();
                ops::mul_add_matrix::<F>(&mut want, row_len, nrows, &terms);

                // Scattered destination: each row separated by an
                // element-aligned gap, seeded with the contiguous row bytes.
                let gap = b * 7;
                let stride = row_len + gap;
                let mut dst = noise(stride * nrows, seed + 0x400);
                let row_starts: Vec<usize> = (0..nrows).map(|j| j * stride).collect();
                for j in 0..nrows {
                    dst[row_starts[j]..row_starts[j] + row_len]
                        .copy_from_slice(&init[j * row_len..(j + 1) * row_len]);
                }
                let before = dst.clone();

                ops::mul_add_matrix_scattered::<F>(&mut dst, row_len, &row_starts, &terms);

                for j in 0..nrows {
                    assert_eq!(
                        &dst[row_starts[j]..row_starts[j] + row_len],
                        &want[j * row_len..(j + 1) * row_len],
                        "{tag}: row {j} rl={row_len} nrows={nrows} nterms={nterms}"
                    );
                    let g = row_starts[j] + row_len;
                    assert_eq!(
                        &dst[g..g + gap],
                        &before[g..g + gap],
                        "{tag}: gap after row {j} was modified"
                    );
                }
            }
        }
    }
}

#[test]
fn gf8_matrix_scattered_matches_contiguous() {
    check_matrix_scattered::<Gf8B>("gf8", 0x5ca7);
}

#[test]
fn gf16_matrix_scattered_matches_contiguous() {
    check_matrix_scattered::<Gf16>("gf16", 0x5ca8);
}

#[test]
fn matrix_scattered_pairs_coefficients_to_out_of_order_rows() {
    // Row offsets need not be monotonic: coefficient `j` binds to
    // `row_starts[j]`, whatever order the offsets appear in.
    let row_len = 64;
    let src = noise(row_len, 0x11);
    let coeffs = [gf8b::Elem(2), gf8b::Elem(9), gf8b::Elem(200)];
    // Three rows placed high-to-low, so offset order is the reverse of index
    // order.
    let row_starts = [2 * row_len, row_len, 0usize];
    let mut dst = vec![0u8; 3 * row_len];
    ops::mul_add_matrix_scattered::<Gf8B>(&mut dst, row_len, &row_starts, &[(&coeffs, &src)]);

    for (j, &start) in row_starts.iter().enumerate() {
        let mut want = vec![0u8; row_len];
        oracle_mul_add::<Gf8B>(&mut want, coeffs[j], &src);
        assert_eq!(&dst[start..start + row_len], &want[..], "row {j}");
    }
}

#[test]
#[should_panic(expected = "overlap")]
fn matrix_scattered_rejects_overlapping_rows() {
    let mut dst = [0u8; 64];
    let coeffs = [gf8b::Elem(1); 2];
    // Second row starts 8 bytes into the first 16-byte row.
    ops::mul_add_matrix_scattered::<Gf8B>(&mut dst, 16, &[0, 8], &[(&coeffs, &[0u8; 16])]);
}

#[test]
#[should_panic(expected = "but dst is")]
fn matrix_scattered_rejects_out_of_bounds_row() {
    let mut dst = [0u8; 32];
    let coeffs = [gf8b::Elem(1); 2];
    ops::mul_add_matrix_scattered::<Gf8B>(&mut dst, 16, &[0, 24], &[(&coeffs, &[0u8; 16])]);
}

#[test]
#[should_panic(expected = "coefficients for")]
fn matrix_scattered_rejects_wrong_coefficient_count() {
    let mut dst = [0u8; 64];
    let coeffs = [gf8b::Elem(1); 2];
    ops::mul_add_matrix_scattered::<Gf8B>(&mut dst, 16, &[0, 16, 32], &[(&coeffs, &[0u8; 16])]);
}

#[test]
fn matrix_leaves_rows_beyond_nrows_untouched() {
    // `rows` may be longer than `nrows * row_len`; the surplus is not ours.
    let row_len = 64;
    let mut buffer = noise(row_len * 8, 0xcc);
    let untouched = buffer[row_len * 3..].to_vec();

    let src = noise(row_len, 0xdd);
    let coeffs = [gf8b::Elem(2), gf8b::Elem(3), gf8b::Elem(4)];
    ops::mul_add_matrix::<Gf8B>(&mut buffer, row_len, 3, &[(&coeffs, &src)]);

    assert_eq!(&buffer[row_len * 3..], &untouched[..]);
}

#[test]
fn gather_matches_summed_mul_add() {
    let len = 300;
    let sources: Vec<Vec<u8>> = (0..6).map(|i| noise(len, 0x300 + i)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();

    // Include zero and one to exercise the short-circuits.
    let coeffs = [0u8, 1, 0x53, 0xff, 2, 0x1d].map(gf8b::Elem);

    let mut got = noise(len, 0xee);
    let mut want = got.clone();

    ops::mul_add_gather::<Gf8B>(&mut got, &coeffs, &refs);
    for (&coeff, &src) in coeffs.iter().zip(&refs) {
        oracle_mul_add::<Gf8B>(&mut want, coeff, src);
    }
    assert_eq!(got, want);
}

#[test]
fn gf16_gather_matches_summed_mul_add() {
    let len = 300;
    let sources: Vec<Vec<u8>> = (0..9).map(|i| noise(len, 0x340 + i)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs = [0u16, 1, 0x0108, 0xffff, 2, 0x1d, 0x2000, 0xabcd, 0x0100].map(gf16::Elem);
    let mut got = noise(len, 0xef);
    let mut want = got.clone();
    ops::mul_add_gather::<Gf16>(&mut got, &coeffs, &refs);
    for (&coeff, &src) in coeffs.iter().zip(&refs) {
        oracle_mul_add::<Gf16>(&mut want, coeff, src);
    }
    assert_eq!(got, want);
}

#[test]
fn dot_product_overwrites_and_matches_gather_from_zero() {
    let len = 96;
    let sources: Vec<Vec<u8>> = (0..6).map(|i| noise(len, 0x348 + i)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs = [0u8, 1, 0x53, 0xff, 2, 0x1d].map(gf8b::Elem);

    let mut want = vec![0; len];
    for (&coeff, &src) in coeffs.iter().zip(&refs) {
        oracle_mul_add::<Gf8B>(&mut want, coeff, src);
    }

    let initial = noise(len, 0x34f);
    let mut got = initial.clone();
    ops::dot_product::<Gf8B>(&mut got, &coeffs, &refs);
    assert_eq!(got, want);

    let mut accumulated = initial.clone();
    ops::mul_add_gather::<Gf8B>(&mut accumulated, &coeffs, &refs);
    for (value, &prefix) in accumulated.iter_mut().zip(&initial) {
        *value ^= prefix;
    }
    assert_eq!(accumulated, want);
}

#[test]
#[cfg(feature = "std")]
fn prepared_dot_product_matches_one_shot_over_gf16() {
    let len = 66;
    let sources = [noise(len, 0x371), noise(len, 0x372), noise(len, 0x373)];
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs = [gf16::Elem(0x0108), gf16::Elem(0), gf16::Elem(0xabcd)];
    let plan = ops::Plan::<Gf16>::new(&coeffs);

    let mut one_shot = noise(len, 0x374);
    let mut prepared = noise(len, 0x375);
    ops::dot_product::<Gf16>(&mut one_shot, &coeffs, &refs);
    ops::dot_product_with::<Gf16>(&mut prepared, &plan, &refs);
    assert_eq!(prepared, one_shot);

    let mut single = noise(len, 0x376);
    let mut scaled = vec![0; len];
    ops::dot_product::<Gf16>(&mut single, &coeffs[..1], &refs[..1]);
    ops::mul_into::<Gf16>(&mut scaled, coeffs[0], refs[0]);
    assert_eq!(single, scaled);
}

#[test]
fn empty_and_zero_dot_products_zero_the_destination() {
    let mut empty_terms = noise(32, 0x377);
    ops::dot_product::<Gf8B>(&mut empty_terms, &[], &[]);
    assert!(empty_terms.iter().all(|&value| value == 0));

    let sources = [noise(32, 0x378), noise(32, 0x379)];
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let mut zero_coefficients = noise(32, 0x37a);
    ops::dot_product::<Gf8B>(&mut zero_coefficients, &[gf8b::Elem::ZERO; 2], &refs);
    assert!(zero_coefficients.iter().all(|&value| value == 0));
}

#[test]
fn prepared_coefficients_match_one_shot_operations() {
    let src8 = noise(258, 0x350);
    for coeff in [0u8, 1, 2, 0x53, 0xff].map(gf8b::Elem) {
        let prepared = ops::Coeff::<Gf8B>::new(coeff);
        assert_eq!(prepared.value(), coeff);

        let mut got = noise(src8.len(), 0x351);
        let mut want = got.clone();
        ops::mul_add_with(&mut got, &prepared, &src8);
        ops::mul_add::<Gf8B>(&mut want, coeff, &src8);
        assert_eq!(got, want, "GF8 prepared AXPY for {coeff:?}");

        let mut got = vec![0; src8.len()];
        let mut want = vec![0; src8.len()];
        ops::mul_into_with(&mut got, &prepared, &src8);
        ops::mul_into::<Gf8B>(&mut want, coeff, &src8);
        assert_eq!(got, want, "GF8 prepared scale for {coeff:?}");
    }

    let src16 = noise(258, 0x352);
    for coeff in [0u16, 1, 2, 0x0108, 0xffff].map(gf16::Elem) {
        let prepared = ops::Coeff::<Gf16>::new(coeff);
        assert_eq!(prepared.value(), coeff);

        let mut got = noise(src16.len(), 0x353);
        let mut want = got.clone();
        ops::mul_add_with(&mut got, &prepared, &src16);
        ops::mul_add::<Gf16>(&mut want, coeff, &src16);
        assert_eq!(got, want, "GF16 prepared AXPY for {coeff:?}");

        let mut got = src16.clone();
        let mut want = src16.clone();
        ops::mul_assign_with(&mut got, &prepared);
        ops::mul_assign::<Gf16>(&mut want, coeff);
        assert_eq!(got, want, "GF16 prepared in-place scale for {coeff:?}");
    }
}

#[test]
#[cfg(feature = "std")]
fn coefficient_plan_preserves_matrix_shape_and_reuses_entries() {
    let values = [
        gf16::Elem(0),
        gf16::Elem(1),
        gf16::Elem(0x0108),
        gf16::Elem(0xffff),
        gf16::Elem(0x2000),
        gf16::Elem(0xabcd),
    ];
    let plan = ops::Plan::<Gf16>::matrix(2, 3, &values);
    assert_eq!(plan.dimensions(), (2, 3));
    assert_eq!(plan.len(), values.len());
    assert!(!plan.is_empty());
    assert_eq!(plan.values().collect::<Vec<_>>(), values);
    assert!(plan.get(values.len()).is_none());

    let coeff = plan.get(2).expect("coefficient exists");
    let src = noise(258, 0x358);
    let mut got = noise(src.len(), 0x359);
    let mut want = got.clone();
    ops::mul_add_with::<Gf16>(&mut got, &coeff, &src);
    ops::mul_add::<Gf16>(&mut want, values[2], &src);
    assert_eq!(got, want);
}

#[test]
#[cfg(feature = "std")]
fn coefficient_plans_drive_all_multi_row_shapes() {
    let row_len = 66;
    let coeffs = [gf16::Elem(0x0108), gf16::Elem(0), gf16::Elem(0xabcd)];
    let vector_plan = ops::Plan::<Gf16>::new(&coeffs);
    let src = noise(row_len, 0x360);

    let mut scatter = noise(row_len * coeffs.len(), 0x361);
    let mut scatter_want = scatter.clone();
    ops::mul_add_scatter_with(&mut scatter, row_len, &vector_plan, &src);
    ops::mul_add_scatter::<Gf16>(&mut scatter_want, row_len, &coeffs, &src);
    assert_eq!(scatter, scatter_want);

    let sources = [
        noise(row_len, 0x362),
        noise(row_len, 0x363),
        noise(row_len, 0x364),
    ];
    let source_refs = sources.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut gather = noise(row_len, 0x365);
    let mut gather_want = gather.clone();
    ops::mul_add_gather_with(&mut gather, &vector_plan, &source_refs);
    ops::mul_add_gather::<Gf16>(&mut gather_want, &coeffs, &source_refs);
    assert_eq!(gather, gather_want);

    let matrix_values = [
        gf16::Elem(1),
        gf16::Elem(0x0108),
        gf16::Elem(2),
        gf16::Elem(3),
        gf16::Elem(0),
        gf16::Elem(0xffff),
    ];
    let matrix_plan = ops::Plan::<Gf16>::matrix(2, 3, &matrix_values);
    let matrix_sources = [&sources[0][..], &sources[1][..]];
    let raw_terms = [
        (&matrix_values[..3], matrix_sources[0]),
        (&matrix_values[3..], matrix_sources[1]),
    ];
    let mut matrix = noise(row_len * 3, 0x366);
    let mut matrix_want = matrix.clone();
    ops::mul_add_matrix_with(&mut matrix, row_len, 3, &matrix_plan, &matrix_sources);
    ops::mul_add_matrix::<Gf16>(&mut matrix_want, row_len, 3, &raw_terms);
    assert_eq!(matrix, matrix_want);

    assert_eq!(
        matrix_plan.get_at(1, 2).map(|coeff| coeff.value()),
        Some(matrix_values[5])
    );
    assert_eq!(
        matrix_plan
            .row(1)
            .expect("row exists")
            .map(|coeff| coeff.value())
            .collect::<Vec<_>>(),
        matrix_values[3..],
    );
}

#[cfg(feature = "std")]
fn assert_blocked_plan_shapes<F: fgf::FieldKernels>(row_len: usize, seed: u64) {
    const NROWS: usize = 5;
    const NTERMS: usize = 9;

    let coefficient_bytes = noise(NROWS * NTERMS * F::BYTES, seed);
    let mut values = coefficient_bytes
        .chunks_exact(F::BYTES)
        .map(F::read)
        .collect::<Vec<_>>();
    values[0] = F::Elem::ZERO;
    values[1] = F::Elem::ONE;

    let vector_plan = ops::Plan::<F>::new(&values[..NROWS]);
    let src = noise(row_len, seed + 1);
    let mut scatter = noise(row_len * NROWS, seed + 2);
    let mut scatter_want = scatter.clone();
    ops::mul_add_scatter_with(&mut scatter, row_len, &vector_plan, &src);
    ops::mul_add_scatter::<F>(&mut scatter_want, row_len, &values[..NROWS], &src);
    assert_eq!(scatter, scatter_want);

    let gather_sources = (0..NROWS)
        .map(|index| noise(row_len, seed + 10 + index as u64))
        .collect::<Vec<_>>();
    let gather_refs = gather_sources.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut gather = noise(row_len, seed + 20);
    let mut gather_want = gather.clone();
    ops::mul_add_gather_with(&mut gather, &vector_plan, &gather_refs);
    ops::mul_add_gather::<F>(&mut gather_want, &values[..NROWS], &gather_refs);
    assert_eq!(gather, gather_want);

    let matrix_plan = ops::Plan::<F>::matrix(NTERMS, NROWS, &values);
    let matrix_sources = (0..NTERMS)
        .map(|index| noise(row_len, seed + 30 + index as u64))
        .collect::<Vec<_>>();
    let matrix_refs = matrix_sources.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let raw_terms = values
        .chunks_exact(NROWS)
        .zip(matrix_refs.iter().copied())
        .collect::<Vec<_>>();
    let mut matrix = noise(row_len * NROWS, seed + 40);
    let mut matrix_want = matrix.clone();
    ops::mul_add_matrix_with(&mut matrix, row_len, NROWS, &matrix_plan, &matrix_refs);
    ops::mul_add_matrix::<F>(&mut matrix_want, row_len, NROWS, &raw_terms);
    assert_eq!(matrix, matrix_want);
}

#[test]
#[cfg(feature = "std")]
fn blocked_plans_match_raw_operations_across_group_boundaries() {
    assert_blocked_plan_shapes::<Gf8B>(79, 0x370);
    assert_blocked_plan_shapes::<Gf16>(78, 0x380);
}

#[test]
#[cfg(feature = "std")]
fn packed_element_helpers_round_trip() {
    let elems = [
        gf16::Elem::ZERO,
        gf16::Elem::ONE,
        gf16::Elem(0x0108),
        gf16::Elem(0xffff),
    ];
    let bytes = ops::pack_to_vec::<Gf16>(&elems);
    assert_eq!(bytes.len(), elems.len() * Gf16::BYTES);

    let mut decoded = [gf16::Elem::ZERO; 4];
    ops::unpack::<Gf16>(&mut decoded, &bytes);
    assert_eq!(decoded, elems);

    let mut repacked = [0u8; 8];
    ops::pack::<Gf16>(&mut repacked, &decoded);
    assert_eq!(repacked.as_slice(), bytes);
}

#[test]
fn elementwise_products_match_field_arithmetic() {
    for len in LENGTHS {
        let a = noise(len, 0x354);
        let b = noise(len, 0x355);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        ops::mul_elementwise::<Gf8B>(&mut got, &a, &b);
        for ((d, &x), &y) in want.iter_mut().zip(&a).zip(&b) {
            *d = gf8b::Elem(x).mul(gf8b::Elem(y)).0;
        }
        assert_eq!(got, want, "GF8 elementwise len {len}");
    }

    for len in LENGTHS.into_iter().filter(|len| len % 2 == 0) {
        let a = noise(len, 0x356);
        let b = noise(len, 0x357);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        ops::mul_elementwise::<Gf16>(&mut got, &a, &b);
        for ((d, x), y) in want
            .chunks_exact_mut(2)
            .zip(a.chunks_exact(2))
            .zip(b.chunks_exact(2))
        {
            d.copy_from_slice(
                &gf16::Elem::from_bytes([x[0], x[1]])
                    .mul(gf16::Elem::from_bytes([y[0], y[1]]))
                    .to_bytes(),
            );
        }
        assert_eq!(got, want, "GF16 elementwise len {len}");
    }
}

#[test]
fn zero_and_one_coefficients_behave() {
    let len = 96;
    let src = noise(len, 0xf0);
    let original = noise(len, 0x0f);

    let mut buffer = original.clone();
    ops::mul_add::<Gf8B>(&mut buffer, gf8b::Elem::ZERO, &src);
    assert_eq!(buffer, original, "zero coefficient must be a no-op");

    let mut buffer = original.clone();
    ops::mul_add::<Gf8B>(&mut buffer, gf8b::Elem::ONE, &src);
    let want: Vec<u8> = original.iter().zip(&src).map(|(a, b)| a ^ b).collect();
    assert_eq!(buffer, want, "unit coefficient must be plain XOR");

    let mut buffer = original.clone();
    ops::mul_assign::<Gf16>(&mut buffer, gf16::Elem::ZERO);
    assert!(buffer.iter().all(|&b| b == 0), "scaling by zero must clear");
}

// ---------------------------------------------------------------------------
// Round trip: the reason this library exists
// ---------------------------------------------------------------------------

/// Encode with a Vandermonde-style parity matrix, then solve a 2x2 system to
/// recover two erased rows. Exercises scatter, matrix, gather, and scalar
/// inversion together, and fails loudly if any of them disagree.
#[test]
fn erasure_round_trip_gf8() {
    let k = 6usize;
    let m = 3usize;
    // Miri interprets every scalar op; keep the round-trip rows small there.
    let row_len = if cfg!(miri) { 64 } else { 1024 };

    let data: Vec<u8> = noise(k * row_len, 0x5150);

    // parity[j] = sum_i coeff(i, j) * data[i], coeff(i, j) = g^((i+1)*(j+1)).
    let coeff = |i: usize, j: usize| Gf8B::GENERATOR.pow(((i + 1) * (j + 1)) as u64);

    let mut parity = vec![0u8; m * row_len];
    for i in 0..k {
        let coeffs: Vec<_> = (0..m).map(|j| coeff(i, j)).collect();
        ops::mul_add_scatter::<Gf8B>(
            &mut parity,
            row_len,
            &coeffs,
            &data[i * row_len..(i + 1) * row_len],
        );
    }

    // Erase data rows 1 and 4. Subtract every surviving data row from the
    // first two parity rows, leaving a 2x2 system in the lost rows.
    let lost = [1usize, 4];
    let mut residual = parity[..2 * row_len].to_vec();
    for i in (0..k).filter(|i| !lost.contains(i)) {
        let coeffs = [coeff(i, 0), coeff(i, 1)];
        ops::mul_add_matrix::<Gf8B>(
            &mut residual,
            row_len,
            2,
            &[(&coeffs, &data[i * row_len..(i + 1) * row_len])],
        );
    }

    // residual[j] = a_j * x + b_j * y, where x, y are the lost rows.
    let (a0, b0) = (coeff(lost[0], 0), coeff(lost[1], 0));
    let (a1, b1) = (coeff(lost[0], 1), coeff(lost[1], 1));
    let det = a0.mul(b1).add(a1.mul(b0));
    assert_ne!(det, gf8b::Elem::ZERO, "chosen submatrix is singular");
    let det_inv = det.inv();

    // Cramer's rule, characteristic two so signs vanish.
    let (r0, r1) = residual.split_at(row_len);
    let mut x = vec![0u8; row_len];
    ops::mul_add_gather::<Gf8B>(&mut x, &[b1.mul(det_inv), b0.mul(det_inv)], &[r0, r1]);
    let mut y = vec![0u8; row_len];
    ops::mul_add_gather::<Gf8B>(&mut y, &[a1.mul(det_inv), a0.mul(det_inv)], &[r0, r1]);

    assert_eq!(x, data[lost[0] * row_len..(lost[0] + 1) * row_len], "row 1");
    assert_eq!(y, data[lost[1] * row_len..(lost[1] + 1) * row_len], "row 4");
}

#[test]
#[should_panic(expected = "dot_product: 2 coefficients for 1 sources")]
fn dot_product_rejects_mismatched_source_count() {
    let mut dst = [0u8; 8];
    ops::dot_product::<Gf8B>(&mut dst, &[gf8b::Elem(2), gf8b::Elem(3)], &[&[0u8; 8]]);
}

#[test]
#[should_panic(expected = "dot_product: source 0 is 7 bytes, expected 8")]
fn dot_product_rejects_wrong_source_length() {
    let mut dst = [0u8; 8];
    ops::dot_product::<Gf8B>(&mut dst, &[gf8b::Elem(2)], &[&[0u8; 7]]);
}

// ---------------------------------------------------------------------------
// Geometry validation
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "dst is 8 bytes but src is 4 bytes")]
fn mul_add_rejects_length_mismatch() {
    let mut dst = [0u8; 8];
    ops::mul_add::<Gf8B>(&mut dst, gf8b::Elem(2), &[0u8; 4]);
}

#[test]
#[should_panic(expected = "whole number of")]
fn gf16_rejects_odd_length() {
    let mut dst = [0u8; 7];
    ops::mul_add::<Gf16>(&mut dst, gf16::Elem(2), &[0u8; 7]);
}

#[test]
#[should_panic(expected = "rows is 30 bytes")]
fn scatter_rejects_wrong_row_count() {
    let mut rows = [0u8; 30];
    let coeffs = [gf8b::Elem(1); 4];
    ops::mul_add_scatter::<Gf8B>(&mut rows, 8, &coeffs, &[0u8; 8]);
}

#[test]
#[should_panic(expected = "coefficients for")]
fn matrix_rejects_wrong_coefficient_count() {
    let mut rows = [0u8; 32];
    let coeffs = [gf8b::Elem(1); 2];
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 4, &[(&coeffs, &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "dot_product_matrix: term supplies 2 coefficients for 4 rows")]
fn dot_product_matrix_rejects_wrong_coefficient_count() {
    let mut rows = [0u8; 32];
    let coeffs = [gf8b::Elem(1); 2];
    ops::dot_product_matrix::<Gf8B>(&mut rows, 8, 4, &[(&coeffs, &[0u8; 8])]);
}

#[cfg(feature = "std")]
#[test]
#[should_panic(expected = "dot_product_matrix_with: plan dimensions do not match")]
fn dot_product_matrix_with_rejects_wrong_plan_dimensions() {
    let mut rows = [0u8; 32];
    let plan = ops::Plan::<Gf8B>::matrix(1, 2, &[gf8b::Elem(1); 2]);
    let sources = [&[0u8; 8][..], &[0u8; 8][..]];
    ops::dot_product_matrix_with::<Gf8B>(&mut rows, 8, 4, &plan, &sources);
}

#[test]
fn empty_buffers_are_no_ops() {
    let mut empty: [u8; 0] = [];
    ops::mul_add::<Gf8B>(&mut empty, gf8b::Elem(7), &[]);
    ops::mul_assign::<Gf16>(&mut empty, gf16::Elem(7));
    ops::mul_add_scatter::<Gf8B>(&mut empty, 0, &[], &[]);
    ops::mul_add_matrix::<Gf8B>(&mut empty, 8, 0, &[]);
    ops::dot_product_matrix::<Gf8B>(&mut empty, 8, 0, &[]);
    ops::mul_add_gather::<Gf8B>(&mut empty, &[], &[]);
}

fn check_wide_field_ops<F: fgf::FieldKernels>(coeffs: &[F::Elem]) {
    let row_len = F::BYTES * 17;
    let src0 = noise(row_len, 0x3200 + u64::from(F::BITS));
    let src1 = noise(row_len, 0x6400 + u64::from(F::BITS));

    let mut got = noise(row_len, 0x1010);
    let mut want = got.clone();
    ops::mul_add::<F>(&mut got, coeffs[1], &src0);
    oracle_mul_add::<F>(&mut want, coeffs[1], &src0);
    assert_eq!(got, want, "{} mul_add", F::NAME);

    let mut into = vec![0u8; row_len];
    ops::mul_into::<F>(&mut into, coeffs[2], &src0);
    let mut assign = src0.clone();
    ops::mul_assign::<F>(&mut assign, coeffs[2]);
    assert_eq!(into, assign, "{} mul_into/mul_assign", F::NAME);

    let mut elementwise = vec![0u8; row_len];
    ops::mul_elementwise::<F>(&mut elementwise, &src0, &src1);
    for ((got, a), b) in elementwise
        .chunks_exact(F::BYTES)
        .zip(src0.chunks_exact(F::BYTES))
        .zip(src1.chunks_exact(F::BYTES))
    {
        assert_eq!(F::read(got), F::read(a).mul(F::read(b)));
    }

    let mut scatter = noise(row_len * coeffs.len(), 0x2020);
    let mut scatter_want = scatter.clone();
    ops::mul_add_scatter::<F>(&mut scatter, row_len, coeffs, &src0);
    for (row, &coeff) in scatter_want.chunks_exact_mut(row_len).zip(coeffs) {
        oracle_mul_add::<F>(row, coeff, &src0);
    }
    assert_eq!(scatter, scatter_want, "{} scatter", F::NAME);

    let sources = [&src0[..], &src1[..]];
    let gather_coeffs = [coeffs[0], coeffs[2]];
    let mut gather = noise(row_len, 0x3030);
    let mut gather_want = gather.clone();
    ops::mul_add_gather::<F>(&mut gather, &gather_coeffs, &sources);
    for (&coeff, src) in gather_coeffs.iter().zip(sources) {
        oracle_mul_add::<F>(&mut gather_want, coeff, src);
    }
    assert_eq!(gather, gather_want, "{} gather", F::NAME);

    let reverse = [coeffs[2], coeffs[1], coeffs[0]];
    let terms = [(&coeffs[..3], &src0[..]), (&reverse[..], &src1[..])];
    let mut matrix = noise(row_len * 3, 0x4040);
    let mut matrix_want = matrix.clone();
    ops::mul_add_matrix::<F>(&mut matrix, row_len, 3, &terms);
    for &(term_coeffs, src) in &terms {
        for (row, &coeff) in matrix_want.chunks_exact_mut(row_len).zip(term_coeffs) {
            oracle_mul_add::<F>(row, coeff, src);
        }
    }
    assert_eq!(matrix, matrix_want, "{} matrix", F::NAME);
}

#[test]
fn tier3_field_public_ops_match_oracle() {
    check_wide_field_ops::<Gf32>(&[gf32::Elem::ZERO, gf32::Elem::ONE, gf32::Elem(0xdead_beef)]);
    check_wide_field_ops::<Gf64>(&[
        gf64::Elem::ZERO,
        gf64::Elem::ONE,
        gf64::Elem(0x0123_4567_89ab_cdef),
    ]);
    check_wide_field_ops::<FanPaar8>(&[
        fan_paar::fp8::Elem::ZERO,
        fan_paar::fp8::Elem::ONE,
        fan_paar::fp8::Elem(0xa5),
    ]);
    check_wide_field_ops::<FanPaar16>(&[
        fan_paar::fp16::Elem::ZERO,
        fan_paar::fp16::Elem::ONE,
        fan_paar::fp16::Elem(0xa55a),
    ]);
    check_wide_field_ops::<FanPaar32>(&[
        fan_paar::fp32::Elem::ZERO,
        fan_paar::fp32::Elem::ONE,
        fan_paar::fp32::Elem(0xa55a_1234),
    ]);
    check_wide_field_ops::<FanPaar64>(&[
        fan_paar::fp64::Elem::ZERO,
        fan_paar::fp64::Elem::ONE,
        fan_paar::fp64::Elem(0xa55a_1234_dead_beef),
    ]);
}

#[test]
fn gf8d_public_ops_match_oracle() {
    // Single-coefficient AXPY over every coefficient and lane boundary: this
    // drives the reused shuffle kernels with the 0x11D bank.
    for len in LENGTHS {
        let src = noise(len, 0x11d0);
        for coeff in (0..=u8::MAX).map(gf8d::Elem) {
            let mut got = noise(len, 0x11d1);
            let mut want = got.clone();
            ops::mul_add::<Gf8D>(&mut got, coeff, &src);
            oracle_mul_add::<Gf8D>(&mut want, coeff, &src);
            assert_eq!(got, want, "gf8d mul_add len {len}, coeff {coeff:?}");
        }
    }

    // Fused mul_into / mul_assign over every coefficient.
    for len in [16usize, 17, 64, 254] {
        let src = noise(len, 0x11d2);
        for coeff in (0..=u8::MAX).map(gf8d::Elem) {
            let mut into = vec![0xaa; len];
            ops::mul_into::<Gf8D>(&mut into, coeff, &src);
            let mut assign = src.clone();
            ops::mul_assign::<Gf8D>(&mut assign, coeff);
            let mut want = vec![0u8; len];
            oracle_mul_add::<Gf8D>(&mut want, coeff, &src);
            assert_eq!(into, want, "gf8d mul_into len {len} coeff {coeff:?}");
            assert_eq!(assign, want, "gf8d mul_assign len {len} coeff {coeff:?}");
        }
    }

    // Elementwise across every boundary.
    for len in LENGTHS {
        let a = noise(len, 0x11d3);
        let b = noise(len, 0x11d4);
        let mut got = vec![0u8; len];
        ops::mul_elementwise::<Gf8D>(&mut got, &a, &b);
        for ((d, &x), &y) in got.iter().zip(&a).zip(&b) {
            assert_eq!(
                *d,
                gf8d::Elem(x).mul(gf8d::Elem(y)).0,
                "gf8d elementwise len {len}"
            );
        }
    }

    // Every multi-row shape against the oracle.
    check_wide_field_ops::<Gf8D>(&[
        gf8d::Elem::ZERO,
        gf8d::Elem::ONE,
        gf8d::Elem(0x53),
        gf8d::Elem(0xff),
        gf8d::Elem(0x02),
    ]);

    // Prepared coefficients: value recovery (via the bank's byte label) and
    // byte-for-byte agreement with the one-shot path.
    for coeff in [0u8, 1, 2, 0x53, 0xff].map(gf8d::Elem) {
        let prepared = ops::Coeff::<Gf8D>::new(coeff);
        assert_eq!(prepared.value(), coeff, "gf8d prepared_coeff recovery");
        let src = noise(64, 0x11d5);
        let mut got = noise(64, 0x11d6);
        let mut want = got.clone();
        ops::mul_add_with::<Gf8D>(&mut got, &prepared, &src);
        ops::mul_add::<Gf8D>(&mut want, coeff, &src);
        assert_eq!(got, want, "gf8d mul_add_with coeff {coeff:?}");
    }
}

#[test]
fn gf8d_matrix_scattered_matches_contiguous() {
    check_matrix_scattered::<Gf8D>("gf8d", 0x8d5c);
}

// ---------------------------------------------------------------------------
// Prime fields GF(2^31 - 1) and GF(2^64 - 2^32 + 1)
// ---------------------------------------------------------------------------
// Byte lengths straddling the SSE (16 B) and AVX2 (32 B) lane boundaries plus
// odd element tails; Mersenne31 is 4-byte, Goldilocks 8-byte. Truncated under
// Miri to the same boundary cases (see `LENGTHS`).
#[cfg(not(miri))]
const M31_LENS: [usize; 12] = [0, 4, 8, 16, 20, 32, 36, 64, 68, 128, 256, 1020];
#[cfg(miri)]
const M31_LENS: [usize; 6] = [0, 4, 8, 16, 20, 36];
#[cfg(not(miri))]
const GLD_LENS: [usize; 11] = [0, 8, 16, 24, 32, 40, 64, 72, 128, 256, 1024];
#[cfg(miri)]
const GLD_LENS: [usize; 5] = [0, 8, 16, 24, 40];

fn oracle_add_assign<F: Field>(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let v = F::read(d).add(F::read(s));
        F::write(d, v);
    }
}

fn oracle_sub_assign<F: Field>(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let v = F::read(d).sub(F::read(s));
        F::write(d, v);
    }
}

fn oracle_mul_assign<F: Field>(dst: &mut [u8], coeff: F::Elem) {
    for d in dst.chunks_exact_mut(F::BYTES) {
        let v = F::read(d).mul(coeff);
        F::write(d, v);
    }
}

fn oracle_mul_elementwise<F: Field>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::write(d, F::read(x).mul(F::read(y)));
    }
}

/// Canonicalize every lane of `buf` in place through the scalar field, giving
/// the canonical bytes the vector kernels must produce.
fn canon<F: Field>(buf: &mut [u8]) {
    let zero = vec![0u8; buf.len()];
    oracle_add_assign::<F>(buf, &zero);
}

/// Exercise the full public operation surface for a prime field against the
/// scalar-field oracle, on whatever backend the host selected.
fn check_prime_ops<F: FieldKernels>(lens: &[usize], coeffs: &[F::Elem]) {
    for &len in lens {
        // Kernel arithmetic is contracted over canonical inputs; raw-lane
        // totality is covered by the scalar algebra suite.
        let mut src = noise(len, 0xa1);
        canon::<F>(&mut src);
        let dst0 = {
            let mut b = noise(len, 0xb2);
            canon::<F>(&mut b);
            b
        };

        // add_assign / sub_assign and their round trip.
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

            let mut rt = dst0.clone();
            ops::add_assign::<F>(&mut rt, &src);
            ops::sub_assign::<F>(&mut rt, &src);
            assert_eq!(rt, dst0, "add then sub restores the addend len {len}");
        }

        // mul_elementwise: both operands vary per lane.
        {
            let mut b = noise(len, 0xc3);
            canon::<F>(&mut b);
            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            ops::mul_elementwise::<F>(&mut got, &src, &b);
            oracle_mul_elementwise::<F>(&mut want, &src, &b);
            assert_eq!(got, want, "mul_elementwise len {len}");
        }

        for &coeff in coeffs {
            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::mul_add::<F>(&mut got, coeff, &src);
            oracle_mul_add::<F>(&mut want, coeff, &src);
            assert_eq!(got, want, "mul_add len {len} coeff {coeff:?}");

            let mut got = dst0.clone();
            let mut want = dst0.clone();
            ops::mul_assign::<F>(&mut got, coeff);
            oracle_mul_assign::<F>(&mut want, coeff);
            assert_eq!(got, want, "mul_assign len {len} coeff {coeff:?}");

            let mut got = dst0.clone();
            let mut want = src.clone();
            ops::mul_into::<F>(&mut got, coeff, &src);
            oracle_mul_assign::<F>(&mut want, coeff);
            assert_eq!(got, want, "mul_into len {len} coeff {coeff:?}");
        }
    }
}

/// Gather/scatter/matrix and an erasure round trip for a prime field.
fn check_prime_shapes<F: FieldKernels>(row_len: usize, coeffs: &[F::Elem]) {
    let nsrc = coeffs.len();
    let srcs: Vec<Vec<u8>> = (0..nsrc)
        .map(|i| {
            let mut s = noise(row_len, 0x100 + i as u64);
            canon::<F>(&mut s);
            s
        })
        .collect();
    let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();

    // gather: dst += sum(coeffs[i] * srcs[i]).
    {
        let mut got = noise(row_len, 0x55);
        let mut want = got.clone();
        ops::mul_add_gather::<F>(&mut got, coeffs, &src_refs);
        for (&c, s) in coeffs.iter().zip(&src_refs) {
            oracle_mul_add::<F>(&mut want, c, s);
        }
        assert_eq!(got, want, "gather");
    }

    // dot_product: overwrite dst = sum(coeffs[i] * srcs[i]).
    {
        let mut got = noise(row_len, 0x55);
        let mut want = vec![0u8; row_len];
        ops::dot_product::<F>(&mut got, coeffs, &src_refs);
        for (&c, s) in coeffs.iter().zip(&src_refs) {
            oracle_mul_add::<F>(&mut want, c, s);
        }
        assert_eq!(got, want, "dot_product");
    }

    // scatter: one source into many rows.
    {
        let mut got = noise(row_len * nsrc, 0x66);
        let mut want = got.clone();
        ops::mul_add_scatter::<F>(&mut got, row_len, coeffs, &srcs[0]);
        for (row, &c) in want.chunks_exact_mut(row_len).zip(coeffs) {
            oracle_mul_add::<F>(row, c, &srcs[0]);
        }
        assert_eq!(got, want, "scatter");
    }

    // matrix: many sources into many rows.
    {
        let nrows = nsrc;
        let terms: Vec<(&[F::Elem], &[u8])> = src_refs.iter().map(|s| (coeffs, *s)).collect();
        let mut got = noise(row_len * nrows, 0x77);
        let mut want = got.clone();
        ops::mul_add_matrix::<F>(&mut got, row_len, nrows, &terms);
        for &(cs, s) in &terms {
            for (row, &c) in want.chunks_exact_mut(row_len).take(nrows).zip(cs) {
                oracle_mul_add::<F>(row, c, s);
            }
        }
        assert_eq!(got, want, "matrix");
    }
}

/// Recover a lost symbol over GF(p): a subtraction the binary fields never do.
fn check_prime_recovery<F: FieldKernels>(row_len: usize, a: F::Elem, b: F::Elem) {
    // parity = a*x + b*y; recover x = (parity - b*y) / a.
    let x = noise(row_len, 0x11);
    let y = noise(row_len, 0x22);
    let mut parity = vec![0u8; row_len];
    oracle_mul_add::<F>(&mut parity, a, &x);
    oracle_mul_add::<F>(&mut parity, b, &y);

    let mut recovered = parity.clone();
    let mut by = vec![0u8; row_len];
    ops::mul_into::<F>(&mut by, b, &y);
    ops::sub_assign::<F>(&mut recovered, &by); // parity - b*y == a*x
    ops::mul_assign::<F>(&mut recovered, a.inv()); // / a

    let mut want = x.to_vec();
    canon::<F>(&mut want);
    assert_eq!(recovered, want, "erasure recovery over GF(p)");
}

fn m31(v: u32) -> mersenne31::Elem {
    mersenne31::Elem(v)
}
fn gld(v: u64) -> goldilocks::Elem {
    goldilocks::Elem(v)
}

#[test]
fn mersenne31_public_ops_match_oracle() {
    let coeffs = [
        m31(0),
        m31(1),
        m31(2),
        m31(7),
        m31(0x5555_5555),
        m31(0x7FFF_FFFE),
        m31(0x7FFF_FFFF), // non-canonical zero
        m31(0xFFFF_FFFF), // non-canonical one
    ];
    check_prime_ops::<Mersenne31>(&M31_LENS, &coeffs);
    check_prime_shapes::<Mersenne31>(64, &coeffs[1..6]);
    check_prime_shapes::<Mersenne31>(20, &coeffs[1..6]);
    check_prime_recovery::<Mersenne31>(64, m31(3), m31(0x1234_5678));
}

fn qm(re: u32, im: u32) -> quad_mersenne31::Elem {
    quad_mersenne31::Elem(re, im)
}

/// 8-byte element lengths straddling the SSE (16 B = 2 elements) and AVX2
/// (32 B = 4 elements) lane boundaries plus odd-element tails. Truncated
/// under Miri to the same boundary cases (see `LENGTHS`).
#[cfg(not(miri))]
const QM_LENS: [usize; 12] = [0, 8, 16, 24, 32, 40, 64, 72, 128, 256, 512, 1016];
#[cfg(miri)]
const QM_LENS: [usize; 5] = [0, 8, 16, 24, 40];

#[test]
fn quad_mersenne31_public_ops_match_oracle() {
    let coeffs = [
        qm(0, 0),
        qm(1, 0),
        qm(0, 1), // i
        qm(2, 7),
        qm(0x5555_5555, 0x1234_5678),
        qm(0x7FFF_FFFE, 0x7FFF_FFFF), // non-canonical zero in im
        qm(0xFFFF_FFFF, 0x8000_0000), // non-canonical limbs
    ];
    check_prime_ops::<QuadMersenne31>(&QM_LENS, &coeffs);
    check_prime_shapes::<QuadMersenne31>(64, &coeffs[1..6]);
    check_prime_shapes::<QuadMersenne31>(24, &coeffs[1..6]);
    // Recovery needs two independent coefficients with a nonzero determinant;
    // (3+i) and (5+2i) qualify.
    check_prime_recovery::<QuadMersenne31>(64, qm(3, 1), qm(5, 2));
}

#[test]
fn goldilocks_public_ops_match_oracle() {
    let coeffs = [
        gld(0),
        gld(1),
        gld(2),
        gld(7),
        gld(0x5555_5555_5555_5555),
        gld(goldilocks::MODULUS - 1),
        gld(goldilocks::MODULUS), // non-canonical zero
        gld(u64::MAX),            // non-canonical
    ];
    check_prime_ops::<Goldilocks>(&GLD_LENS, &coeffs);
    check_prime_shapes::<Goldilocks>(64, &coeffs[1..6]);
    check_prime_shapes::<Goldilocks>(24, &coeffs[1..6]);
    check_prime_recovery::<Goldilocks>(64, gld(3), gld(0x1234_5678_9abc_def0));
}

// ---------------------------------------------------------------------------
// Backend queries and prepared-coefficient plumbing
// ---------------------------------------------------------------------------

#[test]
fn backend_queries_are_consistent_per_field() {
    use fgf::{Backend, KernelBackend, backend, backend_for, has_vector_elementwise};

    let process = backend();
    if cfg!(not(feature = "simd")) {
        assert!(matches!(process, Backend::Scalar), "simd is compiled out");
    }
    if process == Backend::Scalar {
        assert!(!process.has_native_mul(), "scalar has no native multiply");
        assert!(!process.has_blocked_rows(), "scalar has no blocked rows");
    }

    macro_rules! queries {
        ($($field:ty),+ $(,)?) => {$(
            let per_field = backend_for::<$field>();
            // `Backend` order is descending capability (stronger sorts less),
            // so a field's backend is always at or below the process one.
            assert!(
                per_field >= process,
                concat!(stringify!($field), ": field backend exceeds the process backend")
            );
            let vectorized = has_vector_elementwise::<$field>();
            assert!(
                !vectorized || per_field != Backend::Scalar,
                concat!(stringify!($field), ": vectorized elementwise requires a vector field backend")
            );
        )+};
    }
    queries!(
        Gf8B,
        Gf8D,
        Gf16,
        Gf32,
        Gf64,
        FanPaar8,
        FanPaar16,
        FanPaar32,
        FanPaar64,
        Mersenne31,
        Goldilocks,
        QuadMersenne31,
    );
}

/// Every field's `add_assign`/`sub_assign` against the elementwise oracle,
/// including the prime fields where subtraction is a different fold.
#[test]
fn add_and_sub_assign_match_oracle_for_every_field() {
    fn check<F: FieldKernels>(len: usize, coeff: F::Elem) {
        let a = noise(len, 0x11);
        let mut b = noise(len, 0x22);
        // Plant zero element lanes: the prime-field subtraction fold has
        // per-limb zero shortcuts that random bytes never trigger.
        for lane in b.chunks_exact_mut(F::BYTES).step_by(3) {
            lane.fill(0);
        }
        let mut got = a.clone();
        let mut want = a.clone();
        ops::add_assign::<F>(&mut got, &b);
        for (d, s) in want
            .chunks_exact_mut(F::BYTES)
            .zip(b.chunks_exact(F::BYTES))
        {
            let value = F::read(d).add(F::read(s));
            F::write(d, value);
        }
        assert_eq!(got, want, "add_assign over {} bytes", len);

        let mut got = a.clone();
        let mut want = a.clone();
        ops::sub_assign::<F>(&mut got, &b);
        for (d, s) in want
            .chunks_exact_mut(F::BYTES)
            .zip(b.chunks_exact(F::BYTES))
        {
            let value = F::read(d).sub(F::read(s));
            F::write(d, value);
        }
        assert_eq!(got, want, "sub_assign over {} bytes", len);
        let _ = coeff;
    }
    check::<Gf8B>(8, gf8b::Elem(3));
    check::<Gf8D>(8, gf8d::Elem(3));
    check::<Gf16>(8, gf16::Elem(3));
    check::<Gf32>(8, gf32::Elem(3));
    check::<Gf64>(8, gf64::Elem(3));
    check::<FanPaar8>(8, fan_paar::fp8::Elem(3));
    check::<FanPaar16>(8, fan_paar::fp16::Elem(3));
    check::<FanPaar32>(8, fan_paar::fp32::Elem(3));
    check::<FanPaar64>(8, fan_paar::fp64::Elem(3));
    check::<Mersenne31>(8, mersenne31::Elem(3));
    check::<Goldilocks>(8, goldilocks::Elem(3));
    check::<QuadMersenne31>(8, quad_mersenne31::Elem(3, 5));
}

#[test]
#[cfg(feature = "std")]
fn coeff_and_plan_accessors_report_their_contents() {
    let coeffs = [gf16::Elem(0x0102), gf16::Elem(0), gf16::Elem(0xbeef)];

    let coeff = ops::Coeff::<Gf16>::new(coeffs[0]);
    assert_eq!(coeff.value(), coeffs[0]);
    assert_eq!(coeff.clone().value(), coeffs[0]);
    assert!(format!("{coeff:?}").contains("Coeff"), "Coeff Debug");

    let plan = ops::Plan::<Gf16>::new(&coeffs);
    assert_eq!(plan.len(), 3);
    assert!(!plan.is_empty());
    assert_eq!(plan.dimensions(), (1, 3));
    assert_eq!(plan.values().collect::<Vec<_>>(), coeffs);
    assert_eq!(plan.coeffs().map(|c| c.value()).collect::<Vec<_>>(), coeffs);
    assert_eq!(plan.get(1).map(|c| c.value()), Some(coeffs[1]));
    assert!(plan.get(3).is_none());
    assert_eq!(plan.clone().values().collect::<Vec<_>>(), coeffs);
    assert!(format!("{plan:?}").contains("Plan"), "Plan Debug");

    let empty = ops::Plan::<Gf16>::new(&[]);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(empty.get(0).is_none());
    // A 1x0 plan has one (empty) row; every row index reads the same
    // zero-length slice, so out-of-bounds detection needs columns > 0.
    assert_eq!(empty.row(0).map(|row| row.count()), Some(0));

    let matrix = ops::Plan::<Gf16>::matrix(
        2,
        3,
        &[
            coeffs[0], coeffs[1], coeffs[2], coeffs[2], coeffs[1], coeffs[0],
        ],
    );
    assert_eq!(matrix.dimensions(), (2, 3));
    assert_eq!(matrix.get_at(1, 2).map(|c| c.value()), Some(coeffs[0]));
    assert!(matrix.get_at(2, 0).is_none(), "row out of bounds");
    assert!(matrix.get_at(0, 3).is_none(), "column out of bounds");
    assert_eq!(
        matrix
            .row(1)
            .map(|row| row.map(|c| c.value()).collect::<Vec<_>>()),
        Some(vec![coeffs[2], coeffs[1], coeffs[0]])
    );
    assert!(matrix.row(2).is_none(), "row out of bounds");
    // A single-row plan still reads through `row`.
    let vector = ops::Plan::<Gf16>::new(&coeffs);
    assert_eq!(
        vector
            .row(0)
            .map(|row| row.map(|c| c.value()).collect::<Vec<_>>()),
        Some(coeffs.to_vec())
    );
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "Plan::matrix: 3 coefficients for 2x2 matrix")]
fn plan_matrix_rejects_wrong_coefficient_count() {
    let _ = ops::Plan::<Gf8B>::matrix(2, 2, &[gf8b::Elem(1); 3]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "Plan::matrix: dimensions overflow")]
fn plan_matrix_rejects_overflowing_dimensions() {
    let _ = ops::Plan::<Gf8B>::matrix(usize::MAX, 2, &[]);
}

// ---------------------------------------------------------------------------
// Geometry validation and degenerate inputs
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "std")]
fn zero_and_empty_inputs_are_well_defined() {
    // All-zero coefficient plans skip the kernel entirely and leave the
    // destination untouched (or zeroed, for the overwrite shapes).
    let srcs: Vec<Vec<u8>> = [noise(8, 0x31), noise(8, 0x32)].to_vec();
    let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let zero_plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(0); 2]);

    let mut rows = noise(16, 0x33);
    let before = rows.clone();
    ops::mul_add_scatter_with(&mut rows, 8, &zero_plan, &srcs[0]);
    assert_eq!(rows, before, "all-zero scatter plan is a no-op");

    let mut dst = noise(8, 0x34);
    let before = dst.clone();
    ops::mul_add_gather::<Gf8B>(&mut dst, &[gf8b::Elem(0), gf8b::Elem(0)], &refs);
    assert_eq!(dst, before, "all-zero gather is a no-op");
    ops::mul_add_gather_with(&mut dst, &zero_plan, &refs);
    assert_eq!(dst, before, "all-zero gather plan is a no-op");

    // Overwrite shapes zero the destination instead.
    let mut dst = noise(8, 0x35);
    ops::dot_product::<Gf8B>(&mut dst, &[gf8b::Elem(0), gf8b::Elem(0)], &refs);
    assert_eq!(dst, vec![0; 8], "all-zero dot product fills zero");
    ops::dot_product_with(&mut dst, &zero_plan, &refs);
    assert_eq!(dst, vec![0; 8], "all-zero dot product plan fills zero");

    // Empty inputs.
    let mut dst = noise(8, 0x36);
    ops::dot_product::<Gf8B>(&mut dst, &[], &[]);
    assert_eq!(dst, vec![0; 8], "no sources fills zero");
    ops::dot_product_with(&mut dst, &ops::Plan::<Gf8B>::new(&[]), &[]);
    assert_eq!(dst, vec![0; 8], "empty plan fills zero");

    let mut rows = noise(16, 0x37);
    let before = rows.clone();
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 2, &[]);
    assert_eq!(rows, before, "no terms is a no-op");
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 0, &[(&[], &srcs[0])]);
    assert_eq!(rows, before, "zero rows is a no-op");
    ops::mul_add_matrix_with(
        &mut rows,
        8,
        0,
        &ops::Plan::<Gf8B>::matrix(1, 0, &[]),
        &[&srcs[0]],
    );
    assert_eq!(rows, before, "zero-row matrix plan is a no-op");
    ops::mul_add_matrix_with(&mut rows, 8, 2, &ops::Plan::<Gf8B>::matrix(0, 2, &[]), &[]);
    assert_eq!(rows, before, "empty-source matrix plan is a no-op");

    let mut overwritten = vec![1u8; 16];
    ops::dot_product_matrix::<Gf8B>(&mut overwritten, 8, 0, &[(&[], &srcs[0])]);
    ops::dot_product_matrix::<Gf8B>(&mut overwritten, 8, 2, &[]);
    assert_eq!(overwritten, vec![0; 16], "empty terms zero the rows");
    ops::dot_product_matrix_with(
        &mut overwritten,
        8,
        0,
        &ops::Plan::<Gf8B>::matrix(0, 0, &[]),
        &[],
    );
    assert_eq!(overwritten, vec![0; 16], "zero-row matrix plan keeps bytes");
    let mut seeded = vec![1u8; 16];
    ops::dot_product_matrix_with(
        &mut seeded,
        8,
        2,
        &ops::Plan::<Gf8B>::matrix(0, 2, &[]),
        &[],
    );
    assert_eq!(
        seeded,
        vec![0; 16],
        "empty-source matrix plan zeros the rows"
    );

    // Scattered reconstruction with nothing to do.
    ops::mul_add_matrix_scattered::<Gf8B>(&mut rows, 8, &[0, 8], &[]);
    ops::mul_add_matrix_scattered::<Gf8B>(&mut rows, 8, &[], &[(&[], &srcs[0])]);
}

#[test]
#[cfg(feature = "std")]
fn single_term_dot_products_match_mul_into() {
    let src = noise(12, 0x41);
    let coeff = gf16::Elem(0x0a5a);
    let mut dotted = noise(12, 0x42);
    ops::dot_product::<Gf16>(&mut dotted, &[coeff], &[&src]);
    let mut scaled = [0u8; 12];
    ops::mul_into::<Gf16>(&mut scaled, coeff, &src);
    assert_eq!(dotted, scaled);

    let mut planned = noise(12, 0x43);
    ops::dot_product_with::<Gf16>(&mut planned, &ops::Plan::new(&[coeff]), &[&src]);
    assert_eq!(planned, scaled);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "mul_add_scatter_with: rows is 8 bytes")]
fn scatter_with_rejects_short_rows_buffer() {
    let plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(1); 2]);
    let mut rows = vec![0u8; 8];
    ops::mul_add_scatter_with(&mut rows, 8, &plan, &[0u8; 8]);
}

#[test]
#[should_panic(expected = "mul_add_gather: 2 coefficients for 1 sources")]
fn gather_rejects_coefficient_count_mismatch() {
    let src = [0u8; 8];
    let mut dst = vec![0u8; 8];
    ops::mul_add_gather::<Gf8B>(&mut dst, &[gf8b::Elem(1), gf8b::Elem(2)], &[&src]);
}

#[test]
#[should_panic(expected = "mul_add_gather: source is 7 bytes")]
fn gather_rejects_source_length_mismatch() {
    let src = [0u8; 7];
    let mut dst = vec![0u8; 8];
    ops::mul_add_gather::<Gf8B>(&mut dst, &[gf8b::Elem(1)], &[&src]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "mul_add_gather_with: plan has 2 coefficients")]
fn gather_with_rejects_plan_source_mismatch() {
    let src = [0u8; 8];
    let mut dst = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::mul_add_gather_with(&mut dst, &plan, &[&src]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "mul_add_gather_with: source 0 is 7 bytes")]
fn gather_with_rejects_source_length_mismatch() {
    let src = [0u8; 7];
    let mut dst = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(1)]);
    ops::mul_add_gather_with(&mut dst, &plan, &[&src]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "dot_product_with: plan has 2 coefficients")]
fn dot_product_with_rejects_plan_source_mismatch() {
    let src = [0u8; 8];
    let mut dst = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::dot_product_with(&mut dst, &plan, &[&src]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "dot_product_with: source 1 is 9 bytes")]
fn dot_product_with_rejects_source_length_mismatch() {
    let srcs = [vec![0u8; 8], vec![0u8; 9]];
    let mut dst = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::new(&[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::dot_product_with(&mut dst, &plan, &[srcs[0].as_slice(), srcs[1].as_slice()]);
}

#[test]
#[should_panic(expected = "mul_add_matrix: rows is 8 bytes")]
fn matrix_rejects_short_rows_buffer() {
    let mut rows = vec![0u8; 8];
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1); 2], &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "mul_add_matrix: term supplies 1 coefficients for 2 rows")]
fn matrix_rejects_term_coefficient_count() {
    let mut rows = vec![0u8; 16];
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1)], &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "mul_add_matrix: source is 7 bytes")]
fn matrix_rejects_source_length_mismatch() {
    let mut rows = vec![0u8; 16];
    ops::mul_add_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1); 2], &[0u8; 7])]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "mul_add_matrix_with: rows is 8 bytes")]
fn matrix_with_rejects_short_rows_buffer() {
    let mut rows = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::matrix(1, 2, &[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::mul_add_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 8]]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "plan dimensions do not match")]
fn matrix_with_rejects_dimension_mismatch() {
    let mut rows = vec![0u8; 16];
    let plan = ops::Plan::<Gf8B>::matrix(2, 2, &[gf8b::Elem(1); 4]);
    ops::mul_add_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 8]]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "mul_add_matrix_with: source 0 is 7 bytes")]
fn matrix_with_rejects_source_length_mismatch() {
    let mut rows = vec![0u8; 16];
    let plan = ops::Plan::<Gf8B>::matrix(1, 2, &[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::mul_add_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 7]]);
}

#[test]
#[should_panic(expected = "dot_product_matrix: rows is 8 bytes")]
fn dot_product_matrix_rejects_short_rows_buffer() {
    let mut rows = vec![0u8; 8];
    ops::dot_product_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1); 2], &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "dot_product_matrix: term supplies 1 coefficients")]
fn dot_product_matrix_rejects_term_coefficient_count() {
    let mut rows = vec![0u8; 16];
    ops::dot_product_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1)], &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "dot_product_matrix: source is 7 bytes")]
fn dot_product_matrix_rejects_source_length_mismatch() {
    let mut rows = vec![0u8; 16];
    ops::dot_product_matrix::<Gf8B>(&mut rows, 8, 2, &[(&[gf8b::Elem(1); 2], &[0u8; 7])]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "dot_product_matrix_with: rows is 8 bytes")]
fn dot_product_matrix_with_rejects_short_rows_buffer() {
    let mut rows = vec![0u8; 8];
    let plan = ops::Plan::<Gf8B>::matrix(1, 2, &[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::dot_product_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 8]]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "plan dimensions do not match")]
fn dot_product_matrix_with_rejects_dimension_mismatch() {
    let mut rows = vec![0u8; 16];
    let plan = ops::Plan::<Gf8B>::matrix(3, 2, &[gf8b::Elem(1); 6]);
    ops::dot_product_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 8]]);
}

#[test]
#[cfg(feature = "std")]
#[should_panic(expected = "dot_product_matrix_with: source 0 is 7 bytes")]
fn dot_product_matrix_with_rejects_source_length_mismatch() {
    let mut rows = vec![0u8; 16];
    let plan = ops::Plan::<Gf8B>::matrix(1, 2, &[gf8b::Elem(1), gf8b::Elem(2)]);
    ops::dot_product_matrix_with(&mut rows, 8, 2, &plan, &[&[0u8; 7]]);
}

#[test]
#[should_panic(expected = "spans 8..16 but dst is 12 bytes")]
fn scattered_rejects_out_of_bounds_row() {
    let mut dst = vec![0u8; 12];
    ops::mul_add_matrix_scattered::<Gf8B>(
        &mut dst,
        8,
        &[0, 8],
        &[(&[gf8b::Elem(1); 2], &[0u8; 8])],
    );
}

#[test]
#[should_panic(expected = "not a whole number of")]
fn scattered_rejects_misaligned_row_start() {
    let mut dst = vec![0u8; 32];
    ops::mul_add_matrix_scattered::<Gf16>(
        &mut dst,
        8,
        &[0, 13],
        &[(&[gf16::Elem(1); 2], &[0u8; 8])],
    );
}

#[test]
#[should_panic(expected = "mul_add_matrix_scattered: term supplies 1 coefficients")]
fn scattered_rejects_term_coefficient_count() {
    let mut dst = vec![0u8; 16];
    ops::mul_add_matrix_scattered::<Gf8B>(&mut dst, 8, &[0, 8], &[(&[gf8b::Elem(1)], &[0u8; 8])]);
}

#[test]
#[should_panic(expected = "mul_add_matrix_scattered: source is 7 bytes")]
fn scattered_rejects_source_length_mismatch() {
    let mut dst = vec![0u8; 16];
    ops::mul_add_matrix_scattered::<Gf8B>(
        &mut dst,
        8,
        &[0, 8],
        &[(&[gf8b::Elem(1); 2], &[0u8; 7])],
    );
}

#[test]
#[should_panic(expected = "mul_elementwise: dst is 8 bytes but a is 7 bytes")]
fn elementwise_rejects_first_operand_mismatch() {
    let mut dst = vec![0u8; 8];
    ops::mul_elementwise::<Gf8B>(&mut dst, &[0u8; 7], &[0u8; 8]);
}

#[test]
#[should_panic(expected = "mul_elementwise: dst is 8 bytes but b is 9 bytes")]
fn elementwise_rejects_second_operand_mismatch() {
    let mut dst = vec![0u8; 8];
    ops::mul_elementwise::<Gf8B>(&mut dst, &[0u8; 8], &[0u8; 9]);
}

#[test]
#[should_panic(expected = "pack: dst is 2 bytes but 3 GF(2^8) elements")]
fn pack_rejects_destination_mismatch() {
    let mut dst = [0u8; 2];
    ops::pack::<Gf8B>(&mut dst, &[gf8b::Elem(1), gf8b::Elem(2), gf8b::Elem(3)]);
}

#[test]
#[should_panic(expected = "unpack: src is 2 bytes but dst holds 3 GF(2^8) elements")]
fn unpack_rejects_source_mismatch() {
    let mut dst = [gf8b::Elem(0); 3];
    ops::unpack::<Gf8B>(&mut dst, &[0u8; 2]);
}

#[test]
fn gf8d_large_buffer_mul_assign_matches_oracle() {
    // Past 64 KiB the GFNI in-place scale takes its shuffle variant
    // (BENCHMARKS.md); hold both sides of the crossover to the oracle.
    for len in [65_536 - 8, 65_536 + 7] {
        let mut got = noise(len, 0x62);
        let mut want = got.clone();
        ops::mul_assign::<Gf8D>(&mut got, gf8d::Elem(0x53));
        oracle_mul_assign::<Gf8D>(&mut want, gf8d::Elem(0x53));
        assert_eq!(got, want, "in-place scale at {len} bytes");
    }
}

// ---------------------------------------------------------------------------
// Prepared-plan operations match their one-shot forms
// ---------------------------------------------------------------------------

/// Every `_with` shape must produce byte-identical results to the one-shot
/// operation over the same coefficients, for a representative of each kernel
/// family: GFNI-blocked (`Gf8B`), affine-blocked (`Gf8D`), tower (`Gf16`),
/// scalar-defaulted wider towers (`Gf32`), Fan–Paar, and the prime fields.
#[test]
#[cfg(feature = "std")]
fn plan_variants_match_one_shot_operations() {
    fn run<F: FieldKernels>(srcs: Vec<Vec<u8>>, coeffs: Vec<F::Elem>) {
        const ELEMS: usize = 8;
        let row_len = ELEMS * F::BYTES;
        // Kernel arithmetic is contracted over canonical inputs for the prime
        // fields (raw-lane totality is scalar-algebra territory), and this
        // sweep compares kernel paths against each other.
        let seed = || {
            let mut b = noise(2 * row_len, 0xa0);
            canon::<F>(&mut b);
            b
        };
        let fresh = |seed_id: u64| {
            let mut b = noise(row_len, seed_id);
            canon::<F>(&mut b);
            b
        };
        let refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
        let [c0, c1, c2, ..] = coeffs[..] else {
            unreachable!("three coefficients")
        };

        // Scatter: one source into as many rows as the plan has coefficients.
        let mut one_shot = seed();
        let mut planned = one_shot.clone();
        ops::mul_add_scatter::<F>(&mut one_shot, row_len, &coeffs[..2], &srcs[0]);
        ops::mul_add_scatter_with::<F>(
            &mut planned,
            row_len,
            &ops::Plan::new(&coeffs[..2]),
            &srcs[0],
        );
        assert_eq!(one_shot, planned, "scatter");

        // Gather: many sources into one row.
        let mut one_shot = fresh(0xa1);
        let mut planned = one_shot.clone();
        ops::mul_add_gather::<F>(&mut one_shot, &coeffs, &refs);
        ops::mul_add_gather_with::<F>(&mut planned, &ops::Plan::new(&coeffs), &refs);
        assert_eq!(one_shot, planned, "gather");

        // Dot product: overwrite one row.
        let mut one_shot = fresh(0xa2);
        let mut planned = one_shot.clone();
        ops::dot_product::<F>(&mut one_shot, &coeffs, &refs);
        ops::dot_product_with::<F>(&mut planned, &ops::Plan::new(&coeffs), &refs);
        assert_eq!(one_shot, planned, "dot product");

        // Matrix: two terms into two rows, accumulating.
        let terms = &[
            (coeffs[..2].as_ref(), srcs[0].as_slice()),
            (coeffs[1..3].as_ref(), srcs[2].as_slice()),
        ];
        let mut one_shot = seed();
        let mut planned = one_shot.clone();
        ops::mul_add_matrix::<F>(&mut one_shot, row_len, 2, terms);
        let flat = [coeffs[0], coeffs[1], coeffs[1], coeffs[2]];
        let matrix = ops::Plan::<F>::matrix(2, 2, &flat);
        ops::mul_add_matrix_with::<F>(
            &mut planned,
            row_len,
            2,
            &matrix,
            &[srcs[0].as_slice(), srcs[2].as_slice()],
        );
        assert_eq!(one_shot, planned, "matrix");

        // Overwrite matrix: fresh rows from zero.
        let mut one_shot = vec![0u8; 2 * row_len];
        let mut planned = one_shot.clone();
        ops::dot_product_matrix::<F>(&mut one_shot, row_len, 2, terms);
        ops::dot_product_matrix_with::<F>(
            &mut planned,
            row_len,
            2,
            &matrix,
            &[srcs[0].as_slice(), srcs[2].as_slice()],
        );
        assert_eq!(one_shot, planned, "dot product matrix");

        // Single-coefficient prepared forms, through both `Coeff` and a
        // borrowed `CoeffRef` from a plan.
        let mut one_shot = fresh(0xa4);
        let mut from_coeff = one_shot.clone();
        let mut from_coeff_ref = one_shot.clone();
        ops::mul_add::<F>(&mut one_shot, c2, &srcs[1]);
        ops::mul_add_with::<F>(&mut from_coeff, &ops::Coeff::new(c2), &srcs[1]);
        let plan = ops::Plan::<F>::new(&coeffs);
        let borrowed = plan.get(2).expect("three coefficients");
        ops::mul_add_with::<F>(&mut from_coeff_ref, &borrowed, &srcs[1]);
        assert_eq!(one_shot, from_coeff, "mul_add_with through Coeff");
        assert_eq!(one_shot, from_coeff_ref, "mul_add_with through CoeffRef");

        let mut one_shot = fresh(0xa5);
        let mut planned = one_shot.clone();
        ops::mul_into::<F>(&mut one_shot, c1, &srcs[0]);
        ops::mul_into_with::<F>(&mut planned, &ops::Coeff::new(c1), &srcs[0]);
        assert_eq!(one_shot, planned, "mul_into_with");

        let mut one_shot = fresh(0xa6);
        let mut planned = one_shot.clone();
        ops::mul_assign::<F>(&mut one_shot, c1);
        ops::mul_assign_with::<F>(&mut planned, &ops::Coeff::new(c1));
        assert_eq!(one_shot, planned, "mul_assign_with");
        let _ = c0;
    }

    macro_rules! check {
        ($field:ty, $($coeff:expr),+ $(,)?) => {{
            let row_len = 8 * <$field as Field>::BYTES;
            let srcs: Vec<Vec<u8>> = (0..3).map(|i| noise(row_len, 0x90 + i)).collect();
            run::<$field>(srcs, vec![$($coeff),+]);
        }};
    }
    check!(Gf8B, gf8b::Elem(0), gf8b::Elem(1), gf8b::Elem(0x8d));
    check!(Gf8D, gf8d::Elem(0), gf8d::Elem(1), gf8d::Elem(0x53));
    check!(Gf16, gf16::Elem(0), gf16::Elem(1), gf16::Elem(0x0a5a));
    check!(Gf32, gf32::Elem(0), gf32::Elem(1), gf32::Elem(0x0a5a_1234));
    check!(
        Gf64,
        gf64::Elem(0),
        gf64::Elem(1),
        gf64::Elem(0x0a5a_1234_dead_beef)
    );
    check!(
        FanPaar8,
        fan_paar::fp8::Elem(0),
        fan_paar::fp8::Elem(1),
        fan_paar::fp8::Elem(0x8d)
    );
    check!(
        FanPaar16,
        fan_paar::fp16::Elem(0),
        fan_paar::fp16::Elem(1),
        fan_paar::fp16::Elem(0xa55a)
    );
    check!(
        FanPaar32,
        fan_paar::fp32::Elem(0),
        fan_paar::fp32::Elem(1),
        fan_paar::fp32::Elem(0xa55a_1234)
    );
    check!(
        FanPaar64,
        fan_paar::fp64::Elem(0),
        fan_paar::fp64::Elem(1),
        fan_paar::fp64::Elem(0xa55a_1234_dead_beef)
    );
    check!(
        Mersenne31,
        mersenne31::Elem(0),
        mersenne31::Elem(1),
        mersenne31::Elem(0x1234_5678)
    );
    check!(
        Goldilocks,
        goldilocks::Elem(0),
        goldilocks::Elem(1),
        goldilocks::Elem(0x1234_5678_9abc_def0)
    );
    check!(
        QuadMersenne31,
        quad_mersenne31::Elem(0, 0),
        quad_mersenne31::Elem(1, 0),
        quad_mersenne31::Elem(0x1234_5678, 0x9abc_def0)
    );
}
