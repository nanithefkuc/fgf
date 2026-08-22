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
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8B, Gf8D, Gf16, Gf32, Gf64, fan_paar, gf8b, gf8d,
    gf16, gf32, gf64, ops,
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
const LENGTHS: [usize; 12] = [0, 2, 8, 16, 18, 32, 34, 64, 66, 128, 254, 1024];

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
    for row_len in [1usize, 16, 33, 64, 100, 512] {
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
    for &rl_elems in &[1usize, 16, 33, 64, 100, 512] {
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
    for &rl_elems in &[1usize, 16, 33, 64, 100, 512] {
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
    let row_len = 1024;

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
