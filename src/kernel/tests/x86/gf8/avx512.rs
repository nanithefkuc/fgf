//! AVX-512 GF(2^8) differentials: the exhaustive single-row sweeps, the
//! blocked multi-row bodies, the resolve-chunk and column-block term
//! counts, and the alignment peels.

use super::*;

/// The token summon permits direct kernel testing independently of
/// dispatch. The sweep is exhaustive over coefficients in both
/// coefficient forms, and the all-byte-values source covers the products.
#[cfg(feature = "simd512")]
#[test]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
// Exhaustive coefficient forms with all-byte-values sources.
#[allow(clippy::too_many_lines)]
fn gf8_avx512_kernels_match_reference() {
    if tier_gate::summon::<X64V4xToken>().is_none() {
        return;
    }
    // The blocked entries take the coefficient form directly: raw here,
    // and the dispatch-shaped `Prepared` twin in the resolve test below.
    tier_gate::run_entry(x86::gf8::mul_add_avx512, |token, entry| {
        check_gf8_exhaustive_mul_add(
            "gf8 rs avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx512, |token, entry| {
        check_gf8_exhaustive_mul_assign(
            "gf8 rs avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c| {
                entry(token, dst, c);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx512, |token, entry| {
        check_gf8_exhaustive_mul_into(
            "gf8 rs avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    // The AES field through the same 512-bit bodies: every coefficient
    // folds in through its affine map, `GF2P8MULB` never enters.
    tier_gate::run_entry(x86::gf8::mul_add_avx512, |token, entry| {
        check_gf8_exhaustive_mul_add(
            "gf8 aes avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx512, |token, entry| {
        check_gf8_exhaustive_mul_assign(
            "gf8 aes avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c| {
                entry(token, dst, c);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx512, |token, entry| {
        check_gf8_exhaustive_mul_into(
            "gf8 aes avx512",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    // Prefetch coverage: at and above `PREFETCH_MIN` the into body names
    // lines 512 and 640 bytes ahead, so the length must clear the guard
    // plus tiles, lanes, and a sub-lane tail.
    tier_gate::run_entry(x86::gf8::mul_into_avx512, |token, entry| {
        for len in [24 * 1024, 24 * 1024 + 359, 32 * 1024 + 71] {
            let src = noise(len, 0xb20 + len as u64);
            for &coeff in &[
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::ZERO,
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::ONE,
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0x53),
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0xff),
            ] {
                let mut got = noise(len, 0xb21);
                let mut want = src.clone();
                entry(token, &mut got, Prepared::new(coeff), &src);
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(&mut want, coeff);
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 into prefetch: len {len}, coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_add_scatter_avx512, |token, entry| {
        check_scatter(
            "gf8 rs avx512 scatter",
            gf8_coeff_at::<RS>,
            gf8_reference::<RS>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs)[..], src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_avx512, |token, entry| {
        check_gather(
            "gf8 rs avx512 gather",
            gf8_coeff_at::<RS>,
            gf8_reference::<RS>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs)[..], srcs);
            },
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 rs avx512 matrix",
        gf8_coeff_at2::<RS>,
        gf8_reference::<RS>,
        X64V4xToken,
        x86::gf8::mul_add_matrix_avx512_with
    );
    check_prepared_matrix!(
        check_matrix_overwrite,
        "gf8 rs avx512 matrix overwrite",
        gf8_coeff_at2::<RS>,
        gf8_reference::<RS>,
        X64V4xToken,
        x86::gf8::mul_into_matrix_avx512_with
    );
    // The AES field through the generic 512-bit multi-row bodies.
    tier_gate::run_entry(x86::gf8::mul_add_scatter_avx512, |token, entry| {
        check_scatter(
            "gf8 aes avx512 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs)[..], src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_avx512, |token, entry| {
        check_gather(
            "gf8 aes avx512 gather",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs)[..], srcs);
            },
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 aes avx512 matrix",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V4xToken,
        x86::gf8::mul_add_matrix_avx512_with
    );
    check_prepared_matrix!(
        check_matrix_overwrite,
        "gf8 aes avx512 matrix overwrite",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V4xToken,
        x86::gf8::mul_into_matrix_avx512_with
    );
    // 512-bit tile transitions: sub-tile, exact tiles, tile-plus-lanes,
    // and lane-plus-AVX2-ladder remainders, with few and many terms.
    for &row_len in &[64usize, 192, 255, 256, 257, 320, 448, 511, 512, 513, 640] {
        for &nrows in &[1usize, 3, 4, 5, 8, 9] {
            for nterms in [1usize, 3, 9] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0xb00 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..nterms)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                    .collect();
                let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let mut got = noise(row_len * nrows, 0xb10);
                let mut want = got.clone();
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, &mut got, row_len, nrows, &prepared_matrix(&terms));
                    for &(coeffs, src) in &terms {
                        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                            gf8_reference::<RS>(row, coeff, src);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "gf8 rs avx512 tiles: {row_len}B x {nrows} x {nterms}t"
                    );
                });
            }
        }
    }
    if let Some(token) = tier_gate::summon::<X64V4xToken>() {
        let entry = x86::gf8::mul_add_matrix_avx512_with;
        // Empty terms: accumulate leaves rows unchanged.
        for &row_len in &[64usize, 192, 255, 256, 257, 320, 448, 511, 512, 513, 640] {
            for &nrows in &[1usize, 3, 4, 5, 8, 9] {
                let empty: &[(&[Prepared], &[u8])] = &[];
                let mut got = noise(row_len * nrows, 0xb11);
                let want = got.clone();
                entry(token, &mut got, row_len, nrows, empty);
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 empty accumulate: {row_len}B x {nrows}"
                );
            }
        }
    }
    if let Some(token) = tier_gate::summon::<X64V4xToken>() {
        let entry = x86::gf8::mul_into_matrix_avx512_with;
        for &row_len in &[64usize, 192, 255, 256, 257, 320, 448, 511, 512, 513, 640] {
            for &nrows in &[1usize, 3, 4, 5, 8, 9] {
                // Empty overwrite terms zero the rows.
                let mut got = noise(row_len * nrows, 0xb12);
                let empty: &[(&[Prepared], &[u8])] = &[];
                entry(token, &mut got, row_len, nrows, empty);
                assert_eq!(
                    got,
                    vec![0u8; row_len * nrows],
                    "gf8 rs avx512 empty overwrite: {row_len}B x {nrows}"
                );
            }
        }
    }
    // Alignment peel: rows long enough to peel, placed off a 64-byte
    // boundary, so the head runs through the tail helper with a source
    // window shorter than the row. One-, two-, and four-row groups each
    // peel; accumulate and overwrite share the tail. Lengths track the
    // floor so the peel stays exercised if the floor moves. The exact
    // floor length with a single row leaves a one-byte final remainder,
    // pinning the covered-prefix boundary a restructure must honor.
    let floor = x86::gf8::MATRIX_PEEL_MIN;
    for &offset in &[1usize, 16, 48] {
        for &row_len in &[floor - 64, floor, floor + 64] {
            for &nrows in &[1usize, 2, 4, 7] {
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0xb30 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                    .collect();
                let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let span = row_len * nrows;
                let mut backing = noise(span + 128, 0xb31);
                let start = backing.as_ptr().align_offset(64) + offset;
                let mut want = backing[start..start + span].to_vec();
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        gf8_reference::<RS>(row, coeff, src);
                    }
                }
                let rows = &mut backing[start..start + span];
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &prepared_matrix(&terms));
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled accumulate: +{offset} {row_len}B x {nrows}"
                    );
                });
            }
        }
    }
    for &offset in &[1usize, 16, 48] {
        for &row_len in &[floor - 64, floor, floor + 64] {
            for &nrows in &[1usize, 2, 4, 7] {
                let sources: Vec<Vec<u8>> = (0..3usize)
                    .map(|t| noise(row_len, 0xb30 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                    .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                    .collect();
                let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let span = row_len * nrows;
                let mut backing = noise(span + 128, 0xb31);
                let start = backing.as_ptr().align_offset(64) + offset;
                let mut want = vec![0u8; span];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        gf8_reference::<RS>(row, coeff, src);
                    }
                }
                let rows = &mut backing[start..start + span];
                tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &prepared_matrix(&terms));
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled overwrite: +{offset} {row_len}B x {nrows}"
                    );
                });
            }
        }
    }
    // Scatter rows straddle the floor and have both whole-line and
    // sub-line pitches; gather sources exercise the peeled head as well.
    tier_gate::run_entry(x86::gf8::mul_add_scatter_avx512, |token, entry| {
        for &offset in &[1usize, 16, 48] {
            for len in [
                x86::gf8::SCATTER_PEEL_MIN,
                x86::gf8::SCATTER_PEEL_MIN + 16,
                1152,
                1168,
                1184,
                1200,
            ] {
                for &nrows in &[1usize, 2, 3, 4, 6] {
                    let src_backing = noise(len + 128, 0xb40);
                    let s0 = src_backing.as_ptr().align_offset(64) + offset;
                    let src = &src_backing[s0..s0 + len];
                    let coeffs: Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>> =
                        (0..nrows).map(gf8_coeff_at::<RS>).collect();
                    let mut backing = noise(len * nrows + 128, 0xb41);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len * nrows].to_vec();
                    for (row, &coeff) in want.chunks_exact_mut(len).zip(&coeffs) {
                        gf8_reference::<RS>(row, coeff, src);
                    }
                    let rows = &mut backing[start..start + len * nrows];
                    entry(token, rows, len, &prepared_coeffs(&coeffs), src);
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs avx512 peeled scatter: +{offset} {len}B x {nrows}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_avx512, |token, entry| {
        for &offset in &[1usize, 16, 48] {
            for len in [
                x86::gf8::GATHER_PEEL_MIN - 1,
                x86::gf8::GATHER_PEEL_MIN,
                x86::gf8::GATHER_PEEL_MIN + 64,
            ] {
                for &nsrcs in &[1usize, 3, 16] {
                    let backings: Vec<Vec<u8>> = (0..nsrcs)
                        .map(|t| noise(len + 128, 0xb50 + t as u64))
                        .collect();
                    let srcs: Vec<&[u8]> = backings
                        .iter()
                        .map(|b| {
                            let s = b.as_ptr().align_offset(64) + offset;
                            &b[s..s + len]
                        })
                        .collect();
                    let coeffs: Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>> =
                        (0..nsrcs).map(gf8_coeff_at::<RS>).collect();
                    let mut backing = noise(len + 128, 0xb51);
                    let start = backing.as_ptr().align_offset(64) + offset;
                    let mut want = backing[start..start + len].to_vec();
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        gf8_reference::<RS>(&mut want, coeff, src);
                    }
                    let dst = &mut backing[start..start + len];
                    entry(token, dst, &prepared_coeffs(&coeffs), &srcs);
                    assert_eq!(
                        dst,
                        want.as_slice(),
                        "gf8 rs avx512 peeled gather: +{offset} {len}B x {nsrcs}"
                    );
                }
            }
        }
    });
    // Scattered rows: disjoint offsets, blocked affine against per-term AXPY.
    // Even iterations use ascending starts, odd iterations a reversed
    // (non-monotonic) order over the same windows.
    for &row_len in ROW_LENS {
        for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
            let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
            if ri % 2 == 1 {
                starts.reverse();
            }
            let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0x900 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let mut got = noise(span, 0xda);
            let mut want = got.clone();
            tier_gate::run_entry(x86::gf8::mul_add_matrix_at_avx512_with, |token, entry| {
                entry(token, &mut got, row_len, &starts, &prepared_matrix(&terms));
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        scalar::mul_add::<FiniteField<2, 8, Polynomial<RS>>>(
                            &mut want[starts[j]..starts[j] + row_len],
                            coeff,
                            src,
                        );
                    }
                }
                assert_eq!(got, want, "gf8 rs avx512 scattered: {row_len}B x {nrows}");
            });
        }
    }
    if let Some(token) = tier_gate::summon::<X64V4xToken>() {
        let entry = x86::gf8::mul_add_matrix_at_avx512_with;
        // Empty terms leave scattered rows unchanged; zero rows are a no-op.
        for &row_len in ROW_LENS {
            for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
                let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
                if ri % 2 == 1 {
                    starts.reverse();
                }
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
                let mut got = noise(span, 0xdb);
                let want = got.clone();
                let empty: &[(&[Prepared], &[u8])] = &[];
                entry(token, &mut got, row_len, &starts, empty);
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 scattered empty: {row_len}B x {nrows}"
                );
            }
        }
        {
            let mut rows = noise(64, 0xdc);
            let want = rows.clone();
            let empty: &[(&[Prepared], &[u8])] = &[];
            entry(token, &mut rows, 16, &[], empty);
            assert_eq!(rows, want, "gf8 rs avx512 scattered zero rows");
        }
    }
    for &row_len in ROW_LENS {
        for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
            let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
            if ri % 2 == 1 {
                starts.reverse();
            }
            let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0x900 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            // Overwrite from noise equals zeroed accumulation on the named
            // rows; the gaps keep their backing bytes.
            let mut overwrote = noise(span, 0xda1);
            let mut zeroed = overwrote.clone();
            for &start in &starts {
                zeroed[start..start + row_len].fill(0);
            }
            for &(coeffs, src) in &terms {
                for (j, &coeff) in coeffs.iter().enumerate() {
                    scalar::mul_add::<FiniteField<2, 8, Polynomial<RS>>>(
                        &mut zeroed[starts[j]..starts[j] + row_len],
                        coeff,
                        src,
                    );
                }
            }
            tier_gate::run_entry(x86::gf8::mul_into_matrix_at_avx512_with, |token, entry| {
                entry(
                    token,
                    &mut overwrote,
                    row_len,
                    &starts,
                    &prepared_matrix(&terms),
                );
                assert_eq!(
                    overwrote, zeroed,
                    "gf8 rs avx512 scattered overwrite: {row_len}B x {nrows}"
                );
            });
        }
    }
    if let Some(token) = tier_gate::summon::<X64V4xToken>() {
        let entry = x86::gf8::mul_into_matrix_at_avx512_with;
        // Empty overwrite terms zero the named rows.
        for &row_len in ROW_LENS {
            for (ri, &nrows) in ROW_COUNTS.iter().enumerate() {
                let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
                if ri % 2 == 1 {
                    starts.reverse();
                }
                let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
                let empty: &[(&[Prepared], &[u8])] = &[];
                let mut zeroed = noise(span, 0xdb1);
                entry(token, &mut zeroed, row_len, &starts, empty);
                for &start in &starts {
                    assert_eq!(
                        &zeroed[start..start + row_len],
                        &vec![0u8; row_len],
                        "gf8 rs avx512 scattered empty overwrite: {row_len}B x {nrows}"
                    );
                }
            }
        }
    }
    // The AES field through the generic scattered body: same windows.
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let starts: Vec<usize> = (0..nrows).rev().map(|j| j * (row_len + 7)).collect();
            let span = starts.iter().map(|&x| x + row_len).max().unwrap_or(0);
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0x940 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<AES>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let mut got = noise(span, 0xdd);
            let mut want = got.clone();
            tier_gate::run_entry(x86::gf8::mul_add_matrix_at_avx512_with, |token, entry| {
                entry(token, &mut got, row_len, &starts, &prepared_matrix(&terms));
                for &(coeffs, src) in &terms {
                    for (j, &coeff) in coeffs.iter().enumerate() {
                        gf8_reference::<AES>(&mut want[starts[j]..starts[j] + row_len], coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8 aes avx512 scattered: {row_len}B x {nrows}");
            });
        }
    }
    // 64-byte elementwise products for both byte fields: full lanes, the
    // 32/16-byte ladder, and the scalar tail.
    tier_gate::run_entry(x86::gf8::mul_elementwise_avx512::<true>, |token, entry| {
        check_gf8_elementwise::<AES>("gf8 aes avx512 elementwise", |dst, a, b| {
            entry(
                token,
                isomorphism_to_aes::<AES>(),
                Polynomial::<AES>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
    });
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_avx512::<true>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                "gf8 aes avx512 elementwise assign",
                |dst, src| {
                    entry(
                        token,
                        isomorphism_to_aes::<AES>(),
                        Polynomial::<AES>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        },
    );
    tier_gate::run_entry(x86::gf8::mul_elementwise_avx512::<false>, |token, entry| {
        check_gf8_elementwise::<RS>("gf8 rs avx512 elementwise", |dst, a, b| {
            entry(
                token,
                isomorphism_to_aes::<RS>(),
                Polynomial::<RS>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
    });
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_avx512::<false>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<RS>>>(
                "gf8 rs avx512 elementwise assign",
                |dst, src| {
                    entry(
                        token,
                        isomorphism_to_aes::<RS>(),
                        Polynomial::<RS>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        },
    );
    // The `_with` provider path over the flat test-slice terms.
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0xa00 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let flat: Vec<Prepared> = terms
                .iter()
                .flat_map(|&(coeffs, _)| prepared_coeffs(coeffs))
                .collect();
            let flat_srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
            let matrix = crate::kernel::FlatMatrix {
                coefficients: &flat,
                nrows,
                sources: &flat_srcs,
            };
            let mut got = noise(row_len * nrows, 0xdb);
            let mut want = got.clone();
            tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                entry(token, &mut got, row_len, nrows, &matrix);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        scalar::mul_add::<FiniteField<2, 8, Polynomial<RS>>>(row, coeff, src);
                    }
                }
                assert_eq!(got, want, "gf8 rs avx512 with: {row_len}B x {nrows}");
            });
        }
    }
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0xa00 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<RS>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let flat: Vec<Prepared> = terms
                .iter()
                .flat_map(|&(coeffs, _)| prepared_coeffs(coeffs))
                .collect();
            let flat_srcs: Vec<&[u8]> = terms.iter().map(|&(_, s)| s).collect();
            let matrix = crate::kernel::FlatMatrix {
                coefficients: &flat,
                nrows,
                sources: &flat_srcs,
            };
            let mut got = noise(row_len * nrows, 0xdc);
            tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                entry(token, &mut got, row_len, nrows, &matrix);
                let mut want = vec![0u8; row_len * nrows];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        scalar::mul_add::<FiniteField<2, 8, Polynomial<RS>>>(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "gf8 rs avx512 overwrite with: {row_len}B x {nrows}"
                );
            });
        }
    }
}

/// One AVX-512 matrix entry under test: label, whether it overwrites, and
/// the call over contiguous rows with the expected output.
#[cfg(feature = "simd512")]
type MatrixEntry<'a, E> = (
    &'a str,
    bool,
    &'a dyn Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
);

/// Flat row-major coefficients and per-term sources, the layout the
/// provider entries read.
#[cfg(feature = "simd512")]
fn flatten<'a, E: Copy>(terms: &[(&[E], &'a [u8])]) -> (Vec<E>, Vec<&'a [u8]>) {
    let coeffs = terms
        .iter()
        .flat_map(|&(coeffs, _)| coeffs.iter().copied())
        .collect();
    (coeffs, terms.iter().map(|&(_, src)| src).collect())
}

/// Compare AVX-512 matrix entries against a per-term reference sum at term
/// counts straddling the coefficient resolve chunk.
///
/// Row counts cover every mix of four-, two-, and one-row groups. Rows sit
/// 16 bytes past a 64-byte boundary, so the whole-lane floor length runs
/// the head peel; the long lengths keep to few row counts to bound
/// debug-build runtime. Each entry gate receives the expected output of
/// its disposition and contains the entry call with its comparison.
#[cfg(feature = "simd512")]
fn check_matrix_resolve_chunks<E: Copy>(
    field: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    entries: &[MatrixEntry<'_, E>],
) {
    let chunk = x86::gf8::RESOLVE_CHUNK;
    let floor = x86::gf8::MATRIX_PEEL_MIN;
    let every: Vec<usize> = (1..=9).collect();
    let few: &[usize] = &[2, 7];
    // Sub-lane, one lane, tile plus lanes plus sub-lane tail, and the
    // peel floor on both sides of a whole lane.
    let shapes: [(usize, &[usize]); 8] = [
        (40, &every),
        (64, &every),
        (256 + 2 * 64 + 17, &every),
        (512 - 64, &every),
        (512, &every),
        (512 + 64 + 17, &every),
        (floor, few),
        (floor + 1, few),
    ];
    for nterms in [chunk - 1, chunk, chunk + 1, 2 * chunk + 1] {
        for &(row_len, row_counts) in &shapes {
            for &nrows in row_counts {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0xc00 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<E>> = (0..nterms)
                    .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                    .collect();
                let terms: Vec<(&[E], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();
                let span = row_len * nrows;
                let mut sum = vec![0u8; span];
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in sum.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                let mut backing = noise(span + 128, 0xc01);
                let start = backing.as_ptr().align_offset(64) + 16;
                let initial = backing[start..start + span].to_vec();
                // Field addition is XOR, so accumulating the reference
                // sum into the initial rows gives the accumulate result.
                let accumulated: Vec<u8> = initial.iter().zip(&sum).map(|(a, b)| a ^ b).collect();
                for &(label, overwrites, kernel) in entries {
                    let rows = &mut backing[start..start + span];
                    rows.copy_from_slice(&initial);
                    let want = if overwrites { &sum } else { &accumulated };
                    kernel(rows, row_len, nrows, &terms, want);
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "{field} {label}: {row_len}B x {nrows} x {nterms}t"
                    );
                }
            }
        }
    }
}

/// Every AVX-512 matrix entry for both byte fields, at term counts that
/// split the row groups' coefficients across several resolve passes.
#[allow(clippy::too_many_lines)]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
#[cfg(feature = "simd512")]
#[test]
fn gf8_avx512_matrix_straddles_resolve_chunk() {
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] =
            [("mul_add", false, &|rows,
                                  row_len,
                                  nrows,
                                  terms: &[(
                &[Elem<FiniteField<2, 8, Polynomial<RS>>>],
                &[u8],
            )],
                                  want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 rs mul_add: {row_len}B x {nrows}");
                    });
                });
            })];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] =
            [("mul_into", true, &|rows,
                                  row_len,
                                  nrows,
                                  terms: &[(
                &[Elem<FiniteField<2, 8, Polynomial<RS>>>],
                &[u8],
            )],
                                  want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 rs mul_into: {row_len}B x {nrows}");
                    });
                });
            })];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] = [(
            "mul_add with",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 rs mul_add with: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] = [(
            "mul_into with",
            true,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 rs mul_into with: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] = [(
            "mul_add prepared",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 rs mul_add prepared: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] = [(
            "mul_into prepared",
            true,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 rs mul_into prepared: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<RS>>>>; 1] = [(
            "mul_add at",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])],
              want| {
                let starts: Vec<usize> = (0..nrows).map(|j| j * row_len).collect();
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(
                        x86::gf8::mul_add_matrix_at_avx512_with,
                        |token, entry| {
                            entry(token, rows, row_len, &starts, &pairs[..]);
                            assert_eq!(rows, want, "gf8 rs mul_add at: {row_len}B x {nrows}");
                        },
                    );
                });
            },
        )];
        check_matrix_resolve_chunks("gf8 rs", gf8_coeff_at2::<RS>, gf8_reference::<RS>, &entries);
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] =
            [("mul_add", false, &|rows,
                                  row_len,
                                  nrows,
                                  terms: &[(
                &[Elem<FiniteField<2, 8, Polynomial<AES>>>],
                &[u8],
            )],
                                  want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 aes mul_add: {row_len}B x {nrows}");
                    });
                });
            })];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] =
            [("mul_into", true, &|rows,
                                  row_len,
                                  nrows,
                                  terms: &[(
                &[Elem<FiniteField<2, 8, Polynomial<AES>>>],
                &[u8],
            )],
                                  want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 aes mul_into: {row_len}B x {nrows}");
                    });
                });
            })];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] = [(
            "mul_add with",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 aes mul_add with: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] = [(
            "mul_into with",
            true,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 aes mul_into with: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] = [(
            "mul_add prepared",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 aes mul_add prepared: {row_len}B x {nrows}");
                });
            },
        )];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] = [(
            "mul_into prepared",
            true,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])],
              want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(
                        rows, want,
                        "gf8 aes mul_into prepared: {row_len}B x {nrows}"
                    );
                });
            },
        )];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        let entries: [MatrixEntry<'_, Elem<FiniteField<2, 8, Polynomial<AES>>>>; 1] = [(
            "mul_add at",
            false,
            &|rows,
              row_len,
              nrows,
              terms: &[(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])],
              want| {
                let starts: Vec<usize> = (0..nrows).map(|j| j * row_len).collect();
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(
                        x86::gf8::mul_add_matrix_at_avx512_with,
                        |token, entry| {
                            entry(token, rows, row_len, &starts, &pairs[..]);
                            assert_eq!(rows, want, "gf8 aes mul_add at: {row_len}B x {nrows}");
                        },
                    );
                });
            },
        )];
        check_matrix_resolve_chunks(
            "gf8 aes",
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            &entries,
        );
    }
}

/// The AVX-512 matrix walks over rows long enough to split into column
/// blocks. A destination off a 64-byte boundary gives the walk a
/// non-empty leading block before the aligned ones; an aligned
/// destination gives it none.
#[cfg(feature = "simd512")]
#[test]
#[allow(clippy::too_many_lines)]
fn gf8_avx512_matrix_column_blocks_match_reference() {
    if tier_gate::summon::<X64V4xToken>().is_some() {
        for offset in [16usize, 0] {
            check_matrix_column_blocks(
                "gf8 rs avx512 matrix blocks",
                false,
                offset,
                gf8_coeff_at2::<RS>,
                gf8_reference::<RS>,
                |rows, row_len, nrows, terms, want| {
                    prepared_pairs!(terms, |pairs| {
                        tier_gate::run_entry(
                            x86::gf8::mul_add_matrix_avx512_with,
                            |token, entry| {
                                entry(token, rows, row_len, nrows, &pairs[..]);
                                assert_eq!(rows, want, "gf8 rs avx512 matrix blocks");
                            },
                        );
                    });
                },
            );
        }
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        check_matrix_column_blocks(
            "gf8 rs avx512 matrix overwrite blocks",
            true,
            16,
            gf8_coeff_at2::<RS>,
            gf8_reference::<RS>,
            |rows, row_len, nrows, terms, want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_into_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 rs avx512 matrix overwrite blocks");
                    });
                });
            },
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        check_matrix_column_blocks(
            "gf8 aes avx512 matrix blocks",
            false,
            16,
            gf8_coeff_at2::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, nrows, terms, want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "gf8 aes avx512 matrix blocks");
                    });
                });
            },
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        check_matrix_column_blocks(
            "gf8 rs avx512 matrix with blocks",
            false,
            16,
            gf8_coeff_at2::<RS>,
            gf8_reference::<RS>,
            |rows, row_len, nrows, terms, want| {
                let (coefficients, sources) = flatten(terms);
                let prepared: Vec<_> = coefficients.iter().map(|&c| Prepared::new(c)).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &prepared,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf8::mul_add_matrix_avx512_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf8 rs avx512 matrix with blocks");
                });
            },
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        check_matrix_at_column_blocks(
            "gf8 rs avx512 matrix_at blocks",
            false,
            16,
            gf8_coeff_at2::<RS>,
            gf8_reference::<RS>,
            |dst, row_len, starts, terms, want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(
                        x86::gf8::mul_add_matrix_at_avx512_with,
                        |token, entry| {
                            entry(token, dst, row_len, starts, &pairs[..]);
                            assert_eq!(dst, want, "gf8 rs avx512 matrix_at blocks");
                        },
                    );
                });
            },
        );
    }
    if tier_gate::summon::<X64V4xToken>().is_some() {
        check_matrix_at_column_blocks(
            "gf8 rs avx512 matrix_at overwrite blocks",
            true,
            16,
            gf8_coeff_at2::<RS>,
            gf8_reference::<RS>,
            |dst, row_len, starts, terms, want| {
                prepared_pairs!(terms, |pairs| {
                    tier_gate::run_entry(
                        x86::gf8::mul_into_matrix_at_avx512_with,
                        |token, entry| {
                            entry(token, dst, row_len, starts, &pairs[..]);
                            assert_eq!(dst, want, "gf8 rs avx512 matrix_at overwrite blocks");
                        },
                    );
                });
            },
        );
    }
}
