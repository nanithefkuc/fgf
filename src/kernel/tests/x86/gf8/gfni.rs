//! GFNI GF(2^8) differentials: the affine and native `GF2P8MULB` sweeps,
//! the column-block matrices for both named polynomials, the half-lane
//! peel floors, and the scattered-row blocked matrix.

use super::*;
use crate::kernel::tests::tier_gate;

/// Scattered rows at ascending and reversed offsets against scalar arithmetic.
#[allow(clippy::type_complexity)]
fn check_gf8_scattered_gfni_rows<const POLY: u128, const NATIVE: bool>(name: &str) {
    if tier_gate::summon::<X64V3GfniCryptoToken>().is_none() {
        return;
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
            let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<POLY>>>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf8_coeff_at2::<POLY>(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<POLY>>>], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let matrix = prepared_matrix(&terms);
            let mut got = noise(span, 0xda);
            let mut want = got.clone();
            for &(coeffs, src) in &terms {
                for (j, &coeff) in coeffs.iter().enumerate() {
                    scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(
                        &mut want[starts[j]..starts[j] + row_len],
                        coeff,
                        src,
                    );
                }
            }
            tier_gate::run_entry(
                x86::gf8::mul_add_matrix_at_gfni_with::<NATIVE, _>,
                |token, entry| {
                    entry(token, &mut got, row_len, &starts, &matrix);
                    assert_eq!(
                        got, want,
                        "{name} matrix_scattered: row_len {row_len}, nrows {nrows}"
                    );
                },
            );
            let mut overwrote = noise(span, 0xda3);
            let mut zeroed = overwrote.clone();
            for &start in &starts {
                zeroed[start..start + row_len].fill(0);
            }
            for &(coeffs, src) in &terms {
                for (j, &coeff) in coeffs.iter().enumerate() {
                    scalar::mul_add::<FiniteField<2, 8, Polynomial<POLY>>>(
                        &mut zeroed[starts[j]..starts[j] + row_len],
                        coeff,
                        src,
                    );
                }
            }
            tier_gate::run_entry(
                x86::gf8::mul_into_matrix_at_gfni_with::<NATIVE, _>,
                |token, entry| {
                    entry(token, &mut overwrote, row_len, &starts, &matrix);
                    assert_eq!(
                        overwrote, zeroed,
                        "{name} matrix_scattered overwrite: row_len {row_len}, nrows {nrows}"
                    );
                },
            );
        }
    }
}

#[allow(clippy::too_many_lines)]
fn check_gfni_columns<const POLY: u128, const NATIVE: bool>(
    offset: usize,
    flat_overwrite: bool,
    labels: [&str; 5],
) {
    if tier_gate::summon::<X64V3GfniCryptoToken>().is_none() {
        return;
    }
    check_matrix_column_blocks(
        labels[0],
        false,
        offset,
        gf8_coeff_at2::<POLY>,
        gf8_reference::<POLY>,
        |rows, row_len, nrows, terms, want| {
            prepared_pairs!(terms, |pairs| {
                tier_gate::run_entry(
                    x86::gf8::mul_add_matrix_gfni_with::<NATIVE, _>,
                    |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "{}", labels[0]);
                    },
                );
            });
        },
    );
    check_matrix_column_blocks(
        labels[1],
        true,
        offset,
        gf8_coeff_at2::<POLY>,
        gf8_reference::<POLY>,
        |rows, row_len, nrows, terms, want| {
            prepared_pairs!(terms, |pairs| {
                tier_gate::run_entry(
                    x86::gf8::mul_into_matrix_gfni_with::<NATIVE, _>,
                    |token, entry| {
                        entry(token, rows, row_len, nrows, &pairs[..]);
                        assert_eq!(rows, want, "{}", labels[1]);
                    },
                );
            });
        },
    );
    check_matrix_column_blocks(
        labels[2],
        flat_overwrite,
        0,
        gf8_coeff_at2::<POLY>,
        gf8_reference::<POLY>,
        |rows, row_len, nrows, terms, want| {
            with_flat::<POLY>(terms, nrows, |matrix| {
                if flat_overwrite {
                    tier_gate::run_entry(
                        x86::gf8::mul_into_matrix_gfni_with::<NATIVE, _>,
                        |token, entry| {
                            entry(token, rows, row_len, nrows, matrix);
                            assert_eq!(rows, want, "{}", labels[2]);
                        },
                    );
                } else {
                    tier_gate::run_entry(
                        x86::gf8::mul_add_matrix_gfni_with::<NATIVE, _>,
                        |token, entry| {
                            entry(token, rows, row_len, nrows, matrix);
                            assert_eq!(rows, want, "{}", labels[2]);
                        },
                    );
                }
            });
        },
    );
    check_matrix_at_column_blocks(
        labels[3],
        false,
        offset,
        gf8_coeff_at2::<POLY>,
        gf8_reference::<POLY>,
        |dst, row_len, starts, terms, want| {
            prepared_pairs!(terms, |pairs| {
                tier_gate::run_entry(
                    x86::gf8::mul_add_matrix_at_gfni_with::<NATIVE, _>,
                    |token, entry| {
                        entry(token, dst, row_len, starts, &pairs[..]);
                        assert_eq!(dst, want, "{}", labels[3]);
                    },
                );
            });
        },
    );
    check_matrix_at_column_blocks(
        labels[4],
        true,
        offset,
        gf8_coeff_at2::<POLY>,
        gf8_reference::<POLY>,
        |dst, row_len, starts, terms, want| {
            prepared_pairs!(terms, |pairs| {
                tier_gate::run_entry(
                    x86::gf8::mul_into_matrix_at_gfni_with::<NATIVE, _>,
                    |token, entry| {
                        entry(token, dst, row_len, starts, &pairs[..]);
                        assert_eq!(dst, want, "{}", labels[4]);
                    },
                );
            });
        },
    );
}

/// Column-block matrix forms over the AES presentation.
#[test]
fn gfni_matrix_column_blocks_match_reference() {
    check_gfni_columns::<AES, true>(
        0,
        false,
        [
            "gf8 gfni matrix blocks",
            "gf8 gfni matrix overwrite blocks",
            "gf8 gfni matrix with blocks",
            "gf8 gfni matrix_at blocks",
            "gf8 gfni matrix_at overwrite blocks",
        ],
    );
}

/// Column-block matrix forms over the Reed–Solomon presentation.
#[test]
fn gfni_rs_matrix_column_blocks_match_reference() {
    check_gfni_columns::<RS, false>(
        16,
        true,
        [
            "gf8 rs gfni matrix blocks",
            "gf8 rs gfni matrix overwrite blocks",
            "gf8 rs gfni matrix overwrite with blocks",
            "gf8 rs gfni matrix_at blocks",
            "gf8 rs gfni matrix_at overwrite blocks",
        ],
    );
}

/// The Reed–Solomon field's GFNI path: `VGF2P8AFFINEQB` with the
/// const-derived affine bank. The sweep is exhaustive over coefficients,
/// and the all-byte-values source makes it exhaustive over products —
/// this is the silicon-level proof of the affine map's bit-order
/// convention, and the loud failure if `GF2P8MULB` (the AES field) were
/// ever substituted.
#[allow(clippy::too_many_lines)]
#[test]
fn gf8_rs_gfni_kernels_match_reference() {
    // The non-temporal `mul_into` body sits far above the shared lengths;
    // destination offsets 0 and 1 cover both sides of the alignment peel.
    const NT_LEN: usize = (2 << 20) + 130;

    tier_gate::run_entry(x86::gf8::mul_add_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_add(
            "gf8 rs gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_assign(
            "gf8 rs gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c| {
                entry(token, dst, c);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_into(
            "gf8 rs gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<RS>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<false>, |token, entry| {
        let source = noise(NT_LEN + 2, 0x2b8);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            // Noise, not zeros: a streaming body that accumulated into the
            // destination instead of overwriting it must fail this check.
            let mut buffer = noise(NT_LEN + 1, 0x2b9);
            for &coeff in &[
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::ZERO,
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::ONE,
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0x53),
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(0xff),
            ] {
                let mut want = src.to_vec();
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(&mut want, coeff);
                let got = &mut buffer[offset..offset + NT_LEN];
                entry(token, got, Prepared::new(coeff), src);
                assert_eq!(&*got, want, "gf8 rs gfni NT mul_into: offset {offset}");
            }
        }
    });

    // The register-blocked affine multi-row shapes against per-row AXPY,
    // over the same row-group and tile boundaries as the AES kernels.
    tier_gate::run_entry(x86::gf8::mul_add_scatter_gfni::<false>, |token, entry| {
        check_scatter(
            "gf8 rs gfni scatter",
            gf8_coeff_at::<RS>,
            gf8_reference::<RS>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs), src);
            },
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 rs gfni matrix",
        gf8_coeff_at2::<RS>,
        gf8_reference::<RS>,
        X64V3GfniCryptoToken,
        x86::gf8::mul_add_matrix_gfni_with::<false, _>
    );
    check_prepared_matrix!(
        check_matrix_overwrite,
        "gf8 rs gfni matrix overwrite",
        gf8_coeff_at2::<RS>,
        gf8_reference::<RS>,
        X64V3GfniCryptoToken,
        x86::gf8::mul_into_matrix_gfni_with::<false, _>
    );
    check_prepared_prefetch!(
        "gf8 rs gfni matrix prefetch",
        gf8_coeff_at2::<RS>,
        gf8_reference::<RS>,
        x86::gf8::mul_add_matrix_gfni_with::<false, _>,
        x86::gf8::mul_into_matrix_gfni_with::<false, _>
    );
    tier_gate::run_entry(x86::gf8::mul_add_gather_gfni::<false>, |token, entry| {
        check_gather(
            "gf8 rs gfni gather",
            gf8_coeff_at::<RS>,
            gf8_reference::<RS>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
    });
    check_gf8_scattered_gfni_rows::<RS, false>("gf8 rs gfni");
}

/// The `GF2P8MULB` byte kernels across the half-lane peel floor, against
/// the scalar reference for all three single-row forms.
#[test]
fn gf8_gfni_peels_half_lane() {
    let floor = crate::kernel::routing::HALF_LANE_PEEL_MIN;
    tier_gate::run_entry(x86::gf8::mul_add_gfni::<true>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c3);
                let mut dst_storage = noise(len + 128, 0x2c4);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw)
                {
                    let mut want = dst.to_vec();
                    scalar::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff, src);
                    entry(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<true>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c3);
                let mut dst_storage = noise(len + 128, 0x2c4);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw)
                {
                    let mut want = dst.to_vec();
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                    entry(token, dst, Prepared::new(coeff));
                    assert_eq!(&*dst, want, "mul_assign_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<true>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c3);
                let mut dst_storage = noise(len + 128, 0x2c4);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw)
                {
                    let mut want = src.to_vec();
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                    entry(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "mul_into_gfni: len {len}, offset {offset}");
                }
            }
        }
    });
}

/// The `VGF2P8AFFINEQB` byte kernels across the half-lane peel floor,
/// against the scalar reference for all three single-row forms.
#[test]
fn gf8_rs_gfni_peels_half_lane() {
    let floor = crate::kernel::routing::HALF_LANE_PEEL_MIN;
    tier_gate::run_entry(x86::gf8::mul_add_gfni::<false>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c5);
                let mut dst_storage = noise(len + 128, 0x2c6);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw)
                {
                    let mut want = dst.to_vec();
                    scalar::mul_add::<FiniteField<2, 8, Polynomial<RS>>>(&mut want, coeff, src);
                    entry(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "rs mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<false>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c5);
                let mut dst_storage = noise(len + 128, 0x2c6);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw)
                {
                    let mut want = dst.to_vec();
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(&mut want, coeff);
                    entry(token, dst, Prepared::new(coeff));
                    assert_eq!(
                        &*dst, want,
                        "rs mul_assign_gfni: len {len}, offset {offset}"
                    );
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<false>, |token, entry| {
        for len in [floor - 2, floor - 1, floor, floor + 1, floor + 130, 1152] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c5);
                let mut dst_storage = noise(len + 128, 0x2c6);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in
                    [0u8, 1, 0x53, 0xff].map(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw)
                {
                    let mut want = src.to_vec();
                    scalar::mul_assign::<FiniteField<2, 8, Polynomial<RS>>>(&mut want, coeff);
                    entry(token, dst, Prepared::new(coeff), src);
                    assert_eq!(&*dst, want, "rs mul_into_gfni: len {len}, offset {offset}");
                }
            }
        }
    });
}

#[test]
fn scatter_gfni_rejects_short_rows_buffer() {
    // `#[should_panic]` cannot express "skip on hosts that cannot select the
    // tier", so the panic is caught explicitly after the capability guard.
    let Some(token) = tier_gate::summon::<X64V3GfniCryptoToken>() else {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    };
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::mul_add_scatter_gfni::<true>(
            token,
            &mut [0u8; 4],
            4,
            &[
                Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(1)),
                Prepared::new(Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(2)),
            ],
            &[0u8; 4],
        );
    })
    .is_err();
    assert!(
        panicked,
        "mul_add_scatter_gfni must reject a short rows buffer"
    );
}

#[test]
fn scatter_gfni_rs_rejects_short_rows_buffer() {
    let Some(token) = tier_gate::summon::<X64V3GfniCryptoToken>() else {
        eprintln!("skipping: no AVX2+GFNI+VAES on this host");
        return;
    };
    let panicked = std::panic::catch_unwind(|| {
        x86::gf8::mul_add_scatter_gfni::<false>(
            token,
            &mut [0u8; 4],
            4,
            &[
                Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(1)),
                Prepared::new(Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(2)),
            ],
            &[0u8; 4],
        );
    })
    .is_err();
    assert!(
        panicked,
        "mul_add_scatter_gfni must reject a short rows buffer"
    );
}
