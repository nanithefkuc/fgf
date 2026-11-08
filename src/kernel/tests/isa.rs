//! Cross-family ISA differential suites.
//!
//! One differential per instruction-set suite that spans several field
//! families: SSSE3, AVX2, GFNI, and the non-temporal and multi-row payloads
//! several families share. Family-pure and ISA-pure differentials live in
//! the family modules beside this one; every arithmetic kernel invocation
//! here earns its entry receipt through [`tier_gate`].

use super::*;
use crate::kernel::tables::isomorphism_to_aes;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

#[test]
#[allow(clippy::too_many_lines)]
fn gfni_kernels_match_reference() {
    // The single-buffer entries take the prepared coefficient: the AES
    // field runs the native multiply, the affine form follows.
    tier_gate::run_entry(x86::gf8::mul_add_gfni::<true>, |token, entry| {
        check_gf8_exhaustive_mul_add(
            "gf8 aes gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<true>, |token, entry| {
        check_gf8_exhaustive_mul_assign(
            "gf8 aes gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c| {
                entry(token, dst, c);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<true>, |token, entry| {
        check_gf8_exhaustive_mul_into(
            "gf8 aes gfni",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    // The affine multiply on the same field: the dispatch shape every
    // non-AES representation takes, exercised on AES data.
    tier_gate::run_entry(x86::gf8::mul_add_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_add(
            "gf8 aes gfni affine",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_assign(
            "gf8 aes gfni affine",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c| {
                entry(token, dst, c);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<false>, |token, entry| {
        check_gf8_exhaustive_mul_into(
            "gf8 aes gfni affine",
            |c: Elem<FiniteField<2, 8, Polynomial<AES>>>| Prepared::new(c),
            |dst, c, src| {
                entry(token, dst, c, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_gfni, |token, entry| {
        check_gf16_mul_add_tables("gf16 gfni", |dst, tables, src| {
            entry(token, dst, tables.compact(), src);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_assign_gfni, |token, entry| {
        check_gf16_mul_assign_tables("gf16 gfni", |dst, tables| {
            entry(token, dst, tables.compact());
        });
    });

    tier_gate::run_entry(x86::gf8::mul_add_scatter_gfni::<true>, |token, entry| {
        check_scatter(
            "gf8 gfni scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs), src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_gfni, |token, entry| {
        check_scatter(
            "gf16 gfni scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, coeffs, src);
            },
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 gfni matrix",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V3GfniCryptoToken,
        x86::gf8::mul_add_matrix_gfni_with::<true, _>
    );
    check_prepared_matrix!(
        check_matrix_overwrite,
        "gf8 gfni matrix overwrite",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V3GfniCryptoToken,
        x86::gf8::mul_into_matrix_gfni_with::<true, _>
    );
    check_prepared_prefetch!(
        "gf8 gfni matrix prefetch",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        x86::gf8::mul_add_matrix_gfni_with::<true, _>,
        x86::gf8::mul_into_matrix_gfni_with::<true, _>
    );
    tier_gate::run_entry(x86::gf16::mul_add_matrix_gfni, |token, entry| {
        check_matrix(
            "gf16 gfni matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms, _want| {
                entry(token, rows, row_len, nrows, terms);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_gfni::<true>, |token, entry| {
        check_gather(
            "gf8 gfni gather",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_gather_gfni, |token, entry| {
        check_gather_aligned(
            "gf16 gfni gather",
            2,
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| entry(token, dst, coeffs, srcs),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_gfni::<true>, |token, entry| {
        check_gf8_elementwise::<AES>("gf8 gfni elementwise", |dst, a, b| {
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
    tier_gate::run_entry(x86::gf8::mul_elementwise_gfni::<false>, |token, entry| {
        check_gf8_elementwise::<RS>("gf8 rs gfni elementwise", |dst, a, b| {
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
    tier_gate::run_entry(x86::gf16::mul_elementwise_gfni, |token, entry| {
        check_gf16_elementwise("gf16 gfni elementwise", |dst, a, b| {
            entry(token, dst, a, b, DELTA_BYTE, AES_REDUCTION);
        });
    });
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_gfni::<true>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                "gf8 gfni elementwise assign",
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
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_gfni::<false>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<RS>>>(
                "gf8 rs gfni elementwise assign",
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
    tier_gate::run_entry(x86::gf16::mul_elementwise_assign_gfni, |token, entry| {
        check_elementwise_assign::<Rijndael16Field>("gf16 gfni elementwise assign", |dst, src| {
            entry(token, dst, src, DELTA_BYTE, AES_REDUCTION);
        });
    });
    // Level-2/3 tower kernels: the period-2 lane multiply one and two
    // levels up from the GF(2^16) kernel.
    check_tower_gfni_kernels();
}

/// Differential-check the GFNI GF(2^32) and GF(2^64) kernels — the level-2
/// and level-3 tower multiplies — against the portable scalar oracle.
fn check_tower_gfni_kernels() {
    tier_gate::run_entry(x86::gf32::mul_add_gfni, |token, entry| {
        check_tower_mul_add(
            "gf32 gfni mul_add",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_reference,
            |dst, coeff, src| entry(token, dst, coeff, x86::gf32::gf32_tiles(coeff), src),
        );
    });
    tier_gate::run_entry(x86::gf32::mul_assign_gfni, |token, entry| {
        check_tower_mul_assign(
            "gf32 gfni mul_assign",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_assign_reference,
            |dst, coeff| entry(token, dst, coeff, x86::gf32::gf32_tiles(coeff)),
        );
    });
    tier_gate::run_entry(x86::gf32::mul_into_gfni, |token, entry| {
        check_tower_mul_into(
            "gf32 gfni mul_into",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_into_reference,
            |dst, coeff, src| entry(token, dst, coeff, x86::gf32::gf32_tiles(coeff), src),
        );
    });
    // Level-3: the same identity over GF(2^32) lanes.
    tier_gate::run_entry(x86::gf64::mul_add_gfni, |token, entry| {
        check_tower_mul_add(
            "gf64 gfni mul_add",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_reference,
            |dst, coeff, src| entry(token, dst, coeff, x86::gf64::gf64_tiles(coeff), src),
        );
    });
    tier_gate::run_entry(x86::gf64::mul_assign_gfni, |token, entry| {
        check_tower_mul_assign(
            "gf64 gfni mul_assign",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_assign_reference,
            |dst, coeff| entry(token, dst, coeff, x86::gf64::gf64_tiles(coeff)),
        );
    });
    tier_gate::run_entry(x86::gf64::mul_into_gfni, |token, entry| {
        check_tower_mul_into(
            "gf64 gfni mul_into",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_into_reference,
            |dst, coeff, src| entry(token, dst, coeff, x86::gf64::gf64_tiles(coeff), src),
        );
    });
}

#[test]
#[allow(clippy::too_many_lines)]
fn avx2_kernels_match_reference() {
    tier_gate::run_entry(x86::gf8::mul_add_avx2, |token, entry| {
        check_gf8_mul_add("gf8 aes avx2", &gf8_aes_coeffs(), |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx2, |token, entry| {
        check_gf8_mul_assign("gf8 aes avx2", &gf8_aes_coeffs(), |dst, table| {
            entry(token, dst, table);
        });
    });
    // The shuffle kernels are polynomial-agnostic: the coefficient's own
    // bank serves any polynomial, RS included.
    tier_gate::run_entry(x86::gf8::mul_add_avx2, |token, entry| {
        check_gf8_mul_add::<RS>("gf8 rs avx2", &gf8_rs_coeffs(), |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx2, |token, entry| {
        check_gf8_mul_assign::<RS>("gf8 rs avx2", &gf8_rs_coeffs(), |dst, table| {
            entry(token, dst, table);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_add_avx2, |token, entry| {
        check_gf16_mul_add_tables("gf16 avx2", |dst, tables, src| {
            entry(token, dst, tables, src);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_assign_avx2, |token, entry| {
        check_gf16_mul_assign_tables("gf16 avx2", |dst, tables| {
            entry(token, dst, tables);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx2, |token, entry| {
        check_gf8_mul_into("gf8 avx2 mul_into", &gf8_aes_coeffs(), |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx2, |token, entry| {
        check_gf8_mul_into::<RS>(
            "gf8 rs avx2 mul_into",
            &gf8_rs_coeffs(),
            |dst, table, src| {
                entry(token, dst, table, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_into_avx2, |token, entry| {
        check_gf16_mul_into_tables("gf16 avx2 mul_into", |dst, tables, src| {
            entry(token, dst, tables, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_add_scatter_avx2, |token, entry| {
        check_scatter(
            "gf8 avx2 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs), src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_avx2, |token, entry| {
        check_scatter(
            "gf16 avx2 scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, coeffs, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_avx2, |token, entry| {
        check_gather(
            "gf8 avx2 gather",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 avx2 matrix",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V3Token,
        x86::gf8::mul_add_matrix_avx2_with
    );
    tier_gate::run_entry(x86::gf8::mul_elementwise_avx2, |token, entry| {
        check_gf8_elementwise::<AES>("gf8 avx2 elementwise", |dst, a, b| {
            entry(token, Polynomial::<AES>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_avx2, |token, entry| {
        check_gf8_elementwise::<RS>("gf8 rs avx2 elementwise", |dst, a, b| {
            entry(token, Polynomial::<RS>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_gfni::<false>, |token, entry| {
        check_gf8_elementwise::<RS>("gf8 rs gfni elementwise", |dst, a, b| {
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
    tier_gate::run_entry(x86::gf16::mul_elementwise_avx2, |token, entry| {
        check_gf16_elementwise("gf16 avx2 elementwise", |dst, a, b| {
            entry(token, dst, a, b, delta_table(), AES_REDUCTION);
        });
    });
    // Fan–Paar tower (GF(2^16)/32/64): the fp8 nibble tower and its
    // period-2 lane-mul extensions.
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_avx2, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<AES>>>(
            "gf8 avx2 elementwise assign",
            |dst, src| {
                entry(token, Polynomial::<AES>::REDUCTION_LOW, dst, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_avx2, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<RS>>>(
            "gf8 rs avx2 elementwise assign",
            |dst, src| {
                entry(token, Polynomial::<RS>::REDUCTION_LOW, dst, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_elementwise_assign_avx2, |token, entry| {
        check_elementwise_assign::<Rijndael16Field>("gf16 avx2 elementwise assign", |dst, src| {
            entry(token, dst, src, delta_table(), AES_REDUCTION);
        });
    });
    check_fan_paar_avx2_kernels();
}

/// Differential-check the Fan–Paar GF(2^16)/32/64 AVX2 kernels — the fp8
/// nibble tower and its period-2 lane-mul extensions — against the
/// Wiedemann scalar oracle.
fn check_fan_paar_avx2_kernels() {
    tier_gate::run_entry(x86::fan_paar::mul_add_avx2_fp16, |token, entry| {
        check_tower_mul_add(
            "fp16 avx2 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| entry(token, dst, &FpTowerTables::new(coeff), src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_assign_avx2_fp16, |token, entry| {
        check_tower_mul_assign(
            "fp16 avx2 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| entry(token, dst, &FpTowerTables::new(coeff)),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_into_avx2_fp16, |token, entry| {
        check_tower_mul_into(
            "fp16 avx2 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| entry(token, dst, &FpTowerTables::new(coeff), src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_add_avx2_fp32, |token, entry| {
        check_tower_mul_add(
            "fp32 avx2 mul_add",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_reference,
            |dst, coeff, src| entry(token, dst, coeff, src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_assign_avx2_fp32, |token, entry| {
        check_tower_mul_assign(
            "fp32 avx2 mul_assign",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_assign_reference,
            |dst, coeff| entry(token, dst, coeff),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_into_avx2_fp32, |token, entry| {
        check_tower_mul_into(
            "fp32 avx2 mul_into",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_into_reference,
            |dst, coeff, src| entry(token, dst, coeff, src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_add_avx2_fp64, |token, entry| {
        check_tower_mul_add(
            "fp64 avx2 mul_add",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_reference,
            |dst, coeff, src| entry(token, dst, coeff, src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_assign_avx2_fp64, |token, entry| {
        check_tower_mul_assign(
            "fp64 avx2 mul_assign",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_assign_reference,
            |dst, coeff| entry(token, dst, coeff),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_into_avx2_fp64, |token, entry| {
        check_tower_mul_into(
            "fp64 avx2 mul_into",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_into_reference,
            |dst, coeff, src| entry(token, dst, coeff, src),
        );
    });
}

#[test]
#[allow(clippy::too_many_lines)]
fn ssse3_kernels_match_reference() {
    tier_gate::run_entry(x86::gf8::mul_add_ssse3, |token, entry| {
        check_gf8_mul_add("gf8 aes ssse3", &gf8_aes_coeffs(), |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_ssse3, |token, entry| {
        check_gf8_mul_assign("gf8 aes ssse3", &gf8_aes_coeffs(), |dst, table| {
            entry(token, dst, table);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_add_ssse3, |token, entry| {
        check_gf8_mul_add::<RS>("gf8 rs ssse3", &gf8_rs_coeffs(), |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_ssse3, |token, entry| {
        check_gf8_mul_assign::<RS>("gf8 rs ssse3", &gf8_rs_coeffs(), |dst, table| {
            entry(token, dst, table);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_add_ssse3, |token, entry| {
        check_gf16_mul_add_tables("gf16 ssse3", |dst, tables, src| {
            entry(token, dst, tables, src);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_assign_ssse3, |token, entry| {
        check_gf16_mul_assign_tables("gf16 ssse3", |dst, tables| {
            entry(token, dst, tables);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_into_ssse3, |token, entry| {
        check_gf8_mul_into(
            "gf8 ssse3 mul_into",
            &gf8_aes_coeffs(),
            |dst, table, src| {
                entry(token, dst, table, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_ssse3, |token, entry| {
        check_gf8_mul_into::<RS>(
            "gf8 rs ssse3 mul_into",
            &gf8_rs_coeffs(),
            |dst, table, src| {
                entry(token, dst, table, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_into_ssse3, |token, entry| {
        check_gf16_mul_into_tables("gf16 ssse3 mul_into", |dst, tables, src| {
            entry(token, dst, tables, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_add_scatter_ssse3, |token, entry| {
        check_scatter(
            "gf8 ssse3 scatter",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, &prepared_coeffs(coeffs), src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_ssse3, |token, entry| {
        check_scatter(
            "gf16 ssse3 scatter",
            gf16_coeff_at,
            gf16_reference,
            |rows, row_len, coeffs, src| {
                entry(token, rows, row_len, coeffs, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_gather_ssse3, |token, entry| {
        check_gather(
            "gf8 ssse3 gather",
            gf8_coeff_at::<AES>,
            gf8_reference::<AES>,
            |dst, coeffs, srcs| {
                entry(token, dst, &prepared_coeffs(coeffs), srcs);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_add_gather_ssse3, |token, entry| {
        check_gather_aligned(
            "gf16 ssse3 gather",
            2,
            gf16_coeff_at,
            gf16_reference,
            |dst, coeffs, srcs| entry(token, dst, coeffs, srcs),
        );
    });
    check_prepared_matrix!(
        check_matrix,
        "gf8 ssse3 matrix",
        gf8_coeff_at2::<AES>,
        gf8_reference::<AES>,
        X64V2Token,
        x86::gf8::mul_add_matrix_ssse3_with
    );
    tier_gate::run_entry(x86::gf16::mul_add_matrix_ssse3, |token, entry| {
        check_matrix(
            "gf16 ssse3 matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms, _want| {
                entry(token, rows, row_len, nrows, terms);
            },
        );
    });
    // The provider form over flat nibble tables; each case builds its
    // provider before the gate so the entry type borrows live fixtures.
    if tier_gate::summon::<X64V2Token>().is_some() {
        check_matrix(
            "gf16 ssse3 matrix provider",
            gf16_coeff_at2,
            gf16_reference,
            |rows, row_len, nrows, terms, want| {
                let flat: Vec<TowerTables> = terms
                    .iter()
                    .flat_map(|(coeffs, _)| coeffs.iter().copied().map(TowerTables::new))
                    .collect();
                let sources: Vec<&[u8]> = terms.iter().map(|(_, src)| *src).collect();
                let matrix = crate::kernel::FlatMatrix {
                    coefficients: &flat,
                    nrows,
                    sources: &sources,
                };
                tier_gate::run_entry(x86::gf16::mul_add_matrix_ssse3_with, |token, entry| {
                    entry(token, rows, row_len, nrows, &matrix);
                    assert_eq!(rows, want, "gf16 ssse3 matrix provider");
                });
            },
        );
    }
    tier_gate::run_entry(x86::gf8::mul_elementwise_ssse3, |token, entry| {
        check_gf8_elementwise::<AES>("gf8 ssse3 elementwise", |dst, a, b| {
            entry(token, Polynomial::<AES>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_ssse3, |token, entry| {
        check_gf8_elementwise::<RS>("gf8 rs ssse3 elementwise", |dst, a, b| {
            entry(token, Polynomial::<RS>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_elementwise_ssse3, |token, entry| {
        check_gf16_elementwise("gf16 ssse3 elementwise", |dst, a, b| {
            entry(token, dst, a, b, delta_table(), AES_REDUCTION);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_ssse3, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<AES>>>(
            "gf8 ssse3 elementwise assign",
            |dst, src| {
                entry(token, Polynomial::<AES>::REDUCTION_LOW, dst, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_ssse3, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<RS>>>(
            "gf8 rs ssse3 elementwise assign",
            |dst, src| {
                entry(token, Polynomial::<RS>::REDUCTION_LOW, dst, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf16::mul_elementwise_assign_ssse3, |token, entry| {
        check_elementwise_assign::<Rijndael16Field>("gf16 ssse3 elementwise assign", |dst, src| {
            entry(token, dst, src, delta_table(), AES_REDUCTION);
        });
    });
    tier_gate::run_entry(x86::fan_paar::mul_add_ssse3_fp16, |token, entry| {
        check_tower_mul_add(
            "fp16 ssse3 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| entry(token, dst, &FpTowerTables::new(coeff), src),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_assign_ssse3_fp16, |token, entry| {
        check_tower_mul_assign(
            "fp16 ssse3 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| entry(token, dst, &FpTowerTables::new(coeff)),
        );
    });
    tier_gate::run_entry(x86::fan_paar::mul_into_ssse3_fp16, |token, entry| {
        check_tower_mul_into(
            "fp16 ssse3 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| entry(token, dst, &FpTowerTables::new(coeff), src),
        );
    });
}

/// The selected backend proof narrows to every weaker tier's token, and
/// each narrowed token drives that tier's kernels correctly. Dispatch
/// narrows only on its own tier; this proves the narrowing from every
/// stronger selection a forced tier run reaches.
#[test]
fn selected_proof_narrows_to_working_weaker_tiers() {
    use crate::kernel::X86Proof;

    let proof = crate::kernel::x86_proof();
    if matches!(proof, X86Proof::Scalar) {
        eprintln!("skipping: the scalar selection carries no vector proof");
        return;
    }
    let v2 = crate::kernel::x86_v2_token();
    tier_gate::run_entry::<X64V2Token, _>(x86::gf8::mul_add_ssse3, |_, entry| {
        check_gf8_mul_add(
            "gf8 aes ssse3 narrowed",
            &gf8_aes_coeffs(),
            |dst, table, src| {
                entry(v2, dst, table, src);
            },
        );
    });
    if matches!(proof, X86Proof::V2(_)) {
        return;
    }
    let v3 = crate::kernel::x86_v3_token();
    tier_gate::run_entry::<X64V3Token, _>(x86::gf8::mul_add_avx2, |_, entry| {
        check_gf8_mul_add(
            "gf8 aes avx2 narrowed",
            &gf8_aes_coeffs(),
            |dst, table, src| {
                entry(v3, dst, table, src);
            },
        );
    });
}

/// `mul_into` over a buffer past the non-temporal store threshold.
///
/// The shared `LENGTHS` are all far below it, so nothing else reaches the
/// `vmovntdq` body. Destination offsets 0 and 1 cover both sides of the
/// alignment peel — offset 1 leaves the GF(2^16) peel an odd number of
/// bytes, which must fall back to ordinary stores rather than split a
/// buffer mid-element.
#[test]
#[allow(clippy::too_many_lines)]
fn non_temporal_mul_into_matches_reference() {
    const NT_LEN: usize = (2 << 20) + 130;
    // The non-temporal split is a store-side choice, independent of the
    // coefficient, so a zero, a one and a mixed value are enough.
    const GF8_COEFFS: [Elem<FiniteField<2, 8, Polynomial<AES>>>; 3] = [
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::ZERO,
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::ONE,
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
    ];
    const GF16_COEFFS: [Elem<Rijndael16Field>; 3] = [
        Elem::<Rijndael16Field>::from_raw(0),
        Elem::<Rijndael16Field>::from_raw(1),
        Elem::<Rijndael16Field>::from_raw(0x53a7),
    ];

    let source = noise(NT_LEN + 2, 0x1a7);
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<true>, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF8_COEFFS {
                // Noise, not zeros: a streaming body that accumulated into
                // the destination instead of overwriting it must fail here.
                let mut want = src.to_vec();
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                let got = &mut got[offset..offset + NT_LEN];
                entry(token, got, Prepared::new(coeff), src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 gfni nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx2, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF8_COEFFS {
                let table = scale_table(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                let got = &mut got[offset..offset + NT_LEN];
                entry(token, got, table, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 avx2 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_into_ssse3, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF8_COEFFS {
                let table = scale_table(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                let got = &mut got[offset..offset + NT_LEN];
                entry(token, got, table, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });

    // GF(2^16) needs an even length and an even peel.
    tier_gate::run_entry(x86::gf16::mul_into_gfni, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let src = &src[..NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF16_COEFFS {
                let mut want = src.to_vec();
                scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                let got = &mut got[offset..offset + src.len()];
                entry(token, got, TowerCoeff::new(coeff), src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 gfni nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_into_avx2, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let src = &src[..NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF16_COEFFS {
                let tables = TowerTables::new(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                let got = &mut got[offset..offset + src.len()];
                entry(token, got, &tables, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 avx2 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_into_ssse3, |token, entry| {
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let src = &src[..NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];
            for coeff in GF16_COEFFS {
                let tables = TowerTables::new(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                let got = &mut got[offset..offset + src.len()];
                entry(token, got, &tables, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    });
}

#[test]
fn half_lane_payloads_match_reference() {
    let coeff = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53);
    for len in [511usize, 512, 1152, 1168, 1200, 2048] {
        for off in [0usize, 1, 16, 32, 48] {
            let src_backing = noise(len + 128, 0x522);
            let src_start = src_backing.as_ptr().align_offset(64) + (off + 16) % 64;
            let src = &src_backing[src_start..src_start + len];
            let mut backing = noise(len + 128, 0x533);
            let start = backing.as_ptr().align_offset(64) + off;
            let dst = &mut backing[start..start + len];
            let mut want = dst.to_vec();

            tier_gate::run_entry(x86::bytes::xor_avx2, |token, entry| {
                entry(token, dst, src);
                scalar::xor(&mut want, src);
                assert_eq!(dst, want, "xor {len} +{off}");
            });
            tier_gate::run_entry(x86::gf8::mul_add_gfni::<true>, |token, entry| {
                entry(token, dst, Prepared::new(coeff), src);
                scalar::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff, src);
                assert_eq!(dst, want, "mul_add {len} +{off}");
            });
            tier_gate::run_entry(x86::gf8::mul_assign_gfni::<true>, |token, entry| {
                entry(token, dst, Prepared::new(coeff));
                scalar::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(&mut want, coeff);
                assert_eq!(dst, want, "mul_assign {len} +{off}");
            });
        }
    }
}

/// Multi-row scatter across short and long rows, including pitches that
/// alternate the destination alignment within a row group.
///
/// The source and destination start at independent offsets; reference
/// multiplication checks the peeled head and the vector/tail seam.
#[test]
#[allow(clippy::too_many_lines)]
fn aligned_scatter_matches_reference() {
    const ROW_LENS: &[usize] = &[512, 1152, 1168, 1184, 1200, 2048, 4096];

    tier_gate::run_entry(x86::gf8::mul_add_scatter_gfni::<true>, |token, entry| {
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let coeffs: Vec<_> = (0..nrows).map(gf8_coeff_at::<AES>).collect();
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &prepared_coeffs(&coeffs), &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                        gf8_reference::<AES>(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 scatter: {row_len}/{nrows}/{skew}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf8::mul_add_scatter_gfni::<false>, |token, entry| {
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let rs_coeffs: Vec<_> = (0..nrows).map(gf8_coeff_at::<RS>).collect();
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &prepared_coeffs(&rs_coeffs), &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&rs_coeffs) {
                        gf8_reference::<RS>(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf8 rs scatter: {row_len}/{nrows}/{skew}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_gfni, |token, entry| {
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 scatter: {row_len}/{nrows}/{skew}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_avx2, |token, entry| {
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 avx2 scatter: {row_len}/{nrows}/{skew}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_add_scatter_ssse3, |token, entry| {
        for &row_len in ROW_LENS {
            for nrows in [1usize, 4, 6, 9] {
                for skew in [0usize, 1, 16] {
                    let src = noise(row_len, 0x1b8);
                    let mut backing = noise(row_len * nrows + skew, 0x1c9);
                    let rows = &mut backing[skew..];

                    let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 ssse3 scatter: {row_len}/{nrows}/{skew}"
                    );
                }
            }
        }
    });
}

#[test]
fn x86_geometry_guards_accept_zero_length_rows() {
    type E16 = Elem<Rijndael16Field>;

    // `row_len == 0` must fall through the vector prologues untouched.
    let Some(v2) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    };

    let coeffs = [Prepared::new(
        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(3),
    )];
    let empty_aes: &[(&[Prepared], &[u8])] = &[];
    x86::gf8::mul_add_scatter_ssse3(v2, &mut [], 0, &coeffs, &[]);
    let gf16_coeffs = [Elem::<Rijndael16Field>::from_raw(5)];
    x86::gf16::mul_add_scatter_ssse3(v2, &mut [], 0, &gf16_coeffs, &[]);

    if let Some(v3) = tier_gate::summon::<X64V3Token>() {
        x86::gf8::mul_add_scatter_avx2(v3, &mut [], 0, &coeffs, &[]);
        x86::gf16::mul_add_scatter_avx2(v3, &mut [], 0, &gf16_coeffs, &[]);
        x86::gf8::mul_add_matrix_avx2_with(v3, &mut [], 0, 1, empty_aes);
    }
    x86::gf8::mul_add_matrix_ssse3_with(v2, &mut [], 0, 1, empty_aes);
    let no_terms: [(&[E16], &[u8]); 0] = [];
    x86::gf16::mul_add_matrix_ssse3(v2, &mut [], 0, 1, &no_terms);

    // Empty term lists must return before any store, in the checked
    // scattered wrappers as well. The GFNI wrapper runs only where its
    // token summons; the guarded empty-terms differential below covers
    // GFNI hosts.
    if let Some(token) = tier_gate::summon::<X64V3GfniCryptoToken>() {
        let mut dst = [0u8; 8];
        let no_terms: &[(&[Prepared], &[u8])] = &[];
        x86::gf8::mul_add_matrix_at_gfni_with::<true, _>(token, &mut dst, 8, &[0], no_terms);
        assert_eq!(dst, [0; 8]);
    }
}

#[test]
fn gfni_matrix_wrappers_accept_empty_terms() {
    use crate::kernel::FlatMatrix;

    let Some(token) = tier_gate::summon::<X64V3GfniCryptoToken>() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    // Zero-row and empty-source matrices return before any store, through
    // both selection constants.
    let empty: FlatMatrix<'_, Prepared> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    let mut rows = [0u8; 8];
    x86::gf8::mul_add_matrix_gfni_with::<true, _>(token, &mut rows, 8, 0, &empty);
    x86::gf8::mul_add_matrix_gfni_with::<false, _>(token, &mut rows, 8, 0, &empty);
    let gf16_empty: FlatMatrix<'_, TowerCoeff> = FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &gf16_empty);
    assert_eq!(rows, [0; 8]);
}
