//! AVX-512 GF(2^16) differentials: the token-proven wide lanes, the
//! misalignment peels on both sides of the floor, and the wide matrices.

use super::*;
use crate::kernel::tests::tier_gate;

/// Checks the token-proven tower lanes directly against scalar arithmetic.
#[test]
fn gf16_wide_kernels_match_reference() {
    tier_gate::run_entry(x86::gf16::mul_add_avx512, |token, entry| {
        check_gf16_mul_add_tables("gf16 wide", |dst, tables, src| {
            entry(token, dst, tables.compact(), src);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_assign_avx512, |token, entry| {
        check_gf16_mul_assign_tables("gf16 wide", |dst, tables| {
            entry(token, dst, tables.compact());
        });
    });
    tier_gate::run_entry(x86::gf16::mul_into_avx512, |token, entry| {
        for &len in LENGTHS {
            let src = noise(len, 0xb51);
            for coeff in gf16_coeffs() {
                let mut expected = vec![0u8; len];
                scalar::mul_add::<Rijndael16Field>(&mut expected, coeff, &src);
                let mut got = noise(len, 0xb52);
                entry(token, &mut got, TowerCoeff::new(coeff), &src);
                assert_eq!(
                    got, expected,
                    "gf16 wide mul_into len {len} coeff {coeff:?}"
                );
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_elementwise_avx512, |token, entry| {
        check_gf16_elementwise("gf16 wide elementwise", |dst, a, b| {
            entry(token, dst, a, b, DELTA_BYTE, AES_REDUCTION);
        });
    });
    tier_gate::run_entry(x86::gf16::mul_elementwise_assign_avx512, |token, entry| {
        check_elementwise_assign::<Rijndael16Field>("gf16 wide elementwise assign", |dst, src| {
            entry(token, dst, src, DELTA_BYTE, AES_REDUCTION);
        });
    });
}

/// Misaligned destinations on both sides of the `mul_add` peel floor,
/// with even (peelable) and odd (unpeelable) heads.
#[test]
fn gf16_wide_mul_add_peels_misaligned_rows() {
    let floor = x86::gf16::MUL_ADD_PEEL_MIN;
    tier_gate::run_entry(x86::gf16::mul_add_avx512, |token, entry| {
        for len in [floor - 2, floor, floor + 2, floor + 130, 4096 + 66] {
            for offset in [0usize, 1, 2, 16, 34, 62] {
                let src_storage = noise(len + 64, 0xb61);
                let mut dst_storage = noise(len + 128, 0xb62);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = dst.to_vec();
                    scalar::mul_add::<Rijndael16Field>(&mut want, coeff, src);
                    entry(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_add_avx512: len {len}, offset {offset}"
                    );
                }
            }
        }
    });
}

/// Misaligned destinations across the shared 64-byte peel floor for the
/// wide overwrite body, beside the `mul_add` coverage above.
#[test]
fn gf16_wide_mul_into_peels_misaligned_rows() {
    let floor = x86::gf16::MUL_ADD_PEEL_MIN;
    tier_gate::run_entry(x86::gf16::mul_into_avx512, |token, entry| {
        for len in [floor - 2, floor, floor + 2, floor + 130, 4096 + 66] {
            for offset in [0usize, 1, 2, 16, 34, 62] {
                let src_storage = noise(len + 64, 0xb65);
                let mut dst_storage = noise(len + 128, 0xb66);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = src.to_vec();
                    scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                    entry(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_into_avx512: len {len}, offset {offset}"
                    );
                }
            }
        }
    });
}

/// Compares wide matrix tiles and their remainders with independent row sums.
#[test]
fn gf16_wide_matrix_matches_reference() {
    if tier_gate::summon::<X64V4xToken>().is_none() {
        return;
    }
    tier_gate::run_entry(x86::gf16::mul_add_matrix_avx512, |token, entry| {
        check_matrix(
            "gf16 wide matrix",
            gf16_coeff_at2,
            gf16_reference,
            |rows, len, nrows, terms, _want| {
                entry(token, rows, len, nrows, terms);
            },
        );
    });
    check_matrix(
        "gf16 wide matrix provider",
        gf16_coeff_at2,
        gf16_reference,
        |rows, row_len, nrows, terms, want| {
            let flat: Vec<TowerCoeff> = terms
                .iter()
                .flat_map(|(coeffs, _)| coeffs.iter().copied().map(TowerCoeff::new))
                .collect();
            let flat_srcs: Vec<&[u8]> = terms.iter().map(|(_, src)| *src).collect();
            let matrix = crate::kernel::FlatMatrix {
                coefficients: &flat,
                nrows,
                sources: &flat_srcs,
            };
            tier_gate::run_entry(x86::gf16::mul_add_matrix_avx512_with, |token, entry| {
                entry(token, rows, row_len, nrows, &matrix);
                assert_eq!(rows, want, "gf16 wide matrix provider");
            });
        },
    );
}
