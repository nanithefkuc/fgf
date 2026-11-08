//! GF(2^8) differentials that span instruction sets: the additional
//! polynomial sweeps and the column-block matrix geometry helpers its
//! ISA modules share.

use super::*;
use crate::kernel::tables::isomorphism_to_aes;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

#[cfg(feature = "simd512")]
mod avx512;
mod gfni;

/// Column-blocked matrix geometry as `(row_len, nrows, nterms)`.
///
/// Row lengths sit one byte and one lane past two blocks, on a ragged span
/// with a sub-lane remainder, and on an exact multiple of the block; row
/// counts run several row groups. One single-group case at a long row
/// pins the unblocked walk.
fn column_block_cases() -> Vec<(usize, usize, usize)> {
    let block = x86::gf8::MATRIX_COLUMN_BLOCK;
    let mut cases = Vec::new();
    for row_len in [2 * block + 1, 2 * block + 64, 3 * block + 31, 5 * block] {
        for nrows in [3, 5, 6, 8, 9] {
            for nterms in [1, 3, 10] {
                cases.push((row_len, nrows, nterms));
            }
        }
    }
    cases.push((3 * block + 31, 4, 3));
    // Term counts past one resolve chunk over a lane-multiple blocked
    // row, so later chunks accumulate inside every block, including the
    // leading unaligned one.
    let chunk = x86::gf8::RESOLVE_CHUNK;
    for nterms in [chunk + 1, 2 * chunk + 1] {
        for nrows in [3, 5, 7] {
            cases.push((3 * block + 64, nrows, nterms));
        }
    }
    cases
}

/// Deterministic matrix terms: per-term noise sources of `row_len` bytes
/// and `coeff_at(term, row)` coefficients.
fn column_block_terms<E: Copy>(
    row_len: usize,
    nrows: usize,
    nterms: usize,
    coeff_at: impl Fn(usize, usize) -> E,
) -> (Vec<Vec<u8>>, Vec<Vec<E>>) {
    let sources = (0..nterms)
        .map(|t| noise(row_len, 0xc00 + t as u64))
        .collect();
    let coeff_sets = (0..nterms)
        .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
        .collect();
    (sources, coeff_sets)
}

/// Contiguous matrix differential over column-blocked geometry, with the
/// destination `offset` bytes past a 64-byte boundary. Accumulate starts
/// from noise; overwrite starts from noise the kernel must ignore.
fn check_matrix_column_blocks<E: Copy>(
    name: &str,
    overwrite: bool,
    offset: usize,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])], &[u8]),
) {
    for (row_len, nrows, nterms) in column_block_cases() {
        let (sources, coeff_sets) = column_block_terms(row_len, nrows, nterms, &coeff_at);
        let terms: Vec<(&[E], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let span = row_len * nrows;
        let mut backing = noise(span + 128, 0xc10);
        let start = backing.as_ptr().align_offset(64) + offset;
        let mut want = if overwrite {
            vec![0u8; span]
        } else {
            backing[start..start + span].to_vec()
        };
        for &(coeffs, src) in &terms {
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                reference(row, coeff, src);
            }
        }
        let rows = &mut backing[start..start + span];
        kernel(rows, row_len, nrows, &terms, &want);
        assert!(
            *rows == *want,
            "{name}: +{offset} {row_len}B x {nrows} x {nterms}t"
        );
    }
}

/// Scattered-row matrix differential over column-blocked geometry: rows
/// separated by gaps, in reversed order for odd row counts, with the
/// destination `offset` bytes past a 64-byte boundary.
fn check_matrix_at_column_blocks<E: Copy>(
    name: &str,
    overwrite: bool,
    offset: usize,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], usize, &[usize], &[(&[E], &[u8])], &[u8]),
) {
    for (row_len, nrows, nterms) in column_block_cases() {
        let (sources, coeff_sets) = column_block_terms(row_len, nrows, nterms, &coeff_at);
        let terms: Vec<(&[E], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let mut starts: Vec<usize> = (0..nrows).map(|j| j * (row_len + 7)).collect();
        if nrows % 2 == 1 {
            starts.reverse();
        }
        let span = (nrows - 1) * (row_len + 7) + row_len;
        let mut backing = noise(span + 128, 0xc20);
        let start = backing.as_ptr().align_offset(64) + offset;
        // Accumulate starts from the destination bytes; overwrite zeroes
        // only the named rows, leaving the gaps at their backing bytes.
        let mut want = backing[start..start + span].to_vec();
        if overwrite {
            for &row_start in &starts {
                want[row_start..row_start + row_len].fill(0);
            }
        }
        for &(coeffs, src) in &terms {
            for (&row_start, &coeff) in starts.iter().zip(coeffs) {
                reference(&mut want[row_start..row_start + row_len], coeff, src);
            }
        }
        let dst = &mut backing[start..start + span];
        kernel(dst, row_len, &starts, &terms, &want);
        assert!(
            *dst == *want,
            "{name}: +{offset} {row_len}B x {nrows} x {nterms}t"
        );
    }
}

/// Run `f` over a [`FlatMatrix`](crate::kernel::FlatMatrix) holding the
/// same terms as `terms`, so the `_with` forms see a provider other than
/// the term slice.
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn with_flat<const POLY: u128>(
    terms: &[(&[Elem<FiniteField<2, 8, Polynomial<POLY>>>], &[u8])],
    nrows: usize,
    f: impl FnOnce(&crate::kernel::FlatMatrix<'_, Prepared>),
) {
    let coefficients: Vec<Prepared> = terms
        .iter()
        .flat_map(|&(coeffs, _)| prepared_coeffs(coeffs))
        .collect();
    let sources: Vec<&[u8]> = terms.iter().map(|&(_, src)| src).collect();
    f(&crate::kernel::FlatMatrix {
        coefficients: &coefficients,
        nrows,
        sources: &sources,
    });
}

/// Direct differentials for polynomials beyond the two named constants:
/// Data Matrix `0x12D` and the CCSDS Reed–Solomon field `0x187`, exactly
/// the shapes a caller spells as `FiniteField<2, 8, Polynomial<0x12D>>`. Each tier exercises the
/// single-buffer and elementwise kernels at lane/tail boundaries — raw
/// coefficients everywhere, plus the dispatch-shaped `Prepared` form on
/// the blocked backends — and skips through its token convention when
/// the host cannot prove the instructions.
#[test]
fn gf8_additional_polynomials_match_reference() {
    check_gf8_polynomial_kernels::<0x12D, false>("0x12d");
    check_gf8_polynomial_kernels::<0x187, false>("0x187");
}

#[allow(clippy::too_many_lines)]
fn check_gf8_polynomial_kernels<const POLY: u128, const NATIVE: bool>(name: &str) {
    let coeffs = gf8_coeffs_of::<POLY>();

    // GFNI: the affine multiply folds any polynomial's coefficient in.
    // The dispatch shape selects the native body only for the AES
    // polynomial, which none of these sweeps name.
    tier_gate::run_entry(x86::gf8::mul_add_gfni::<false>, |token, entry| {
        check_tower_mul_add(
            &format!("{name} gfni mul_add"),
            LENGTHS,
            &coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| entry(token, dst, Prepared::new(c), src),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_gfni::<false>, |token, entry| {
        check_tower_mul_assign(
            &format!("{name} gfni mul_assign"),
            LENGTHS,
            &coeffs,
            scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>,
            |dst, c| entry(token, dst, Prepared::new(c)),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_gfni::<false>, |token, entry| {
        check_tower_mul_into(
            &format!("{name} gfni mul_into"),
            LENGTHS,
            &coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| entry(token, dst, Prepared::new(c), src),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_gfni::<NATIVE>, |token, entry| {
        check_gf8_elementwise::<POLY>(&format!("{name} gfni elementwise"), |dst, a, b| {
            entry(
                token,
                isomorphism_to_aes::<POLY>(),
                Polynomial::<POLY>::REDUCTION_LOW,
                dst,
                a,
                b,
            );
        });
    });
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_gfni::<NATIVE>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<POLY>>>(
                &format!("{name} gfni elementwise assign"),
                |dst, src| {
                    entry(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Polynomial::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        },
    );

    // AVX2 and SSSE3: nibble tables from the polynomial's own bank.
    tier_gate::run_entry(x86::gf8::mul_add_avx2, |token, entry| {
        check_gf8_mul_add::<POLY>(&format!("{name} avx2"), &coeffs, |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx2, |token, entry| {
        check_gf8_mul_assign::<POLY>(&format!("{name} avx2"), &coeffs, |dst, table| {
            entry(token, dst, table);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_avx2, |token, entry| {
        check_gf8_elementwise::<POLY>(&format!("{name} avx2 elementwise"), |dst, a, b| {
            entry(token, Polynomial::<POLY>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_avx2, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<POLY>>>(
            &format!("{name} avx2 elementwise assign"),
            |dst, src| {
                entry(token, Polynomial::<POLY>::REDUCTION_LOW, dst, src);
            },
        );
    });
    tier_gate::run_entry(x86::gf8::mul_add_ssse3, |token, entry| {
        check_gf8_mul_add::<POLY>(&format!("{name} ssse3"), &coeffs, |dst, table, src| {
            entry(token, dst, table, src);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_assign_ssse3, |token, entry| {
        check_gf8_mul_assign::<POLY>(&format!("{name} ssse3"), &coeffs, |dst, table| {
            entry(token, dst, table);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_ssse3, |token, entry| {
        check_gf8_elementwise::<POLY>(&format!("{name} ssse3 elementwise"), |dst, a, b| {
            entry(token, Polynomial::<POLY>::REDUCTION_LOW, dst, a, b);
        });
    });
    tier_gate::run_entry(x86::gf8::mul_elementwise_assign_ssse3, |token, entry| {
        check_elementwise_assign::<FiniteField<2, 8, Polynomial<POLY>>>(
            &format!("{name} ssse3 elementwise assign"),
            |dst, src| {
                entry(token, Polynomial::<POLY>::REDUCTION_LOW, dst, src);
            },
        );
    });

    // AVX-512: the affine multiply at 64-byte lanes over the prepared
    // coefficient.
    #[cfg(feature = "simd512")]
    check_gf8_polynomial_avx512::<POLY, NATIVE>(name, &coeffs);
    #[cfg(not(feature = "simd512"))]
    let _ = name;
}

/// The 512-bit tier of [`check_gf8_polynomial_kernels`], compiled only
/// where the dispatch tier exists.
#[cfg(feature = "simd512")]
fn check_gf8_polynomial_avx512<const POLY: u128, const NATIVE: bool>(
    name: &str,
    coeffs: &[Elem<FiniteField<2, 8, Polynomial<POLY>>>],
) {
    tier_gate::run_entry(x86::gf8::mul_add_avx512, |token, entry| {
        check_tower_mul_add(
            &format!("{name} avx512 mul_add"),
            LENGTHS,
            coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| entry(token, dst, Prepared::new(c), src),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_assign_avx512, |token, entry| {
        check_tower_mul_assign(
            &format!("{name} avx512 mul_assign"),
            LENGTHS,
            coeffs,
            scalar::mul_assign::<FiniteField<2, 8, Polynomial<POLY>>>,
            |dst, c| entry(token, dst, Prepared::new(c)),
        );
    });
    tier_gate::run_entry(x86::gf8::mul_into_avx512, |token, entry| {
        check_tower_mul_into(
            &format!("{name} avx512 mul_into"),
            LENGTHS,
            coeffs,
            gf8_reference::<POLY>,
            |dst, c, src| entry(token, dst, Prepared::new(c), src),
        );
    });
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_avx512::<NATIVE>,
        |token, entry| {
            check_gf8_elementwise::<POLY>(&format!("{name} avx512 elementwise"), |dst, a, b| {
                entry(
                    token,
                    isomorphism_to_aes::<POLY>(),
                    Polynomial::<POLY>::REDUCTION_LOW,
                    dst,
                    a,
                    b,
                );
            });
        },
    );
    tier_gate::run_entry(
        x86::gf8::mul_elementwise_assign_avx512::<NATIVE>,
        |token, entry| {
            check_elementwise_assign::<FiniteField<2, 8, Polynomial<POLY>>>(
                &format!("{name} avx512 elementwise assign"),
                |dst, src| {
                    entry(
                        token,
                        isomorphism_to_aes::<POLY>(),
                        Polynomial::<POLY>::REDUCTION_LOW,
                        dst,
                        src,
                    );
                },
            );
        },
    );
}
