//! GFNI GF(2^16) differentials: the half-lane peel floor, the staged
//! blocked tails, the provider-form matrix, and the degenerate-geometry
//! wrapper contracts.

use super::*;
use crate::kernel::tests::tier_gate;

/// Row lengths whose GFNI scatter/gather/matrix tail is 17-30 bytes past
/// the last 32-byte window: the regression geometry for the staged
/// `mul_add_tail_gfni` chunking.
const GFNI_TAIL_LENGTHS: &[usize] = &[50, 62, 94, 126, 158, 190, 222, 254, 350, 510, 1086];

/// The GF(2^16) GFNI kernels across the half-lane peel floor: even
/// lengths over every destination residue, so declined odd heads stay
/// covered beside the peeled 16-mod-32 ones.
#[test]
fn gf16_gfni_peels_half_lane() {
    let floor = crate::kernel::routing::HALF_LANE_PEEL_MIN;
    tier_gate::run_entry(x86::gf16::mul_add_gfni, |token, entry| {
        for len in [floor - 2, floor, floor + 2, floor + 130, 1152, 4096 + 66] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c7);
                let mut dst_storage = noise(len + 128, 0x2c8);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = dst.to_vec();
                    scalar::mul_add::<Rijndael16Field>(&mut want, coeff, src);
                    entry(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(&*dst, want, "gf16 mul_add_gfni: len {len}, offset {offset}");
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<Rijndael16Field>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_assign_gfni, |token, entry| {
        for len in [floor - 2, floor, floor + 2, floor + 130, 1152, 4096 + 66] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c7);
                let mut dst_storage = noise(len + 128, 0x2c8);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = dst.to_vec();
                    scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                    entry(token, dst, TowerCoeff::new(coeff));
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_assign_gfni: len {len}, offset {offset}"
                    );
                    dst.copy_from_slice(src);
                    scalar::mul_assign::<Rijndael16Field>(dst, coeff);
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_into_gfni, |token, entry| {
        for len in [floor - 2, floor, floor + 2, floor + 130, 1152, 4096 + 66] {
            for offset in [0usize, 1, 2, 15, 16, 17, 31, 32, 47] {
                let src_storage = noise(len + 64, 0x2c7);
                let mut dst_storage = noise(len + 128, 0x2c8);
                let base = dst_storage.as_ptr().align_offset(64);
                let dst = &mut dst_storage[base + offset..base + offset + len];
                let src = &src_storage[offset..offset + len];
                for coeff in gf16_coeffs() {
                    let mut want = src.to_vec();
                    scalar::mul_assign::<Rijndael16Field>(&mut want, coeff);
                    entry(token, dst, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        &*dst, want,
                        "gf16 mul_into_gfni: len {len}, offset {offset}"
                    );
                }
            }
        }
    });
}

/// The staged GFNI tail past whole 32-byte windows: 17-30-byte
/// remainders chunked at 16 bytes, across aligned and misaligned
/// destinations. The single 16-byte staging of `mul_add_tail_gfni`
/// overflowed on exactly this geometry; the blocked GFNI routes hand
/// it remainders up to 30 bytes.
#[test]
fn gf16_gfni_blocked_tails_match_scalar() {
    tier_gate::run_entry(x86::gf16::mul_add_scatter_gfni, |token, entry| {
        for &row_len in GFNI_TAIL_LENGTHS {
            for nrows in [1usize, 2, 3, 5] {
                for skew in [0usize, 1] {
                    let src = noise(row_len, 0x311);
                    let coeffs: Vec<Elem<Rijndael16Field>> =
                        [0x0000u16, 0x0001, 0x0100, 0x53a7, 0xbeef, 0xffff]
                            .iter()
                            .copied()
                            .map(Elem::<Rijndael16Field>::from_raw)
                            .collect();
                    let row_coeffs: Vec<Elem<Rijndael16Field>> =
                        (0..nrows).map(|j| coeffs[j % coeffs.len()]).collect();
                    let mut backing = noise(row_len * nrows + skew, 0x322);
                    let rows = &mut backing[skew..skew + row_len * nrows];
                    let mut want = rows.to_vec();
                    entry(token, rows, row_len, &row_coeffs, &src);
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&row_coeffs) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 gfni scatter tail: row_len {row_len}, nrows {nrows}, skew {skew}"
                    );
                }
            }
        }
    });
    tier_gate::run_entry(x86::gf16::mul_add_gather_gfni, |token, entry| {
        for &row_len in GFNI_TAIL_LENGTHS {
            let sources: Vec<Vec<u8>> = (0..5usize)
                .map(|t| noise(row_len, 0x333 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<Elem<Rijndael16Field>> =
                [0x0000u16, 0x0001, 0x0100, 0x53a7, 0xbeef, 0xffff]
                    .iter()
                    .copied()
                    .map(Elem::<Rijndael16Field>::from_raw)
                    .collect();
            let gather_coeffs: Vec<Elem<Rijndael16Field>> =
                (0..5).map(|t| coeffs[t % coeffs.len()]).collect();
            let mut got = noise(row_len, 0x344);
            let mut want = got.clone();
            entry(token, &mut got, &gather_coeffs, &srcs);
            for (&coeff, &src) in gather_coeffs.iter().zip(&srcs) {
                gf16_reference(&mut want, coeff, src);
            }
            assert_eq!(got, want, "gf16 gfni gather tail: row_len {row_len}");
        }
    });
    tier_gate::run_entry(x86::gf16::mul_add_matrix_gfni, |token, entry| {
        for &row_len in GFNI_TAIL_LENGTHS {
            let sources: Vec<Vec<u8>> = (0..5usize)
                .map(|t| noise(row_len, 0x333 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let nrows = 5;
            let coeff_sets: Vec<Vec<Elem<Rijndael16Field>>> = (0..3)
                .map(|t| (0..nrows).map(|j| gf16_coeff_at2(t, j)).collect())
                .collect();
            let terms: Vec<(&[Elem<_>], &[u8])> = coeff_sets
                .iter()
                .zip(&srcs)
                .map(|(c, s)| (c.as_slice(), *s))
                .collect();
            let mut got = noise(row_len * nrows, 0x355);
            let mut want = got.clone();
            entry(token, &mut got, row_len, nrows, &terms);
            for &(coeffs, src) in &terms {
                for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                    gf16_reference(row, coeff, src);
                }
            }
            assert_eq!(got, want, "gf16 gfni matrix tail: row_len {row_len}");
        }
    });
}

/// Compares the provider-form GFNI matrix with independent row sums.
#[test]
fn gf16_gfni_matrix_with_matches_reference() {
    if tier_gate::summon::<X64V3GfniCryptoToken>().is_none() {
        return;
    }
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let sources: Vec<Vec<u8>> = (0..3usize)
                .map(|t| noise(row_len, 0x360 + t as u64))
                .collect();
            let coeff_sets: Vec<Vec<Elem<Rijndael16Field>>> = (0..3usize)
                .map(|t| (0..nrows).map(|j| gf16_coeff_at2(t, j)).collect())
                .collect();
            let flat: Vec<TowerCoeff> = coeff_sets
                .iter()
                .flatten()
                .map(|&coeff| TowerCoeff::new(coeff))
                .collect();
            let flat_srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let matrix = crate::kernel::FlatMatrix {
                coefficients: &flat,
                nrows,
                sources: &flat_srcs,
            };
            tier_gate::run_entry(x86::gf16::mul_add_matrix_gfni_with, |token, entry| {
                let mut got = noise(row_len * nrows, 0x361);
                let mut want = got.clone();
                entry(token, &mut got, row_len, nrows, &matrix);
                for (term, src) in sources.iter().enumerate() {
                    for (row, coeff) in want.chunks_exact_mut(row_len).zip(&coeff_sets[term]) {
                        gf16_reference(row, *coeff, src);
                    }
                }
                assert_eq!(got, want, "gf16 gfni matrix with: {row_len}B x {nrows}");
            });
        }
    }
}

#[test]
fn gf16_gfni_wrappers_tolerate_degenerate_geometry() {
    let Some(token) = tier_gate::summon::<X64V3GfniCryptoToken>() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };
    // The gf16 wrappers clamp rather than panic: zero-length rows, zero
    // rows, and room for fewer rows than coefficients are all no-ops.
    x86::gf16::mul_add_scatter_gfni(
        token,
        &mut [],
        0,
        &[Elem::<Rijndael16Field>::from_raw(1)],
        &[],
    );
    let mut rows = [0u8; 8];
    x86::gf16::mul_add_scatter_gfni(
        token,
        &mut rows,
        0,
        &[Elem::<Rijndael16Field>::from_raw(1)],
        &[],
    );
    assert_eq!(rows, [0; 8]);

    let empty: crate::kernel::FlatMatrix<'_, TowerCoeff> = crate::kernel::FlatMatrix {
        coefficients: &[],
        nrows: 0,
        sources: &[],
    };
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 2, &empty);
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 0, 2, &empty);
    x86::gf16::mul_add_matrix_gfni_with(token, &mut rows, 8, 0, &empty);
    assert_eq!(rows, [0; 8]);
}
