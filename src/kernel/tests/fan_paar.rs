//! Fan-Paar tower differentials outside the cross-family suites: the
//! prepared matrix plan against one dispatched row application per term.

#[cfg(feature = "alloc")]
use super::*;

#[cfg(feature = "alloc")]
#[test]
fn macro_kernels_matrix_plan_matches_per_row_application() {
    // The prepared matrix plan applies the same prepared coefficients row by
    // row; compare against one `mul_add` per row and term.
    let row_len = 8;
    let srcs: Vec<Vec<u8>> = vec![noise(row_len, 0x71), noise(row_len, 0x72)];
    let coeffs = [
        Elem::<FanPaar8Field>::from_raw(1),
        Elem::<FanPaar8Field>::from_raw(0x8d),
    ];
    let prepared: Vec<_> = coeffs
        .iter()
        .copied()
        .map(|coeff| <FanPaar8Field>::prepare(RawDispatch, coeff))
        .collect();
    // Source-major layout: term 0 uses (prepared[0], prepared[1]), term 1
    // uses (prepared[1], prepared[0]).
    let values = [coeffs[0], coeffs[1], coeffs[1], coeffs[0]];
    let flat = [prepared[0], prepared[1], prepared[1], prepared[0]];

    let mut got = noise(2 * row_len, 0x73);
    let mut want = got.clone();
    <FanPaar8Field as KernelDispatch>::mul_add_matrix_plan(
        RawDispatch,
        &mut got,
        row_len,
        2,
        &values,
        &flat,
        &[&srcs[0], &srcs[1]],
    );
    for (term, src) in srcs.iter().enumerate() {
        let row_coeffs = if term == 0 {
            [&prepared[0], &prepared[1]]
        } else {
            [&prepared[1], &prepared[0]]
        };
        for (row, coeff) in want.chunks_exact_mut(row_len).zip(row_coeffs) {
            <FanPaar8Field as KernelDispatch>::mul_add(RawDispatch, row, coeff, src);
        }
    }
    assert_eq!(got, want, "prepared matrix");
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[test]
fn fan_paar_byte_shuffle_matches_descriptor() {
    use super::*;
    use crate::field::binary::tower::{FANPAAR8_DESC, FanPaar8Field};
    use crate::kernel::tables::{FpScaleTable, fp_scale_table};
    use crate::kernel::x86;

    fn check(
        add: impl Fn(&mut [u8], &FpScaleTable, &[u8]),
        into: impl Fn(&mut [u8], &FpScaleTable, &[u8]),
        assign: impl Fn(&mut [u8], &FpScaleTable),
    ) {
        for &len in XOR_LENGTHS {
            let src = noise(len, 0x4de);
            let initial = noise(len, 0xf04);
            for coeff in [0, 1, 0x10, 0x2d, 0x80, 0xff] {
                let table = fp_scale_table(Elem::<FanPaar8Field>::from_raw(coeff));
                let expected: Vec<_> = src
                    .iter()
                    .map(|&x| {
                        FANPAAR8_DESC
                            .mul(u64::from(coeff), u64::from(x))
                            .to_le_bytes()[0]
                    })
                    .collect();
                let mut dst = initial.clone();
                add(&mut dst, table, &src);
                for ((&out, &before), &product) in dst.iter().zip(&initial).zip(&expected) {
                    assert_eq!(out, before ^ product);
                }
                into(&mut dst, table, &src);
                assert_eq!(dst, expected);
                dst.copy_from_slice(&src);
                assign(&mut dst, table);
                assert_eq!(dst, expected);
            }
        }
    }

    if let Some(token) = tier_gate::summon::<X64V3Token>() {
        eprintln!("kernel_differential executing tier=v3 field=fan-paar8");
        check(
            |dst, table, src| x86::gf8::mul_add_avx2(token, dst, table, src),
            |dst, table, src| x86::gf8::mul_into_avx2(token, dst, table, src),
            |dst, table| x86::gf8::mul_assign_avx2(token, dst, table),
        );
    } else {
        eprintln!("skipping Fan-Paar byte AVX2: unsupported");
    }
    if let Some(token) = tier_gate::summon::<X64V2Token>() {
        eprintln!("kernel_differential executing tier=v2 field=fan-paar8");
        check(
            |dst, table, src| x86::gf8::mul_add_ssse3(token, dst, table, src),
            |dst, table, src| x86::gf8::mul_into_ssse3(token, dst, table, src),
            |dst, table| x86::gf8::mul_assign_ssse3(token, dst, table),
        );
    } else {
        eprintln!("skipping Fan-Paar byte SSSE3: unsupported");
    }
}
