//! Fan-Paar tower differentials outside the cross-family suites: the
//! prepared-terms matrix override the macro kernels install.

use super::*;

#[test]
fn macro_kernels_matrix_with_matches_per_row_application() {
    // `impl_field_kernels!`'s prepared-terms matrix override (FanPaar8Field):
    // apply the same prepared coefficients row by row and compare.
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

    let mut got = noise(2 * row_len, 0x73);
    let mut want = got.clone();
    <FanPaar8Field as KernelDispatch>::mul_add_matrix_with(
        RawDispatch,
        &mut got,
        row_len,
        2,
        &[
            (&[prepared[0], prepared[1]], &srcs[0]),
            (&[prepared[1], prepared[0]], &srcs[1]),
        ],
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
