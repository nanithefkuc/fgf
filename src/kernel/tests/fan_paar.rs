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
