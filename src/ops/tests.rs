//! Cross-shape checks at the common parent of the operation modules.

use super::*;
use crate::field::binary::Polynomial;
use crate::field::binary::poly::AES;
use crate::field::binary::tower::Rijndael16Field;
use crate::field::{Elem as E8, FiniteField};

/// `row_len == 0` with otherwise valid geometry is a no-op everywhere —
/// including the scalar backend, whose zero-length row chunking is an
/// empty loop rather than a panic.
#[test]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn zero_row_length_is_a_no_op() {
    let coeffs = [E8::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x07); 3];
    let mut rows = [0xA5u8; 8];
    mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &coeffs, &[]);
    assert_eq!(rows, [0xA5; 8]);

    let coeffs_vec = CoeffVec::<FiniteField<2, 8, Polynomial<AES>>>::new(&coeffs);
    mul_add_scatter_with::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut rows,
        0,
        coeffs_vec.as_ref(),
        &[],
    );
    assert_eq!(rows, [0xA5; 8]);

    let terms: &[(&[E8<FiniteField<2, 8, Polynomial<AES>>>], &[u8])] = &[(&coeffs, &[])];
    mul_add_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, 3, terms);
    assert_eq!(rows, [0xA5; 8]);
    mul_into_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, 3, terms);
    assert_eq!(rows, [0xA5; 8]);
    mul_add_matrix_at::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut rows,
        0,
        &[1, 4],
        &[(&coeffs[..2], &[])],
    );
    assert_eq!(rows, [0xA5; 8]);

    let matrix =
        CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(1, 3, &coeffs);
    mul_add_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &matrix, &[&[]]);
    assert_eq!(rows, [0xA5; 8]);
    mul_into_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, 0, &matrix, &[&[]]);
    assert_eq!(rows, [0xA5; 8]);

    // GF(2^16): the field whose odd-length buffers make the element
    // check matter.
    let wide = [E8::<Rijndael16Field>::from_raw(0x0103); 2];
    let mut rows16 = [0x5Au8; 4];
    mul_add_scatter::<Rijndael16Field>(&mut rows16, 0, &wide, &[]);
    assert_eq!(rows16, [0x5A; 4]);
}
