#![cfg(feature = "std")]

use fgf::poly::{AES, REED_SOLOMON};
use fgf::{Elem, FieldKernels, Gf, Gf8, Goldilocks, Mersenne31, Poly, ops};

#[path = "common/zero_alloc.rs"]
mod common;
use common::{TEST_LOCK, count_allocations, noise};

#[test]
fn mul_into_gather_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let len = 96;
    let sources: Vec<Vec<u8>> = (0..8).map(|index| noise(len, 0x700 + index)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs: Vec<_> = (0..8)
        .map(|index| {
            Elem::<Gf<8, Poly<AES>>>::from_raw((index as u8).wrapping_mul(37).wrapping_add(2))
        })
        .collect();
    let vector = ops::CoeffVec::<Gf8<Poly<AES>>>::new(&coeffs);
    let mut dst = noise(len, 0x800);

    // Resolve backend selection and warm every code path before counting.
    ops::mul_into_gather::<Gf8<Poly<AES>>>(&mut dst, &coeffs, &refs);
    ops::mul_into_gather_with::<Gf8<Poly<AES>>>(&mut dst, vector.as_ref(), &refs);

    let one_shot =
        count_allocations(|| ops::mul_into_gather::<Gf8<Poly<AES>>>(&mut dst, &coeffs, &refs));
    assert_eq!(one_shot, 0, "one-shot dot product allocated");

    let prepared = count_allocations(|| {
        ops::mul_into_gather_with::<Gf8<Poly<AES>>>(&mut dst, vector.as_ref(), &refs)
    });
    assert_eq!(prepared, 0, "prepared dot product allocated");
}

#[test]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn mul_into_matrix_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    const ROW_LEN: usize = 4096;
    const NROWS: usize = 4;
    const NTERMS: usize = 10;

    let sources: Vec<Vec<u8>> = (0..NTERMS)
        .map(|index| noise(ROW_LEN, 0x900 + index as u64))
        .collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<AES>>>>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| {
                    Elem::<Gf<8, Poly<AES>>>::from_raw(((term * 37 + row * 19 + 2) & 0xff) as u8)
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<Gf<8, Poly<AES>>>], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let flat: Vec<Elem<Gf<8, Poly<AES>>>> = coeff_sets.iter().flatten().copied().collect();
    let matrix = ops::CoeffMatrix::<Gf8<Poly<AES>>>::from_source_major(NTERMS, NROWS, &flat);
    let mut rows = noise(ROW_LEN * NROWS, 0xa00);

    // Resolve backend selection and warm both paths before counting.
    ops::mul_into_matrix::<Gf8<Poly<AES>>>(&mut rows, ROW_LEN, NROWS, &terms);
    ops::mul_into_matrix_with::<Gf8<Poly<AES>>>(&mut rows, ROW_LEN, &matrix, &refs);

    let one_shot = count_allocations(|| {
        ops::mul_into_matrix::<Gf8<Poly<AES>>>(&mut rows, ROW_LEN, NROWS, &terms);
    });
    assert_eq!(one_shot, 0, "one-shot overwrite matrix allocated");

    let prepared = count_allocations(|| {
        ops::mul_into_matrix_with::<Gf8<Poly<AES>>>(&mut rows, ROW_LEN, &matrix, &refs);
    });
    assert_eq!(prepared, 0, "prepared overwrite matrix allocated");
}

#[test]
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn mul_into_matrix_reed_solomon_chunk_boundary_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    // 33 sources straddle the 32-term resolve chunk: the second chunk and its
    // stack scratch must not allocate.
    const ROW_LEN: usize = 4096;
    const NROWS: usize = 4;
    const NTERMS: usize = 33;

    let sources: Vec<Vec<u8>> = (0..NTERMS)
        .map(|index| noise(ROW_LEN, 0xb00 + index as u64))
        .collect();
    let coeff_sets: Vec<Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| {
                    Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(
                        ((term * 41 + row * 23 + 3) & 0xff) as u8,
                    )
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<Gf<8, Poly<REED_SOLOMON>>>], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let mut rows = noise(ROW_LEN * NROWS, 0xc00);

    ops::mul_into_matrix::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, NROWS, &terms);
    let steady = count_allocations(|| {
        ops::mul_into_matrix::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, NROWS, &terms);
    });
    assert_eq!(steady, 0, "reed-solomon chunk-boundary matrix allocated");
}

#[test]
fn coeff_matrix_reed_solomon_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    const ROW_LEN: usize = 4096;
    const NROWS: usize = 4;
    const NTERMS: usize = 33;

    let sources: Vec<Vec<u8>> = (0..NTERMS)
        .map(|index| noise(ROW_LEN, 0xd00 + index as u64))
        .collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let flat: Vec<Elem<Gf<8, Poly<REED_SOLOMON>>>> = (0..NTERMS)
        .flat_map(|term| {
            (0..NROWS).map(move |row| {
                Elem::<Gf<8, Poly<REED_SOLOMON>>>::from_raw(
                    ((term * 43 + row * 7 + 5) & 0xff) as u8,
                )
            })
        })
        .collect();
    let matrix =
        ops::CoeffMatrix::<Gf8<Poly<REED_SOLOMON>>>::from_source_major(NTERMS, NROWS, &flat);
    let mut rows = noise(ROW_LEN * NROWS, 0xe00);

    ops::mul_into_matrix_with::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, &matrix, &refs);
    ops::mul_add_matrix_with::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, &matrix, &refs);

    let overwrite = count_allocations(|| {
        ops::mul_into_matrix_with::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, &matrix, &refs);
    });
    assert_eq!(
        overwrite, 0,
        "reed-solomon prepared overwrite matrix allocated"
    );

    let accumulate = count_allocations(|| {
        ops::mul_add_matrix_with::<Gf8<Poly<REED_SOLOMON>>>(&mut rows, ROW_LEN, &matrix, &refs);
    });
    assert_eq!(
        accumulate, 0,
        "reed-solomon prepared accumulate matrix allocated"
    );
}

/// Steady-state prime-field ops must not allocate on the hot path.
fn assert_prime_steady_state_zero_alloc<F: FieldKernels>(coeff: Elem<F>) {
    let len = 4096;
    let src = noise(len, 0x111);
    let b = noise(len, 0x222);
    let mut dst = noise(len, 0x333);
    let mut ew = vec![0u8; len];
    let sources: Vec<Vec<u8>> = (0..4).map(|i| noise(len, 0x400 + i)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs = vec![coeff; 4];

    // Warm dispatch and every code path before counting.
    ops::mul_add::<F>(&mut dst, coeff, &src);
    ops::mul_assign::<F>(&mut dst, coeff);
    ops::add_assign::<F>(&mut dst, &src);
    ops::sub_assign::<F>(&mut dst, &src);
    ops::mul_add_gather::<F>(&mut dst, &coeffs, &refs);
    ops::mul_elementwise::<F>(&mut ew, &src, &b);

    let allocations = count_allocations(|| {
        ops::mul_add::<F>(&mut dst, coeff, &src);
        ops::mul_assign::<F>(&mut dst, coeff);
        ops::add_assign::<F>(&mut dst, &src);
        ops::sub_assign::<F>(&mut dst, &src);
        ops::mul_add_gather::<F>(&mut dst, &coeffs, &refs);
        ops::mul_elementwise::<F>(&mut ew, &src, &b);
    });
    assert_eq!(allocations, 0, "prime steady-state op allocated");
}

#[test]
fn mersenne31_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_prime_steady_state_zero_alloc::<Mersenne31>(Elem::<Mersenne31>::from_raw(7));
}

#[test]
fn goldilocks_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_prime_steady_state_zero_alloc::<Goldilocks>(Elem::<Goldilocks>::from_raw(7));
}

#[test]
fn bits_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let bits = 4096;
    let len = fgf::bits::bytes_for(bits);
    let a = noise(len, 0x100);
    let b = noise(len, 0x200);
    let mut dst = noise(len, 0x300);
    let sources: Vec<Vec<u8>> = (0..4).map(|i| noise(len, 0x400 + i)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();

    // Warm dispatch before counting.
    fgf::bits::xor_assign(&mut dst, &a);
    fgf::bits::xor_range(&mut dst, &a, bits, 100, 4000);
    fgf::bits::and_into(&mut dst, &a, &b);
    fgf::bits::and_assign(&mut dst, &a);
    fgf::bits::andnot_assign(&mut dst, &a);
    fgf::bits::clear_range(&mut dst, bits, 0, 64);
    fgf::bits::set_range(&mut dst, bits, 0, 64);
    let _ = fgf::bits::weight(&dst, bits);
    let _ = fgf::bits::dot_product(&dst, &a, bits);
    fgf::bits::xor_gather(&mut dst, bits, 0b0101, &refs);

    let allocations = count_allocations(|| {
        fgf::bits::xor_assign(&mut dst, &a);
        fgf::bits::xor_range(&mut dst, &a, bits, 100, 4000);
        fgf::bits::and_into(&mut dst, &a, &b);
        fgf::bits::and_assign(&mut dst, &a);
        fgf::bits::andnot_assign(&mut dst, &a);
        fgf::bits::clear_range(&mut dst, bits, 0, 64);
        fgf::bits::set_range(&mut dst, bits, 0, 64);
        let _ = fgf::bits::weight(&dst, bits);
        let _ = fgf::bits::dot_product(&dst, &a, bits);
        fgf::bits::xor_gather(&mut dst, bits, 0b0101, &refs);
    });
    assert_eq!(allocations, 0, "bits steady-state op allocated");
}

#[test]
fn add_assign_rows_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let row_len = 256;
    let rows = 8;
    let len = row_len * rows;
    let src = noise(len, 0xa00);
    let mut dst = noise(len, 0xa01);
    // A prime-field control: `add_assign_rows` keeps the defaulted semantic
    // path there, which must be just as allocation-free.
    let src_p = noise(len, 0xa02);
    let mut dst_p = noise(len, 0xa03);

    // Resolve backend selection and warm every code path before counting.
    ops::add_assign_rows::<Gf8<Poly<AES>>>(&mut dst, row_len, &src);
    ops::add_assign_rows::<Mersenne31>(&mut dst_p, row_len, &src_p);

    let binary = count_allocations(|| {
        ops::add_assign_rows::<Gf8<Poly<AES>>>(&mut dst, row_len, &src);
    });
    assert_eq!(binary, 0, "row-interleaved binary add allocated");

    let prime = count_allocations(|| {
        ops::add_assign_rows::<Mersenne31>(&mut dst_p, row_len, &src_p);
    });
    assert_eq!(prime, 0, "prime-field row add allocated");
}
