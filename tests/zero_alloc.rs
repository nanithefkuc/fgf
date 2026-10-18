#![cfg(feature = "std")]

use fgf::poly::{AES, RS};
use fgf::{
    Components, Elem, Embedding, Extension, ExtensionSpec, FieldBuffer, FieldElem, FieldKernels,
    FiniteField, Goldilocks, Isomorphism, Mersenne31, Normal, Packed, Polynomial, QuadMersenne31,
    Relation, Residue, ops,
};

#[path = "common/zero_alloc.rs"]
mod common;
use common::{TEST_LOCK, count_allocations, noise};

type Gf1 = fgf::FiniteField<2, 1, fgf::Polynomial<3>>;
type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;
type FanPaar64Field = fgf::FiniteField<2, 64, fgf::Extension<fgf::FanPaar64, 2>>;
type FanPaar8Row = fgf::FiniteField<2, 8, fgf::Extension<fgf::FanPaar8, 2>>;
type ResiduePrime = FiniteField<1_000_000_007, 1, Residue>;
type ResidueTwo = FiniteField<2, 1, Residue>;

/// A custom degree-32 extension over GF(2^16): `t^2 + t + 0x2001`, distinct
/// from every pinned presentation.
#[derive(Clone, Copy)]
struct Custom32Spec;
impl ExtensionSpec<2> for Custom32Spec {
    type Base = Gf16;
    type Storage = Packed<u32>;
    const NAME: &'static str = "custom GF(2^32) extension";
    const RELATION: Relation = Relation::from_binary_words::<Gf16>(&[0x2001, 0x0001]);
}
type Custom32Row = FiniteField<2, 32, Extension<Custom32Spec, 2>>;

/// Steady-state prepared application over one fallback class: preparation
/// allocates by contract, application must not.
#[allow(clippy::too_many_lines)]
fn assert_fallback_steady_state_zero_alloc<F: FieldKernels>(coeffs: &[Elem<F>]) {
    const ROW_LEN: usize = 1024;
    let src = noise(ROW_LEN, 0xF00);
    let sources: Vec<Vec<u8>> = (0..coeffs.len())
        .map(|index| noise(ROW_LEN, 0xF10 + index as u64))
        .collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let vector = ops::CoeffVec::<F>::new(coeffs);
    let flat: Vec<Elem<F>> = (0..2).flat_map(|_| coeffs.iter().copied()).collect();
    let matrix = ops::CoeffMatrix::<F>::from_source_major(2, coeffs.len(), &flat);
    let matrix_refs: Vec<&[u8]> = vec![src.as_slice(), src.as_slice()];
    let nrows = coeffs.len();
    let mut rows = noise(ROW_LEN * nrows, 0xF20);
    let mut row = noise(ROW_LEN, 0xF30);

    // Resolve backend selection and warm every path before counting.
    ops::mul_add_scatter::<F>(&mut rows, ROW_LEN, coeffs, &src);
    ops::mul_add_gather::<F>(&mut row, coeffs, &refs);
    ops::mul_add_scatter_with::<F>(&mut rows, ROW_LEN, vector.as_ref(), &src);
    ops::mul_add_gather_with::<F>(&mut row, vector.as_ref(), &refs);
    ops::mul_into_gather_with::<F>(&mut row, vector.as_ref(), &refs);
    ops::mul_add_matrix_with::<F>(&mut rows, ROW_LEN, &matrix, &matrix_refs);
    ops::mul_into_matrix_with::<F>(&mut rows, ROW_LEN, &matrix, &matrix_refs);

    let allocations = count_allocations(|| {
        ops::mul_add_scatter_with::<F>(&mut rows, ROW_LEN, vector.as_ref(), &src);
        ops::mul_add_gather_with::<F>(&mut row, vector.as_ref(), &refs);
        ops::mul_into_gather_with::<F>(&mut row, vector.as_ref(), &refs);
        ops::mul_add_matrix_with::<F>(&mut rows, ROW_LEN, &matrix, &matrix_refs);
        ops::mul_into_matrix_with::<F>(&mut rows, ROW_LEN, &matrix, &matrix_refs);
    });
    assert_eq!(allocations, 0, "fallback prepared steady state allocated");
}

#[test]
fn basis8_prepared_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let coeffs: Vec<Elem<FiniteField<2, 8, Normal<0x11B, 0x20>>>> = (0..4u8)
        .map(|index| {
            Elem::<FiniteField<2, 8, Normal<0x11B, 0x20>>>::from_raw(index.wrapping_mul(71))
        })
        .collect();
    assert_fallback_steady_state_zero_alloc(&coeffs);
}

#[test]
fn custom_tower32_prepared_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let coeffs: Vec<Elem<Custom32Row>> = (0..4u32)
        .map(|index| Elem::<Custom32Row>::from_raw(index.wrapping_mul(0x1F3B_5A79)))
        .collect();
    assert_fallback_steady_state_zero_alloc(&coeffs);
}

#[test]
fn gf64_prepared_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let coeffs: Vec<Elem<Gf64>> = (0..4u64)
        .map(|index| Elem::<Gf64>::from_raw(index.wrapping_mul(0x00C0_FFEE_00C0_FFEE)))
        .collect();
    assert_fallback_steady_state_zero_alloc(&coeffs);
}

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
            Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                (index as u8).wrapping_mul(37).wrapping_add(2),
            )
        })
        .collect();
    let vector = ops::CoeffVec::<FiniteField<2, 8, Polynomial<AES>>>::new(&coeffs);
    let mut dst = noise(len, 0x800);

    // Resolve backend selection and warm every code path before counting.
    ops::mul_into_gather::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, &coeffs, &refs);
    ops::mul_into_gather_with::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut dst,
        vector.as_ref(),
        &refs,
    );

    let one_shot = count_allocations(|| {
        ops::mul_into_gather::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, &coeffs, &refs)
    });
    assert_eq!(one_shot, 0, "one-shot dot product allocated");

    let prepared = count_allocations(|| {
        ops::mul_into_gather_with::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut dst,
            vector.as_ref(),
            &refs,
        )
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
    let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| {
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                        ((term * 37 + row * 19 + 2) & 0xff) as u8,
                    )
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let flat: Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>> =
        coeff_sets.iter().flatten().copied().collect();
    let matrix = ops::CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(
        NTERMS, NROWS, &flat,
    );
    let mut rows = noise(ROW_LEN * NROWS, 0xa00);
    let gap = ROW_LEN / 2;
    let stride = ROW_LEN + gap;
    let row_starts: Vec<usize> = (0..NROWS).map(|row| row * stride).collect();
    let mut frame = noise(stride * NROWS, 0xa01);

    // Resolve backend selection and warm both paths before counting.
    ops::mul_into_matrix::<FiniteField<2, 8, Polynomial<AES>>>(&mut rows, ROW_LEN, NROWS, &terms);
    ops::mul_into_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut rows, ROW_LEN, &matrix, &refs,
    );
    ops::mul_into_matrix_at::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut frame,
        ROW_LEN,
        &row_starts,
        &terms,
    );
    ops::mul_into_matrix_at_with::<FiniteField<2, 8, Polynomial<AES>>>(
        &mut frame,
        ROW_LEN,
        &row_starts,
        &matrix,
        &refs,
    );

    let one_shot = count_allocations(|| {
        ops::mul_into_matrix::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut rows, ROW_LEN, NROWS, &terms,
        );
    });
    assert_eq!(one_shot, 0, "one-shot overwrite matrix allocated");

    let prepared = count_allocations(|| {
        ops::mul_into_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut rows, ROW_LEN, &matrix, &refs,
        );
    });
    assert_eq!(prepared, 0, "prepared overwrite matrix allocated");

    let scattered = count_allocations(|| {
        ops::mul_into_matrix_at::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut frame,
            ROW_LEN,
            &row_starts,
            &terms,
        );
        ops::mul_add_matrix_at::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut frame,
            ROW_LEN,
            &row_starts,
            &terms,
        );
        ops::mul_into_matrix_at_with::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut frame,
            ROW_LEN,
            &row_starts,
            &matrix,
            &refs,
        );
        ops::mul_add_matrix_at_with::<FiniteField<2, 8, Polynomial<AES>>>(
            &mut frame,
            ROW_LEN,
            &row_starts,
            &matrix,
            &refs,
        );
    });
    assert_eq!(scattered, 0, "scattered matrix steady state allocated");
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
    let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| {
                    Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(
                        ((term * 41 + row * 23 + 3) & 0xff) as u8,
                    )
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<RS>>>], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let mut rows = noise(ROW_LEN * NROWS, 0xc00);

    ops::mul_into_matrix::<FiniteField<2, 8, Polynomial<RS>>>(&mut rows, ROW_LEN, NROWS, &terms);
    let steady = count_allocations(|| {
        ops::mul_into_matrix::<FiniteField<2, 8, Polynomial<RS>>>(
            &mut rows, ROW_LEN, NROWS, &terms,
        );
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
    let flat: Vec<Elem<FiniteField<2, 8, Polynomial<RS>>>> = (0..NTERMS)
        .flat_map(|term| {
            (0..NROWS).map(move |row| {
                Elem::<FiniteField<2, 8, Polynomial<RS>>>::from_raw(
                    ((term * 43 + row * 7 + 5) & 0xff) as u8,
                )
            })
        })
        .collect();
    let matrix = ops::CoeffMatrix::<FiniteField<2, 8, Polynomial<RS>>>::from_source_major(
        NTERMS, NROWS, &flat,
    );
    let mut rows = noise(ROW_LEN * NROWS, 0xe00);

    ops::mul_into_matrix_with::<FiniteField<2, 8, Polynomial<RS>>>(
        &mut rows, ROW_LEN, &matrix, &refs,
    );
    ops::mul_add_matrix_with::<FiniteField<2, 8, Polynomial<RS>>>(
        &mut rows, ROW_LEN, &matrix, &refs,
    );

    let overwrite = count_allocations(|| {
        ops::mul_into_matrix_with::<FiniteField<2, 8, Polynomial<RS>>>(
            &mut rows, ROW_LEN, &matrix, &refs,
        );
    });
    assert_eq!(
        overwrite, 0,
        "reed-solomon prepared overwrite matrix allocated"
    );

    let accumulate = count_allocations(|| {
        ops::mul_add_matrix_with::<FiniteField<2, 8, Polynomial<RS>>>(
            &mut rows, ROW_LEN, &matrix, &refs,
        );
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

/// Residue rows: the borrowed steady-state path is allocation-free, the
/// characteristic-two row included.
#[test]
fn residue_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_prime_steady_state_zero_alloc::<ResiduePrime>(Elem::<ResiduePrime>::from_raw(7));
    assert_prime_steady_state_zero_alloc::<ResidueTwo>(Elem::<ResidueTwo>::ONE);
}

/// Residue rows: prepared coefficients resolve into the row's one-word
/// backend form (the canonical element on byte rows, the Shoup pair on
/// word rows, the Montgomery word on doubleword rows), so preparation and
/// construction allocate nothing and every prepared application runs
/// allocation-free. Construction sits outside the counted interval; the
/// counted window covers the reused application paths.
#[test]
fn residue_prepared_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let coeffs: Vec<Elem<ResiduePrime>> = (0..4u64)
        .map(|index| Elem::<ResiduePrime>::from_raw(index.wrapping_mul(71) + 1))
        .collect();
    assert_fallback_steady_state_zero_alloc::<ResiduePrime>(&coeffs);

    // A single prepared coefficient reused across short buffers: the
    // counted window starts after construction.
    let src = noise(64, 0x1500);
    let mut dst = noise(64, 0x1501);
    let prepared = ops::Coeff::<ResiduePrime>::new(coeffs[2]);
    ops::mul_add_with::<ResiduePrime>(&mut dst, &prepared, &src);
    let allocations = count_allocations(|| {
        ops::mul_add_with::<ResiduePrime>(&mut dst, &prepared, &src);
        ops::mul_into_with::<ResiduePrime>(&mut dst, &prepared, &src);
    });
    assert_eq!(allocations, 0, "residue prepared reuse allocated");
}

/// Warmed prepared residue ops on one row per width plus the overwrite
/// matrix on a word row: preparation happens before the counted window,
/// every application inside it allocates nothing.
#[test]
fn residue_prepared_widths_steady_state_allocate_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    type Byte = FiniteField<101, 1, Residue>;
    type Short = FiniteField<65_521, 1, Residue>;
    type Full = FiniteField<4_294_967_291, 1, Residue>;
    type Wide = FiniteField<0xFFFF_FFFF_FFFF_FFC5, 1, Residue>;
    fn warmed<F: FieldKernels>(coeff: Elem<F>) {
        let row_len = 64 * F::BYTES;
        let src = noise(row_len, 0x1600);
        let mut dst = noise(row_len, 0x1601);
        let prepared = ops::Coeff::<F>::new(coeff);
        ops::mul_into_with::<F>(&mut dst, &prepared, &src);
        ops::mul_add_with::<F>(&mut dst, &prepared, &src);
        let allocations = count_allocations(|| {
            ops::mul_into_with::<F>(&mut dst, &prepared, &src);
            ops::mul_add_with::<F>(&mut dst, &prepared, &src);
        });
        assert_eq!(allocations, 0, "warmed residue prepared ops allocated");
    }
    warmed::<Byte>(Elem::<Byte>::from_raw(37));
    warmed::<Short>(Elem::<Short>::from_raw(41_000));
    warmed::<ResiduePrime>(Elem::<ResiduePrime>::from_raw(123_456_789));
    warmed::<Full>(Elem::<Full>::from_raw(4_000_000_007));
    warmed::<Wide>(Elem::<Wide>::from_raw(0xFFFF_FFFF_FFFF_FFC4));
    warmed::<ResidueTwo>(Elem::<ResidueTwo>::ONE);

    // Overwrite matrix on a word row.
    let row_len = 64 * ResiduePrime::BYTES;
    let sources = [noise(row_len, 0x1610), noise(row_len, 0x1611)];
    let refs: [&[u8]; 2] = [&sources[0], &sources[1]];
    let flat = [
        Elem::<ResiduePrime>::from_raw(3),
        Elem::<ResiduePrime>::from_raw(5),
        Elem::<ResiduePrime>::from_raw(7),
        Elem::<ResiduePrime>::from_raw(11),
    ];
    let matrix = ops::CoeffMatrix::<ResiduePrime>::from_source_major(2, 2, &flat);
    let mut rows = noise(row_len * 2, 0x1612);
    ops::mul_into_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
    let allocations = count_allocations(|| {
        ops::mul_into_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
    });
    assert_eq!(allocations, 0, "residue overwrite matrix allocated");
}

/// Warmed routed delayed-reduction matrix on a four-byte row: four
/// outputs, 256-byte rows, sixteen sources at `K = 8`, every output column
/// nonzero. That depth routes through the delayed tile on capable hosts
/// for both overwrite and accumulation, and allocates nothing in steady
/// state.
#[test]
fn residue_delayed_matrix_steady_state_allocates_nothing() {
    type DelayedPrime = FiniteField<536_870_909, 1, Residue>;
    const SOURCES: usize = 16;
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let row_len = 64 * DelayedPrime::BYTES;
    let sources: Vec<Vec<u8>> = (0..SOURCES as u64)
        .map(|index| noise(row_len, 0x1620 + index))
        .collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let flat: Vec<Elem<DelayedPrime>> = (0..4 * SOURCES as u64)
        .map(|index| Elem::<DelayedPrime>::from_raw(index.wrapping_mul(71) + 1))
        .collect();
    let matrix = ops::CoeffMatrix::<DelayedPrime>::from_source_major(SOURCES, 4, &flat);
    let mut rows = noise(row_len * 4, 0x1623);
    ops::mul_into_matrix_with::<DelayedPrime>(&mut rows, row_len, &matrix, &refs);
    ops::mul_add_matrix_with::<DelayedPrime>(&mut rows, row_len, &matrix, &refs);
    let allocations = count_allocations(|| {
        ops::mul_into_matrix_with::<DelayedPrime>(&mut rows, row_len, &matrix, &refs);
        ops::mul_add_matrix_with::<DelayedPrime>(&mut rows, row_len, &matrix, &refs);
    });
    assert_eq!(allocations, 0, "residue delayed matrix allocated");
}

/// A fallback-geometry residue matrix stays allocation-free: one output
/// row never reaches the delayed tile.
#[test]
fn residue_fallback_matrix_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let row_len = 64 * ResiduePrime::BYTES;
    let sources = [
        noise(row_len, 0x1630),
        noise(row_len, 0x1631),
        noise(row_len, 0x1632),
    ];
    let refs: [&[u8]; 3] = [&sources[0], &sources[1], &sources[2]];
    let flat: Vec<Elem<ResiduePrime>> = (0..3u64)
        .map(|index| Elem::<ResiduePrime>::from_raw(index.wrapping_mul(71) + 1))
        .collect();
    let matrix = ops::CoeffMatrix::<ResiduePrime>::from_source_major(3, 1, &flat);
    let mut rows = noise(row_len, 0x1633);
    ops::mul_into_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
    ops::mul_add_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
    let allocations = count_allocations(|| {
        ops::mul_into_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
        ops::mul_add_matrix_with::<ResiduePrime>(&mut rows, row_len, &matrix, &refs);
    });
    assert_eq!(allocations, 0, "residue fallback matrix allocated");
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
    ops::add_assign_rows::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, row_len, &src);
    ops::add_assign_rows::<Mersenne31>(&mut dst_p, row_len, &src_p);

    let binary = count_allocations(|| {
        ops::add_assign_rows::<FiniteField<2, 8, Polynomial<AES>>>(&mut dst, row_len, &src);
    });
    assert_eq!(binary, 0, "row-interleaved binary add allocated");

    let prime = count_allocations(|| {
        ops::add_assign_rows::<Mersenne31>(&mut dst_p, row_len, &src_p);
    });
    assert_eq!(prime, 0, "prime-field row add allocated");
}

#[test]
fn embedding_construction_and_application_allocate_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let byte_tower = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf64>::new().unwrap();
    let fanpaar_tower = Embedding::<FanPaar8Row, Gf64>::new().unwrap();
    let absolute = Embedding::<Gf1, Gf64>::new().unwrap();
    let source = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53);
    let target = Elem::<Gf64>::from_raw(u64::MAX);

    // Warm every method before counting.
    let lifted = byte_tower.embed(source);
    let _ = byte_tower.contains(lifted);
    let _ = byte_tower.restrict(lifted);
    let _ = byte_tower.frobenius(target);
    let _ = byte_tower.trace(target);
    let _ = byte_tower.norm(target);
    let _ = fanpaar_tower.trace(target);
    let _ = absolute.norm(target);
    let fanpaar_pair = Isomorphism::<FanPaar64Field, Gf64>::new().unwrap();
    let source_pair = Elem::<FanPaar64Field>::from_raw(u64::MAX);
    let _ = fanpaar_pair.apply(source_pair);
    let _ = fanpaar_pair.unapply(target);

    let constructions = count_allocations(|| {
        let _ = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf64>::new().unwrap();
        let _ = Embedding::<FanPaar8Row, Gf64>::new().unwrap();
        let _ = Embedding::<Gf1, Gf64>::new().unwrap();
        let _ = Embedding::<Gf64, FiniteField<2, 8, Polynomial<AES>>>::new();
        let _ = Isomorphism::<FanPaar64Field, Gf64>::new().unwrap();
    });
    assert_eq!(constructions, 0, "embedding construction allocated");

    let applications = count_allocations(|| {
        let lifted = byte_tower.embed(source);
        let _ = byte_tower.contains(lifted);
        let _ = byte_tower.restrict(lifted);
        let _ = byte_tower.frobenius(target);
        let _ = byte_tower.trace(target);
        let _ = byte_tower.norm(target);
        let _ = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf64>::generator_count();
        let _ = byte_tower.relative_degree();
        let _ = fanpaar_tower.trace(target);
        let _ = fanpaar_tower.norm(target);
        let _ = absolute.trace(target);
        let _ = absolute.norm(target);
        // The generic relation surface: image construction, composition,
        // and the isomorphism round trip.
        let rebuilt = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf64>::from_images(
            byte_tower.to_images(),
        )
        .unwrap();
        let _ = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf64>::generator(0);
        let wide = Isomorphism::<Gf64, Gf64>::new().unwrap();
        let _ = rebuilt.then(&wide);
        let byte_iso = Isomorphism::<
            FiniteField<2, 8, Polynomial<AES>>,
            FiniteField<2, 8, Polynomial<AES>>,
        >::new()
        .unwrap();
        let _ = byte_iso.then_embedding(&byte_tower);
        let iso = Isomorphism::<Gf64, Gf64>::new().unwrap();
        let _ = iso.apply(target);
        let _ = iso.unapply(target);
        let _ = iso.inverse();
        let _ = iso.to_images();
        let _ = Isomorphism::<Gf64, Gf64>::from_images(iso.to_images()).unwrap();
        let _ = iso.then(&iso);
        // An odd-characteristic isomorphism under the allocator.
        let qiso = Isomorphism::<QuadMersenne31, QuadMersenne31>::new().unwrap();
        let z = Elem::<QuadMersenne31>::from_raw([3, 5]);
        let _ = qiso.apply(z);
        let _ = qiso.unapply(qiso.apply(z));
        let _ = qiso.inverse();
        let _ = qiso.to_images();
    });
    assert_eq!(applications, 0, "embedding application allocated");
}

/// A generic odd-characteristic extension row: GF(49) as `t^2 - 3` over
/// the residue row at seven, component storage with a byte form.
#[derive(Clone, Copy)]
struct Gf49Spec;
impl ExtensionSpec<2> for Gf49Spec {
    type Base = FiniteField<7, 1>;
    type Storage = Components;
    const NAME: &'static str = "GF(49)";
    const RELATION: Relation = Relation::from_coordinates(&[4, 0]);
}
type Gf49Row = FiniteField<7, 2, Extension<Gf49Spec, 2>>;

/// A generic characteristic-two extension row: GF(2^16) as `t^2 + t +
/// 0x21` over the AES field, packed storage, distinct from every pinned
/// presentation.
#[derive(Clone, Copy)]
struct Packed16Spec;
impl ExtensionSpec<2> for Packed16Spec {
    type Base = FiniteField<2, 8, Polynomial<AES>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<AES>>>(&[0x21, 0x01]);
}
type Packed16Row = FiniteField<2, 16, Extension<Packed16Spec, 2>>;

/// The whole checked packed surface the generic family serves, inside one
/// counted window per row: every borrowed operation, the scatter, matrix,
/// and offset-addressed forms, the broadcast scalar forms, and the
/// prepared coefficient and prepared collections.
#[allow(clippy::too_many_lines)]
fn assert_extension_packed_surface_zero_alloc<F: FieldKernels>(coeff: Elem<F>) {
    let row_len = 64;
    let src = noise(row_len, 0x611);
    let other = noise(row_len, 0x622);
    let mut dst = noise(row_len, 0x633);
    let mut gather_dst = noise(row_len, 0x644);
    let mut ew = noise(row_len, 0x655);
    let region = noise(4 * row_len, 0x666);
    let sources = [
        noise(row_len, 0x671),
        noise(row_len, 0x672),
        noise(row_len, 0x673),
        noise(row_len, 0x674),
    ];
    let refs: [&[u8]; 4] = [&sources[0], &sources[1], &sources[2], &sources[3]];
    let coeffs = [coeff; 4];
    let ones = [Elem::<F>::ONE; 4];
    let mut scatter_rows = noise(4 * row_len, 0x681);
    let mut matrix_rows = noise(4 * row_len, 0x682);
    let mut at_rows = noise(4 * row_len, 0x683);
    let row_starts = [0, row_len, 2 * row_len, 3 * row_len];
    let offsets = [0u32, 64, 128, 192];
    let terms = [(&coeffs[..], &src[..]), (&ones[..], &other[..])];
    let prepared = ops::Coeff::<F>::new(coeff);
    let vector = ops::CoeffVec::<F>::new(&coeffs);
    let flat = [coeff; 8];
    let matrix = ops::CoeffMatrix::<F>::from_source_major(2, 4, &flat);
    let matrix_refs: [&[u8]; 2] = [&src, &other];

    // Warm dispatch and every path before counting.
    ops::add_assign::<F>(&mut dst, &src);
    ops::add_assign_rows::<F>(&mut dst, row_len, &src);
    ops::add_gather_offsets::<F>(&mut dst, &region, &offsets);
    ops::sub_assign::<F>(&mut dst, &src);
    ops::mul_add::<F>(&mut dst, coeff, &src);
    ops::mul_into::<F>(&mut ew, coeff, &src);
    ops::mul_assign::<F>(&mut ew, coeff);
    ops::mul_add_scatter::<F>(&mut scatter_rows, row_len, &coeffs, &src);
    ops::mul_add_gather::<F>(&mut gather_dst, &coeffs, &refs);
    ops::mul_into_gather::<F>(&mut gather_dst, &coeffs, &refs);
    ops::mul_add_matrix::<F>(&mut matrix_rows, row_len, 4, &terms);
    ops::mul_into_matrix::<F>(&mut matrix_rows, row_len, 4, &terms);
    ops::mul_into_matrix_at::<F>(&mut at_rows, row_len, &row_starts, &terms);
    ops::mul_elementwise::<F>(&mut ew, &src, &other);
    ops::mul_elementwise_assign::<F>(&mut ew, &other);
    ops::add_assign_scalar::<F>(&mut dst, coeff);
    ops::sub_assign_scalar::<F>(&mut dst, coeff);
    ops::mul_add_with::<F>(&mut dst, &prepared, &src);
    ops::mul_into_with::<F>(&mut dst, &prepared, &src);
    ops::mul_assign_with::<F>(&mut dst, &prepared);
    ops::mul_add_scatter_with::<F>(&mut scatter_rows, row_len, vector.as_ref(), &src);
    ops::mul_add_gather_with::<F>(&mut gather_dst, vector.as_ref(), &refs);
    ops::mul_into_gather_with::<F>(&mut gather_dst, vector.as_ref(), &refs);
    ops::mul_add_matrix_with::<F>(&mut matrix_rows, row_len, &matrix, &matrix_refs);
    ops::mul_add_matrix_at_with::<F>(&mut at_rows, row_len, &row_starts, &matrix, &matrix_refs);
    ops::mul_into_matrix_at_with::<F>(&mut at_rows, row_len, &row_starts, &matrix, &matrix_refs);

    let allocations = count_allocations(|| {
        ops::add_assign::<F>(&mut dst, &src);
        ops::add_assign_rows::<F>(&mut dst, row_len, &src);
        ops::add_gather_offsets::<F>(&mut dst, &region, &offsets);
        ops::sub_assign::<F>(&mut dst, &src);
        ops::mul_add::<F>(&mut dst, coeff, &src);
        ops::mul_into::<F>(&mut ew, coeff, &src);
        ops::mul_assign::<F>(&mut ew, coeff);
        ops::mul_add_scatter::<F>(&mut scatter_rows, row_len, &coeffs, &src);
        ops::mul_add_gather::<F>(&mut gather_dst, &coeffs, &refs);
        ops::mul_into_gather::<F>(&mut gather_dst, &coeffs, &refs);
        ops::mul_add_matrix::<F>(&mut matrix_rows, row_len, 4, &terms);
        ops::mul_into_matrix::<F>(&mut matrix_rows, row_len, 4, &terms);
        ops::mul_into_matrix_at::<F>(&mut at_rows, row_len, &row_starts, &terms);
        ops::mul_elementwise::<F>(&mut ew, &src, &other);
        ops::mul_elementwise_assign::<F>(&mut ew, &other);
        ops::add_assign_scalar::<F>(&mut dst, coeff);
        ops::sub_assign_scalar::<F>(&mut dst, coeff);
        ops::mul_add_with::<F>(&mut dst, &prepared, &src);
        ops::mul_into_with::<F>(&mut dst, &prepared, &src);
        ops::mul_assign_with::<F>(&mut dst, &prepared);
        ops::mul_add_scatter_with::<F>(&mut scatter_rows, row_len, vector.as_ref(), &src);
        ops::mul_add_gather_with::<F>(&mut gather_dst, vector.as_ref(), &refs);
        ops::mul_into_gather_with::<F>(&mut gather_dst, vector.as_ref(), &refs);
        ops::mul_add_matrix_with::<F>(&mut matrix_rows, row_len, &matrix, &matrix_refs);
        ops::mul_add_matrix_at_with::<F>(&mut at_rows, row_len, &row_starts, &matrix, &matrix_refs);
        ops::mul_into_matrix_at_with::<F>(
            &mut at_rows,
            row_len,
            &row_starts,
            &matrix,
            &matrix_refs,
        );
    });
    assert_eq!(allocations, 0, "generic extension packed surface allocated");
}

/// Generic extension rows: the whole checked packed surface runs
/// allocation-free, and the prepared coefficient is the element itself,
/// so preparation outside the counted window buys allocation-free reuse.
#[test]
fn generic_extension_rows_steady_state_allocate_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");

    // Odd characteristic and characteristic two over packed storage.
    let odd_coeff = Elem::<Gf49Row>::from_raw([3, 5]);
    assert_extension_packed_surface_zero_alloc::<Gf49Row>(odd_coeff);
    let binary_coeff = Elem::<Packed16Row>::from_raw(0x1B70);
    assert_extension_packed_surface_zero_alloc::<Packed16Row>(binary_coeff);
}

/// Byte-lane dot products return by value and fold scalar lanes, so steady
/// state allocates nothing on any family.
#[test]
fn bulk_dot_product_steady_state_allocates_nothing() {
    use fgf::FiniteField;
    use fgf::Polynomial;
    use fgf::poly::AES;

    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    type Gf8 = FiniteField<2, 8, Polynomial<AES>>;
    let a = noise(1024, 0xd20);
    let b = noise(1024, 0xd21);
    let a31 = noise(1020, 0xd22);
    let b31 = noise(1020, 0xd23);
    // Warm dispatch before counting.
    let _ = ops::dot_product::<Gf8>(&a, &b);
    let _ = ops::dot_product::<Mersenne31>(&a31, &b31);
    let _ = ops::dot_product::<Goldilocks>(&a, &b);

    let allocations = count_allocations(|| {
        let _ = ops::dot_product::<Gf8>(&a, &b);
        let _ = ops::dot_product::<Mersenne31>(&a31, &b31);
        let _ = ops::dot_product::<Goldilocks>(&a, &b);
        let _ = ops::dot_product::<ResiduePrime>(&a31, &b31);
    });
    assert_eq!(allocations, 0, "byte-lane dot product allocated");
}
