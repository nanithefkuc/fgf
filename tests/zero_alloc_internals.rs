#![cfg(all(
    feature = "std",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]

#[path = "common/zero_alloc.rs"]
mod common;

use common::{TEST_LOCK, count_allocations, noise};
use fgf::{Gf8B, ops};

/// The token-proven `internals` wrappers replaced eager `format!`
/// validation with `format_args`; successful direct calls must stay
/// allocation-free, including many-source gather and overwrite-matrix
/// shapes. Tables, tokens, terms, and buffers are prepared before the
/// counting window; observable output is asserted outside it.
#[test]
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn proven_internals_gather_and_overwrite_allocate_nothing() {
    use fgf::internals::kernel::tables::{TowerCoeff, TowerTables};
    use fgf::internals::kernel::{SimdToken, X64V3GfniCryptoToken, x86};
    use fgf::{gf8b, gf16};

    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let Some(token) = X64V3GfniCryptoToken::summon() else {
        eprintln!("skipping: no AVX2+GFNI on this host");
        return;
    };

    const LEN: usize = 4096;
    const SOURCES: usize = 17;
    const NROWS: usize = 6;
    const TERMS: usize = 5;

    let sources: Vec<Vec<u8>> = (0..SOURCES)
        .map(|index| noise(LEN, 0x900 + index as u64))
        .collect();
    let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs: Vec<gf8b::Elem> = (0..SOURCES)
        .map(|index| gf8b::Elem::from_raw((index as u8).wrapping_mul(37).wrapping_add(2)))
        .collect();

    let columns: Vec<Vec<gf8b::Elem>> = (0..TERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| gf8b::Elem::from_raw(((term * 31 + row * 29) % 256) as u8))
                .collect()
        })
        .collect();
    let terms: Vec<(&[gf8b::Elem], &[u8])> = columns
        .iter()
        .zip(&srcs)
        .map(|(column, src)| (column.as_slice(), *src))
        .collect();
    let mut gather_dst = noise(LEN, 0xa40);
    let mut matrix_rows = noise(NROWS * LEN, 0xa50);

    // Warm dispatch and validation paths before counting, then reset so the
    // counted call folds into the original seed exactly once.
    x86::gf8::mul_add_gather_gfni(token, &mut gather_dst, &coeffs, &srcs);
    x86::gf8::mul_into_matrix_gfni(token, &mut matrix_rows, LEN, NROWS, &terms);
    gather_dst = noise(LEN, 0xa40);

    let gather_count = count_allocations(|| {
        x86::gf8::mul_add_gather_gfni(token, &mut gather_dst, &coeffs, &srcs);
    });
    assert_eq!(gather_count, 0, "proven gather allocated");

    let matrix_count = count_allocations(|| {
        x86::gf8::mul_into_matrix_gfni(token, &mut matrix_rows, LEN, NROWS, &terms);
    });
    assert_eq!(matrix_count, 0, "proven overwrite matrix allocated");

    // Observable output outside the counting window: gather is an AXPY fold
    // into the seeded destination.
    let mut gather_want = noise(LEN, 0xa40);
    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
        ops::mul_add::<Gf8B>(&mut gather_want, coeff, src);
    }
    assert_eq!(gather_dst, gather_want, "proven gather output");

    let mut matrix_want = noise(NROWS * LEN, 0xa50);
    for row in 0..NROWS {
        let row_coeffs: Vec<gf8b::Elem> = (0..TERMS)
            .map(|term| gf8b::Elem::from_raw(((term * 31 + row * 29) % 256) as u8))
            .collect();
        let target = &mut matrix_want[row * LEN..(row + 1) * LEN];
        target.fill(0);
        ops::mul_into_gather::<Gf8B>(target, &row_coeffs, &srcs[..TERMS]);
    }
    assert_eq!(matrix_rows, matrix_want, "proven overwrite matrix output");

    // Tower kernels: the GFNI tower coefficient and the shuffle tables are
    // built outside the window; the calls themselves must not allocate.
    let tower_src = noise(512, 0xa60);
    let mut tower_dst = noise(512, 0xa61);
    let coeff = gf16::Elem::from_raw(0xbeef);
    let tower_coeff = TowerCoeff::new(coeff);
    let tower_tables = TowerTables::new(coeff);
    x86::gf16::mul_add_gfni(token, &mut tower_dst, tower_coeff, &tower_src);
    let tower_count = count_allocations(|| {
        x86::gf16::mul_add_gfni(token, &mut tower_dst, tower_coeff, &tower_src);
    });
    assert_eq!(tower_count, 0, "proven gf16 mul_add allocated");
    let v3 = token.v3();
    x86::gf16::mul_add_avx2(v3, &mut tower_dst, &tower_tables, &tower_src);
    let tower_table_count = count_allocations(|| {
        x86::gf16::mul_add_avx2(v3, &mut tower_dst, &tower_tables, &tower_src);
    });
    assert_eq!(tower_table_count, 0, "proven gf16 table mul_add allocated");
}
