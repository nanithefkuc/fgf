#![cfg(feature = "std")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

use fgf::{
    FieldKernels, Gf8B, Gf8D, Goldilocks, Mersenne31, gf8b, gf8d, goldilocks, mersenne31, ops,
};

struct CountingAllocator;

// Per-thread tracking: a process-global counter would also count allocations
// made by other libtest threads (result formatting, timers) that race into the
// counted window, producing false positives under CI load. Thread-locals keep
// the count to the test thread alone.
thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
static TEST_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACKING.with(|t| {
            if t.get() {
                ALLOCATIONS.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        TRACKING.with(|t| {
            if t.get() {
                ALLOCATIONS.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TRACKING.with(|t| {
            if t.get() {
                ALLOCATIONS.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

fn count_allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    body();
    TRACKING.with(|t| t.set(false));
    ALLOCATIONS.with(|c| c.get())
}

#[test]
fn dot_product_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let len = 96;
    let sources: Vec<Vec<u8>> = (0..8).map(|index| noise(len, 0x700 + index)).collect();
    let refs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    let coeffs: Vec<_> = (0..8)
        .map(|index| gf8b::Elem::from_raw((index as u8).wrapping_mul(37).wrapping_add(2)))
        .collect();
    let plan = ops::Plan::<Gf8B>::new(&coeffs);
    let mut dst = noise(len, 0x800);

    // Resolve backend selection and warm every code path before counting.
    ops::dot_product::<Gf8B>(&mut dst, &coeffs, &refs);
    ops::dot_product_with::<Gf8B>(&mut dst, &plan, &refs);

    let one_shot = count_allocations(|| ops::dot_product::<Gf8B>(&mut dst, &coeffs, &refs));
    assert_eq!(one_shot, 0, "one-shot dot product allocated");

    let prepared = count_allocations(|| ops::dot_product_with::<Gf8B>(&mut dst, &plan, &refs));
    assert_eq!(prepared, 0, "prepared dot product allocated");
}

#[test]
fn dot_product_matrix_steady_state_allocates_nothing() {
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
    let coeff_sets: Vec<Vec<gf8b::Elem>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| gf8b::Elem::from_raw(((term * 37 + row * 19 + 2) & 0xff) as u8))
                .collect()
        })
        .collect();
    let terms: Vec<(&[gf8b::Elem], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let flat: Vec<gf8b::Elem> = coeff_sets.iter().flatten().copied().collect();
    let plan = ops::Plan::<Gf8B>::matrix(NTERMS, NROWS, &flat);
    let mut rows = noise(ROW_LEN * NROWS, 0xa00);

    // Resolve backend selection and warm both paths before counting.
    ops::dot_product_matrix::<Gf8B>(&mut rows, ROW_LEN, NROWS, &terms);
    ops::dot_product_matrix_with::<Gf8B>(&mut rows, ROW_LEN, &plan, &refs);

    let one_shot = count_allocations(|| {
        ops::dot_product_matrix::<Gf8B>(&mut rows, ROW_LEN, NROWS, &terms);
    });
    assert_eq!(one_shot, 0, "one-shot overwrite matrix allocated");

    let prepared = count_allocations(|| {
        ops::dot_product_matrix_with::<Gf8B>(&mut rows, ROW_LEN, &plan, &refs);
    });
    assert_eq!(prepared, 0, "prepared overwrite matrix allocated");
}

#[test]
fn dot_product_matrix_gf8d_chunk_boundary_allocates_nothing() {
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
    let coeff_sets: Vec<Vec<gf8d::Elem>> = (0..NTERMS)
        .map(|term| {
            (0..NROWS)
                .map(|row| gf8d::Elem::from_raw(((term * 41 + row * 23 + 3) & 0xff) as u8))
                .collect()
        })
        .collect();
    let terms: Vec<(&[gf8d::Elem], &[u8])> = coeff_sets
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let mut rows = noise(ROW_LEN * NROWS, 0xc00);

    ops::dot_product_matrix::<Gf8D>(&mut rows, ROW_LEN, NROWS, &terms);
    let steady = count_allocations(|| {
        ops::dot_product_matrix::<Gf8D>(&mut rows, ROW_LEN, NROWS, &terms);
    });
    assert_eq!(steady, 0, "gf8d chunk-boundary matrix allocated");
}

#[test]
fn plan_matrix_gf8d_steady_state_allocates_nothing() {
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
    let flat: Vec<gf8d::Elem> = (0..NTERMS)
        .flat_map(|term| {
            (0..NROWS)
                .map(move |row| gf8d::Elem::from_raw(((term * 43 + row * 7 + 5) & 0xff) as u8))
        })
        .collect();
    let plan = ops::Plan::<Gf8D>::matrix(NTERMS, NROWS, &flat);
    let mut rows = noise(ROW_LEN * NROWS, 0xe00);

    ops::dot_product_matrix_with::<Gf8D>(&mut rows, ROW_LEN, &plan, &refs);
    ops::mul_add_matrix_with::<Gf8D>(&mut rows, ROW_LEN, &plan, &refs);

    let overwrite = count_allocations(|| {
        ops::dot_product_matrix_with::<Gf8D>(&mut rows, ROW_LEN, &plan, &refs);
    });
    assert_eq!(overwrite, 0, "gf8d prepared overwrite matrix allocated");

    let accumulate = count_allocations(|| {
        ops::mul_add_matrix_with::<Gf8D>(&mut rows, ROW_LEN, &plan, &refs);
    });
    assert_eq!(accumulate, 0, "gf8d prepared accumulate matrix allocated");
}

/// Steady-state prime-field ops must not allocate on the hot path.
fn assert_prime_steady_state_zero_alloc<F: FieldKernels>(coeff: F::Elem) {
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
    assert_prime_steady_state_zero_alloc::<Mersenne31>(mersenne31::Elem::from_raw(7));
}

#[test]
fn goldilocks_steady_state_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    assert_prime_steady_state_zero_alloc::<Goldilocks>(goldilocks::Elem::from_raw(7));
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
    fgf::bits::xor(&mut dst, &a);
    fgf::bits::xor_range(&mut dst, &a, bits, 100, 4000);
    fgf::bits::and_into(&mut dst, &a, &b);
    fgf::bits::and_assign(&mut dst, &a);
    fgf::bits::andnot_assign(&mut dst, &a);
    fgf::bits::clear_range(&mut dst, bits, 0, 64);
    fgf::bits::set_range(&mut dst, bits, 0, 64);
    let _ = fgf::bits::weight(&dst, bits);
    let _ = fgf::bits::parity_dot(&dst, &a, bits);
    fgf::bits::xor_gather(&mut dst, &refs, 0b0101, bits);

    let allocations = count_allocations(|| {
        fgf::bits::xor(&mut dst, &a);
        fgf::bits::xor_range(&mut dst, &a, bits, 100, 4000);
        fgf::bits::and_into(&mut dst, &a, &b);
        fgf::bits::and_assign(&mut dst, &a);
        fgf::bits::andnot_assign(&mut dst, &a);
        fgf::bits::clear_range(&mut dst, bits, 0, 64);
        fgf::bits::set_range(&mut dst, bits, 0, 64);
        let _ = fgf::bits::weight(&dst, bits);
        let _ = fgf::bits::parity_dot(&dst, &a, bits);
        fgf::bits::xor_gather(&mut dst, &refs, 0b0101, bits);
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
    ops::add_assign_rows::<Gf8B>(&mut dst, &src, row_len);
    ops::add_assign_rows::<Mersenne31>(&mut dst_p, &src_p, row_len);

    let binary = count_allocations(|| {
        ops::add_assign_rows::<Gf8B>(&mut dst, &src, row_len);
    });
    assert_eq!(binary, 0, "row-interleaved binary add allocated");

    let prime = count_allocations(|| {
        ops::add_assign_rows::<Mersenne31>(&mut dst_p, &src_p, row_len);
    });
    assert_eq!(prime, 0, "prime-field row add allocated");
}

/// The token-proven `internals` wrappers replaced eager `format!`
/// validation with `format_args`; successful direct calls must stay
/// allocation-free, including many-source gather and overwrite-matrix
/// shapes. Tables, tokens, terms, and buffers are prepared before the
/// counting window; observable output is asserted outside it.
#[test]
#[cfg(all(
    feature = "internals",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn proven_internals_gather_and_overwrite_allocate_nothing() {
    use fgf::kernel::tables::{TowerCoeff, TowerTables};
    use fgf::kernel::{SimdToken, X64V3GfniCryptoToken, x86};
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
    x86::proven::gf8::gather_gfni(token, &mut gather_dst, &coeffs, &srcs);
    x86::proven::gf8::matrix_overwrite_gfni(token, &mut matrix_rows, LEN, NROWS, &terms);
    gather_dst = noise(LEN, 0xa40);

    let gather_count = count_allocations(|| {
        x86::proven::gf8::gather_gfni(token, &mut gather_dst, &coeffs, &srcs);
    });
    assert_eq!(gather_count, 0, "proven gather allocated");

    let matrix_count = count_allocations(|| {
        x86::proven::gf8::matrix_overwrite_gfni(token, &mut matrix_rows, LEN, NROWS, &terms);
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
        ops::dot_product::<Gf8B>(target, &row_coeffs, &srcs[..TERMS]);
    }
    assert_eq!(matrix_rows, matrix_want, "proven overwrite matrix output");

    // Tower kernels: the GFNI tower coefficient and the shuffle tables are
    // built outside the window; the calls themselves must not allocate.
    let tower_src = noise(512, 0xa60);
    let mut tower_dst = noise(512, 0xa61);
    let coeff = gf16::Elem::from_raw(0xbeef);
    let tower_coeff = TowerCoeff::new(coeff);
    let tower_tables = TowerTables::new(coeff);
    x86::proven::gf16::mul_add_gfni(token, &mut tower_dst, tower_coeff, &tower_src);
    let tower_count = count_allocations(|| {
        x86::proven::gf16::mul_add_gfni(token, &mut tower_dst, tower_coeff, &tower_src);
    });
    assert_eq!(tower_count, 0, "proven gf16 mul_add allocated");
    let v3 = token.v3();
    x86::proven::gf16::mul_add_avx2(v3, &mut tower_dst, &tower_tables, &tower_src);
    let tower_table_count = count_allocations(|| {
        x86::proven::gf16::mul_add_avx2(v3, &mut tower_dst, &tower_tables, &tower_src);
    });
    assert_eq!(tower_table_count, 0, "proven gf16 table mul_add allocated");
}
