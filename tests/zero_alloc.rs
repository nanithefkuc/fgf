#![cfg(feature = "std")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

use fgf::{Gf8B, gf8b, ops};

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
        .map(|index| gf8b::Elem((index as u8).wrapping_mul(37).wrapping_add(2)))
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
                .map(|row| gf8b::Elem(((term * 37 + row * 19 + 2) & 0xff) as u8))
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
    ops::dot_product_matrix_with::<Gf8B>(&mut rows, ROW_LEN, NROWS, &plan, &refs);

    let one_shot = count_allocations(|| {
        ops::dot_product_matrix::<Gf8B>(&mut rows, ROW_LEN, NROWS, &terms);
    });
    assert_eq!(one_shot, 0, "one-shot overwrite matrix allocated");

    let prepared = count_allocations(|| {
        ops::dot_product_matrix_with::<Gf8B>(&mut rows, ROW_LEN, NROWS, &plan, &refs);
    });
    assert_eq!(prepared, 0, "prepared overwrite matrix allocated");
}
