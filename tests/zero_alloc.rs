#![cfg(feature = "std")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fgf::{Gf8B, gf8b, ops};

struct CountingAllocator;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the allocator contract unchanged to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
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
    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    body();
    TRACKING.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn dot_product_steady_state_allocates_nothing() {
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
