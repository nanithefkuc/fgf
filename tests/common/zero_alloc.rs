use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

struct CountingAllocator;

// Per-thread tracking: a process-global counter would also count allocations
// made by other libtest threads (result formatting, timers) that race into the
// counted window, producing false positives under CI load. Thread-locals keep
// the count to the test thread alone.
thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
pub(super) static TEST_LOCK: Mutex<()> = Mutex::new(());

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

#[path = "../../src/kernel/tests/fixtures.rs"]
mod fixtures;

pub(super) use fixtures::noise_odd_seed as noise;

pub(super) fn count_allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    body();
    TRACKING.with(|t| t.set(false));
    ALLOCATIONS.with(|c| c.get())
}
