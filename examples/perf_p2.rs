//! Focused perf harness for the two-output-row overwrite matrix (`p2`) path.
//!
//! ```sh
//! taskset -c 3 perf stat -M TopdownL1,TopdownL2 -- \
//!   cargo run --release --features internals --example perf_p2 -- 8b 2 20000
//! ```
//!
//! argv: field (`8b`|`8d`), nrows, iters. Prints throughput; the tight measured
//! loop dominates the process so `perf stat` over the whole run attributes to
//! the kernel.
//!
//! The candidate kernels are x86 GFNI-only, so the body compiles only on x86;
//! other targets print a stub so `cargo test --all-targets` stays buildable.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use std::hint::black_box;
    use std::time::Instant;

    use fgf::kernel::tables::{ScaleTable, scale_table, scale_table_8d};
    use fgf::{Gf8B, Gf8D, backend, gf8b, gf8d, ops};

    const BYTES: usize = 64 * 1024;
    const SOURCES: usize = 10;

    /// Pack a scale table into ISA-L's contiguous `[lo;16, hi;16]` record.
    fn pack(table: &ScaleTable) -> [u8; 32] {
        let mut packed = [0u8; 32];
        packed[..16].copy_from_slice(&table.lo);
        packed[16..].copy_from_slice(&table.hi);
        packed
    }

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    /// 64-byte aligned buffer matching the comparison harness.
    struct AlignedBuf {
        ptr: *mut u8,
        len: usize,
        layout: std::alloc::Layout,
    }

    impl AlignedBuf {
        fn noise(len: usize, seed: u64) -> Self {
            let layout = std::alloc::Layout::from_size_align(len.max(1), 64).unwrap();
            // SAFETY: nonzero size, valid alignment.
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null());
            let bytes = noise(len, seed);
            // SAFETY: `ptr` owns `len` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len) };
            Self { ptr, len, layout }
        }
        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    pub(super) fn main() {
        let args: Vec<String> = std::env::args().collect();
        let field = args.get(1).map(String::as_str).unwrap_or("8b");
        let nrows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
        let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20_000);
        let variant = args.get(4).map(String::as_str).unwrap_or("prod");

        let sources: Vec<AlignedBuf> = (0..SOURCES)
            .map(|t| AlignedBuf::noise(BYTES, 0xa00 + t as u64))
            .collect();
        let source_refs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();

        let cols_8b: Vec<Vec<gf8b::Elem>> = (0..SOURCES)
            .map(|t| {
                (0..nrows)
                    .map(|r| gf8b::Elem((1 + ((r * SOURCES + t) * 97 + 13) % 255) as u8))
                    .collect()
            })
            .collect();
        let cols_8d: Vec<Vec<gf8d::Elem>> = (0..SOURCES)
            .map(|t| {
                (0..nrows)
                    .map(|r| gf8d::Elem((1 + ((r * SOURCES + t) * 97 + 13) % 255) as u8))
                    .collect()
            })
            .collect();
        let terms_8b: Vec<(&[gf8b::Elem], &[u8])> = cols_8b
            .iter()
            .zip(&source_refs)
            .map(|(c, s)| (c.as_slice(), *s))
            .collect();
        let terms_8d: Vec<(&[gf8d::Elem], &[u8])> = cols_8d
            .iter()
            .zip(&source_refs)
            .map(|(c, s)| (c.as_slice(), *s))
            .collect();

        let packed_8b: Vec<[u8; 32]> = cols_8b
            .iter()
            .flat_map(|c| c.iter().map(|&e| pack(scale_table(e))))
            .collect();
        let packed_8d: Vec<[u8; 32]> = cols_8d
            .iter()
            .flat_map(|c| c.iter().map(|&e| pack(scale_table_8d(e))))
            .collect();

        let mut dst = AlignedBuf::noise(BYTES * nrows, 0xb00);

        eprintln!(
            "backend {} field {field} nrows {nrows} iters {iters} variant {variant}",
            backend().name()
        );

        let srcs = source_refs.as_slice();
        let run = |variant: &str, dst: &mut [u8]| {
            use fgf::kernel::x86::gf8::{
                matrix_overwrite2_8b, matrix_overwrite2_8d, matrix_overwrite2_shuffle_packed,
            };
            match (variant, field) {
                ("prod", "8b") => ops::dot_product_matrix::<Gf8B>(dst, BYTES, nrows, &terms_8b),
                ("prod", "8d") => ops::dot_product_matrix::<Gf8D>(dst, BYTES, nrows, &terms_8d),
                ("t4", "8b") => matrix_overwrite2_8b(dst, BYTES, &terms_8b, false, 4),
                ("t4", "8d") => matrix_overwrite2_8d(dst, BYTES, &terms_8d, false, 4),
                ("t2", "8b") => matrix_overwrite2_8b(dst, BYTES, &terms_8b, false, 2),
                ("t2", "8d") => matrix_overwrite2_8d(dst, BYTES, &terms_8d, false, 2),
                ("nt4", "8b") => matrix_overwrite2_8b(dst, BYTES, &terms_8b, true, 4),
                ("nt4", "8d") => matrix_overwrite2_8d(dst, BYTES, &terms_8d, true, 4),
                ("nt2", "8b") => matrix_overwrite2_8b(dst, BYTES, &terms_8b, true, 2),
                ("nt2", "8d") => matrix_overwrite2_8d(dst, BYTES, &terms_8d, true, 2),
                ("sh1", "8b") => matrix_overwrite2_shuffle_packed(dst, BYTES, &packed_8b, srcs, 1),
                ("sh1", "8d") => matrix_overwrite2_shuffle_packed(dst, BYTES, &packed_8d, srcs, 1),
                ("sh2", "8b") => matrix_overwrite2_shuffle_packed(dst, BYTES, &packed_8b, srcs, 2),
                ("sh2", "8d") => matrix_overwrite2_shuffle_packed(dst, BYTES, &packed_8d, srcs, 2),
                _ => panic!("unknown variant/field {variant}/{field}"),
            }
        };

        // Validate the variant against production before timing.
        if nrows == 2 {
            let mut want = AlignedBuf::noise(BYTES * nrows, 0xc00);
            run("prod", want.as_mut_slice());
            run(variant, dst.as_mut_slice());
            assert_eq!(
                want.as_slice(),
                dst.as_slice(),
                "variant {variant} differs from production"
            );
        }

        // Warmup.
        for _ in 0..64 {
            run(variant, dst.as_mut_slice());
        }

        let start = Instant::now();
        for _ in 0..iters {
            run(variant, black_box(dst.as_mut_slice()));
        }
        let elapsed = start.elapsed();

        let logical = (BYTES * SOURCES * iters) as f64;
        let gib = logical / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        eprintln!("{gib:.2} GiB/s ({:.3?} total)", elapsed);
        // Keep the destination live.
        black_box(dst.as_slice());
    }
}

fn main() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    x86::main();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    eprintln!("perf_p2: x86 GFNI-only; not built for this target");
}
