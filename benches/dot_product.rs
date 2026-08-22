//! Direct GF(2^8) N-to-1 baseline: accumulate versus today's overwrite composition.
//!
//! The raw samples bypass public dispatch through `internals`. `zero_then_gather`
//! is intentionally not called a dot-product kernel: it times the current honest
//! overwrite composition, including destination zeroing, until a zero-accumulator
//! body exists. Every fixture is validated before timing.
//!
//! ```sh
//! cargo bench --features internals --bench dot_product
//! cargo bench --features internals --bench dot_product -- --smoke
//! cargo bench --features internals --bench dot_product -- --tails
//! ```

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod imp {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use fgf::kernel::x86::gf8 as x86_gf8;
    use fgf::{Backend, Gf8B, backend_for, gf8b, ops};

    const FULL_LENGTHS: &[usize] = &[
        15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 160, 192, 224, 255, 256,
        257, 511, 512, 513, 1_024, 2_048, 3_072, 4_096, 4_128, 4_160, 4_192, 6_144, 8_192, 16_384,
        32_768, 65_536,
    ];
    const FULL_SOURCE_COUNTS: &[usize] = &[1, 2, 3, 4, 8, 12, 16, 24, 32];
    const SMOKE_LENGTHS: &[usize] = &[
        16, 31, 32, 64, 95, 96, 97, 127, 128, 129, 160, 192, 224, 255, 4_096, 4_160,
    ];
    const SMOKE_SOURCE_COUNTS: &[usize] = &[1, 4, 16];
    const TAIL_LENGTHS: &[usize] = &[16, 32, 64, 96, 128];
    const TAIL_SOURCE_COUNTS: &[usize] = &[1, 2, 3, 4, 8, 16, 32];

    struct AlignedBuf {
        storage: Vec<u8>,
        start: usize,
        len: usize,
    }

    impl AlignedBuf {
        fn noise(len: usize, residue: usize, seed: u64) -> Self {
            assert!(residue < 32);
            let mut storage = vec![0; len + 31];
            let base = storage.as_ptr() as usize;
            let start = (residue + 32 - (base & 31)) & 31;
            let bytes = noise(len, seed);
            storage[start..start + len].copy_from_slice(&bytes);
            let result = Self {
                storage,
                start,
                len,
            };
            assert_eq!((result.as_slice().as_ptr() as usize) & 31, residue);
            result
        }

        fn as_slice(&self) -> &[u8] {
            &self.storage[self.start..self.start + self.len]
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            &mut self.storage[self.start..self.start + self.len]
        }
    }

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

    fn dense_coefficients(count: usize) -> Vec<gf8b::Elem> {
        (0..count)
            .map(|index| gf8b::Elem(2 + ((index * 73 + 19) % 254) as u8))
            .collect()
    }

    fn reference(mut dst: Vec<u8>, coeffs: &[gf8b::Elem], srcs: &[&[u8]]) -> Vec<u8> {
        for (&coefficient, &source) in coeffs.iter().zip(srcs) {
            for (&value, output) in source.iter().zip(&mut dst) {
                *output ^= coefficient.mul(gf8b::Elem(value)).0;
            }
        }
        dst
    }

    fn run_batch(body: &mut impl FnMut(), repetitions: usize) -> Duration {
        let start = Instant::now();
        for _ in 0..repetitions {
            body();
        }
        start.elapsed()
    }

    fn calibrated_repetitions(body: &mut impl FnMut()) -> usize {
        let mut repetitions = 1usize;
        loop {
            if run_batch(body, repetitions) >= Duration::from_micros(50) || repetitions >= 1 << 20 {
                return repetitions;
            }
            repetitions *= 2;
        }
    }

    fn record_sample(samples: &mut Vec<Duration>, repetitions: usize, body: &mut impl FnMut()) {
        samples.push(run_batch(body, repetitions));
    }

    fn median_seconds(mut samples: Vec<Duration>, repetitions: usize) -> f64 {
        samples.sort_unstable();
        samples[samples.len() / 2].as_secs_f64() / repetitions as f64
    }

    fn bench_case(
        mut raw: impl FnMut(),
        mut public: impl FnMut(),
        mut overwrite: impl FnMut(),
    ) -> [f64; 3] {
        for _ in 0..8 {
            raw();
            public();
            overwrite();
        }

        let raw_repetitions = calibrated_repetitions(&mut raw);
        let public_repetitions = calibrated_repetitions(&mut public);
        let overwrite_repetitions = calibrated_repetitions(&mut overwrite);
        let mut raw_samples = Vec::with_capacity(32);
        let mut public_samples = Vec::with_capacity(32);
        let mut overwrite_samples = Vec::with_capacity(32);
        let deadline = Instant::now() + Duration::from_millis(100);
        let mut round = 0usize;
        while Instant::now() < deadline && raw_samples.len() < 32 {
            match round % 3 {
                0 => {
                    record_sample(&mut raw_samples, raw_repetitions, &mut raw);
                    record_sample(&mut public_samples, public_repetitions, &mut public);
                    record_sample(
                        &mut overwrite_samples,
                        overwrite_repetitions,
                        &mut overwrite,
                    );
                }
                1 => {
                    record_sample(&mut public_samples, public_repetitions, &mut public);
                    record_sample(
                        &mut overwrite_samples,
                        overwrite_repetitions,
                        &mut overwrite,
                    );
                    record_sample(&mut raw_samples, raw_repetitions, &mut raw);
                }
                _ => {
                    record_sample(
                        &mut overwrite_samples,
                        overwrite_repetitions,
                        &mut overwrite,
                    );
                    record_sample(&mut raw_samples, raw_repetitions, &mut raw);
                    record_sample(&mut public_samples, public_repetitions, &mut public);
                }
            }
            round += 1;
        }

        [
            median_seconds(raw_samples, raw_repetitions),
            median_seconds(public_samples, public_repetitions),
            median_seconds(overwrite_samples, overwrite_repetitions),
        ]
    }

    fn bench_pair(mut control: impl FnMut(), mut candidate: impl FnMut()) -> [f64; 2] {
        for _ in 0..8 {
            control();
            candidate();
        }

        let control_repetitions = calibrated_repetitions(&mut control);
        let candidate_repetitions = calibrated_repetitions(&mut candidate);
        let mut control_samples = Vec::with_capacity(32);
        let mut candidate_samples = Vec::with_capacity(32);
        let deadline = Instant::now() + Duration::from_millis(100);
        let mut round = 0usize;
        while Instant::now() < deadline && control_samples.len() < 32 {
            if round & 1 == 0 {
                record_sample(&mut control_samples, control_repetitions, &mut control);
                record_sample(
                    &mut candidate_samples,
                    candidate_repetitions,
                    &mut candidate,
                );
            } else {
                record_sample(
                    &mut candidate_samples,
                    candidate_repetitions,
                    &mut candidate,
                );
                record_sample(&mut control_samples, control_repetitions, &mut control);
            }
            round += 1;
        }

        [
            median_seconds(control_samples, control_repetitions),
            median_seconds(candidate_samples, candidate_repetitions),
        ]
    }

    fn print_timing(label: &str, logical_bytes: usize, seconds: f64) {
        let gib_per_sec = logical_bytes as f64 / seconds / (1024.0 * 1024.0 * 1024.0);
        if seconds < 1e-6 {
            println!(
                "    {label:<30} {:>8.2}ns  {gib_per_sec:>7.2} GiB/s",
                seconds * 1e9
            );
        } else {
            println!(
                "    {label:<30} {:>8.2}µs  {gib_per_sec:>7.2} GiB/s",
                seconds * 1e6
            );
        }
    }

    fn check_fixture(
        initial: &[u8],
        expected_accumulate: &[u8],
        expected_overwrite: &[u8],
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
    ) {
        let mut raw = initial.to_vec();
        x86_gf8::gather_gfni(&mut raw, coeffs, srcs);
        assert_eq!(raw, expected_accumulate, "raw accumulate fixture mismatch");

        let mut control = initial.to_vec();
        x86_gf8::gather_gfni_axpy_tail(&mut control, coeffs, srcs);
        assert_eq!(
            control, expected_accumulate,
            "AXPY-tail control fixture mismatch"
        );

        let mut public = initial.to_vec();
        ops::mul_add_gather::<Gf8B>(&mut public, coeffs, srcs);
        assert_eq!(
            public, expected_accumulate,
            "public accumulate fixture mismatch"
        );

        let mut overwrite = initial.to_vec();
        overwrite.fill(0);
        x86_gf8::gather_gfni(&mut overwrite, coeffs, srcs);
        assert_eq!(
            overwrite, expected_overwrite,
            "zero-then-gather fixture mismatch"
        );
    }

    pub fn main() {
        let has_gfni = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("gfni");
        if !has_gfni {
            eprintln!("skipping: direct dot-product baseline requires AVX2+GFNI");
            return;
        }

        let selected = backend_for::<Gf8B>();
        if selected != Backend::V3GfniCrypto {
            eprintln!(
                "skipping: public backend is {}, expected v3_gfni_crypto; remove SIMD_BACKEND downgrade",
                selected.name()
            );
            return;
        }

        let arguments: Vec<_> = std::env::args().collect();
        let smoke = arguments.iter().any(|argument| argument == "--smoke");
        let tails = arguments.iter().any(|argument| argument == "--tails");
        let (lengths, source_counts) = if tails {
            (TAIL_LENGTHS, TAIL_SOURCE_COUNTS)
        } else if smoke {
            (SMOKE_LENGTHS, SMOKE_SOURCE_COUNTS)
        } else {
            (FULL_LENGTHS, FULL_SOURCE_COUNTS)
        };

        println!(
            "GF(2^8) direct N-to-1 baseline — public backend: {}",
            selected.name()
        );
        println!("  coefficients: dense nontrivial; alignment: 32-byte; cache mode: hot");
        println!("  overwrite baseline includes zeroing; logical throughput counts source bytes\n");

        for &len in lengths {
            for &source_count in source_counts {
                let sources: Vec<AlignedBuf> = (0..source_count)
                    .map(|index| AlignedBuf::noise(len, 0, 0x1000 + index as u64 * 17 + len as u64))
                    .collect();
                let srcs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
                let coeffs = dense_coefficients(source_count);
                let initial = noise(len, 0x2000 + len as u64 + source_count as u64);
                let expected_accumulate = reference(initial.clone(), &coeffs, &srcs);
                let expected_overwrite = reference(vec![0; len], &coeffs, &srcs);
                check_fixture(
                    &initial,
                    &expected_accumulate,
                    &expected_overwrite,
                    &coeffs,
                    &srcs,
                );

                let mut raw_dst = AlignedBuf::noise(len, 0, 0x3000 + len as u64);
                let mut control_dst = AlignedBuf::noise(len, 0, 0x2800 + len as u64);
                let mut public_dst = AlignedBuf::noise(len, 0, 0x4000 + len as u64);
                let mut overwrite_dst = AlignedBuf::noise(len, 0, 0x5000 + len as u64);
                raw_dst.as_mut_slice().copy_from_slice(&initial);
                control_dst.as_mut_slice().copy_from_slice(&initial);
                public_dst.as_mut_slice().copy_from_slice(&initial);
                overwrite_dst.as_mut_slice().copy_from_slice(&initial);
                let logical_bytes = len * source_count;

                println!("  {len:>5} B x {source_count:>2} sources:");
                let [control, fused] = bench_pair(
                    || {
                        x86_gf8::gather_gfni_axpy_tail(
                            black_box(control_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        x86_gf8::gather_gfni(
                            black_box(raw_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing("raw AXPY-tail control", logical_bytes, control);
                print_timing("raw fused-tail candidate", logical_bytes, fused);
                println!("      fused-tail speedup {:.2}x", control / fused);
                let [raw, public, overwrite] = bench_case(
                    || {
                        x86_gf8::gather_gfni(
                            black_box(raw_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        ops::mul_add_gather::<Gf8B>(
                            black_box(public_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        overwrite_dst.as_mut_slice().fill(0);
                        x86_gf8::gather_gfni(
                            black_box(overwrite_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing("raw accumulate", logical_bytes, raw);
                print_timing("public accumulate", logical_bytes, public);
                print_timing("raw zero_then_gather", logical_bytes, overwrite);
                println!(
                    "      public/raw {:.2}x, zero_then_gather/raw {:.2}x",
                    public / raw,
                    overwrite / raw
                );
            }
        }
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {
    imp::main();
}

#[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
fn main() {
    eprintln!("skipping: direct dot-product baseline is x86 SIMD-only");
}
