//! Direct GF(2^8) N-to-1 benchmarks for accumulate, overwrite, short-row
//! fusion, prepared-affine multiplication, and gather tile/page topology.
//!
//! Raw samples bypass public dispatch through `internals`. Overwrite controls
//! time destination zeroing. Every fixture is validated before timing.
//!
//! ```sh
//! cargo bench --features internals --bench dot_product
//! cargo bench --features internals --bench dot_product -- --smoke
//! cargo bench --features internals --bench dot_product -- --tails
//! cargo bench --features internals --bench dot_product -- --affine
//! cargo bench --features internals --bench dot_product -- --tiles
//! cargo bench --features internals --bench dot_product -- --tiles --counter --tile-lanes=4
//! cargo bench --features internals --bench dot_product -- --tiles --counter --tile-split
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
    const AFFINE_LENGTHS: &[usize] = &[16, 32, 64, 96, 128, 256, 1_024, 4_096, 4_160, 16_384];
    const AFFINE_SOURCE_COUNTS: &[usize] = &[1, 2, 3, 4, 8, 12, 16, 24, 32];
    const TILE_LENGTHS: &[usize] = &[128, 256, 1_024, 4_096, 4_160, 16_384];
    const TILE_SOURCE_COUNTS: &[usize] = &[4, 8, 16, 32];
    const TILE_TOPOLOGY_LAYOUTS: &[&str] = &[
        "src+1",
        "dst+1",
        "both+1",
        "page-zero",
        "page-end",
        "page-stagger",
    ];

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

        fn page_noise(len: usize, page_offset: usize, seed: u64) -> Self {
            assert!(page_offset < 4_096);
            let mut storage = vec![0; len + 4_095];
            let base = storage.as_ptr() as usize;
            let start = (page_offset + 4_096 - (base & 4_095)) & 4_095;
            let bytes = noise(len, seed);
            storage[start..start + len].copy_from_slice(&bytes);
            let result = Self {
                storage,
                start,
                len,
            };
            assert_eq!((result.as_slice().as_ptr() as usize) & 4_095, page_offset);
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

        let mut public_dot = initial.to_vec();
        ops::dot_product::<Gf8B>(&mut public_dot, coeffs, srcs);
        assert_eq!(
            public_dot, expected_overwrite,
            "public dot-product fixture mismatch"
        );
    }

    fn usize_argument(arguments: &[String], prefix: &str) -> Option<usize> {
        arguments.iter().find_map(|argument| {
            argument.strip_prefix(prefix).map(|value| {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid integer in {argument}"))
            })
        })
    }

    fn string_argument<'a>(arguments: &'a [String], prefix: &str) -> Option<&'a str> {
        arguments
            .iter()
            .find_map(|argument| argument.strip_prefix(prefix))
    }

    fn tile_buffer(len: usize, layout: &str, source: Option<usize>, seed: u64) -> AlignedBuf {
        match layout {
            "aligned" => AlignedBuf::noise(len, 0, seed),
            "src+1" => AlignedBuf::noise(len, usize::from(source.is_some()), seed),
            "dst+1" => AlignedBuf::noise(len, usize::from(source.is_none()), seed),
            "both+1" => AlignedBuf::noise(len, 1, seed),
            "page-zero" => AlignedBuf::page_noise(len, 0, seed),
            "page-end" => AlignedBuf::page_noise(len, 4_032, seed),
            "page-stagger" => {
                let offset = source.map_or(0, |index| (index * 64) & 4_095);
                AlignedBuf::page_noise(len, offset, seed)
            }
            _ => panic!("unknown tile layout {layout}"),
        }
    }

    fn apply_tile(lanes: usize, dst: &mut [u8], coeffs: &[gf8b::Elem], srcs: &[&[u8]]) {
        match lanes {
            1 => x86_gf8::gather_gfni_tile::<1>(dst, coeffs, srcs),
            2 => x86_gf8::gather_gfni_tile::<2>(dst, coeffs, srcs),
            3 => x86_gf8::gather_gfni_tile::<3>(dst, coeffs, srcs),
            4 => x86_gf8::gather_gfni_tile::<4>(dst, coeffs, srcs),
            _ => panic!("tile lane count must be 1–4"),
        }
    }

    fn bench_tile_pair<const LANES: usize>(
        control: &mut AlignedBuf,
        candidate: &mut AlignedBuf,
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
    ) -> [f64; 2] {
        bench_pair(
            || {
                x86_gf8::gather_gfni_tile::<4>(
                    black_box(control.as_mut_slice()),
                    black_box(coeffs),
                    black_box(srcs),
                );
            },
            || {
                x86_gf8::gather_gfni_tile::<LANES>(
                    black_box(candidate.as_mut_slice()),
                    black_box(coeffs),
                    black_box(srcs),
                );
            },
        )
    }

    fn bench_split_pair(
        control: &mut AlignedBuf,
        candidate: &mut AlignedBuf,
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
    ) -> [f64; 2] {
        bench_pair(
            || {
                x86_gf8::gather_gfni(
                    black_box(control.as_mut_slice()),
                    black_box(coeffs),
                    black_box(srcs),
                );
            },
            || {
                x86_gf8::gather_gfni_split(
                    black_box(candidate.as_mut_slice()),
                    black_box(coeffs),
                    black_box(srcs),
                );
            },
        )
    }

    fn run_counter<const LANES: usize>(
        dst: &mut AlignedBuf,
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
        iterations: usize,
    ) {
        for _ in 0..iterations {
            x86_gf8::gather_gfni_tile::<LANES>(
                black_box(dst.as_mut_slice()),
                black_box(coeffs),
                black_box(srcs),
            );
        }
    }

    fn run_split_counter(
        dst: &mut AlignedBuf,
        coeffs: &[gf8b::Elem],
        srcs: &[&[u8]],
        iterations: usize,
    ) {
        for _ in 0..iterations {
            x86_gf8::gather_gfni_split(
                black_box(dst.as_mut_slice()),
                black_box(coeffs),
                black_box(srcs),
            );
        }
    }

    fn bench_tile_case(
        len: usize,
        source_count: usize,
        layout: &str,
        counter_lanes: Option<usize>,
        counter_split: bool,
    ) {
        let sources: Vec<AlignedBuf> = (0..source_count)
            .map(|index| {
                tile_buffer(
                    len,
                    layout,
                    Some(index),
                    0x8100 + index as u64 * 17 + len as u64,
                )
            })
            .collect();
        let srcs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
        let coeffs = dense_coefficients(source_count);
        let initial = noise(len, 0x8200 + len as u64 + source_count as u64);
        let expected = reference(initial.clone(), &coeffs, &srcs);
        for lanes in 1..=4 {
            let mut got = initial.clone();
            apply_tile(lanes, &mut got, &coeffs, &srcs);
            assert_eq!(
                got, expected,
                "{lanes}-lane tile fixture mismatch: {len} B x {source_count}, {layout}"
            );
        }
        let mut split = initial.clone();
        x86_gf8::gather_gfni_split(&mut split, &coeffs, &srcs);
        assert_eq!(
            split, expected,
            "split tile fixture mismatch: {len} B x {source_count}, {layout}"
        );

        if counter_lanes.is_some() || counter_split {
            let mut dst = tile_buffer(len, layout, None, 0x8300 + len as u64);
            dst.as_mut_slice().copy_from_slice(&initial);
            const ITERATIONS: usize = 1 << 20;
            let start = Instant::now();
            if counter_split {
                run_split_counter(&mut dst, &coeffs, &srcs, ITERATIONS);
            } else {
                match counter_lanes.expect("counter lane checked above") {
                    1 => run_counter::<1>(&mut dst, &coeffs, &srcs, ITERATIONS),
                    2 => run_counter::<2>(&mut dst, &coeffs, &srcs, ITERATIONS),
                    3 => run_counter::<3>(&mut dst, &coeffs, &srcs, ITERATIONS),
                    4 => run_counter::<4>(&mut dst, &coeffs, &srcs, ITERATIONS),
                    _ => panic!("tile lane count must be 1–4"),
                }
            }
            let checksum = dst
                .as_slice()
                .iter()
                .fold(0u8, |accumulator, &byte| accumulator ^ byte);
            let counter_label = if counter_split {
                "split accumulators"
            } else {
                match counter_lanes.expect("counter lane checked above") {
                    1 => "1 lane",
                    2 => "2 lanes",
                    3 => "3 lanes",
                    4 => "4 lanes",
                    _ => unreachable!(),
                }
            };
            println!(
                "tile counter: {len} B x {source_count}, {layout}, {}, \
                 {ITERATIONS} iterations, {:?}, checksum {checksum:#04x}",
                counter_label,
                start.elapsed()
            );
            return;
        }

        let logical_bytes = len * source_count;
        println!("  {len:>5} B x {source_count:>2} sources, {layout}:");
        for lanes in 1..=3 {
            let mut control =
                tile_buffer(len, layout, None, 0x8400 + len as u64 + lanes as u64 * 31);
            let mut candidate =
                tile_buffer(len, layout, None, 0x8500 + len as u64 + lanes as u64 * 31);
            control.as_mut_slice().copy_from_slice(&initial);
            candidate.as_mut_slice().copy_from_slice(&initial);
            let [production, candidate] = match lanes {
                1 => bench_tile_pair::<1>(&mut control, &mut candidate, &coeffs, &srcs),
                2 => bench_tile_pair::<2>(&mut control, &mut candidate, &coeffs, &srcs),
                3 => bench_tile_pair::<3>(&mut control, &mut candidate, &coeffs, &srcs),
                _ => unreachable!(),
            };
            print_timing("128-byte production", logical_bytes, production);
            print_timing(
                match lanes {
                    1 => "32-byte candidate",
                    2 => "64-byte candidate",
                    3 => "96-byte candidate",
                    _ => unreachable!(),
                },
                logical_bytes,
                candidate,
            );
            println!(
                "      {}-byte speedup {:.2}x",
                lanes * 32,
                production / candidate
            );
        }
        let mut control = tile_buffer(len, layout, None, 0x8600 + len as u64);
        let mut split = tile_buffer(len, layout, None, 0x8700 + len as u64);
        control.as_mut_slice().copy_from_slice(&initial);
        split.as_mut_slice().copy_from_slice(&initial);
        let [production, candidate] = bench_split_pair(&mut control, &mut split, &coeffs, &srcs);
        print_timing("128-byte production", logical_bytes, production);
        print_timing("128-byte split candidate", logical_bytes, candidate);
        println!("      split speedup {:.2}x", production / candidate);
    }

    fn bench_tiles(arguments: &[String]) {
        let requested_len = usize_argument(arguments, "--tile-len=");
        let requested_sources = usize_argument(arguments, "--tile-sources=");
        let requested_layout = string_argument(arguments, "--tile-layout=");
        let counter = arguments.iter().any(|argument| argument == "--counter");
        let counter_lanes = usize_argument(arguments, "--tile-lanes=");
        let counter_split = arguments.iter().any(|argument| argument == "--tile-split");

        if counter {
            assert!(
                counter_split || counter_lanes.is_some(),
                "--counter requires --tile-lanes=1..4 or --tile-split"
            );
            bench_tile_case(
                requested_len.unwrap_or(4_096),
                requested_sources.unwrap_or(16),
                requested_layout.unwrap_or("aligned"),
                counter_lanes,
                counter_split,
            );
            return;
        }

        println!("GF(2^8) native GFNI gather tile sweep");
        println!("  candidates: 32/64/96 B; production control: 128 B");
        println!("  speedup is production time / candidate time; values above 1 favor candidate\n");

        let mut cases = Vec::new();
        let base_layout = requested_layout.unwrap_or("aligned");
        for &len in TILE_LENGTHS {
            for &source_count in TILE_SOURCE_COUNTS {
                if requested_len.is_none_or(|requested| requested == len)
                    && requested_sources.is_none_or(|requested| requested == source_count)
                {
                    cases.push((len, source_count, base_layout));
                }
            }
        }
        if requested_layout.is_none() {
            for &len in &[4_096, 4_160, 16_384] {
                for &layout in TILE_TOPOLOGY_LAYOUTS {
                    if requested_len.is_none_or(|requested| requested == len)
                        && requested_sources.is_none_or(|requested| requested == 16)
                    {
                        cases.push((len, 16, layout));
                    }
                }
            }
        }
        assert!(
            !cases.is_empty(),
            "tile filters selected no benchmark cases"
        );
        for (len, source_count, layout) in cases {
            bench_tile_case(len, source_count, layout, None, false);
        }
    }

    fn bench_affine() {
        println!("GF(2^8) native-versus-affine N-to-1 prototype");
        println!("  coefficients: dense nontrivial; alignment: 32-byte; cache mode: hot");
        println!("  speedup is native time / affine time; values above 1 favor affine\n");

        for &len in AFFINE_LENGTHS {
            for &source_count in AFFINE_SOURCE_COUNTS {
                let sources: Vec<AlignedBuf> = (0..source_count)
                    .map(|index| AlignedBuf::noise(len, 0, 0x7100 + index as u64 * 17 + len as u64))
                    .collect();
                let srcs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
                let coeffs = dense_coefficients(source_count);
                let factors: Vec<_> = coeffs
                    .iter()
                    .copied()
                    .map(x86_gf8::prepare_affine_8b)
                    .collect();
                let initial = noise(len, 0x7200 + len as u64 + source_count as u64);
                let expected_accumulate = reference(initial.clone(), &coeffs, &srcs);
                let expected_overwrite = reference(vec![0; len], &coeffs, &srcs);

                let mut affine_check = initial.clone();
                x86_gf8::gather_affine_8b(&mut affine_check, &factors, &srcs);
                assert_eq!(
                    affine_check, expected_accumulate,
                    "prepared affine accumulate fixture mismatch"
                );
                affine_check.fill(0);
                x86_gf8::gather_affine_8b(&mut affine_check, &factors, &srcs);
                assert_eq!(
                    affine_check, expected_overwrite,
                    "prepared affine overwrite fixture mismatch"
                );

                let mut native_dst = AlignedBuf::noise(len, 0, 0x7300 + len as u64);
                let mut affine_dst = AlignedBuf::noise(len, 0, 0x7400 + len as u64);
                let mut native_overwrite = AlignedBuf::noise(len, 0, 0x7500 + len as u64);
                let mut affine_overwrite = AlignedBuf::noise(len, 0, 0x7600 + len as u64);
                let mut native_one_shot = AlignedBuf::noise(len, 0, 0x7700 + len as u64);
                let mut affine_one_shot = AlignedBuf::noise(len, 0, 0x7800 + len as u64);
                native_dst.as_mut_slice().copy_from_slice(&initial);
                affine_dst.as_mut_slice().copy_from_slice(&initial);
                let mut one_shot_factors = factors.clone();
                let logical_bytes = len * source_count;

                println!("  {len:>5} B x {source_count:>2} sources:");
                let [native, affine] = bench_pair(
                    || {
                        x86_gf8::gather_gfni(
                            black_box(native_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        x86_gf8::gather_affine_8b(
                            black_box(affine_dst.as_mut_slice()),
                            black_box(&factors),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing("native accumulate", logical_bytes, native);
                print_timing("prepared affine accumulate", logical_bytes, affine);
                println!("      prepared affine speedup {:.2}x", native / affine);

                let [native, affine] = bench_pair(
                    || {
                        native_overwrite.as_mut_slice().fill(0);
                        x86_gf8::gather_gfni(
                            black_box(native_overwrite.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        affine_overwrite.as_mut_slice().fill(0);
                        x86_gf8::gather_affine_8b(
                            black_box(affine_overwrite.as_mut_slice()),
                            black_box(&factors),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing("native overwrite", logical_bytes, native);
                print_timing("prepared affine overwrite", logical_bytes, affine);
                println!("      affine overwrite speedup {:.2}x", native / affine);

                let [native, affine] = bench_pair(
                    || {
                        x86_gf8::gather_gfni(
                            black_box(native_one_shot.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        for (factor, &coeff) in one_shot_factors.iter_mut().zip(&coeffs) {
                            *factor = x86_gf8::prepare_affine_8b(black_box(coeff));
                        }
                        x86_gf8::gather_affine_8b(
                            black_box(affine_one_shot.as_mut_slice()),
                            black_box(&one_shot_factors),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing("native one-shot", logical_bytes, native);
                print_timing("prepare + affine gather", logical_bytes, affine);
                println!("      affine one-shot speedup {:.2}x", native / affine);
            }
        }
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
        let affine = arguments.iter().any(|argument| argument == "--affine");
        let tiles = arguments.iter().any(|argument| argument == "--tiles");
        if tiles {
            bench_tiles(&arguments);
            return;
        }
        if affine {
            bench_affine();
            return;
        }
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
                let mut public_dot_dst = AlignedBuf::noise(len, 0, 0x6000 + len as u64);
                raw_dst.as_mut_slice().copy_from_slice(&initial);
                control_dst.as_mut_slice().copy_from_slice(&initial);
                public_dst.as_mut_slice().copy_from_slice(&initial);
                overwrite_dst.as_mut_slice().copy_from_slice(&initial);
                public_dot_dst.as_mut_slice().copy_from_slice(&initial);
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
                let [zero_then_gather, public_dot] = bench_pair(
                    || {
                        public_dst.as_mut_slice().fill(0);
                        ops::mul_add_gather::<Gf8B>(
                            black_box(public_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                    || {
                        ops::dot_product::<Gf8B>(
                            black_box(public_dot_dst.as_mut_slice()),
                            black_box(&coeffs),
                            black_box(&srcs),
                        );
                    },
                );
                print_timing(
                    "public zero_then_gather ctl",
                    logical_bytes,
                    zero_then_gather,
                );
                print_timing("public overwrite dot", logical_bytes, public_dot);
                println!(
                    "      public overwrite/composition {:.2}x",
                    public_dot / zero_then_gather
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
