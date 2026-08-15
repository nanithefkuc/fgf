//! Criterion harness for the `0x11D` GFNI affine decision.
//!
//! Measures `Gf8D`'s candidate GFNI kernels — `VGF2P8AFFINEQB` with the
//! const-derived affine bank — against the AVX2 nibble shuffle a GFNI host
//! dispatches to today, per operation shape and buffer size. `Gf8B`'s native
//! `GF2P8MULB` runs as a control: it is the single-instruction ceiling the
//! affine map should reach, and an unchanged operation that exposes machine
//! noise if it drifts. The kernels are called directly through `internals`
//! because dispatch is process-global — the two candidates cannot coexist in
//! one `ops` process.
//!
//! ```sh
//! cargo bench --features internals --bench affine
//! ```

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod imp {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
    use fgf::gf8b;
    use fgf::gf8d;
    use fgf::kernel::tables::{affine_8d, scale_table_8d};
    use fgf::kernel::x86::gf8 as x86_gf8;

    /// The crate-wide deterministic source (same LCG as the tests and the other
    /// benches), so benchmark bytes match what the differential tests exercise.
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

    /// Sub-lane tails, network payloads, L1/L2 residents, and one DRAM-size row.
    const SIZES: &[usize] = &[64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

    /// Stop early with a clear message rather than SIGILL on a GFNI-less host.
    fn require_gfni() -> bool {
        let capable =
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni");
        if !capable {
            eprintln!("skipping: no AVX2+GFNI on this host");
        }
        capable
    }

    fn group<'a>(
        c: &'a mut Criterion,
        name: &str,
    ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
        let mut g = c.benchmark_group(name);
        g.warm_up_time(Duration::from_secs(1));
        g.measurement_time(Duration::from_secs(3));
        g
    }

    fn bench_mul_add(c: &mut Criterion) {
        if !require_gfni() {
            return;
        }
        let coeff = gf8d::Elem(0x53);
        let table = scale_table_8d(coeff);
        let map = affine_8d(coeff);
        let mut g = group(c, "gf8d mul_add");
        for &len in SIZES {
            let src = noise(len, 0x700 + len as u64);
            let mut dst = noise(len, 0x800 + len as u64);
            g.throughput(Throughput::Bytes(len as u64));
            g.bench_function(BenchmarkId::new("shuffle_avx2", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_add_avx2(black_box(dst.as_mut_slice()), table, black_box(&src))
                });
            });
            g.bench_function(BenchmarkId::new("affine_gfni", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_add_affine(
                        black_box(dst.as_mut_slice()),
                        map,
                        table,
                        black_box(&src),
                    );
                });
            });
            g.bench_function(BenchmarkId::new("native_gfni_0x11b", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_add_gfni(
                        black_box(dst.as_mut_slice()),
                        gf8b::Elem(0x53),
                        black_box(&src),
                    );
                });
            });
        }
        g.finish();
    }

    fn bench_mul_assign(c: &mut Criterion) {
        if !require_gfni() {
            return;
        }
        let coeff = gf8d::Elem(0x53);
        let table = scale_table_8d(coeff);
        let map = affine_8d(coeff);
        let mut g = group(c, "gf8d mul_assign");
        for &len in SIZES {
            let mut dst = noise(len, 0x900 + len as u64);
            g.throughput(Throughput::Bytes(len as u64));
            g.bench_function(BenchmarkId::new("shuffle_avx2", len), |b| {
                b.iter(|| x86_gf8::mul_assign_avx2(black_box(dst.as_mut_slice()), table));
            });
            g.bench_function(BenchmarkId::new("affine_gfni", len), |b| {
                b.iter(|| x86_gf8::mul_assign_affine(black_box(dst.as_mut_slice()), map, table));
            });
            g.bench_function(BenchmarkId::new("native_gfni_0x11b", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_assign_gfni(black_box(dst.as_mut_slice()), gf8b::Elem(0x53))
                });
            });
        }
        g.finish();
    }

    fn bench_mul_into(c: &mut Criterion) {
        if !require_gfni() {
            return;
        }
        let coeff = gf8d::Elem(0x53);
        let table = scale_table_8d(coeff);
        let map = affine_8d(coeff);
        let mut g = group(c, "gf8d mul_into");
        // 4 MiB past the non-temporal store threshold closes the sweep.
        for &len in SIZES.iter().chain(&[4_194_304]) {
            let src = noise(len, 0xa00 + len as u64);
            let mut dst = noise(len, 0xb00 + len as u64);
            g.throughput(Throughput::Bytes(len as u64));
            if len >= 4_194_304 {
                g.sample_size(20);
            }
            g.bench_function(BenchmarkId::new("shuffle_avx2", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_into_avx2(black_box(dst.as_mut_slice()), table, black_box(&src))
                });
            });
            g.bench_function(BenchmarkId::new("affine_gfni", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_into_affine(
                        black_box(dst.as_mut_slice()),
                        map,
                        table,
                        black_box(&src),
                    );
                });
            });
            g.bench_function(BenchmarkId::new("native_gfni_0x11b", len), |b| {
                b.iter(|| {
                    x86_gf8::mul_into_gfni(
                        black_box(dst.as_mut_slice()),
                        gf8b::Elem(0x53),
                        black_box(&src),
                    );
                });
            });
        }
        g.finish();
    }

    /// The multi-row shapes compose the single-coefficient kernel per row, so
    /// their ratio must equal `mul_add`'s; two geometries confirm the
    /// composition itself adds nothing.
    fn bench_scatter(c: &mut Criterion) {
        if !require_gfni() {
            return;
        }
        let mut g = group(c, "gf8d scatter");
        for &row_len in &[16_384usize, 65_536] {
            for &nrows in &[4usize, 16] {
                let src = noise(row_len, 0xc00 + row_len as u64);
                let mut rows = noise(row_len * nrows, 0xd00 + row_len as u64);
                let coeffs: Vec<gf8d::Elem> =
                    (0..nrows).map(|j| gf8d::Elem(0x53 ^ j as u8)).collect();
                let label = format!("{nrows}x{row_len}");
                g.throughput(Throughput::Bytes((row_len * nrows) as u64));
                g.bench_function(BenchmarkId::new("shuffle_avx2", &label), |b| {
                    b.iter(|| {
                        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs) {
                            x86_gf8::mul_add_avx2(
                                black_box(row),
                                scale_table_8d(coeff),
                                black_box(&src),
                            );
                        }
                    });
                });
                g.bench_function(BenchmarkId::new("affine_gfni", &label), |b| {
                    b.iter(|| {
                        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs) {
                            x86_gf8::mul_add_affine(
                                black_box(row),
                                affine_8d(coeff),
                                scale_table_8d(coeff),
                                black_box(&src),
                            );
                        }
                    });
                });
            }
        }
        g.finish();
    }

    criterion_group!(
        benches,
        bench_mul_add,
        bench_mul_assign,
        bench_mul_into,
        bench_scatter
    );
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
criterion::criterion_main!(imp::benches);

#[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
fn main() {
    eprintln!("the affine kernels are x86 GFNI only; nothing to measure on this target");
}
