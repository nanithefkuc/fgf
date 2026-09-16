//! Criterion harness for the QuadMersenne31 AVX2 candidate against the
//! unchanged public scalar control.
//!
//! One binary, interleaved per row: the public [`fgf::ops`] dispatch (the
//! scalar control QuadMersenne31 still resolves to), the identical control
//! again under a second id (a duplicate exposing run-to-run noise), and the
//! direct AVX2 candidate reached through `internals`. Every row's outputs
//! are validated byte for byte against the control before any timing; a
//! mismatch panics before a single sample is taken. The resolved process and
//! field backends are reported so a downgraded or scalar run cannot pass
//! unnoticed as vector coverage.
//!
//! ```sh
//! FEC_GOLDEN_CORE=3 just _bench-run prime_ntt
//! FEC_GOLDEN_CORE=3 just _bench-run prime_ntt mul_into
//! FEC_GOLDEN_CORE=3 just _bench-run prime_ntt mul_into/2048
//! ```

#[cfg(all(
    feature = "internals",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod imp {
    use std::hint::black_box;
    use std::sync::Once;
    use std::time::Duration;

    use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
    use fgf::kernel::x86;
    use fgf::kernel::{SimdToken, X64V3Token};
    use fgf::quad_mersenne31::Elem;
    use fgf::{QuadMersenne31, backend, backend_for, has_vector_elementwise, ops};

    /// Row lengths in bytes: sub-vector elements, exact and partial vectors,
    /// the NTT-relevant small rows, and one L2-resident row.
    const ROWS: &[usize] = &[8, 16, 24, 32, 40, 64, 128, 256, 2048];

    /// The benchmark coefficient: canonical, neither zero nor one, both
    /// limbs nonzero so every multiply path does full work.
    const COEFF: Elem = Elem::from_raw(0x1234_5678, 0x3FFF_FFF1);

    static REPORTED: Once = Once::new();

    /// Report the actual process and field backends once, so an unpinned or
    /// downgraded run is visible in the record next to the numbers.
    fn report_backends_once() {
        REPORTED.call_once(|| {
            println!("process backend: {}", backend());
            println!(
                "QuadMersenne31 field backend: {} (has_vector_elementwise: {})",
                backend_for::<QuadMersenne31>(),
                has_vector_elementwise::<QuadMersenne31>()
            );
        });
    }

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (state >> 33).to_le_bytes()[0]
            })
            .collect()
    }

    /// A fixture whose limbs are canonical, meeting the packed canonical
    /// lane contract the multiply kernels are defined on.
    fn canonical(len: usize, seed: u64) -> Vec<u8> {
        let raw = noise(len, seed);
        let mut out = Vec::with_capacity(len);
        for chunk in raw.as_chunks::<8>().0 {
            let e = Elem::from_bytes(*chunk);
            out.extend_from_slice(&e.canonical().to_bytes());
        }
        out
    }

    fn group<'a>(
        c: &'a mut Criterion,
        name: &str,
    ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
        let mut g = c.benchmark_group(name);
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(300));
        g.measurement_time(Duration::from_secs(1));
        g
    }

    /// Validate a candidate output against the control, then register the
    /// control, the duplicate control, and the candidate for timing. Panics
    /// on any byte mismatch before timing begins.
    #[allow(clippy::too_many_arguments)]
    fn bench_row(
        g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        len: usize,
        op: &str,
        start: &[u8],
        mut control: impl FnMut(&mut [u8]),
        mut candidate: impl FnMut(&mut [u8]),
    ) {
        g.throughput(Throughput::Bytes(len as u64));

        let mut want = start.to_vec();
        control(&mut want);
        let mut got = start.to_vec();
        candidate(&mut got);
        assert_eq!(got, want, "{op} candidate mismatch at {len} bytes");
        let mut dup = start.to_vec();
        control(&mut dup);
        assert_eq!(dup, want, "{op} duplicate control mismatch at {len} bytes");

        let mut d1 = start.to_vec();
        g.bench_function(BenchmarkId::new("scalar", len), |b| {
            b.iter(|| control(black_box(&mut d1)))
        });
        let mut d2 = start.to_vec();
        g.bench_function(BenchmarkId::new("scalar_control_2", len), |b| {
            b.iter(|| control(black_box(&mut d2)))
        });
        let mut d3 = start.to_vec();
        g.bench_function(BenchmarkId::new("avx2_candidate", len), |b| {
            b.iter(|| candidate(black_box(&mut d3)))
        });
    }

    fn bench_add(c: &mut Criterion) {
        report_backends_once();
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping add_assign: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "add_assign");
        for &len in ROWS {
            let src = canonical(len, 0x7000 + len as u64);
            let base = canonical(len, 0x8000 + len as u64);
            bench_row(
                &mut g,
                len,
                "add_assign",
                &base,
                |dst| ops::add_assign::<QuadMersenne31>(dst, &src),
                |dst| x86::prime::add_assign_qm31_avx2(token, dst, &src),
            );
        }
        g.finish();
    }

    fn bench_sub(c: &mut Criterion) {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping sub_assign: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "sub_assign");
        for &len in ROWS {
            let src = canonical(len, 0x7000 + len as u64);
            let base = canonical(len, 0x8000 + len as u64);
            bench_row(
                &mut g,
                len,
                "sub_assign",
                &base,
                |dst| ops::sub_assign::<QuadMersenne31>(dst, &src),
                |dst| x86::prime::sub_assign_qm31_avx2(token, dst, &src),
            );
        }
        g.finish();
    }

    fn bench_mul_add(c: &mut Criterion) {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping mul_add: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "mul_add");
        for &len in ROWS {
            let src = canonical(len, 0x7000 + len as u64);
            let base = canonical(len, 0x8000 + len as u64);
            bench_row(
                &mut g,
                len,
                "mul_add",
                &base,
                |dst| ops::mul_add::<QuadMersenne31>(dst, COEFF, &src),
                |dst| x86::prime::mul_add_qm31_avx2(token, dst, COEFF, &src),
            );
        }
        g.finish();
    }

    fn bench_mul_assign(c: &mut Criterion) {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping mul_assign: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "mul_assign");
        for &len in ROWS {
            let base = canonical(len, 0x8000 + len as u64);
            bench_row(
                &mut g,
                len,
                "mul_assign",
                &base,
                |dst| ops::mul_assign::<QuadMersenne31>(dst, COEFF),
                |dst| x86::prime::mul_assign_qm31_avx2(token, dst, COEFF),
            );
        }
        g.finish();
    }

    fn bench_mul_into(c: &mut Criterion) {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping mul_into: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "mul_into");
        for &len in ROWS {
            let src = canonical(len, 0x7000 + len as u64);
            // The destination is write-only: both sides start from the same
            // ignored bytes.
            let base = vec![0xA5u8; len];
            bench_row(
                &mut g,
                len,
                "mul_into",
                &base,
                |dst| ops::mul_into::<QuadMersenne31>(dst, COEFF, &src),
                |dst| x86::prime::mul_into_qm31_avx2(token, dst, COEFF, &src),
            );
        }
        g.finish();
    }

    fn bench_mul_elementwise(c: &mut Criterion) {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("skipping mul_elementwise: no AVX2 on this host");
            return;
        };
        let mut g = group(c, "mul_elementwise");
        for &len in ROWS {
            let a = canonical(len, 0x7000 + len as u64);
            let b = canonical(len, 0x9000 + len as u64);
            let base = vec![0u8; len];
            bench_row(
                &mut g,
                len,
                "mul_elementwise",
                &base,
                |dst| ops::mul_elementwise::<QuadMersenne31>(dst, &a, &b),
                |dst| x86::prime::mul_elementwise_qm31_avx2(token, dst, &a, &b),
            );
        }
        g.finish();
    }

    criterion_group!(
        benches,
        bench_add,
        bench_sub,
        bench_mul_add,
        bench_mul_assign,
        bench_mul_into,
        bench_mul_elementwise
    );
}

#[cfg(all(
    feature = "internals",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
criterion::criterion_main!(imp::benches);

#[cfg(not(all(
    feature = "internals",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
fn main() {
    eprintln!("skipping: QM31 AVX2 comparison is x86 SIMD-only");
}
