//! Prime-field NTT consumer validation and schedule tuning.
//!
//! Drives the real `butterfly-fft` transform surface — `NttPlan` forward and
//! inverse execution plus the `internals` schedule controls
//! `ntt_forward_fused` and `ntt_forward_packed` — over the local `fgf`
//! working copy, for `Goldilocks` and `QuadMersenne31` at sizes 64, 1024,
//! and 65536 with 1, 2, 4, and 16 lanes per row.
//!
//! Every geometry is validated before any of its arms are timed: the forced
//! packed forward and the production forward must reproduce the forced fused
//! forward byte for byte, the inverse must return the canonical input
//! unchanged, and sizes up to [`DFT_MAX`] are additionally checked against a
//! direct `Y[k] = Σ_j x[j] · ω^(j·k)` evaluation that shares no indexing with
//! the FFT, with `ω` derived from the field generator and cross-checked
//! against the plan's root. A failing check panics, so a red run never
//! produces numbers.
//!
//! Benchmark ids follow `field/nSIZE_lLANES/arm` with arms `fused`,
//! `fused_control` (a duplicated unchanged control), `packed`, `forward`,
//! and `inverse`; any id prefix filters, e.g. `-- goldilocks/n1024` or
//! `-- qm31/n64_l2/packed`. `--test` runs each arm once for a quick
//! correctness pass without timing. Arms iterate in place on evolving data,
//! matching the consumer benchmark convention; transform cost is
//! data-independent, so every iteration performs the same work.
// `as_chunks_mut::<F::BYTES>()` would need a const generic argument that
// depends on `F`, which stable Rust does not accept.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::internals::{ntt_forward_fused, ntt_forward_packed};
use butterfly_fft::ntt::NttPlan;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Goldilocks, QuadMersenne31, backend, backend_for, has_vector_elementwise};

/// Transform lengths spanning the consumer's measured range.
const SIZES: [usize; 3] = [64, 1024, 65536];
/// Independent transform lanes packed into one row.
const LANES: [usize; 4] = [1, 2, 4, 16];
/// Largest size checked against the direct DFT; above this the fused
/// schedule, the packed schedule, production, and the inverse roundtrip
/// remain the oracles.
const DFT_MAX: usize = 1024;
/// LCG seed matching the consumer benchmark, so the inputs here are the
/// inputs the consumer measures.
const SEED: u64 = 0x1234_5678_9abc_def1;

/// Fixed-seed LCG fill, reduced into the field by the canonicalizing decode.
fn fill<F: Field>(bytes: &mut [u8], seed: u64) {
    let mut state = seed;
    for slot in bytes.chunks_exact_mut(F::BYTES) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value = F::decode(&state.to_le_bytes()[..F::BYTES]);
        F::encode(slot, value.add(F::Elem::ZERO));
    }
}

/// Direct `Y[k] = Σ_j x[j] · ω^(j·k)` over decoded elements, sharing no
/// permutation, staging, or twiddle layout with the FFT.
///
/// The root is `GENERATOR^((|F| - 1) / size)` from the field API; a
/// disagreement with the plan's root panics before any output is compared.
fn direct_dft<F: FieldKernels>(plan: &NttPlan<F>, input: &[u8], row_len: usize) -> Vec<u8> {
    let size = plan.size();
    let exponent = u64::try_from((F::ORDER - 1) / u128::try_from(size).expect("size fits u128"))
        .expect("group quotient fits u64");
    let root = F::GENERATOR.pow(exponent);
    assert_eq!(root, plan.root(), "field-derived root must match the plan");

    // `powers[k]` is ω^k, built by running multiplication.
    let mut powers = Vec::with_capacity(size);
    let mut power = F::Elem::ONE;
    for _ in 0..size {
        powers.push(power);
        power = power.mul(root);
    }

    let mut output = vec![0u8; input.len()];
    for lane in 0..row_len / F::BYTES {
        let column: Vec<F::Elem> = (0..size)
            .map(|row| {
                let at = row * row_len + lane * F::BYTES;
                F::decode(&input[at..at + F::BYTES])
            })
            .collect();
        for (index, slot) in output.chunks_exact_mut(row_len).enumerate() {
            let step = powers[index];
            let mut weight = F::Elem::ONE;
            let mut acc = F::Elem::ZERO;
            for &value in &column {
                acc = acc.add(value.mul(weight));
                weight = weight.mul(step);
            }
            F::encode(&mut slot[lane * F::BYTES..][..F::BYTES], acc);
        }
    }
    output
}

/// Validate one geometry completely, then register its timed arms.
///
/// The plan, scratch rows, and every arm's buffer exist before timing; the
/// timed closures contain nothing but the transform call.
fn geometry<F: FieldKernels>(criterion: &mut Criterion, plan: &NttPlan<F>, label: &str) {
    let size = plan.size();
    for lanes in LANES {
        let row_len = lanes * F::BYTES;
        let mut canonical = vec![0u8; size * row_len];
        fill::<F>(&mut canonical, SEED);

        let mut scratch = plan.scratch(row_len).expect("scratch");
        let mut fused_out = canonical.clone();
        ntt_forward_fused(plan, &mut fused_out, row_len, &mut scratch).expect("fused forward");

        let mut packed_out = canonical.clone();
        ntt_forward_packed(plan, &mut packed_out, row_len, &mut scratch).expect("packed forward");
        assert_eq!(packed_out, fused_out, "packed schedule must match fused");

        let mut production = canonical.clone();
        plan.forward_bytes_scratch(&mut production, row_len, &mut scratch)
            .expect("production forward");
        assert_eq!(production, fused_out, "production forward must match fused");

        let mut roundtrip = fused_out.clone();
        plan.inverse_bytes_scratch(&mut roundtrip, row_len, &mut scratch)
            .expect("inverse");
        assert_eq!(
            roundtrip, canonical,
            "inverse(forward(x)) must return the canonical input"
        );

        let mut checks = "packed, production forward, inverse roundtrip";
        if size <= DFT_MAX {
            assert_eq!(
                direct_dft(plan, &canonical, row_len),
                fused_out,
                "fused forward must match the direct DFT"
            );
            checks = "packed, production forward, inverse roundtrip, direct DFT";
        }
        println!("{label}/n{size}_l{lanes}: validated ({checks})");

        let mut group = criterion.benchmark_group(format!("{label}/n{size}_l{lanes}"));
        group
            .sample_size(10)
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_secs(1))
            .throughput(Throughput::Elements((size * lanes) as u64));

        {
            let mut rows = canonical.clone();
            let mut scratch = plan.scratch(row_len).expect("scratch");
            group.bench_function("fused", |bencher| {
                bencher.iter(|| {
                    ntt_forward_fused(
                        black_box(plan),
                        black_box(&mut rows),
                        black_box(row_len),
                        black_box(&mut scratch),
                    )
                    .expect("fused forward");
                });
            });
        }

        {
            let mut rows = canonical.clone();
            let mut scratch = plan.scratch(row_len).expect("scratch");
            group.bench_function("fused_control", |bencher| {
                bencher.iter(|| {
                    ntt_forward_fused(
                        black_box(plan),
                        black_box(&mut rows),
                        black_box(row_len),
                        black_box(&mut scratch),
                    )
                    .expect("fused forward");
                });
            });
        }

        {
            let mut rows = canonical.clone();
            let mut scratch = plan.scratch(row_len).expect("scratch");
            group.bench_function("packed", |bencher| {
                bencher.iter(|| {
                    ntt_forward_packed(
                        black_box(plan),
                        black_box(&mut rows),
                        black_box(row_len),
                        black_box(&mut scratch),
                    )
                    .expect("packed forward");
                });
            });
        }

        {
            let mut rows = canonical.clone();
            let mut scratch = plan.scratch(row_len).expect("scratch");
            group.bench_function("forward", |bencher| {
                bencher.iter(|| {
                    plan.forward_bytes_scratch(
                        black_box(&mut rows),
                        black_box(row_len),
                        black_box(&mut scratch),
                    )
                    .expect("production forward");
                });
            });
        }

        {
            let mut rows = fused_out.clone();
            let mut scratch = plan.scratch(row_len).expect("scratch");
            group.bench_function("inverse", |bencher| {
                bencher.iter(|| {
                    plan.inverse_bytes_scratch(
                        black_box(&mut rows),
                        black_box(row_len),
                        black_box(&mut scratch),
                    )
                    .expect("inverse");
                });
            });
        }

        group.finish();
    }
}

/// Register one field's panel and report the resolved backends.
fn field_panel<F: FieldKernels>(criterion: &mut Criterion, label: &str) {
    println!(
        "{label} ({}): process backend {}, field backend {}, vector elementwise {}",
        F::NAME,
        backend(),
        backend_for::<F>(),
        has_vector_elementwise::<F>(),
    );
    for size in SIZES {
        let plan = NttPlan::<F>::new(size).expect("plan");
        geometry(criterion, &plan, label);
    }
}

fn goldilocks(criterion: &mut Criterion) {
    field_panel::<Goldilocks>(criterion, "goldilocks");
}

fn quad_mersenne31(criterion: &mut Criterion) {
    field_panel::<QuadMersenne31>(criterion, "qm31");
}

criterion_group!(ntt_consumer, goldilocks, quad_mersenne31);
criterion_main!(ntt_consumer);
