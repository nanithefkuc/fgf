//! Throughput harness for the vector kernels.
//!
//! No criterion: these kernels are memory-bound straight-line loops with no
//! branchy tail, so a warm loop and a wall-clock median is enough to see the
//! effects that matter — backend, buffer size relative to cache, and whether
//! the blocked multi-row shapes actually beat repeated single-row AXPY.
//!
//! ```sh
//! cargo bench --features internals --bench kernels
//! SIMD_BACKEND=v3     cargo bench --features internals --bench kernels # compare backends
//! SIMD_BACKEND=scalar cargo bench --features internals --bench kernels
//! ```
//!
//! Every timing helper prints a machine record beside its human line:
//! `SAMPLE\t{label}\t{logical_bytes}\t{ns_per_call:.9}` for region shapes
//! and `SCALAR\t{label}\t{ns_per_scalar_op:.9}` for scalar element chains.
//! Process entry prints `FIELD_BACKEND\t{label}\t{backend_name}` for every
//! measured field — each field's own routing, not the process backend —
//! plus the host's actual CPU affinity and scheduler policy. Each process
//! times an unchanged byte-XOR control at start and end
//! (`control byte xor start`, `control byte xor end`), so campaign drift is
//! separable from field speed. `--blocked-diagnostic` runs the
//! internals-only blocked-versus-AXPY probe; the public panels never
//! invoke it.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::poly::AES;
use fgf::{
    Elem, Extension, ExtensionSpec, FieldBuffer, FieldElem, FiniteField, Goldilocks, Mersenne31,
    Packed, Polynomial, QuadMersenne31, RS, Relation, backend, backend_for, ops,
};

type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf32 = fgf::FiniteField<2, 32, fgf::Extension<fgf::Rijndael32, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;
type FanPaar16Field = fgf::FiniteField<2, 16, fgf::Extension<fgf::FanPaar16, 2>>;
type FanPaar32Field = fgf::FiniteField<2, 32, fgf::Extension<fgf::FanPaar32, 2>>;
type FanPaar64Field = fgf::FiniteField<2, 64, fgf::Extension<fgf::FanPaar64, 2>>;
type Word1e9p7 = FiniteField<1_000_000_007, 1>;
type WordMax = FiniteField<4_294_967_291, 1>;
type Byte101 = FiniteField<101, 1>;
type Byte127 = FiniteField<127, 1>;
type Byte131 = FiniteField<131, 1>;
type Half65521 = FiniteField<65_521, 1>;
type Word1280000003 = FiniteField<1_280_000_003, 1>;
type RowP2 = FiniteField<2, 1>;
type WideMax = FiniteField<18_446_744_073_709_551_557, 1>;

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

/// Format a nanosecond count the way `Duration`'s debug output scales its
/// unit, but from an `f64`: `Duration` cannot hold sub-nanosecond values, so
/// dividing one by a repetition count quantizes every estimate to whole
/// nanoseconds.
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.2}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

/// Target wall-clock duration of one sample batch: long enough that a short
/// workload's batch sits above clock and scheduler jitter.
const BATCH_TARGET_NS: f64 = 100_000.0;
/// Upper bound on repetitions in one sample batch.
const MAX_BATCH_REPS: u64 = 1 << 20;
/// Logical byte count of the drift control's XOR pass.
const CONTROL_BYTES: usize = 256 * 1024;
/// Iterations in every scalar element chain body.
const SCALAR_ITERS: usize = 1 << 20;

/// Repetitions per sample batch for a body whose calibration pass took
/// `est_ns`: the batch grows to [`BATCH_TARGET_NS`] for short workloads and
/// never drops below the helper's historical fixed batch.
fn batch_reps(est_ns: f64, min_reps: u64) -> u64 {
    ((BATCH_TARGET_NS / est_ns.max(1.0)).ceil() as u64).clamp(min_reps, MAX_BATCH_REPS)
}

/// The drift control: a plain byte XOR over black-boxed buffers. The body
/// names no field code and is identical in every ported harness, so
/// start/end movement measures the measurement environment, not a field.
fn control_body(dst: &mut [u8], src: &[u8]) {
    let dst = black_box(dst);
    let src = black_box(src);
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// The process's actual CPU affinity and scheduling policy, read from the
/// kernel: every recorded run states where its numbers came from.
fn print_host_lines() {
    let status = std::fs::read_to_string("/proc/self/status").ok();
    let cpus = status
        .as_deref()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
        })
        .map(|line| line["Cpus_allowed_list:".len()..].trim().to_string());
    // /proc/self/stat field 41 carries the scheduling policy number. The
    // comm field may contain spaces and parentheses, so parsing starts
    // after the last ')'; state is field 3, which puts policy at index 38.
    let stat = std::fs::read_to_string("/proc/self/stat").ok();
    let policy = stat.as_deref().and_then(|text| {
        text.rsplit_once(')')?
            .1
            .split_whitespace()
            .nth(38)
            .and_then(|field| field.parse::<u32>().ok())
    });
    let policy_name = match policy {
        Some(0) => "normal",
        Some(1) => "fifo",
        Some(2) => "rr",
        Some(3) => "batch",
        Some(5) => "idle",
        Some(6) => "deadline",
        Some(_) => "unknown",
        None => "unavailable",
    };
    println!(
        "HOST\tcpus_allowed_list={}\tsched_policy={} {policy_name}",
        cpus.as_deref().unwrap_or("unavailable"),
        policy.map_or("n/a".to_string(), |p| p.to_string()),
    );
}

/// One `FIELD_BACKEND` record per measured field: where that field's bulk
/// operations actually run in this process. Fields with narrower kernel
/// routes report their own answer from `backend_for`, not the process
/// backend.
fn print_field_backends() {
    fn field<F: fgf::FieldKernels>(label: &str) {
        println!("FIELD_BACKEND\t{label}\t{}", backend_for::<F>().name());
    }
    field::<FiniteField<2, 8, Polynomial<AES>>>("gf8b");
    field::<FiniteField<2, 8, Polynomial<RS>>>("gf8d");
    field::<Gf16>("gf16b");
    field::<Gf32>("gf32b");
    field::<Gf64>("gf64b");
    field::<FanPaar16Field>("fp16");
    field::<FanPaar32Field>("fp32");
    field::<FanPaar64Field>("fp64");
    field::<Mersenne31>("m31");
    field::<QuadMersenne31>("qm31");
    field::<Goldilocks>("gld");
    field::<Byte101>("byte101");
    field::<Byte127>("byte127");
    field::<Byte131>("byte131");
    field::<Half65521>("half65521");
    field::<Word1e9p7>("word1e9p7");
    field::<Word1280000003>("word1280000003");
    field::<WordMax>("wordmax");
    field::<WideMax>("widemax");
    field::<RowP2>("rowp2");
    field::<Gf16d>("gf16d");
}

/// The Mersenne prime `2^31 - 1`: the m31 element bound and each qm31 limb
/// bound.
const M31_PRIME: u32 = 0x7fff_ffff;
/// The Goldilocks prime `2^64 - 2^32 + 1`.
const GOLDILOCKS_PRIME: u64 = 0xffff_ffff_0000_0001;

/// Reduce every 4-byte lane to the canonical m31 range `0..p`. Random
/// fixture bytes can exceed the prime, and a qm31 lane is two such limbs,
/// so one pass canonicalizes both fields. Runs outside every timed region.
fn canonicalize_m31_lanes(buf: &mut [u8]) {
    for lane in buf.chunks_exact_mut(4) {
        let value = u32::from_le_bytes(lane.try_into().unwrap()) % M31_PRIME;
        lane.copy_from_slice(&value.to_le_bytes());
    }
}

/// Reduce every 8-byte lane to the canonical Goldilocks range `0..p`.
/// Runs outside every timed region.
fn canonicalize_goldilocks_lanes(buf: &mut [u8]) {
    for lane in buf.chunks_exact_mut(8) {
        let value = u64::from_le_bytes(lane.try_into().unwrap()) % GOLDILOCKS_PRIME;
        lane.copy_from_slice(&value.to_le_bytes());
    }
}

/// Run `body` until it has been timed enough times to trust the median, and
/// report bytes per second over `bytes` of logical traffic per iteration.
///
/// Returns the median per-iteration time in nanoseconds so callers can print
/// a ratio between two shapes measured back to back in the same process.
fn bench(label: &str, bytes: usize, mut body: impl FnMut()) -> f64 {
    // Warm caches, branch predictors, and the lazy backend detection.
    for _ in 0..16 {
        body();
    }
    // One calibration call sizes every sample batch: short workloads run
    // enough repetitions to sit above clock and scheduler jitter, long
    // workloads keep the historical fixed batch.
    let reps = batch_reps(
        {
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64() * 1e9
        },
        32,
    );

    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        for _ in 0..reps {
            body();
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / reps as f64);
    }
    samples.sort_unstable_by(f64::total_cmp);
    let median = samples[samples.len() / 2];

    let gib_per_sec = bytes as f64 / (median / 1e9) / (1024.0 * 1024.0 * 1024.0);
    println!(
        "  {label:<44} {:>9}  {gib_per_sec:>7.2} GiB/s",
        fmt_ns(median)
    );
    println!("SAMPLE\t{label}\t{bytes}\t{median:.9}");
    median
}

/// Row lengths for the preparation crossover sweep: below one lane through
/// well past the point where the byte loop dominates table derivation.
const CROSSOVER_LENGTHS: &[usize] = &[
    16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 16_384, 65_536,
];

/// Where does preparing a coefficient stop paying for itself?
///
/// One-shot `mul_add` derives the backend's coefficient form on every call —
/// two broadcast words on GFNI, four nibble tables on a shuffle backend. The
/// `_with` form derives it once.
fn bench_preparation_crossover() {
    println!("preparation crossover — one-shot vs prepared, by row length:");
    for &len in CROSSOVER_LENGTHS {
        let src = noise(len, 0xa00 + len as u64);
        let mut dst = noise(len, 0xb00 + len as u64);
        let coeff8 = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53);
        let coeff16 = Elem::<Gf16>::from_raw(0x53a7);
        let prepared8 = ops::Coeff::<FiniteField<2, 8, Polynomial<AES>>>::new(coeff8);
        let prepared16 = ops::Coeff::<Gf16>::new(coeff16);

        println!("  row {len} B:");
        let one8 = bench("  mul_add one-shot           gf8b", len, || {
            ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                coeff8,
                black_box(&src),
            );
        });
        let with8 = bench("  mul_add prepared           gf8b", len, || {
            ops::mul_add_with::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                &prepared8,
                black_box(&src),
            );
        });
        let one16 = bench("  mul_add one-shot          gf16b", len, || {
            ops::mul_add::<Gf16>(black_box(&mut dst), coeff16, black_box(&src));
        });
        let with16 = bench("  mul_add prepared          gf16b", len, || {
            ops::mul_add_with::<Gf16>(black_box(&mut dst), &prepared16, black_box(&src));
        });
        println!(
            "    one-shot/prepared: gf8b {:.2}x  gf16b {:.2}x",
            one8 / with8,
            one16 / with16
        );
    }
    println!();
}

/// Payload lengths used by network-facing consumers.
///
/// The run around 1,200 bytes alternates exact 32-byte lanes with 16-byte
/// remainders so SIMD tail cliffs stay visible.
const NETWORK_LENGTHS: &[usize] = &[
    64, 256, 512, 1_152, 1_168, 1_184, 1_200, 1_216, 1_232, 1_248, 1_400,
];

fn bench_network_payloads() {
    println!("network-size GF(2^8) payloads:");
    for &len in NETWORK_LENGTHS {
        println!("  payload {len} B:");
        let src = noise(len, 0x700 + len as u64);
        let mut dst = noise(len, 0x800 + len as u64);

        bench("xor", len, || {
            ops::add_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                black_box(&src),
            );
        });
        bench("mul_add", len, || {
            ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                black_box(&src),
            );
        });
        bench("mul_assign", len, || {
            ops::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
            );
        });

        for nrows in [4usize, 16] {
            let coeffs: Vec<_> = (0..nrows)
                .map(|row| {
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                        (row as u8).wrapping_mul(37).wrapping_add(2),
                    )
                })
                .collect();
            let mut rows = noise(len * nrows, 0x900 + nrows as u64);
            let label = format!("scatter ({nrows} rows)");
            bench(&label, len * nrows, || {
                ops::mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut rows),
                    len,
                    black_box(&coeffs),
                    black_box(&src),
                );
            });
        }
    }
    println!();
}

/// Long batches isolate the payload loop from the clock and scheduler.
fn bench_network(label: &str, bytes: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..64 {
        body();
    }
    let reps = batch_reps(
        {
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64() * 1e9
        },
        512,
    );
    let mut samples = [0.0; 64];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..reps {
            body();
        }
        *sample = start.elapsed().as_secs_f64() * 1e9 / reps as f64;
    }
    samples.sort_unstable_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    println!(
        "  {label:<44} {:>9}  {:>7.2} GiB/s",
        fmt_ns(median),
        bytes as f64 / (median / 1e9) / (1024.0 * 1024.0 * 1024.0)
    );
    println!("SAMPLE\t{label}\t{bytes}\t{median:.9}");
    median
}

/// Time a scalar element chain: fixed `SCALAR_ITERS`-iteration bodies, one
/// timed call per sample. Emits a `SCALAR` machine record instead of
/// `SAMPLE`; a chain has no logical byte count.
fn bench_scalar(label: &str, mut body: impl FnMut()) {
    for _ in 0..4 {
        body();
    }
    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        body();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let ns = median.as_secs_f64() * 1e9 / SCALAR_ITERS as f64;
    let mops = SCALAR_ITERS as f64 / median.as_secs_f64() / 1e6;
    println!("  {label:<44} {ns:>9.2} ns/op  {mops:>7.2} Mops/s");
    println!("SCALAR\t{label}\t{ns:.9}");
}

/// Pinned diagnostic: fixed allocation alignment separates row pitch from
/// the allocator's base address.
fn bench_network_alignment() {
    for &len in &[1152usize, 1168, 1184, 1200] {
        let src_storage = noise(len + 128, 0x700 + len as u64);
        let mut dst_storage = noise(len + 128, 0x800 + len as u64);
        for &off in &[0usize, 16, 32, 48] {
            let s = src_storage.as_ptr().align_offset(64) + off;
            let d = dst_storage.as_ptr().align_offset(64) + off;
            let src = &src_storage[s..s + len];
            let dst = &mut dst_storage[d..d + len];
            println!(
                "network diagnostic {len} B off={off} dst={}",
                (dst.as_ptr() as usize) & 63
            );
            bench_network("xor", len, || {
                ops::add_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut *dst),
                    black_box(src),
                )
            });
            bench_network("mul_add", len, || {
                ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut *dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                    black_box(src),
                )
            });
            bench_network("mul_assign", len, || {
                ops::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut *dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                )
            });
            for nrows in [4usize, 16] {
                let coeffs: Vec<_> = (0..nrows)
                    .map(|row| {
                        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                            (row as u8).wrapping_mul(37).wrapping_add(2),
                        )
                    })
                    .collect();
                let mut storage = noise(len * nrows + 128, 0x900 + nrows as u64);
                let r = storage.as_ptr().align_offset(64) + off;
                println!(
                    "rows={nrows} base={}",
                    (storage[r..].as_ptr() as usize) & 63
                );
                let rows = &mut storage[r..r + len * nrows];
                bench_network(&format!("scatter {nrows}"), len * nrows, || {
                    ops::mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(
                        black_box(&mut *rows),
                        len,
                        black_box(&coeffs),
                        black_box(src),
                    )
                });
            }
        }
    }
}

/// Blocked multi-row GF(2^16) kernels measured against repeated single-row
/// AXPY, bypassing dispatch.
///
/// The SSSE3 and GFNI tiers dispatch the blocked gather. This harness times
/// each against repeated AXPY in one process through `internals`
/// token-proven entries. Diagnostics only: run it explicitly with
/// `--blocked-diagnostic`; the public panels never invoke it.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn bench_blocked_vs_axpy() {
    use fgf::internals::kernel::tables::{TowerCoeff, TowerTables};
    use fgf::internals::kernel::x86;
    use fgf::internals::kernel::{SimdToken, X64V2Token, X64V3GfniCryptoToken};

    // Genuine capability tokens, summoned once: each tier that the host
    // cannot prove is skipped rather than SIGILLed.
    let Some(ssse3) = X64V2Token::summon() else {
        return;
    };
    let gfni = X64V3GfniCryptoToken::summon();

    println!("blocked vs AXPY — direct GF(2^16) kernel calls (dispatch bypassed):");
    for &row_len in &[4 * 1024usize, 16 * 1024, 64 * 1024] {
        for &nsrc in &[2usize, 4, 8, 16] {
            let sources: Vec<Vec<u8>> = (0..nsrc)
                .map(|t| noise(row_len, 0xc00 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<Elem<Gf16>> = (0..nsrc)
                .map(|t| Elem::<Gf16>::from_raw(((t * 7919) as u16).wrapping_add(3)))
                .collect();
            let mut dst = noise(row_len, 0xd00);
            let traffic = row_len * nsrc;

            println!("  gather {nsrc} sources x {} KiB:", row_len / 1024);
            let blocked = bench("  gather blocked           ssse3", traffic, || {
                x86::gf16::mul_add_gather_ssse3(
                    ssse3,
                    black_box(&mut dst),
                    &coeffs,
                    black_box(&srcs),
                );
            });
            let axpy = bench("  gather AXPY              ssse3", traffic, || {
                for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                    x86::gf16::mul_add_ssse3(
                        ssse3,
                        black_box(&mut dst),
                        &TowerTables::new(coeff),
                        black_box(src),
                    );
                }
            });
            println!("    ssse3 blocked/AXPY: {:.2}x", axpy / blocked);

            if let Some(gfni) = gfni {
                let blocked = bench("  gather blocked            gfni", traffic, || {
                    x86::gf16::mul_add_gather_gfni(
                        gfni,
                        black_box(&mut dst),
                        &coeffs,
                        black_box(&srcs),
                    );
                });
                let axpy = bench("  gather AXPY               gfni", traffic, || {
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        x86::gf16::mul_add_gfni(
                            gfni,
                            black_box(&mut dst),
                            TowerCoeff::new(coeff),
                            black_box(src),
                        );
                    }
                });
                println!("    gfni  blocked/AXPY: {:.2}x", axpy / blocked);
            }
        }
    }
    println!();
}

/// Destination sizes straddling the non-temporal store threshold
/// (`x86::NT_STORE_MIN`, 2 MiB): one below it as a control, two above.
const LARGE_DST_LENGTHS: &[usize] = &[1 << 20, 8 << 20, 32 << 20];

/// `mul_into` over destinations large enough for the store policy to matter.
///
/// `mul_into` never reads its destination, so an ordinary store still fetches
/// every line it overwrites. Past 2 MiB the x86 kernels switch to `vmovntdq`,
/// which skips that fetch and does not keep the destination cached. The
/// read-back row is the workload that trade pessimizes: encode, then consume
/// the result immediately. `mul_add` is the control — it reads its
/// destination, so it is unaffected either way.
fn bench_large_destination() {
    println!("large destinations — mul_into store policy:");
    for &len in LARGE_DST_LENGTHS {
        let src = noise(len, 0xd00);
        let mut dst = noise(len, 0xd01);
        let mib = len / (1024 * 1024);

        bench(&format!("{mib:3} MiB mul_into          gf8b"), len, || {
            ops::mul_into::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                black_box(&src),
            );
        });
        bench(&format!("{mib:3} MiB mul_into         gf16b"), len, || {
            ops::mul_into::<Gf16>(
                black_box(&mut dst),
                Elem::<Gf16>::from_raw(0x53a7),
                black_box(&src),
            );
        });
        bench(&format!("{mib:3} MiB mul_into+read     gf8b"), len, || {
            ops::mul_into::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                black_box(&src),
            );
            let mut acc = 0u64;
            for word in dst.chunks_exact(8) {
                acc ^= u64::from_le_bytes(word.try_into().unwrap());
            }
            black_box(acc);
        });
        bench(&format!("{mib:3} MiB mul_add           gf8b"), len, || {
            ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                black_box(&src),
            );
        });
    }
    println!();
}

/// Multi-row scatter with the destination at each 32-byte residue.
///
/// A 32-byte access at an odd multiple of 32 touches two cache lines, and a
/// scatter loads and stores every row of a group once per source window, so
/// the destination side is where that doubles. The GFNI and AVX2 kernels peel
/// up to 31 bytes per row group to align the rest of the pass; the skew rows
/// below are what set that policy, and an allocator-fresh buffer alone will
/// not show it because its residue is fixed.
fn bench_destination_alignment() {
    println!("destination alignment — multi-row scatter:");
    let nrows = 8;
    for &row_len in &[64 * 1024usize, 256 * 1024] {
        let src = noise(row_len, 0xe00);
        let coeffs8: Vec<_> = (0..nrows)
            .map(|j| {
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                    (j as u8).wrapping_mul(37).wrapping_add(2),
                )
            })
            .collect();
        let coeffs16: Vec<_> = (0..nrows)
            .map(|j| Elem::<Gf16>::from_raw((j as u16).wrapping_mul(9871).wrapping_add(2)))
            .collect();
        let mut backing = noise(row_len * nrows + 79, 0xe01);
        let aligned = backing.as_ptr().align_offset(64);
        let kib = row_len / 1024;
        for skew in [0usize, 16] {
            let start = aligned + skew;
            let rows = &mut backing[start..start + row_len * nrows];
            assert_eq!(rows.as_ptr() as usize % 64, skew);
            let traffic = row_len * nrows;
            bench(
                &format!("{kib:4} KiB rows, skew {skew:2}   gf8b"),
                traffic,
                || {
                    ops::mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(
                        black_box(rows),
                        row_len,
                        &coeffs8,
                        black_box(&src),
                    );
                },
            );
            bench(
                &format!("{kib:4} KiB rows, skew {skew:2}  gf16b"),
                traffic,
                || {
                    ops::mul_add_scatter::<Gf16>(
                        black_box(rows),
                        row_len,
                        &coeffs16,
                        black_box(&src),
                    );
                },
            );
        }
    }
    println!();
}

/// Row lengths where a GF(2^16) coefficient's four nibble tables are a
/// visible share of the work, plus one length where they are not.
const SMALL_ROW_LENGTHS: &[usize] = &[64, 256, 1_024, 16_384];

/// Multi-row GF(2^16) shapes at row lengths where preparation dominates.
///
/// A GF(2^16) coefficient costs four derived nibble tables. Multi-row kernels
/// that rebuild them per `(term, row)` pay
/// that per row instead of once, which only shows up when the row is short
/// enough that the byte loop cannot hide it. Coefficient sets deliberately
/// include zeros and ones, the case the zero/one specializations exist for.
fn bench_small_row_shapes() {
    println!("small-row GF(2^16) multi-row shapes — preparation-dominated:");
    let nrows = 16;
    let nsrc = 8;
    for &row_len in SMALL_ROW_LENGTHS {
        let sources: Vec<Vec<u8>> = (0..nsrc)
            .map(|t| noise(row_len, 0xf00 + t as u64))
            .collect();
        let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        // Every fourth coefficient is zero and every fifth is one.
        let coeff_at = |i: usize| match i % 5 {
            0 => Elem::<Gf16>::ZERO,
            1 => Elem::<Gf16>::ONE,
            _ => Elem::<Gf16>::from_raw(((i * 7919) as u16) | 0x0100),
        };
        let row_coeffs: Vec<Elem<Gf16>> = (0..nrows).map(coeff_at).collect();
        let coeff_sets: Vec<Vec<Elem<Gf16>>> = (0..nsrc)
            .map(|t| (0..nrows).map(|j| coeff_at(t * 3 + j)).collect())
            .collect();
        let terms: Vec<(&[Elem<Gf16>], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let mut rows = noise(row_len * nrows, 0x1000);
        let mut dst = noise(row_len, 0x1100);

        println!("  row {row_len} B, {nrows} rows, {nsrc} sources:");
        bench("  scatter                   gf16b", row_len * nrows, || {
            ops::mul_add_scatter::<Gf16>(
                black_box(&mut rows),
                row_len,
                &row_coeffs,
                black_box(srcs[0]),
            );
        });
        bench("  gather                    gf16b", row_len * nsrc, || {
            ops::mul_add_gather::<Gf16>(black_box(&mut dst), &row_coeffs[..nsrc], black_box(&srcs));
        });
        bench(
            "  matrix                    gf16b",
            row_len * nrows * nsrc,
            || {
                ops::mul_add_matrix::<Gf16>(black_box(&mut rows), row_len, nrows, &terms);
            },
        );

        // Fused `mul_into` against the copy-then-scale it replaces: the
        // fused kernel touches the destination once.
        let coeff = Elem::<Gf16>::from_raw(0x53a7);
        let fused = bench("  mul_into fused            gf16b", row_len, || {
            ops::mul_into::<Gf16>(black_box(&mut dst), coeff, black_box(srcs[0]));
        });
        let copied = bench("  mul_into copy+scale       gf16b", row_len, || {
            let dst = black_box(&mut dst);
            dst.copy_from_slice(black_box(srcs[0]));
            ops::mul_assign::<Gf16>(dst, coeff);
        });
        println!("    copy+scale/fused: {:.2}x", copied / fused);
    }
    println!();
}

/// The benchmark-owned gf16d presentation: GF(2^16) as `a + b*t` over
/// `FiniteField<2, 8, Polynomial<RS>>` under the relation `t^2 + 2*t + 0x80`. The
/// raw word packs the low component first (`a | b << 8`). The crate's
/// kernel tests exercise the same relation; no shipped instance matches it,
/// so every bulk operation routes through the typed scalar fallback and
/// `backend_for` reports `Scalar` on every host.
struct Rs16Spec;

impl ExtensionSpec<2> for Rs16Spec {
    type Base = FiniteField<2, 8, Polynomial<RS>>;
    type Storage = Packed<u16>;
    const NAME: &'static str = "GF(2^16)/0x11D";
    const RELATION: Relation =
        Relation::from_binary_words::<FiniteField<2, 8, Polynomial<RS>>>(&[0x80, 2]);
}

/// The gf16d field: a degree-16 extension over the Reed–Solomon byte field.
type Gf16d = FiniteField<2, 16, Extension<Rs16Spec, 2>>;

/// The coefficient the gf16d regions multiply by: nonzero in both
/// components.
const GF16D_COEFF_RAW: u16 = 0x53a7;

/// Row lengths shared by the custom gf16d one-shot and prepared panels.
const GF16D_LENGTHS: &[usize] = &[64, 1_024, 16_384, 65_536];

/// Independent GF(2^8) multiplication by `0x11D`: carry-less multiplication
/// with interleaved reduction, sharing no code with the crate's arithmetic.
fn oracle_gf8_mul(a: u8, b: u8) -> u8 {
    let mut acc = 0u16;
    let mut a = u16::from(a);
    let mut b = u16::from(b);
    loop {
        if b & 1 == 1 {
            acc ^= a;
        }
        b >>= 1;
        if b == 0 {
            return acc as u8;
        }
        a <<= 1;
        if a & 0x100 != 0 {
            a ^= 0x11d;
        }
    }
}

/// Independent gf16d multiplication: component schoolbook over
/// `t^2 = 2*t + 0x80` in the oracle byte field, low component first.
fn oracle_gf16d_mul(x: u16, y: u16) -> u16 {
    let (x0, x1) = (x as u8, (x >> 8) as u8);
    let (y0, y1) = (y as u8, (y >> 8) as u8);
    let cross = oracle_gf8_mul(x1, y1);
    let lo = oracle_gf8_mul(x0, y0) ^ oracle_gf8_mul(0x80, cross);
    let hi = oracle_gf8_mul(x0, y1) ^ oracle_gf8_mul(x1, y0) ^ oracle_gf8_mul(2, cross);
    u16::from(lo) | (u16::from(hi) << 8)
}

/// Pre-timing oracle check: the crate's gf16d scalar multiply must agree
/// with the independent relation product on the measured coefficient plus
/// boundary and pseudo-random values. A mismatch aborts the run before any
/// timed number exists.
fn assert_gf16d_oracle() {
    let mut cases = vec![
        GF16D_COEFF_RAW,
        0x0000,
        0x0001,
        0x0002,
        0x00ff,
        0x0100,
        0x8000,
        0xffff,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..8 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        cases.push((state >> 48) as u16);
    }
    for &x in &cases {
        for &y in &cases {
            let got = Elem::<Gf16d>::from_raw(x)
                .mul(Elem::<Gf16d>::from_raw(y))
                .to_raw();
            let want = oracle_gf16d_mul(x, y);
            assert_eq!(
                got, want,
                "gf16d oracle mismatch: {x:#06x} * {y:#06x} = {got:#06x}, want {want:#06x}"
            );
        }
    }
}

/// The gf16d panel: one-shot and prepared `mul_add` over representative row
/// lengths plus the scalar element chain. Preparation happens outside the
/// timed regions, and the oracle check above gates every number the panel
/// prints.
fn bench_gf16d_custom_tower() {
    println!("custom scalar-routed tower gf16d — GF(2^16)/0x11D, A=2 B=0x80:");
    assert_gf16d_oracle();
    let coeff = Elem::<Gf16d>::from_raw(GF16D_COEFF_RAW);
    let prepared = ops::Coeff::<Gf16d>::new(coeff);
    for &len in GF16D_LENGTHS {
        let src = noise(len, 0x3a00 + len as u64);
        let mut dst = noise(len, 0x3b00 + len as u64);
        println!("  row {len} B:");
        bench("  mul_add one-shot          gf16d", len, || {
            ops::mul_add::<Gf16d>(black_box(&mut dst), coeff, black_box(&src));
        });
        bench("  mul_add prepared          gf16d", len, || {
            ops::mul_add_with::<Gf16d>(black_box(&mut dst), &prepared, black_box(&src));
        });
    }
    bench_scalar("gf16d scalar mul", || {
        let mut x = Elem::<Gf16d>::from_raw(0x1234);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(Elem::<Gf16d>::from_raw((0x53a7u16).wrapping_add(i as u16)));
        }
        black_box(x);
    });
    println!();
}

/// Bit-packed GF(2) surface: xor (dispatched byte XOR), the portable AND
/// word loop, the popcount folds, and a masked range — against the
/// byte-per-element GF(2^8) add as the density control. The packing itself
/// is the expected win (8x the elements per byte); the interesting number
/// is whether the portable loops sit at the same memory ceiling as the
/// dispatched kernel.
fn bench_gf2_bits() {
    println!("bit-packed GF(2) — packed vs byte-per-element control:");
    // L1-resident, L2-resident, and DRAM-resident buffer sizes.
    for &bytes in &[4 * 1024usize, 256 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024] {
        let bits = 8 * bytes;
        let human = if bytes >= 1024 * 1024 {
            format!("{} MiB", bytes / (1024 * 1024))
        } else {
            format!("{} KiB", bytes / 1024)
        };
        let a = noise(bytes, 0x2100 + bytes as u64);
        let b = noise(bytes, 0x2200 + bytes as u64);
        let mut dst = noise(bytes, 0x2300 + bytes as u64);
        println!("  {human} buffers ({bits} elements):");
        bench("  bits::xor_assign              packed", bytes, || {
            fgf::bits::xor_assign(black_box(&mut dst), black_box(&a));
        });
        bench("  bits::and_into         packed", bytes, || {
            fgf::bits::and_into(black_box(&mut dst), black_box(&a), black_box(&b));
        });
        bench("  bits::weight           packed", bytes, || {
            black_box(fgf::bits::weight(black_box(&a), bits));
        });
        bench("  bits::dot_product       packed", bytes, || {
            black_box(fgf::bits::dot_product(black_box(&a), black_box(&b), bits));
        });
        let from = bits / 4;
        let to = bits - bits / 8;
        bench("  bits::xor_range  5/8    packed", bytes, || {
            fgf::bits::xor_range(black_box(&mut dst), black_box(&a), bits, from, to);
        });
        let plan = fgf::bits::XorRange::new(bits, from, to);
        bench("  bits::XorRange   5/8    packed", bytes, || {
            fgf::bits::xor_range_with(black_box(&mut dst), black_box(&a), black_box(&plan));
        });
        bench("  ops::add_assign  gf8b control", bytes, || {
            ops::add_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut dst),
                black_box(&a),
            );
        });
    }
    println!();
}

/// The GF(2) elimination's per-call shape: short rows, sub-word ranges.
///
/// An elimination XORs an eight-bit range of one row into another, once
/// per row per pivot. Bandwidth is irrelevant at this size — the cost is
/// the per-call surface — so this section compares the checked one-shot
/// `xor_range` against the prepared `XorRange` + `xor_range_with` split
/// over the same buffer pairs. The range `[61, 69)` straddles a byte
/// boundary, the worst two-byte window.
fn bench_gf2_short_rows() {
    const ROWS: usize = 512;
    const ROW_BYTES: usize = 16; // 128 elements: the 128-column row shape
    let mut dst = noise(ROWS * ROW_BYTES, 0x5100);
    let src = noise(ROWS * ROW_BYTES, 0x5200);
    let bits = ROW_BYTES * 8;
    let pairs = ROWS / 2;
    println!("  {pairs} row pairs, 16-byte rows, bits 61..69:");
    let one_shot = bench("  bits::xor_range  8-bit row", ROW_BYTES * pairs, || {
        for i in 0..pairs {
            let off = i * 2 * ROW_BYTES;
            fgf::bits::xor_range(
                black_box(&mut dst[off..off + ROW_BYTES]),
                black_box(&src[off..off + ROW_BYTES]),
                bits,
                61,
                69,
            );
        }
    });
    let plan = fgf::bits::XorRange::new(bits, 61, 69);
    let prepared = bench("  bits::XorRange   8-bit row", ROW_BYTES * pairs, || {
        for i in 0..pairs {
            let off = i * 2 * ROW_BYTES;
            fgf::bits::xor_range_with(
                black_box(&mut dst[off..off + ROW_BYTES]),
                black_box(&src[off..off + ROW_BYTES]),
                black_box(&plan),
            );
        }
    });
    let per_call = |d: f64| d / pairs as f64;
    println!(
        "  per call: one-shot {:.2} ns, prepared {:.2} ns ({:.2}x)\n",
        per_call(one_shot),
        per_call(prepared),
        one_shot / prepared
    );
}

/// The row-interleaved addition matrix: flat `add_assign` against
/// `add_assign_rows` and a literal row-by-row loop, over the row counts and
/// row lengths where four-stream interleaving should pay (and its edges,
/// where it must not regress). Rerun under each `SIMD_BACKEND` tier for the
/// per-backend numbers; the crossover thresholds in `kernel::xor_rows` were
/// set from this matrix on the reference host.
fn bench_add_assign_rows<F: fgf::FieldKernels>(name: &str, canon: Option<fn(&mut [u8])>) {
    println!("add_assign_rows — {name}:");
    for &row_len in &[64usize, 1024, 64 * 1024] {
        for &rows in &[1usize, 2, 4, 8, 16, 32] {
            let len = row_len * rows;
            let mut src = noise(len, 0x2a00 + row_len as u64);
            let mut dst = noise(len, 0x2b00 + rows as u64);
            // Prime fields need canonical lanes on every timed input; the
            // binary fields read the same bytes as pure XOR vectors.
            if let Some(canon) = canon {
                canon(&mut src);
                canon(&mut dst);
            }

            let flat = bench("  flat add_assign", len, || {
                ops::add_assign::<F>(black_box(&mut dst), black_box(&src));
            });
            let interleaved = bench("  add_assign_rows", len, || {
                ops::add_assign_rows::<F>(black_box(&mut dst), row_len, black_box(&src));
            });
            let looped = bench("  add_assign per row", len, || {
                let dst = black_box(&mut dst);
                let src = black_box(&src);
                for (d, s) in dst.chunks_exact_mut(row_len).zip(src.chunks_exact(row_len)) {
                    ops::add_assign::<F>(d, s);
                }
            });
            println!(
                "    rows/flat: {:.2}x, rows/loop: {:.2}x  ({rows} rows x {row_len} B)",
                flat / interleaved,
                looped / interleaved,
            );
        }
    }
    println!();
}

fn main() {
    println!("fgf kernel benchmark — backend: {}", backend().name());
    print_host_lines();
    print_field_backends();
    println!("control byte xor — drift check:");
    let control_src = noise(CONTROL_BYTES, 0x57a7);
    let mut control_dst = noise(CONTROL_BYTES, 0x57a8);
    bench("control byte xor start", CONTROL_BYTES, || {
        control_body(&mut control_dst, &control_src)
    });
    run_panels();
    println!("control byte xor — drift check:");
    bench("control byte xor end", CONTROL_BYTES, || {
        control_body(&mut control_dst, &control_src)
    });
}

/// Independent lane word: little-endian bytes combined with no crate
/// arithmetic, so the panel oracles share no code with the kernels.
#[allow(clippy::cast_possible_truncation)]
fn prime_lane(buf: &[u8], bytes: usize, lane: usize) -> u64 {
    let mut word = 0u64;
    for b in 0..bytes {
        word |= u64::from(buf[lane * bytes + b]) << (8 * b);
    }
    word
}

/// Assert every lane of `buf` holds exactly its canonical expectation.
/// Inputs may be any lane bit pattern (lanes canonicalize on load), but
/// every measured output path stores canonical lanes, so the comparison
/// is exact rather than modular.
fn prime_check(buf: &[u8], bytes: usize, expect: &[u64], what: &str) {
    assert_eq!(buf.len(), expect.len() * bytes, "{what}: length");
    for (i, &want) in expect.iter().enumerate() {
        let got = prime_lane(buf, bytes, i);
        assert!(got == want, "{what} lane {i}: got {got} want {want}");
    }
}

/// One aligned backing region per panel; every buffer is a slice of it at
/// a fixed relative offset, so allocator placement is the only thing the
/// placement rotation varies. Returns the store and its first
/// 64-byte-aligned base.
fn prime_arena(total: usize) -> (Vec<u8>, usize) {
    let backing = vec![0u8; total + 64];
    let misalignment = backing.as_ptr() as usize % 64;
    (backing, (64 - misalignment) % 64)
}

/// Placement rotation: the whole panel footprint shifts through these
/// relative offsets, and each panel reports the median over placements.
const PRIME_PLACEMENTS: &[usize] = &[0, 16, 32];

/// Median of per-placement medians.
fn prime_median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}
/// Time `body` once per placement over freshly staged slices and report
/// the median over placements. Staging (pristine copies into the
/// placement's slices) stays outside the timed region.
#[allow(clippy::too_many_arguments)]
fn prime_time_placed(
    label: &str,
    len: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    src0: &[u8],
    dst0: &[u8],
    mut body: impl FnMut(&mut [u8], &[u8]),
) {
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        let region = &mut backing[base + off..base + off + need];
        let (dst, rest) = region.split_at_mut(len);
        let (src, _) = rest.split_at_mut(len);
        dst.copy_from_slice(dst0);
        src.copy_from_slice(src0);
        medians.push(bench(&format!("{label} place{off}"), len, || {
            body(black_box(&mut *dst), black_box(&*src))
        }));
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{len}\t{median:.9}");
}

/// Three-buffer form of [`prime_time_placed`], for elementwise panels.
#[allow(clippy::too_many_arguments)]
fn prime_time3_placed(
    label: &str,
    len: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    dst0: &[u8],
    a0: &[u8],
    b0: &[u8],
    mut body: impl FnMut(&mut [u8], &[u8], &[u8]),
) {
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        let region = &mut backing[base + off..base + off + need];
        let (dst, rest) = region.split_at_mut(len);
        let (a, b) = rest.split_at_mut(len);
        dst.copy_from_slice(dst0);
        a.copy_from_slice(a0);
        b.copy_from_slice(b0);
        medians.push(bench(&format!("{label} place{off}"), len, || {
            body(black_box(&mut *dst), black_box(&*a), black_box(&*b))
        }));
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{len}\t{median:.9}");
}
/// Direct column helpers below call the x86 token-gated entries where
/// available and the portable body elsewhere; every helper runs on all
/// targets so the panel structure stays identical.
/// Direct column for one elementwise shape: the narrow-Barrett body called
/// directly through `internals` with the exact token, or the portable body
/// for classes below the floor. Times through the same placement rotation
/// as the production column.
#[allow(clippy::too_many_arguments)]
fn prime_direct_elementwise<F>(
    label: &str,
    len: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    dst0: &[u8],
    a0: &[u8],
    b0: &[u8],
    p: u64,
    assign: bool,
) where
    F: fgf::FieldKernels + FieldBuffer,
{
    // The direct column calls the kernel's own body wherever its
    // preconditions allow (four-byte rows at any length), independent of
    // the routing selector. `takes` below only selects the oracle path.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let takes = F::BYTES == 4;
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let takes = false;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let params = fgf::internals::kernel::x86::residue::U32Params::new(p as u32);
    // Rows the kernel rejects print no direct cell.
    let kernel_accepts = F::BYTES == 4;
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        if !kernel_accepts {
            break;
        }
        let region = &mut backing[base + off..base + off + need];
        let (dst, rest) = region.split_at_mut(len);
        let (a, b) = rest.split_at_mut(len);
        dst.copy_from_slice(dst0);
        a.copy_from_slice(a0);
        b.copy_from_slice(b0);
        medians.push(bench(&format!("{label} place{off}"), len, || {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                use fgf::internals::kernel::{SimdToken, X64V3Token};
                if takes {
                    let Some(token) = X64V3Token::summon() else {
                        return;
                    };
                    {
                        use fgf::internals::kernel::x86::residue as direct;
                        if assign {
                            direct::mul_elementwise_assign_u32_avx2(
                                token,
                                black_box(&mut *dst),
                                black_box(&*b),
                                params,
                            );
                        } else {
                            direct::mul_elementwise_u32_avx2(
                                token,
                                black_box(&mut *dst),
                                black_box(&*a),
                                black_box(&*b),
                                params,
                            );
                        }
                        return;
                    }
                }
            }
            if assign {
                fgf::internals::kernel::scalar::mul_elementwise_assign::<F>(
                    black_box(&mut *dst),
                    black_box(&*b),
                );
            } else {
                fgf::internals::kernel::prime::mul_elementwise::<F>(
                    black_box(&mut *dst),
                    black_box(&*a),
                    black_box(&*b),
                );
            }
        }));
    }
    if medians.is_empty() {
        println!("DIRECT_SKIP\t{label}\t{len}");
        return;
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{len}\t{median:.9}");
}
/// Direct column for one matrix shape: the delayed body called directly
/// through `internals` with the exact token and a `FlatMatrix` over the
/// row's prepared words, or the portable composition below the tile shape.
/// Times through the same placement rotation as the production column.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn prime_direct_matrix<const P: u64>(
    label: &str,
    traffic: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    row_len: usize,
    nsources: usize,
    src0: &[Vec<u8>],
    dst0: &[u8],
    coeff_raw: &[u64],
    gather_vecs: &[ops::CoeffVec<FiniteField<P, 1>>],
    add: bool,
) {
    // The direct column calls the delayed body wherever the kernel's own
    // preconditions allow (four-byte row, delay depth at least two, four
    // outputs), independent of the routing selector.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let takes = {
        let k = ((1u128 << 32) / P as u128) as u64;
        k >= 2
    };
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let takes = false;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let flat: Vec<fgf::internals::kernel::x86::residue::PreparedResidue> = coeff_raw
        .iter()
        .map(|&c| {
            let q = ((c as u128 * (1u128 << 32)) / P as u128) as u64;
            fgf::internals::kernel::x86::residue::PreparedResidue::from_word(c | (q << 32))
        })
        .collect();
    // Rows the kernel rejects (delay depth below two) print no direct
    // cell; the production and gathers columns still time.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let kernel_accepts = {
        let k = ((1u128 << 32) / P as u128) as u64;
        k >= 2
    };
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let kernel_accepts = false;
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        if !kernel_accepts {
            break;
        }
        let region = &mut backing[base + off..base + off + need];
        let (src_area, rows_area) = region.split_at_mut(nsources * row_len);
        for (slot, staged) in src_area.chunks_exact_mut(row_len).zip(src0) {
            slot.copy_from_slice(staged);
        }
        rows_area.copy_from_slice(dst0);
        let src_refs: Vec<&[u8]> = src_area.chunks_exact(row_len).collect();
        medians.push(bench(&format!("{label} place{off}"), traffic, || {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                use fgf::internals::kernel::{FlatMatrix, SimdToken, X64V3Token};
                if takes {
                    let Some(token) = X64V3Token::summon() else {
                        return;
                    };
                    {
                        use fgf::internals::kernel::x86::residue as direct;
                        let terms = FlatMatrix {
                            coefficients: &flat,
                            nrows: 4,
                            sources: &src_refs,
                        };
                        if add {
                            direct::mul_add_matrix_u32_avx2_with(
                                token,
                                black_box(&mut *rows_area),
                                row_len,
                                4,
                                &terms,
                                P as u32,
                            );
                        } else {
                            direct::mul_into_matrix_u32_avx2_with(
                                token,
                                black_box(&mut *rows_area),
                                row_len,
                                4,
                                &terms,
                                P as u32,
                            );
                        }
                        return;
                    }
                }
            }
            // Portable body: the public gather composition the dispatch
            // runs below the tile shape (repeated single-row path), over
            // the same prepared per-output vectors as the control column.
            for (j, vec) in gather_vecs.iter().enumerate() {
                if add {
                    ops::mul_add_gather_with::<FiniteField<P, 1>>(
                        &mut rows_area[j * row_len..(j + 1) * row_len],
                        vec.as_ref(),
                        &src_refs,
                    );
                } else {
                    ops::mul_into_gather_with::<FiniteField<P, 1>>(
                        &mut rows_area[j * row_len..(j + 1) * row_len],
                        vec.as_ref(),
                        &src_refs,
                    );
                }
            }
        }));
    }
    if medians.is_empty() {
        println!("DIRECT_SKIP\t{label}\t{traffic}");
        return;
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{traffic}\t{median:.9}");
}

/// Matrix form: `nsources` source rows then four output rows, timed as one
/// body over the whole footprint. Traffic counts row bytes times sources
/// times outputs.
#[allow(clippy::too_many_arguments)]
fn prime_time_matrix_placed(
    label: &str,
    traffic: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    row_len: usize,
    nsources: usize,
    src0: &[Vec<u8>],
    dst0: &[u8],
    mut body: impl FnMut(&mut [u8], &[&[u8]]),
) {
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        let region = &mut backing[base + off..base + off + need];
        let (src_area, rows_area) = region.split_at_mut(nsources * row_len);
        for (slot, staged) in src_area.chunks_exact_mut(row_len).zip(src0) {
            slot.copy_from_slice(staged);
        }
        rows_area.copy_from_slice(dst0);
        let src_refs: Vec<&[u8]> = src_area.chunks_exact(row_len).collect();
        medians.push(bench(&format!("{label} place{off}"), traffic, || {
            body(black_box(&mut *rows_area), black_box(&src_refs))
        }));
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{traffic}\t{median:.9}");
}
/// Which single-row operation the direct column times.
#[derive(Clone, Copy)]
enum DirectOp {
    Into,
    Add,
    Assign,
    Once,
}

/// Direct column for one single-row shape: the vector body called directly
/// through `internals` with the exact token, or the portable body for
/// classes below a floor or without a token on this host. Times through
/// the same placement rotation as the production column. The `assign`
/// body restores the staged destination after each call exactly as the
/// production column does, so both columns time the same reset shape.
/// The one-shot `Once` column prepares once outside timing and reuses the
/// prepared word, matching the production one-shot's steady state without
/// its per-call preparation.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn prime_direct_single<F>(
    label: &str,
    len: usize,
    backing: &mut [u8],
    base: usize,
    need: usize,
    src0: &[u8],
    dst0: &[u8],
    coeff: Elem<F>,
    c: u64,
    p: u64,
    op: DirectOp,
) where
    F: fgf::FieldKernels + FieldBuffer,
{
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use fgf::internals::kernel::residue as selector;
    let bytes = F::BYTES;
    // The direct column calls the vector body for every kernel-covered
    // class at every measured length, independent of the routing floors.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let takes = selector::routes_vector(bytes, p);
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let takes = false;
    // Prepared word parts for the direct vector call, derived exactly as
    // dispatch derives them: Shoup pair halves on four-byte rows,
    // Montgomery word on eight-byte odd rows, canonical byte on byte rows.
    let (c32, q32, c_mont, c8) = direct_coefficient_parts(c, p, bytes);
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let params = direct_u32_params(p);
    // Rows the kernel rejects print no direct cell.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let kernel_accepts = selector::routes_vector(F::BYTES, p);
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let kernel_accepts = false;
    let mut medians = Vec::with_capacity(PRIME_PLACEMENTS.len());
    for &off in PRIME_PLACEMENTS {
        if !kernel_accepts {
            break;
        }
        let region = &mut backing[base + off..base + off + need];
        let (dst, rest) = region.split_at_mut(len);
        let (src, _) = rest.split_at_mut(len);
        dst.copy_from_slice(dst0);
        src.copy_from_slice(src0);
        medians.push(bench(&format!("{label} place{off}"), len, || {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            direct_single_body::<F>(
                black_box(&mut *dst),
                black_box(&*src),
                black_box(dst0),
                coeff,
                c32,
                q32,
                c_mont,
                c8,
                params,
                p,
                bytes,
                takes,
                op,
            );
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                use fgf::internals::kernel::prime as portable;
                match op {
                    DirectOp::Into | DirectOp::Once => {
                        portable::mul_into::<F>(black_box(&mut *dst), coeff, black_box(&*src))
                    }
                    DirectOp::Add => {
                        portable::mul_add::<F>(black_box(&mut *dst), coeff, black_box(&*src))
                    }
                    DirectOp::Assign => {
                        portable::mul_assign::<F>(black_box(&mut *dst), coeff);
                        black_box(&mut *dst).copy_from_slice(black_box(dst0));
                    }
                }
            }
        }));
    }
    if medians.is_empty() {
        println!("DIRECT_SKIP\t{label}\t{len}");
        return;
    }
    let median = prime_median(&mut medians);
    println!("PLACEMENT\t{label}\t{len}\t{median:.9}");
}

/// The four-byte modulus parameters for the direct elementwise call,
/// derived exactly as the row's monomorphized body derives them.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn direct_u32_params(p: u64) -> fgf::internals::kernel::x86::residue::U32Params {
    fgf::internals::kernel::x86::residue::U32Params::new(p as u32)
}

/// Split the reduced coefficient into the direct call's word parts.
fn direct_coefficient_parts(c: u64, p: u64, bytes: usize) -> (u32, u32, u64, u8) {
    if bytes == 4 {
        let q = ((c as u128 * (1u128 << 32)) / p as u128) as u64;
        (c as u32, q as u32, 0, 0)
    } else if bytes == 8 && p % 2 == 1 {
        (0, 0, ((c as u128 * (1u128 << 64)) % p as u128) as u64, 0)
    } else {
        (0, 0, 0, c as u8)
    }
}

/// One timed direct call for the single-row direct column.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn direct_single_body<F>(
    dst: &mut [u8],
    src: &[u8],
    staged: &[u8],
    coeff: Elem<F>,
    c32: u32,
    q32: u32,
    c_mont: u64,
    c8: u8,
    params: fgf::internals::kernel::x86::residue::U32Params,
    p: u64,
    bytes: usize,
    takes: bool,
    op: DirectOp,
) where
    F: fgf::FieldKernels + FieldBuffer,
{
    use fgf::internals::kernel::prime as portable;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        use fgf::internals::kernel::{SimdToken, X64V3Token};
        if takes {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            {
                use fgf::internals::kernel::x86::residue as direct;
                match (bytes, p, op) {
                    (1, _, DirectOp::Into) | (1, _, DirectOp::Once) => {
                        direct::mul_into_u8_avx2(token, dst, src, c8, p as u8);
                        return;
                    }
                    (1, _, DirectOp::Add) => {
                        direct::mul_add_u8_avx2(token, dst, src, c8, p as u8);
                        return;
                    }
                    (1, _, DirectOp::Assign) => {
                        direct::mul_assign_u8_avx2(token, dst, c8, p as u8);
                        dst.copy_from_slice(staged);
                        return;
                    }
                    (4, _, DirectOp::Into) | (4, _, DirectOp::Once) => {
                        direct::mul_into_u32_avx2(token, dst, src, c32, q32, p as u32);
                        return;
                    }
                    (4, _, DirectOp::Add) => {
                        direct::mul_add_u32_avx2(token, dst, src, c32, q32, p as u32);
                        return;
                    }
                    (4, _, DirectOp::Assign) => {
                        direct::mul_assign_u32_avx2(token, dst, c32, q32, p as u32);
                        dst.copy_from_slice(staged);
                        return;
                    }
                    (8, _, DirectOp::Into) | (8, _, DirectOp::Once) => {
                        direct::mul_into_u64_avx2(token, dst, src, c_mont, p);
                        return;
                    }
                    (8, _, DirectOp::Add) => {
                        direct::mul_add_u64_avx2(token, dst, src, c_mont, p);
                        return;
                    }
                    (8, _, DirectOp::Assign) => {
                        direct::mul_assign_u64_avx2(token, dst, c_mont, p);
                        dst.copy_from_slice(staged);
                        return;
                    }
                    _ => {}
                }
            }
        }
        let _ = params;
    }
    // Portable body: the exact fallback dispatch runs below a floor.
    match op {
        DirectOp::Into | DirectOp::Once => portable::mul_into::<F>(dst, coeff, src),
        DirectOp::Add => portable::mul_add::<F>(dst, coeff, src),
        DirectOp::Assign => {
            portable::mul_assign::<F>(dst, coeff);
            dst.copy_from_slice(staged);
        }
    }
}

/// Single-row panels for one residue row: each of prepared `mul_into_with`,
/// `mul_add_with`, and `mul_assign_with`, plus one-shot `mul_into`, times
/// three columns in one process — production (the public `ops` call,
/// whatever the selector picks), direct (the vector body called directly
/// through `internals` with the exact token, or the portable body for
/// classes below a floor), and control. For single-row shapes the control
/// is the `SIMD_BACKEND=scalar` arm of the same artifact; the routed arm
/// still prints it as the no-extra-work reference. `c` is the reduced
/// coefficient value; the one-shot coefficient is never 0 or 1 except on
/// the `P = 2` control, whose only residues are 0 and 1.
///
/// Each row prints its `backend_for` and, from the library's own selector,
/// whether the production path takes the vector body for that geometry.
/// Fixtures are raw noise (lanes canonicalize on load); the `P = 2`
/// fixtures are canonical single bits so the unit shortcut the control
/// must take stays byte-comparable. Overwrite and accumulation bodies run
/// back to back with no reset: every iteration keeps canonical lanes, so
/// the steady state is the timed shape. Before timing, every operation's
/// output is checked against the independent `u128` schoolbook.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn prime_single_row_panel<F>(row: &str, p: u64, coeff: Elem<F>, c: u64, seed: u64)
where
    F: fgf::FieldKernels + FieldBuffer,
{
    let bytes = F::BYTES;
    println!("prime row {row} — backend {}:", backend_for::<F>().name());
    println!("PRIME_ROW\t{row}\t{}", backend_for::<F>().name());
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let _v3 = {
        use fgf::internals::kernel::{SimdToken, X64V3Token};
        X64V3Token::summon()
    };
    let prepared = ops::Coeff::<F>::new(coeff);
    for &lanes in &[8usize, 16, 32, 64, 256, 1024] {
        let len = lanes * bytes;
        let mut src0 = noise(len, seed);
        let mut dst0 = noise(len, seed ^ 0x9e37);
        if p == 2 {
            for buf in [&mut src0, &mut dst0] {
                for lane in buf.chunks_exact_mut(bytes) {
                    let bit = lane[0] & 1;
                    lane.fill(0);
                    lane[0] = bit;
                }
            }
        }
        let modulus = p as u128;
        let mul = |x: u64| ((x as u128 * c as u128) % modulus) as u64;
        let src_r: Vec<u64> = (0..lanes)
            .map(|i| prime_lane(&src0, bytes, i) % p)
            .collect();
        let dst_r: Vec<u64> = (0..lanes)
            .map(|i| prime_lane(&dst0, bytes, i) % p)
            .collect();
        let exp_into: Vec<u64> = src_r.iter().map(|&s| mul(s)).collect();
        let exp_add: Vec<u64> = dst_r
            .iter()
            .zip(&src_r)
            .map(|(&d, &s)| ((d as u128 + mul(s) as u128) % modulus) as u64)
            .collect();
        let exp_assign: Vec<u64> = dst_r.iter().map(|&d| mul(d)).collect();
        let mut probe = dst0.clone();
        ops::mul_into_with::<F>(&mut probe, &prepared, &src0);
        prime_check(
            &probe,
            bytes,
            &exp_into,
            &format!("prime oracle with {row} into {lanes}"),
        );
        probe.copy_from_slice(&dst0);
        ops::mul_add_with::<F>(&mut probe, &prepared, &src0);
        prime_check(
            &probe,
            bytes,
            &exp_add,
            &format!("prime oracle with {row} add {lanes}"),
        );
        probe.copy_from_slice(&dst0);
        ops::mul_assign_with::<F>(&mut probe, &prepared);
        prime_check(
            &probe,
            bytes,
            &exp_assign,
            &format!("prime oracle with {row} assign {lanes}"),
        );
        // Direct-column oracle: the same bodies the timed direct column
        // calls, checked against the same expectations before timing.
        // Below a floor the direct column runs the portable body, which
        // the production oracle above already covers.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            use fgf::internals::kernel::residue as selector_oracle;
            if !selector_oracle::routes_vector(bytes, p) {
                // Kernel rejects this class; no direct cell exists and the
                // production oracle above covers the portable path.
            } else {
                let (c32, q32, c_mont, c8) = direct_coefficient_parts(c, p, bytes);
                let params = direct_u32_params(p);
                let mut direct = dst0.clone();
                let mut direct_src = src0.clone();
                direct_single_body::<F>(
                    &mut direct,
                    &direct_src,
                    &dst0,
                    coeff,
                    c32,
                    q32,
                    c_mont,
                    c8,
                    params,
                    p,
                    bytes,
                    true,
                    DirectOp::Into,
                );
                prime_check(
                    &direct,
                    bytes,
                    &exp_into,
                    &format!("prime direct oracle with {row} into {lanes}"),
                );
                direct.copy_from_slice(&dst0);
                direct_src.copy_from_slice(&src0);
                direct_single_body::<F>(
                    &mut direct,
                    &direct_src,
                    &dst0,
                    coeff,
                    c32,
                    q32,
                    c_mont,
                    c8,
                    params,
                    p,
                    bytes,
                    true,
                    DirectOp::Add,
                );
                prime_check(
                    &direct,
                    bytes,
                    &exp_add,
                    &format!("prime direct oracle with {row} add {lanes}"),
                );
                direct.copy_from_slice(&dst0);
                direct_src.copy_from_slice(&src0);
                // The assign direct body restores the staged destination after
                // each call, so probe it without the restore here.
                direct.copy_from_slice(&dst0);
                ops::mul_assign_with::<F>(&mut direct, &prepared);
                prime_check(
                    &direct,
                    bytes,
                    &exp_assign,
                    &format!("prime direct oracle with {row} assign {lanes}"),
                );
            }
        }
        probe.copy_from_slice(&dst0);
        ops::mul_into::<F>(&mut probe, coeff, &src0);
        prime_check(
            &probe,
            bytes,
            &exp_into,
            &format!("prime oracle once {row} into {lanes}"),
        );
        let need = 2 * len;
        println!(
            "PRIME_LAYOUT\tprime {row} {lanes}\tregion={need}B align=64 dst=0 src={len} placements=0,16,32"
        );
        // Routing report from the library's own selector: whether the
        // production column takes the vector body for this geometry.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            use fgf::internals::kernel::residue as selector;
            use fgf::internals::kernel::residue::VectorOp;
            let into_backend = selector::single_row_backend(bytes, p, len, VectorOp::Into);
            let add_backend = selector::single_row_backend(bytes, p, len, VectorOp::Add);
            let assign_backend = selector::single_row_backend(bytes, p, len, VectorOp::Assign);
            let takes_into = into_backend != fgf::Backend::Scalar;
            let takes_add = add_backend != fgf::Backend::Scalar;
            let takes_assign = assign_backend != fgf::Backend::Scalar;
            println!(
                "PRIME_ROUTES\tprime with {row} into {lanes}\tbackend={} takes_vector={takes_into} via=selector",
                into_backend.name()
            );
            println!(
                "PRIME_ROUTES\tprime with {row} add {lanes}\tbackend={} takes_vector={takes_add} via=selector",
                add_backend.name()
            );
            println!(
                "PRIME_ROUTES\tprime with {row} assign {lanes}\tbackend={} takes_vector={takes_assign} via=selector",
                assign_backend.name()
            );
            println!(
                "PRIME_ROUTES\tprime once {row} into {lanes}\tbackend={} takes_vector={takes_into} via=selector",
                into_backend.name()
            );
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        {
            println!(
                "PRIME_ROUTES\tprime with {row} into {lanes}\tbackend={} takes_vector=false via=selector",
                backend_for::<F>().name()
            );
            println!(
                "PRIME_ROUTES\tprime with {row} add {lanes}\tbackend={} takes_vector=false via=selector",
                backend_for::<F>().name()
            );
            println!(
                "PRIME_ROUTES\tprime with {row} assign {lanes}\tbackend={} takes_vector=false via=selector",
                backend_for::<F>().name()
            );
            println!(
                "PRIME_ROUTES\tprime once {row} into {lanes}\tbackend={} takes_vector=false via=selector",
                backend_for::<F>().name()
            );
        }
        let (mut backing, base) = prime_arena(need + 32);
        // Production column: the public `ops` call, whatever the
        // selector picks.
        prime_time_placed(
            &format!("prime with {row} into {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            |dst, src| {
                ops::mul_into_with::<F>(dst, &prepared, src);
            },
        );
        // Direct column: the vector body through `internals` with the
        // exact token, or the portable body below a floor. The control
        // column for single-row shapes is the scalar arm of this same
        // artifact; nothing extra runs in-process here.
        prime_direct_single::<F>(
            &format!("prime direct with {row} into {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            coeff,
            c,
            p,
            DirectOp::Into,
        );
        prime_time_placed(
            &format!("prime with {row} add {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            |dst, src| {
                ops::mul_add_with::<F>(dst, &prepared, src);
            },
        );
        prime_direct_single::<F>(
            &format!("prime direct with {row} add {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            coeff,
            c,
            p,
            DirectOp::Add,
        );
        prime_time_placed(
            &format!("prime with {row} assign {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            |dst, _| {
                ops::mul_assign_with::<F>(dst, &prepared);
                dst.copy_from_slice(&dst0);
            },
        );
        prime_direct_single::<F>(
            &format!("prime direct with {row} assign {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            coeff,
            c,
            p,
            DirectOp::Assign,
        );
        prime_time_placed(
            &format!("prime once {row} into {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            |dst, src| {
                ops::mul_into::<F>(dst, coeff, src);
            },
        );
        prime_direct_single::<F>(
            &format!("prime direct once {row} into {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &src0,
            &dst0,
            coeff,
            c,
            p,
            DirectOp::Once,
        );
    }
}

/// Elementwise panels for one four-byte row: `mul_elementwise` and
/// `mul_elementwise_assign` at short lengths, with the same
/// region, placement, and oracle discipline as the single-row panels.
#[allow(clippy::cast_possible_truncation)]
fn prime_elementwise_panel<F>(row: &str, p: u64, seed: u64)
where
    F: fgf::FieldKernels + FieldBuffer,
{
    let bytes = F::BYTES;
    println!(
        "prime elementwise {row} — backend {}:",
        backend_for::<F>().name()
    );
    println!(
        "PRIME_ROW\telementwise-{row}\t{}",
        backend_for::<F>().name()
    );
    for &lanes in &[8usize, 16, 32, 64, 256, 1024] {
        let len = lanes * bytes;
        let a0 = noise(len, seed);
        let b0 = noise(len, seed ^ 0x51ab);
        let modulus = p as u128;
        let a_r: Vec<u64> = (0..lanes).map(|i| prime_lane(&a0, bytes, i) % p).collect();
        let b_r: Vec<u64> = (0..lanes).map(|i| prime_lane(&b0, bytes, i) % p).collect();
        let exp: Vec<u64> = a_r
            .iter()
            .zip(&b_r)
            .map(|(&a, &b)| ((a as u128 * b as u128) % modulus) as u64)
            .collect();
        let mut probe = vec![0u8; len];
        ops::mul_elementwise::<F>(&mut probe, &a0, &b0);
        prime_check(
            &probe,
            bytes,
            &exp,
            &format!("prime oracle elementwise {row} {lanes}"),
        );
        probe.copy_from_slice(&a0);
        ops::mul_elementwise_assign::<F>(&mut probe, &b0);
        prime_check(
            &probe,
            bytes,
            &exp,
            &format!("prime oracle elementwise_assign {row} {lanes}"),
        );
        // Direct-column oracle for both elementwise forms, gated on
        // the selector like the timed column.
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            use fgf::internals::kernel::residue as selector_oracle;
            if !selector_oracle::routes_vector(bytes, p) || bytes != 4 {
                // Kernel rejects this class; no direct cell exists.
            } else {
                let params = direct_u32_params(p);
                let mut direct = vec![0u8; len];
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    use fgf::internals::kernel::{SimdToken, X64V3Token};
                    if let Some(token) = X64V3Token::summon() {
                        use fgf::internals::kernel::x86::residue as direct_body;
                        direct_body::mul_elementwise_u32_avx2(token, &mut direct, &a0, &b0, params);
                        prime_check(
                            &direct,
                            bytes,
                            &exp,
                            &format!("prime direct oracle elementwise {row} {lanes}"),
                        );
                        direct.copy_from_slice(&a0);
                        direct_body::mul_elementwise_assign_u32_avx2(
                            token,
                            &mut direct,
                            &b0,
                            params,
                        );
                        prime_check(
                            &direct,
                            bytes,
                            &exp,
                            &format!("prime direct oracle elementwise_assign {row} {lanes}"),
                        );
                    }
                }
            }
        }
        let zeros = vec![0u8; len];
        let need = 3 * len;
        println!(
            "PRIME_LAYOUT\tprime elementwise {row} {lanes}\tregion={need}B align=64 dst=0 a={len} b={} placements=0,16,32",
            2 * len
        );
        {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            use fgf::internals::kernel::residue as selector;
            use fgf::internals::kernel::residue::VectorOp;
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let operation_backend =
                selector::single_row_backend(bytes, p, len, VectorOp::Elementwise);
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            let assign_backend =
                selector::single_row_backend(bytes, p, len, VectorOp::ElementwiseAssign);
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let operation_backend = fgf::Backend::Scalar;
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            let assign_backend = fgf::Backend::Scalar;
            let takes = operation_backend != fgf::Backend::Scalar;
            let takes_assign = assign_backend != fgf::Backend::Scalar;
            println!(
                "PRIME_ROUTES\tprime elementwise {row} {lanes}\tbackend={} takes_vector={takes} via=selector",
                operation_backend.name()
            );
            println!(
                "PRIME_ROUTES\tprime elementwise_assign {row} {lanes}\tbackend={} takes_vector={takes_assign} via=selector",
                assign_backend.name()
            );
        }
        let (mut backing, base) = prime_arena(need + 32);
        prime_time3_placed(
            &format!("prime elementwise {row} {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &zeros,
            &a0,
            &b0,
            |dst, a, b| {
                ops::mul_elementwise::<F>(dst, a, b);
            },
        );
        // Direct column: the vector body through `internals` with the
        // exact token, or the portable body below the floor. The control
        // column for elementwise shapes is the scalar arm of this same
        // artifact; nothing extra runs in-process here.
        prime_direct_elementwise::<F>(
            &format!("prime direct elementwise {row} {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &zeros,
            &a0,
            &b0,
            p,
            false,
        );
        prime_time3_placed(
            &format!("prime elementwise_assign {row} {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &a0,
            &a0,
            &b0,
            |dst, _, b| {
                ops::mul_elementwise_assign::<F>(dst, b);
            },
        );
        prime_direct_elementwise::<F>(
            &format!("prime direct elementwise_assign {row} {lanes}"),
            len,
            &mut backing,
            base,
            need,
            &a0,
            &a0,
            &b0,
            p,
            true,
        );
    }
}

/// Matrix panels for the four-byte rows: prepared overwrite
/// (`mul_into_matrix_with`) or accumulate (`mul_add_matrix_with`) over
/// four outputs at 64 and 4096 lanes (the 64-lane row is the 256-byte
/// floor), beside the same work as four prepared per-output-row gathers as
/// the in-tier control, plus a one-shot panel that builds its `CoeffMatrix`
/// inside the timed region.
///
/// Each row prints its selected kernel tier and, from the library's own
/// selectors, whether it uses the delayed body or single-row composition.
/// Before timing, every path's output is checked against the independent
/// `u128` schoolbook.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
fn prime_matrix_panel<const P: u64>(row: &str, counts: &[usize], seed: u64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    use fgf::internals::kernel::residue as selector;
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let k = selector::delay_depth::<P>();
    #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
    let k = ((1u128 << 32) / P as u128) as u64;
    println!(
        "prime matrix {row} — backend {} K={k}:",
        backend_for::<FiniteField<P, 1>>().name()
    );
    println!(
        "PRIME_ROW\tmatrix-{row}\t{} K={k}",
        backend_for::<FiniteField<P, 1>>().name()
    );
    for add in [false, true] {
        let disposition = if add { "add" } else { "into" };
        for &nsources in counts {
            for &lanes in &[64usize, 4096] {
                let row_len = lanes * 4;
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                let matrix_backend = selector::delayed_matrix_backend(4, k, row_len, nsources, add);
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                )))]
                let matrix_backend = fgf::Backend::Scalar;
                let routes = matrix_backend != fgf::Backend::Scalar;
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                let operation_backend = if routes {
                    matrix_backend
                } else {
                    selector::single_row_backend(4, P, row_len, selector::VectorOp::Add)
                };
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                )))]
                let operation_backend = fgf::Backend::Scalar;
                let geometry = format!("prime matrix {disposition} {row} s{nsources} o4 L{lanes}");
                let once_geometry =
                    format!("prime matrix once {disposition} {row} s{nsources} o4 L{lanes}");
                let gather_geometry =
                    format!("prime gather {disposition} {row} s{nsources} o4 L{lanes}");
                let direct_geometry =
                    format!("prime direct matrix {disposition} {row} s{nsources} o4 L{lanes}");
                println!(
                    "PRIME_ROUTES\t{geometry}\tbackend={} delayed_backend={} K={k} routes_delayed={routes} via=selector",
                    operation_backend.name(),
                    matrix_backend.name()
                );
                println!(
                    "PRIME_ROUTES\t{once_geometry}\tbackend={} delayed_backend={} K={k} routes_delayed={routes} via=selector",
                    operation_backend.name(),
                    matrix_backend.name()
                );
                let src0: Vec<Vec<u8>> = (0..nsources)
                    .map(|t| noise(row_len, seed + t as u64))
                    .collect();
                let src_refs0: Vec<&[u8]> = src0.iter().map(Vec::as_slice).collect();
                let dst0 = noise(row_len * 4, seed ^ 0x51f7);
                let coeff_raw: Vec<u64> = (0..nsources * 4)
                    .map(|i| {
                        let mut c = (i as u64).wrapping_mul(53).wrapping_add(17) % P;
                        if c < 2 {
                            c += 2;
                        }
                        c
                    })
                    .collect();
                let coeff_elems: Vec<Elem<FiniteField<P, 1>>> = coeff_raw
                    .iter()
                    .map(|&c| Elem::<FiniteField<P, 1>>::from_raw(c))
                    .collect();
                let modulus = P as u128;
                let mut exp = vec![0u64; 4 * lanes];
                for j in 0..4 {
                    for i in 0..lanes {
                        let mut acc = 0u128;
                        for t in 0..nsources {
                            let s = u128::from(prime_lane(&src0[t], 4, i) % P);
                            acc = (acc + s * u128::from(coeff_raw[t * 4 + j])) % modulus;
                        }
                        if add {
                            let d = u128::from(prime_lane(&dst0, 4, j * lanes + i) % P);
                            acc = (acc + d) % modulus;
                        }
                        exp[j * lanes + i] = acc as u64;
                    }
                }
                let check_rows = |buf: &[u8], what: &str| {
                    for j in 0..4 {
                        prime_check(
                            &buf[j * row_len..(j + 1) * row_len],
                            4,
                            &exp[j * lanes..(j + 1) * lanes],
                            &format!("{what} output {j}"),
                        );
                    }
                };
                let matrix = ops::CoeffMatrix::<FiniteField<P, 1>>::from_source_major(
                    nsources,
                    4,
                    &coeff_elems,
                );
                let gather_vecs: Vec<ops::CoeffVec<FiniteField<P, 1>>> = (0..4)
                    .map(|j| {
                        let col: Vec<Elem<FiniteField<P, 1>>> =
                            (0..nsources).map(|t| coeff_elems[t * 4 + j]).collect();
                        ops::CoeffVec::new(&col)
                    })
                    .collect();
                let mut probe = dst0.clone();
                if add {
                    ops::mul_add_matrix_with::<FiniteField<P, 1>>(
                        &mut probe, row_len, &matrix, &src_refs0,
                    );
                } else {
                    ops::mul_into_matrix_with::<FiniteField<P, 1>>(
                        &mut probe, row_len, &matrix, &src_refs0,
                    );
                }
                check_rows(&probe, &format!("prime oracle {geometry}"));
                probe.copy_from_slice(&dst0);
                for (j, vec) in gather_vecs.iter().enumerate() {
                    if add {
                        ops::mul_add_gather_with::<FiniteField<P, 1>>(
                            &mut probe[j * row_len..(j + 1) * row_len],
                            vec.as_ref(),
                            &src_refs0,
                        );
                    } else {
                        ops::mul_into_gather_with::<FiniteField<P, 1>>(
                            &mut probe[j * row_len..(j + 1) * row_len],
                            vec.as_ref(),
                            &src_refs0,
                        );
                    }
                }
                check_rows(&probe, &format!("prime oracle {gather_geometry}"));
                probe.copy_from_slice(&dst0);
                if add {
                    let once_matrix = ops::CoeffMatrix::<FiniteField<P, 1>>::from_source_major(
                        nsources,
                        4,
                        &coeff_elems,
                    );
                    ops::mul_add_matrix_with::<FiniteField<P, 1>>(
                        &mut probe,
                        row_len,
                        &once_matrix,
                        &src_refs0,
                    );
                } else {
                    let once_matrix = ops::CoeffMatrix::<FiniteField<P, 1>>::from_source_major(
                        nsources,
                        4,
                        &coeff_elems,
                    );
                    ops::mul_into_matrix_with::<FiniteField<P, 1>>(
                        &mut probe,
                        row_len,
                        &once_matrix,
                        &src_refs0,
                    );
                }
                check_rows(&probe, &format!("prime oracle {once_geometry}"));
                // Direct-column oracle: the same delayed-or-portable body
                // the timed direct column calls.
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    let flat: Vec<fgf::internals::kernel::x86::residue::PreparedResidue> =
                        coeff_raw
                            .iter()
                            .map(|&c| {
                                let q = ((c as u128 * (1u128 << 32)) / P as u128) as u64;
                                fgf::internals::kernel::x86::residue::PreparedResidue::from_word(
                                    c | (q << 32),
                                )
                            })
                            .collect();
                    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                        use fgf::internals::kernel::{FlatMatrix, SimdToken, X64V3Token};
                        // Oracle the delayed body wherever the kernel
                        // accepts it (depth at least two), independent of
                        // the routing floors.
                        let k = ((1u128 << 32) / P as u128) as u64;
                        if k >= 2 {
                            let Some(token) = X64V3Token::summon() else {
                                return;
                            };
                            {
                                use fgf::internals::kernel::x86::residue as direct_body;
                                let terms = FlatMatrix {
                                    coefficients: &flat,
                                    nrows: 4,
                                    sources: &src_refs0,
                                };
                                probe.copy_from_slice(&dst0);
                                if add {
                                    direct_body::mul_add_matrix_u32_avx2_with(
                                        token, &mut probe, row_len, 4, &terms, P as u32,
                                    );
                                } else {
                                    direct_body::mul_into_matrix_u32_avx2_with(
                                        token, &mut probe, row_len, 4, &terms, P as u32,
                                    );
                                }
                                check_rows(&probe, &format!("prime direct oracle {geometry}"));
                            }
                        }
                    }
                }
                let need = (nsources + 4) * row_len;
                let traffic = row_len * nsources * 4;
                println!(
                    "PRIME_LAYOUT\t{geometry}\tregion={need}B align=64 srcs=0 rows={} placements=0,16,32",
                    nsources * row_len
                );
                let (mut backing, base) = prime_arena(need + 32);
                if add {
                    prime_time_matrix_placed(
                        &geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            ops::mul_add_matrix_with::<FiniteField<P, 1>>(
                                rows, row_len, &matrix, srcs,
                            );
                        },
                    );
                    // Direct column: the delayed body through `internals`
                    // with the exact token, or the portable composition
                    // below the tile shape.
                    prime_direct_matrix::<P>(
                        &direct_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        &coeff_raw,
                        &gather_vecs,
                        true,
                    );
                    prime_time_matrix_placed(
                        &gather_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            for (j, vec) in gather_vecs.iter().enumerate() {
                                ops::mul_add_gather_with::<FiniteField<P, 1>>(
                                    &mut rows[j * row_len..(j + 1) * row_len],
                                    vec.as_ref(),
                                    srcs,
                                );
                            }
                        },
                    );
                    prime_time_matrix_placed(
                        &once_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            let once_matrix =
                                ops::CoeffMatrix::<FiniteField<P, 1>>::from_source_major(
                                    nsources,
                                    4,
                                    &coeff_elems,
                                );
                            ops::mul_add_matrix_with::<FiniteField<P, 1>>(
                                rows,
                                row_len,
                                &once_matrix,
                                srcs,
                            );
                        },
                    );
                } else {
                    prime_time_matrix_placed(
                        &geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            ops::mul_into_matrix_with::<FiniteField<P, 1>>(
                                rows, row_len, &matrix, srcs,
                            );
                        },
                    );
                    // Direct column: the delayed body through `internals`
                    // with the exact token, or the portable composition
                    // below the tile shape.
                    prime_direct_matrix::<P>(
                        &direct_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        &coeff_raw,
                        &gather_vecs,
                        false,
                    );
                    prime_time_matrix_placed(
                        &gather_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            for (j, vec) in gather_vecs.iter().enumerate() {
                                ops::mul_into_gather_with::<FiniteField<P, 1>>(
                                    &mut rows[j * row_len..(j + 1) * row_len],
                                    vec.as_ref(),
                                    srcs,
                                );
                            }
                        },
                    );
                    prime_time_matrix_placed(
                        &once_geometry,
                        traffic,
                        &mut backing,
                        base,
                        need,
                        row_len,
                        nsources,
                        &src0,
                        &dst0,
                        |rows, srcs| {
                            let once_matrix =
                                ops::CoeffMatrix::<FiniteField<P, 1>>::from_source_major(
                                    nsources,
                                    4,
                                    &coeff_elems,
                                );
                            ops::mul_into_matrix_with::<FiniteField<P, 1>>(
                                rows,
                                row_len,
                                &once_matrix,
                                srcs,
                            );
                        },
                    );
                }
            }
        }
    }
}

/// Every residue row's single-row panels.
fn bench_prime_single_rows() {
    prime_single_row_panel::<Byte101>("byte101", 101, Elem::<Byte101>::from_raw(50), 50, 0xc101);
    prime_single_row_panel::<Byte131>("byte131", 131, Elem::<Byte131>::from_raw(65), 65, 0xc102);
    prime_single_row_panel::<Half65521>(
        "half65521",
        65_521,
        Elem::<Half65521>::from_raw(32_760),
        32_760,
        0xc103,
    );
    prime_single_row_panel::<Word1e9p7>(
        "word1e9p7",
        1_000_000_007,
        Elem::<Word1e9p7>::from_raw(123_456_789),
        123_456_789,
        0xc104,
    );
    prime_single_row_panel::<Word1280000003>(
        "word1280000003",
        1_280_000_003,
        Elem::<Word1280000003>::from_raw(640_000_001),
        640_000_001,
        0xc105,
    );
    prime_single_row_panel::<WordMax>(
        "wordmax",
        4_294_967_291,
        Elem::<WordMax>::from_raw(4_000_000_007),
        4_000_000_007,
        0xc106,
    );
    prime_single_row_panel::<WideMax>(
        "widemax",
        18_446_744_073_709_551_557,
        Elem::<WideMax>::from_raw(0xFFFF_FFFF_FFFF_FFC4),
        18_446_744_073_709_551_556,
        0xc107,
    );
    prime_single_row_panel::<RowP2>("rowp2", 2, Elem::<RowP2>::from_raw(1), 1, 0xc108);
}

/// Four-byte generic residue elementwise products at two resident lengths.
/// Fixtures are raw noise: the kernels canonicalize every lane bit pattern
/// on load.
fn bench_prime_elementwise() {
    prime_elementwise_panel::<Word1e9p7>("word1e9p7", 1_000_000_007, 0xe1e1);
    prime_elementwise_panel::<Word1280000003>("word1280000003", 1_280_000_003, 0xe1e2);
    prime_elementwise_panel::<WordMax>("wordmax", 4_294_967_291, 0xe1e3);
}

/// Prepared matrices on the four-byte rows: the `K = 4` row at 4/16 sources,
/// every four-byte row (`K = 1, 3, 4, 5, 8, 16, 64, 65535`) at sources
/// 4, 8, 16, 32, four outputs, 64 and 4096 lanes, each beside its in-tier
/// four-prepared-gathers control. Rows the kernel rejects print no direct
/// cell; the production column keeps the selector.
fn bench_prime_matrix() {
    prime_matrix_panel::<1_000_000_007>("word1e9p7", &[4, 8, 16, 32], 0xd001);
    prime_matrix_panel::<65_537>("word65537", &[4, 8, 16, 32], 0xd004);
    prime_matrix_panel::<1_280_000_003>("word1280000003", &[4, 8, 16, 32], 0xd002);
    prime_matrix_panel::<4_294_967_291>("wordmax", &[4, 8, 16, 32], 0xd003);
    prime_matrix_panel::<858_993_433>("word858m", &[4, 8, 16, 32], 0xd005);
    prime_matrix_panel::<536_870_909>("word536m", &[4, 8, 16, 32], 0xd006);
    prime_matrix_panel::<268_435_399>("word268m", &[4, 8, 16, 32], 0xd007);
    prime_matrix_panel::<67_108_859>("word67m", &[4, 8, 16, 32], 0xd008);
}

/// Every panel, behind `main`'s drift controls. The diagnostic flags return
/// early and only skip the remaining panels, never the end control.
fn run_panels() {
    // Family flags `--gf`, `--gdl`, `--m31`, `--gf2`, and `--prime` run only
    // that family's panels. Unknown flags are ignored so wrapper arguments
    // pass through unchanged, and no family flag runs every panel.
    let args: Vec<String> = std::env::args().collect();
    let has = |flag: &str| args.iter().any(|arg| arg.as_str() == flag);
    let named = has("--gf") || has("--gdl") || has("--m31") || has("--gf2") || has("--prime");
    let gf = !named || has("--gf");
    let gdl = !named || has("--gdl");
    let m31 = !named || has("--m31");
    let gf2 = !named || has("--gf2");
    let prime = !named || has("--prime");
    if has("--network-diagnostic") {
        bench_network_alignment();
        return;
    }
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if has("--blocked-diagnostic") {
        bench_blocked_vs_axpy();
        return;
    }
    println!("  (override with SIMD_BACKEND=v3_gfni_crypto|v3|v2|neon|scalar)\n");

    if gf {
        bench_preparation_crossover();
        bench_small_row_shapes();
        bench_gf16d_custom_tower();
        bench_add_assign_rows::<FiniteField<2, 8, Polynomial<AES>>>("gf8b", None);
        bench_add_assign_rows::<Gf16>("gf16b", None);
        bench_large_destination();
        bench_destination_alignment();
        bench_network_payloads();
    }
    if m31 {
        bench_add_assign_rows::<Mersenne31>(
            "m31 (prime control, flat default)",
            Some(canonicalize_m31_lanes),
        );
    }
    if gf2 {
        bench_gf2_bits();
        bench_gf2_short_rows();
    }
    if prime {
        bench_prime_single_rows();
        bench_prime_elementwise();
        bench_prime_matrix();
    }

    // L1-resident, L2-resident, and DRAM-resident. The panel's rows all
    // belong to the gf and prime families, so a gf2-only run skips it.
    for &len in &[4 * 1024usize, 256 * 1024, 8 * 1024 * 1024] {
        if !(gf || m31 || gdl) {
            continue;
        }
        let human = if len >= 1024 * 1024 {
            format!("{} MiB", len / (1024 * 1024))
        } else {
            format!("{} KiB", len / 1024)
        };
        println!("buffer {human}:");

        let mut src = noise(len, 1);
        let mut dst = noise(len, 2);
        let prepared16 = ops::Coeff::<Gf16>::new(Elem::<Gf16>::from_raw(0x53a7));
        let mut rhs = noise(len, 0x602);
        let mut product = vec![0; len];

        if gf {
            bench("xor                       gf8b", len, || {
                ops::add_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut dst),
                    black_box(&src),
                );
            });
            bench("mul_add                   gf8b", len, || {
                ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                    black_box(&src),
                );
            });
            bench("mul_add                  gf16b", len, || {
                ops::mul_add::<Gf16>(
                    black_box(&mut dst),
                    Elem::<Gf16>::from_raw(0x53a7),
                    black_box(&src),
                );
            });
            bench("mul_add prepared         gf16b", len, || {
                ops::mul_add_with::<Gf16>(black_box(&mut dst), &prepared16, black_box(&src));
            });
            bench("mul_into                  gf8b", len, || {
                ops::mul_into::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut product),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                    black_box(&src),
                );
            });
            bench("mul_into                 gf16b", len, || {
                ops::mul_into::<Gf16>(
                    black_box(&mut product),
                    Elem::<Gf16>::from_raw(0x53a7),
                    black_box(&src),
                );
            });
            bench("mul_assign                gf8b", len, || {
                ops::mul_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                );
            });
            bench("elementwise                gf8b", len, || {
                ops::mul_elementwise::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut product),
                    black_box(&src),
                    black_box(&rhs),
                );
            });
            bench("elementwise               gf16b", len, || {
                ops::mul_elementwise::<Gf16>(
                    black_box(&mut product),
                    black_box(&src),
                    black_box(&rhs),
                );
            });
            bench("elementwise_assign        gf8b", len, || {
                ops::mul_elementwise_assign::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut product),
                    black_box(&src),
                );
            });
            bench("elementwise_assign       gf16b", len, || {
                ops::mul_elementwise_assign::<Gf16>(black_box(&mut product), black_box(&src));
            });
            bench("mul_assign               gf16b", len, || {
                ops::mul_assign::<Gf16>(black_box(&mut dst), Elem::<Gf16>::from_raw(0x53a7));
            });
            bench("add_assign                gf16b", len, || {
                ops::add_assign::<Gf16>(black_box(&mut dst), black_box(&src));
            });
            bench("add_assign_scalar          gf8b", len, || {
                ops::add_assign_scalar::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                );
            });
            bench("sub_assign_scalar          gf8b", len, || {
                ops::sub_assign_scalar::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut dst),
                    Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53),
                );
            });
            bench("add_assign_scalar         gf16b", len, || {
                ops::add_assign_scalar::<Gf16>(black_box(&mut dst), Elem::<Gf16>::from_raw(0x53a7));
            });
            bench("sub_assign_scalar         gf16b", len, || {
                ops::sub_assign_scalar::<Gf16>(black_box(&mut dst), Elem::<Gf16>::from_raw(0x53a7));
            });
        }

        // Prime-field elementwise and broadcast scalar ops. `add_assign` and
        // `elementwise` are the unchanged controls for the new assign forms.
        // Every prime input and destination holds canonical lanes before
        // the timed region; prime arithmetic preserves canonical values,
        // so in-place panels stay canonical across batches.
        if m31 {
            for buf in [&mut src, &mut dst, &mut rhs, &mut product] {
                canonicalize_m31_lanes(buf);
            }
            bench("add_assign                   m31", len, || {
                ops::add_assign::<Mersenne31>(black_box(&mut dst), black_box(&src));
            });
            bench("add_assign_scalar            m31", len, || {
                ops::add_assign_scalar::<Mersenne31>(
                    black_box(&mut dst),
                    Elem::<Mersenne31>::from_raw(0x1234_5678),
                );
            });
            bench("sub_assign_scalar            m31", len, || {
                ops::sub_assign_scalar::<Mersenne31>(
                    black_box(&mut dst),
                    Elem::<Mersenne31>::from_raw(0x1234_5678),
                );
            });
            bench("elementwise                  m31", len, || {
                ops::mul_elementwise::<Mersenne31>(
                    black_box(&mut product),
                    black_box(&src),
                    black_box(&rhs),
                );
            });
            bench("elementwise_assign           m31", len, || {
                ops::mul_elementwise_assign::<Mersenne31>(black_box(&mut product), black_box(&src));
            });
        }
        if gdl {
            for buf in [&mut src, &mut dst, &mut rhs, &mut product] {
                canonicalize_goldilocks_lanes(buf);
            }
            bench("add_assign                   gld", len, || {
                ops::add_assign::<Goldilocks>(black_box(&mut dst), black_box(&src));
            });
            bench("add_assign_scalar            gld", len, || {
                ops::add_assign_scalar::<Goldilocks>(
                    black_box(&mut dst),
                    Elem::<Goldilocks>::from_raw(0x1234_5678_9abc_def0),
                );
            });
            bench("sub_assign_scalar            gld", len, || {
                ops::sub_assign_scalar::<Goldilocks>(
                    black_box(&mut dst),
                    Elem::<Goldilocks>::from_raw(0x1234_5678_9abc_def0),
                );
            });
            bench("elementwise                  gld", len, || {
                ops::mul_elementwise::<Goldilocks>(
                    black_box(&mut product),
                    black_box(&src),
                    black_box(&rhs),
                );
            });
            bench("elementwise_assign           gld", len, || {
                ops::mul_elementwise_assign::<Goldilocks>(black_box(&mut product), black_box(&src));
            });
        }
        if m31 {
            for buf in [&mut src, &mut dst, &mut rhs, &mut product] {
                canonicalize_m31_lanes(buf);
            }
            bench("add_assign                  qm31", len, || {
                ops::add_assign::<QuadMersenne31>(black_box(&mut dst), black_box(&src));
            });
            bench("add_assign_scalar           qm31", len, || {
                ops::add_assign_scalar::<QuadMersenne31>(
                    black_box(&mut dst),
                    Elem::<QuadMersenne31>::from_raw([0x1234_5678, 0x0987_6543]),
                );
            });
            bench("sub_assign_scalar           qm31", len, || {
                ops::sub_assign_scalar::<QuadMersenne31>(
                    black_box(&mut dst),
                    Elem::<QuadMersenne31>::from_raw([0x1234_5678, 0x0987_6543]),
                );
            });
            bench("elementwise                 qm31", len, || {
                ops::mul_elementwise::<QuadMersenne31>(
                    black_box(&mut product),
                    black_box(&src),
                    black_box(&rhs),
                );
            });
            bench("elementwise_assign          qm31", len, || {
                ops::mul_elementwise_assign::<QuadMersenne31>(
                    black_box(&mut product),
                    black_box(&src),
                );
            });
        }
        println!();
    }

    // The remaining panels cover the binary tower fields only.
    if !gf {
        return;
    }

    // Multi-row shapes. Geometry is a realistic erasure code: k data rows
    // folded into m parity rows, each row a 64 KiB symbol.
    let row_len = 64 * 1024;
    for &nrows in &[2usize, 4, 8, 16] {
        println!("scatter/matrix — {nrows} rows x {} KiB:", row_len / 1024);

        let src = noise(row_len, 3);
        let mut rows = noise(row_len * nrows, 4);
        let coeffs8: Vec<_> = (0..nrows)
            .map(|j| {
                Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                    (j as u8).wrapping_mul(37).wrapping_add(2),
                )
            })
            .collect();
        let coeffs16: Vec<_> = (0..nrows)
            .map(|j| Elem::<Gf16>::from_raw((j as u16).wrapping_mul(9871).wrapping_add(2)))
            .collect();
        let scatter_coeffs8 = ops::CoeffVec::<FiniteField<2, 8, Polynomial<AES>>>::new(&coeffs8);
        let scatter_coeffs16 = ops::CoeffVec::<Gf16>::new(&coeffs16);

        let traffic = row_len * nrows;
        bench("scatter                   gf8b", traffic, || {
            ops::mul_add_scatter::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut rows),
                row_len,
                &coeffs8,
                black_box(&src),
            );
        });
        bench("scatter                  gf16b", traffic, || {
            ops::mul_add_scatter::<Gf16>(black_box(&mut rows), row_len, &coeffs16, black_box(&src));
        });
        bench("scatter prepared          gf8b", traffic, || {
            ops::mul_add_scatter_with::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut rows),
                row_len,
                scatter_coeffs8.as_ref(),
                black_box(&src),
            );
        });
        bench("scatter prepared         gf16b", traffic, || {
            ops::mul_add_scatter_with::<Gf16>(
                black_box(&mut rows),
                row_len,
                scatter_coeffs16.as_ref(),
                black_box(&src),
            );
        });
        bench("scatter (unblocked)       gf8b", traffic, || {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs8) {
                ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(row),
                    coeff,
                    black_box(&src),
                );
            }
        });
        bench("scatter (unblocked)      gf16b", traffic, || {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs16) {
                ops::mul_add::<Gf16>(black_box(row), coeff, black_box(&src));
            }
        });

        // The blocked-vs-unblocked comparison: 8 sources into `nrows` rows,
        // as one matrix call versus eight scatter calls. Same arithmetic,
        // different destination memory traffic.
        let sources: Vec<Vec<u8>> = (0..16).map(|t| noise(row_len, 100 + t as u64)).collect();
        let coeff_sets: Vec<Vec<Elem<FiniteField<2, 8, Polynomial<AES>>>>> = (0..8)
            .map(|t| {
                (0..nrows)
                    .map(|j| {
                        Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(
                            ((t * 31 + j * 17) as u8).wrapping_add(1),
                        )
                    })
                    .collect()
            })
            .collect();
        let coeff_sets16: Vec<Vec<Elem<Gf16>>> = (0..8)
            .map(|t| {
                (0..nrows)
                    .map(|j| Elem::<Gf16>::from_raw(((t * 7919 + j * 613) as u16).wrapping_add(1)))
                    .collect()
            })
            .collect();
        // Term geometry nests the unified element spelling; the slices stay slices.
        #[allow(clippy::type_complexity)]
        let terms: Vec<(&[Elem<FiniteField<2, 8, Polynomial<AES>>>], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let terms16: Vec<(&[Elem<Gf16>], &[u8])> = coeff_sets16
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let matrix_coeffs8: Vec<_> = coeff_sets.iter().flatten().copied().collect();
        let matrix_coeffs16: Vec<_> = coeff_sets16.iter().flatten().copied().collect();
        let matrix8 = ops::CoeffMatrix::<FiniteField<2, 8, Polynomial<AES>>>::from_source_major(
            8,
            nrows,
            &matrix_coeffs8,
        );
        let matrix16 = ops::CoeffMatrix::<Gf16>::from_source_major(8, nrows, &matrix_coeffs16);
        let matrix_srcs: Vec<&[u8]> = sources.iter().take(8).map(Vec::as_slice).collect();

        let traffic = row_len * nrows * 8;
        bench("matrix (selected)          gf8b", traffic, || {
            ops::mul_add_matrix::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut rows),
                row_len,
                nrows,
                &terms,
            );
        });
        bench("matrix (unblocked AXPY)   gf8b", traffic, || {
            for &(coeffs, src) in &terms {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(black_box(row), coeff, src);
                }
            }
        });
        bench("matrix (selected)         gf16b", traffic, || {
            ops::mul_add_matrix::<Gf16>(black_box(&mut rows), row_len, nrows, &terms16);
        });
        bench("matrix prepared           gf8b", traffic, || {
            ops::mul_add_matrix_with::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut rows),
                row_len,
                &matrix8,
                black_box(&matrix_srcs),
            );
        });
        bench("matrix prepared          gf16b", traffic, || {
            ops::mul_add_matrix_with::<Gf16>(
                black_box(&mut rows),
                row_len,
                &matrix16,
                black_box(&matrix_srcs),
            );
        });
        bench("matrix (unblocked AXPY)  gf16b", traffic, || {
            for &(coeffs, src) in &terms16 {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    ops::mul_add::<Gf16>(black_box(row), coeff, src);
                }
            }
        });

        let gather_srcs: Vec<&[u8]> = sources.iter().take(nrows).map(Vec::as_slice).collect();
        let mut gathered = noise(row_len, 5);
        let gather_coeffs8 = ops::CoeffVec::<FiniteField<2, 8, Polynomial<AES>>>::new(&coeffs8);
        let gather_coeffs16 = ops::CoeffVec::<Gf16>::new(&coeffs16);
        let gather_traffic = row_len * nrows;
        bench("gather (selected)          gf8b", gather_traffic, || {
            ops::mul_add_gather::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut gathered),
                &coeffs8,
                black_box(&gather_srcs),
            );
        });
        bench("gather (unblocked)        gf8b", gather_traffic, || {
            for (&coeff, &source) in coeffs8.iter().zip(&gather_srcs) {
                ops::mul_add::<FiniteField<2, 8, Polynomial<AES>>>(
                    black_box(&mut gathered),
                    coeff,
                    black_box(source),
                );
            }
        });
        bench("gather (selected)         gf16b", gather_traffic, || {
            ops::mul_add_gather::<Gf16>(
                black_box(&mut gathered),
                &coeffs16,
                black_box(&gather_srcs),
            );
        });
        bench("gather prepared           gf8b", gather_traffic, || {
            ops::mul_add_gather_with::<FiniteField<2, 8, Polynomial<AES>>>(
                black_box(&mut gathered),
                gather_coeffs8.as_ref(),
                black_box(&gather_srcs),
            );
        });
        bench("gather prepared          gf16b", gather_traffic, || {
            ops::mul_add_gather_with::<Gf16>(
                black_box(&mut gathered),
                gather_coeffs16.as_ref(),
                black_box(&gather_srcs),
            );
        });
        bench("gather (unblocked)       gf16b", gather_traffic, || {
            for (&coeff, &source) in coeffs16.iter().zip(&gather_srcs) {
                ops::mul_add::<Gf16>(black_box(&mut gathered), coeff, black_box(source));
            }
        });
        println!();
    }

    // Same byte volume means the same information volume: one GF(2^32)
    // symbol occupies exactly two GF(2^16) symbols. This makes the cost of
    // choosing a wider field directly visible rather than hiding it behind a
    // symbol-count comparison.
    let tier3_len = 256 * 1024;
    let tier3_src = noise(tier3_len, 0x900);
    let mut tier3_dst = noise(tier3_len, 0x901);
    println!("Tier 3 field cost — {tier3_len} bytes:");
    bench("mul_add polynomial tower     gf16b", tier3_len, || {
        ops::mul_add::<Gf16>(
            black_box(&mut tier3_dst),
            Elem::<Gf16>::from_raw(0x53a7),
            black_box(&tier3_src),
        );
    });
    bench("mul_add polynomial tower     gf32b", tier3_len, || {
        ops::mul_add::<Gf32>(
            black_box(&mut tier3_dst),
            Elem::<Gf32>::from_raw(0xdead_beef),
            black_box(&tier3_src),
        );
    });
    bench("mul_add polynomial tower     gf64b", tier3_len, || {
        ops::mul_add::<Gf64>(
            black_box(&mut tier3_dst),
            Elem::<Gf64>::from_raw(0x0123_4567_89ab_cdef),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp16", tier3_len, || {
        ops::mul_add::<FanPaar16Field>(
            black_box(&mut tier3_dst),
            Elem::<FanPaar16Field>::from_raw(0xe2de),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp32", tier3_len, || {
        ops::mul_add::<FanPaar32Field>(
            black_box(&mut tier3_dst),
            Elem::<FanPaar32Field>::from_raw(0x03e2_1cea),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp64", tier3_len, || {
        ops::mul_add::<FanPaar64Field>(
            black_box(&mut tier3_dst),
            Elem::<FanPaar64Field>::from_raw(0x070f_870d_cd9c_1d88),
            black_box(&tier3_src),
        );
    });
}
