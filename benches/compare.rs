//! `fgf` self-numbers over the competitor-harness fixture family.
//!
//! Measures scalar and 64 KiB single-source region operations, 64 KiB
//! elementwise products, prepared 16-source overwrite dot products at 4 and
//! 16 KiB, and 64 KiB ten-source erasure encodes over 2/4/6 output rows in
//! raw, prepared, and scattered-row forms. The alignment sweep times
//! misaligned and aligned scatter, gather, and matrix rows on both sides of
//! each peel floor. Buffers are 64-byte aligned to match the ISA-L harness.
//!
//! ```sh
//! just bench compare [--peel-diagnostic]
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
//! separable from field speed.

use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::{
    Binary, Elem, Goldilocks, Mersenne31, Polynomial, QuadMersenne31, backend, backend_for, ops,
};

use fgf::poly::{AES, RS};

type Gf16 = fgf::Binary<16, fgf::Tower<fgf::Rijndael16>>;
type Gf32 = fgf::Binary<32, fgf::Tower<fgf::Rijndael32>>;
type Gf64 = fgf::Binary<64, fgf::Tower<fgf::Rijndael64>>;
type FanPaar16Field = fgf::Binary<16, fgf::Tower<fgf::FanPaar16>>;
type FanPaar32Field = fgf::Binary<32, fgf::Tower<fgf::FanPaar32>>;
type FanPaar64Field = fgf::Binary<64, fgf::Tower<fgf::FanPaar64>>;

const BYTES: usize = 64 * 1024;
const SCALAR_ITERS: usize = 1 << 20;
const DOT_SOURCES: usize = 16;
const DOT_LENGTHS: &[usize] = &[4 * 1024, 16 * 1024];
const ENCODE_SOURCES: usize = 10;
const ENCODE_ROWS: &[usize] = &[2, 4, 6];

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

/// Target wall-clock duration of one sample batch: long enough that a short
/// workload's batch sits above clock and scheduler jitter.
const BATCH_TARGET_NS: f64 = 100_000.0;
/// Upper bound on repetitions in one sample batch.
const MAX_BATCH_REPS: u64 = 1 << 20;
/// Logical byte count of the drift control's XOR pass.
const CONTROL_BYTES: usize = 256 * 1024;

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
    field::<Binary<8, Polynomial<AES>>>("gf8b");
    field::<Binary<8, Polynomial<RS>>>("gf8d");
    field::<Gf16>("gf16b");
    field::<Gf32>("gf32b");
    field::<Gf64>("gf64b");
    field::<FanPaar16Field>("fp16");
    field::<FanPaar32Field>("fp32");
    field::<FanPaar64Field>("fp64");
    field::<Mersenne31>("m31");
    field::<QuadMersenne31>("qm31");
    field::<Goldilocks>("gld");
}

struct AlignedBuf {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    fn noise(len: usize, seed: u64) -> Self {
        let mut storage = vec![0; len + 63];
        let base = storage.as_ptr() as usize;
        let start = (64 - (base & 63)) & 63;
        storage[start..start + len].copy_from_slice(&noise(len, seed));
        let result = Self {
            storage,
            start,
            len,
        };
        assert_eq!((result.as_slice().as_ptr() as usize) & 63, 0);
        result
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }
}

fn bench_mul_into_gather(len: usize) {
    let sources: Vec<AlignedBuf> = (0..DOT_SOURCES)
        .map(|index| AlignedBuf::noise(len, 0x800 + index as u64))
        .collect();
    let srcs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
    let coefficient_bytes: Vec<u8> = (0..DOT_SOURCES)
        .map(|index| 2 + ((index * 73 + 19) % 254) as u8)
        .collect();
    let coeffs_8b: Vec<_> = coefficient_bytes
        .iter()
        .copied()
        .map(Elem::<Binary<8, Polynomial<AES>>>::from_raw)
        .collect();
    let coeffs_8d: Vec<_> = coefficient_bytes
        .iter()
        .copied()
        .map(Elem::<Binary<8, Polynomial<RS>>>::from_raw)
        .collect();
    let prepared_8b = ops::CoeffVec::<Binary<8, Polynomial<AES>>>::new(&coeffs_8b);
    let prepared_8d = ops::CoeffVec::<Binary<8, Polynomial<RS>>>::new(&coeffs_8d);
    let mut dst_8b = AlignedBuf::noise(len, 0x900);
    let mut dst_8d = AlignedBuf::noise(len, 0x901);

    // Raw-coefficient overwrite gather (not the prepared `_with` form):
    // the candidate one-row-matrix route fires here.
    let coeff_bytes: Vec<_> = coefficient_bytes
        .iter()
        .copied()
        .map(Elem::<Binary<8, Polynomial<RS>>>::from_raw)
        .collect();
    println!("{len} B x {DOT_SOURCES} sources overwrite dot product (raw)");
    bench_region("fgf gf8d mul_into_gather", len * DOT_SOURCES, || {
        ops::mul_into_gather::<Binary<8, Polynomial<RS>>>(
            black_box(dst_8d.as_mut_slice()),
            &coeff_bytes,
            black_box(&srcs),
        );
    });
    // Accumulate-form prepared gather: hits the `_with` default path.
    bench_region("fgf gf8d mul_add_gather_with", len * DOT_SOURCES, || {
        ops::mul_add_gather_with::<Binary<8, Polynomial<RS>>>(
            black_box(dst_8d.as_mut_slice()),
            prepared_8d.as_ref(),
            black_box(&srcs),
        );
    });
    println!("{len} B x {DOT_SOURCES} sources overwrite dot product");
    let logical_bytes = len * DOT_SOURCES;
    bench_region("fgf gf8b mul_into_gather_with", logical_bytes, || {
        ops::mul_into_gather_with::<Binary<8, Polynomial<AES>>>(
            black_box(dst_8b.as_mut_slice()),
            prepared_8b.as_ref(),
            black_box(&srcs),
        );
    });
    bench_region("fgf gf8d mul_into_gather_with", logical_bytes, || {
        ops::mul_into_gather_with::<Binary<8, Polynomial<RS>>>(
            black_box(dst_8d.as_mut_slice()),
            prepared_8d.as_ref(),
            black_box(&srcs),
        );
    });
}

// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn bench_encode(nrows: usize) {
    let sources: Vec<AlignedBuf> = (0..ENCODE_SOURCES)
        .map(|term| AlignedBuf::noise(BYTES, 0xa00 + term as u64))
        .collect();
    let columns_8b: Vec<Vec<Elem<Binary<8, Polynomial<AES>>>>> = (0..ENCODE_SOURCES)
        .map(|term| {
            (0..nrows)
                .map(|row| {
                    Elem::<Binary<8, Polynomial<AES>>>::from_raw(
                        (1 + ((row * ENCODE_SOURCES + term) * 97 + 13) % 255) as u8,
                    )
                })
                .collect()
        })
        .collect();
    let columns_8d: Vec<Vec<Elem<Binary<8, Polynomial<RS>>>>> = (0..ENCODE_SOURCES)
        .map(|term| {
            (0..nrows)
                .map(|row| {
                    Elem::<Binary<8, Polynomial<RS>>>::from_raw(
                        (1 + ((row * ENCODE_SOURCES + term) * 97 + 13) % 255) as u8,
                    )
                })
                .collect()
        })
        .collect();
    let terms_8b: Vec<(&[Elem<Binary<8, Polynomial<AES>>>], &[u8])> = columns_8b
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let terms_8d: Vec<(&[Elem<Binary<8, Polynomial<RS>>>], &[u8])> = columns_8d
        .iter()
        .zip(&sources)
        .map(|(coeffs, src)| (coeffs.as_slice(), src.as_slice()))
        .collect();
    let mut old_8b = AlignedBuf::noise(BYTES * nrows, 0xb00);
    let mut new_8b = AlignedBuf::noise(BYTES * nrows, 0xb01);
    let mut old_8d = AlignedBuf::noise(BYTES * nrows, 0xb02);
    let mut new_8d = AlignedBuf::noise(BYTES * nrows, 0xb03);

    old_8b.as_mut_slice().fill(0);
    ops::mul_add_matrix::<Binary<8, Polynomial<AES>>>(
        old_8b.as_mut_slice(),
        BYTES,
        nrows,
        &terms_8b,
    );
    ops::mul_into_matrix::<Binary<8, Polynomial<AES>>>(
        new_8b.as_mut_slice(),
        BYTES,
        nrows,
        &terms_8b,
    );
    assert_eq!(
        old_8b.as_slice(),
        new_8b.as_slice(),
        "Binary<8, Polynomial<AES>> matrix overwrite differs"
    );

    old_8d.as_mut_slice().fill(0);
    ops::mul_add_matrix::<Binary<8, Polynomial<RS>>>(
        old_8d.as_mut_slice(),
        BYTES,
        nrows,
        &terms_8d,
    );
    ops::mul_into_matrix::<Binary<8, Polynomial<RS>>>(
        new_8d.as_mut_slice(),
        BYTES,
        nrows,
        &terms_8d,
    );
    assert_eq!(
        old_8d.as_slice(),
        new_8d.as_slice(),
        "Binary<8, Polynomial<RS>> matrix overwrite differs"
    );

    println!("{BYTES} B x {ENCODE_SOURCES} sources -> {nrows} rows encode");
    let logical_bytes = BYTES * ENCODE_SOURCES;
    bench_region("fgf gf8b fill + mul_add_matrix", logical_bytes, || {
        old_8b.as_mut_slice().fill(0);
        ops::mul_add_matrix::<Binary<8, Polynomial<AES>>>(
            black_box(old_8b.as_mut_slice()),
            BYTES,
            nrows,
            black_box(&terms_8b),
        );
    });
    bench_region("fgf gf8b mul_into_matrix", logical_bytes, || {
        ops::mul_into_matrix::<Binary<8, Polynomial<AES>>>(
            black_box(new_8b.as_mut_slice()),
            BYTES,
            nrows,
            black_box(&terms_8b),
        );
    });
    bench_region("fgf gf8d fill + mul_add_matrix", logical_bytes, || {
        old_8d.as_mut_slice().fill(0);
        ops::mul_add_matrix::<Binary<8, Polynomial<RS>>>(
            black_box(old_8d.as_mut_slice()),
            BYTES,
            nrows,
            black_box(&terms_8d),
        );
    });
    bench_region("fgf gf8d mul_into_matrix", logical_bytes, || {
        ops::mul_into_matrix::<Binary<8, Polynomial<RS>>>(
            black_box(new_8d.as_mut_slice()),
            BYTES,
            nrows,
            black_box(&terms_8d),
        );
    });

    // Prepared and scattered-row forms of the same encode, validated against
    // the raw overwrite result before timing.
    let source_refs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
    let flat_8b: Vec<Elem<Binary<8, Polynomial<AES>>>> =
        columns_8b.iter().flatten().copied().collect();
    let prepared_8b = ops::CoeffMatrix::<Binary<8, Polynomial<AES>>>::from_source_major(
        ENCODE_SOURCES,
        nrows,
        &flat_8b,
    );
    let row_starts: Vec<usize> = (0..nrows).map(|row| row * BYTES).collect();
    let mut with_8b = AlignedBuf::noise(BYTES * nrows, 0xb08);
    let mut at_8b = AlignedBuf::noise(BYTES * nrows, 0xb09);
    ops::mul_into_matrix_with::<Binary<8, Polynomial<AES>>>(
        with_8b.as_mut_slice(),
        BYTES,
        &prepared_8b,
        &source_refs,
    );
    assert_eq!(
        new_8b.as_slice(),
        with_8b.as_slice(),
        "Binary<8, Polynomial<AES>> prepared overwrite differs"
    );
    at_8b.as_mut_slice().fill(0);
    ops::mul_add_matrix_at::<Binary<8, Polynomial<AES>>>(
        at_8b.as_mut_slice(),
        BYTES,
        &row_starts,
        &terms_8b,
    );
    assert_eq!(
        new_8b.as_slice(),
        at_8b.as_slice(),
        "Binary<8, Polynomial<AES>> scattered rows differ"
    );
    bench_region("fgf gf8b mul_into_matrix_with", logical_bytes, || {
        ops::mul_into_matrix_with::<Binary<8, Polynomial<AES>>>(
            black_box(with_8b.as_mut_slice()),
            BYTES,
            black_box(&prepared_8b),
            black_box(&source_refs),
        );
    });
    bench_region("fgf gf8b mul_add_matrix_with", logical_bytes, || {
        ops::mul_add_matrix_with::<Binary<8, Polynomial<AES>>>(
            black_box(with_8b.as_mut_slice()),
            BYTES,
            black_box(&prepared_8b),
            black_box(&source_refs),
        );
    });
    bench_region("fgf gf8b mul_add_matrix_at", logical_bytes, || {
        ops::mul_add_matrix_at::<Binary<8, Polynomial<AES>>>(
            black_box(at_8b.as_mut_slice()),
            BYTES,
            black_box(&row_starts),
            black_box(&terms_8b),
        );
    });
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

fn bench_region(label: &str, bytes: usize, mut body: impl FnMut()) {
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
    let gib = bytes as f64 / (median / 1e9) / (1024.0 * 1024.0 * 1024.0);
    println!("  {label:<44} {:>9}  {gib:>7.2} GiB/s", fmt_ns(median));
    println!("SAMPLE\t{label}\t{bytes}\t{median:.9}");
}

/// Fixed batches amortize the clock call when comparing peel variants.
fn bench_peel_region(label: &str, bytes: usize, mut body: impl FnMut()) {
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
    let gib = bytes as f64 / (median / 1e9) / (1024.0 * 1024.0 * 1024.0);
    println!("  {label:<44} {:>9}  {gib:>7.2} GiB/s", fmt_ns(median));
    println!("SAMPLE\t{label}\t{bytes}\t{median:.9}");
}

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

/// Offset-sweep probe for the 512-bit alignment question: the same
/// `Binary<8, Polynomial<RS>>` `mul_add` region with destination/source buffer bases
/// shifted by `OFF mod 64`. Buffer bases are 64-byte aligned (`AlignedBuf`),
/// so the offset is exact. Glibc-sized allocations land 16 mod 64 in
/// practice; the sweep names the realistic line-split cost before any
/// alignment-step work.
fn bench_offset_sweep() {
    const LEN: usize = 16 * 1024;
    let c = Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x53);
    let backing_src = AlignedBuf::noise(LEN + 64, 0x700);
    let mut backing_dst = AlignedBuf::noise(LEN + 64, 0x701);
    println!("gf8d mul_add 16 KiB destination offset sweep (dst/src shift mod 64)");
    for &off in &[0usize, 16, 32, 48] {
        let src = &backing_src.as_slice()[off..off + LEN];
        let dst = &mut backing_dst.as_mut_slice()[off..off + LEN];
        bench_region(&format!("gf8d mul_add off={off}"), LEN, || {
            ops::mul_add::<Binary<8, Polynomial<RS>>>(black_box(dst), c, black_box(src));
        });
    }

    // Prepared scatter: raw values vs prepared coefficients through the plan.
    {
        let nrows = 4usize;
        let row_len = 16384usize;
        let src = AlignedBuf::noise(row_len, 0x720);
        let mut rows = AlignedBuf::noise(row_len * nrows, 0x721);
        let values: Vec<_> = (0..nrows)
            .map(|j| {
                Elem::<Binary<8, Polynomial<RS>>>::from_raw(
                    (j as u8).wrapping_mul(37).wrapping_add(2),
                )
            })
            .collect();
        let prepared = ops::CoeffVec::<Binary<8, Polynomial<RS>>>::new(&values);
        bench_region("scatter 4x16K raw", row_len * nrows, || {
            ops::mul_add_scatter::<Binary<8, Polynomial<RS>>>(
                black_box(rows.as_mut_slice()),
                row_len,
                &values,
                black_box(src.as_slice()),
            );
        });
        bench_region("scatter 4x16K prepared", row_len * nrows, || {
            ops::mul_add_scatter_with::<Binary<8, Polynomial<RS>>>(
                black_box(rows.as_mut_slice()),
                row_len,
                prepared.as_ref(),
                black_box(src.as_slice()),
            );
        });
    }
}

/// Misaligned (16 mod 64) and aligned rows on both sides of each 64-byte
/// peel floor: four-row scatter, sixteen-source gather, and a ten-source
/// four-row matrix. Rows, sources, and destinations share the offset.
// Term geometry nests the unified element spelling; the slices stay slices.
#[allow(clippy::type_complexity)]
fn bench_peel_floors() {
    const SCATTER_ROWS: usize = 4;
    const GATHER_SOURCES: usize = 16;
    const MATRIX_TERMS: usize = 10;
    const MATRIX_ROWS: usize = 4;
    let coeff =
        |i: usize| Elem::<Binary<8, Polynomial<RS>>>::from_raw((1 + (i * 97 + 13) % 255) as u8);
    println!("gf8d peel floor sweep (row bytes, base offset mod 64)");
    for &off in &[16usize, 0] {
        for &len in &[256usize, 512, 1024, 2048, 3072, 4096, 8192, 16384] {
            let src = AlignedBuf::noise(len + 64, 0x710);
            let src = &src.as_slice()[off..off + len];
            let mut rows = AlignedBuf::noise(len * SCATTER_ROWS + 64, 0x711);
            let rows = &mut rows.as_mut_slice()[off..off + len * SCATTER_ROWS];
            let scatter: Vec<_> = (0..SCATTER_ROWS).map(coeff).collect();
            bench_peel_region(
                &format!("scatter {SCATTER_ROWS}x{len} off={off}"),
                len * SCATTER_ROWS,
                || {
                    ops::mul_add_scatter::<Binary<8, Polynomial<RS>>>(
                        black_box(&mut *rows),
                        len,
                        &scatter,
                        black_box(src),
                    );
                },
            );

            let gather_bufs: Vec<AlignedBuf> = (0..GATHER_SOURCES)
                .map(|t| AlignedBuf::noise(len + 64, 0x720 + t as u64))
                .collect();
            let gather_srcs: Vec<&[u8]> = gather_bufs
                .iter()
                .map(|b| &b.as_slice()[off..off + len])
                .collect();
            let gather: Vec<_> = (0..GATHER_SOURCES).map(coeff).collect();
            let mut dst = AlignedBuf::noise(len + 64, 0x730);
            let dst = &mut dst.as_mut_slice()[off..off + len];
            bench_peel_region(
                &format!("gather {GATHER_SOURCES}x{len} off={off}"),
                len * GATHER_SOURCES,
                || {
                    ops::mul_add_gather::<Binary<8, Polynomial<RS>>>(
                        black_box(&mut *dst),
                        &gather,
                        black_box(&gather_srcs),
                    );
                },
            );

            let matrix_bufs: Vec<AlignedBuf> = (0..MATRIX_TERMS)
                .map(|t| AlignedBuf::noise(len + 64, 0x740 + t as u64))
                .collect();
            let matrix_coeffs: Vec<_> = (0..MATRIX_TERMS * MATRIX_ROWS).map(coeff).collect();
            let terms: Vec<(&[Elem<Binary<8, Polynomial<RS>>>], &[u8])> = matrix_coeffs
                .chunks(MATRIX_ROWS)
                .zip(matrix_bufs.iter().map(|b| &b.as_slice()[off..off + len]))
                .collect();
            let mut matrix = AlignedBuf::noise(len * MATRIX_ROWS + 64, 0x750);
            let matrix = &mut matrix.as_mut_slice()[off..off + len * MATRIX_ROWS];
            bench_peel_region(
                &format!("matrix {MATRIX_TERMS}->{MATRIX_ROWS}x{len} off={off}"),
                len * MATRIX_TERMS,
                || {
                    ops::mul_add_matrix::<Binary<8, Polynomial<RS>>>(
                        black_box(&mut *matrix),
                        len,
                        MATRIX_ROWS,
                        black_box(&terms),
                    );
                },
            );
        }
    }
}

fn main() {
    println!("fgf — backend: {}", backend().name());
    print_host_lines();
    print_field_backends();
    println!("control byte xor — drift check:");
    let control_src = noise(CONTROL_BYTES, 0x57a7);
    let mut control_dst = noise(CONTROL_BYTES, 0x57a8);
    bench_region("control byte xor start", CONTROL_BYTES, || {
        control_body(&mut control_dst, &control_src)
    });
    run_panels();
    println!("control byte xor — drift check:");
    bench_region("control byte xor end", CONTROL_BYTES, || {
        control_body(&mut control_dst, &control_src)
    });
}

/// Every panel, behind `main`'s drift controls. The `--peel-diagnostic`
/// flag returns early and only skips the remaining panels, never the end
/// control.
fn run_panels() {
    if std::env::args().any(|arg| arg == "--peel-diagnostic") {
        bench_peel_floors();
        return;
    }
    bench_offset_sweep();

    let src8 = AlignedBuf::noise(BYTES, 0x600);
    let mut dst8 = AlignedBuf::noise(BYTES, 0x601);
    let src16 = AlignedBuf::noise(BYTES, 0x602);
    let mut dst16 = AlignedBuf::noise(BYTES, 0x603);

    // gf8b
    let c8 = Elem::<Binary<8, Polynomial<AES>>>::from_raw(0x53);
    bench_scalar("gf8b scalar mul", || {
        let mut x = Elem::<Binary<8, Polynomial<AES>>>::from_raw(0xA5);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(Elem::<Binary<8, Polynomial<AES>>>::from_raw(
                (0x53u8).wrapping_add(i as u8),
            ));
        }
        black_box(x);
    });
    bench_region("gf8b mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Binary<8, Polynomial<AES>>>(
            black_box(dst8.as_mut_slice()),
            c8,
            black_box(src8.as_slice()),
        );
    });
    bench_region("gf8b mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Binary<8, Polynomial<AES>>>(
            black_box(dst8.as_mut_slice()),
            c8,
            black_box(src8.as_slice()),
        );
    });

    // Bit-compatible GF(2^8)/0x11D used by ISA-L and klauspost/reedsolomon.
    let c8d = Elem::<Binary<8, Polynomial<RS>>>::from_raw(0x53);
    bench_scalar("gf8d scalar mul", || {
        let mut x = Elem::<Binary<8, Polynomial<RS>>>::from_raw(0xA5);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(Elem::<Binary<8, Polynomial<RS>>>::from_raw(
                (0x53u8).wrapping_add(i as u8),
            ));
        }
        black_box(x);
    });
    bench_region("gf8d mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Binary<8, Polynomial<RS>>>(
            black_box(dst8.as_mut_slice()),
            c8d,
            black_box(src8.as_slice()),
        );
    });
    bench_region("gf8d mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Binary<8, Polynomial<RS>>>(
            black_box(dst8.as_mut_slice()),
            c8d,
            black_box(src8.as_slice()),
        );
    });

    // In-place scale sweep: the V4x-unconditional question for `Binary<8, Polynomial<AES>>`.
    for &len in &[4096usize, 16384, 65536, 262144, 1048576] {
        let mut dst = AlignedBuf::noise(len, 0x610 + len as u64);
        bench_region(&format!("gf8b assign {len}B"), len, || {
            ops::mul_assign::<Binary<8, Polynomial<AES>>>(black_box(dst.as_mut_slice()), c8);
        });
        let mut dst = AlignedBuf::noise(len, 0x620 + len as u64);
        bench_region(&format!("gf8d assign {len}B"), len, || {
            ops::mul_assign::<Binary<8, Polynomial<RS>>>(black_box(dst.as_mut_slice()), c8d);
        });
    }
    // 4 MiB overwrite: the streaming-store threshold shape.
    {
        let src4 = AlignedBuf::noise(4 * 1024 * 1024, 0x604);
        let mut dst4 = AlignedBuf::noise(4 * 1024 * 1024, 0x605);
        bench_region("gf8d mul_into 4MiB (dst = c*src)", 4 * 1024 * 1024, || {
            ops::mul_into::<Binary<8, Polynomial<RS>>>(
                black_box(dst4.as_mut_slice()),
                c8d,
                black_box(src4.as_slice()),
            );
        });
    }

    // Elementwise products, both byte fields and both forms.
    // `Binary<8, Polynomial<RS>>` conjugates `GF2P8MULB` by the field isomorphism on
    // GFNI tiers.
    {
        let a = AlignedBuf::noise(BYTES, 0x606);
        let b = AlignedBuf::noise(BYTES, 0x607);
        let mut dst = AlignedBuf::noise(BYTES, 0x608);
        bench_region("gf8d elementwise (dst = a*b)", BYTES, || {
            ops::mul_elementwise::<Binary<8, Polynomial<RS>>>(
                black_box(dst.as_mut_slice()),
                black_box(a.as_slice()),
                black_box(b.as_slice()),
            );
        });
        bench_region("gf8d elementwise assign (dst *= src)", BYTES, || {
            ops::mul_elementwise_assign::<Binary<8, Polynomial<RS>>>(
                black_box(dst.as_mut_slice()),
                black_box(b.as_slice()),
            );
        });
        bench_region("gf8b elementwise (dst = a*b)", BYTES, || {
            ops::mul_elementwise::<Binary<8, Polynomial<AES>>>(
                black_box(dst.as_mut_slice()),
                black_box(a.as_slice()),
                black_box(b.as_slice()),
            );
        });
        bench_region("gf8b elementwise assign (dst *= src)", BYTES, || {
            ops::mul_elementwise_assign::<Binary<8, Polynomial<AES>>>(
                black_box(dst.as_mut_slice()),
                black_box(b.as_slice()),
            );
        });
    }

    // gf16b
    let c16 = Elem::<Gf16>::from_raw(0x53A7);
    bench_scalar("gf16b scalar mul", || {
        let mut x = Elem::<Gf16>::from_raw(0x1234);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(Elem::<Gf16>::from_raw((0x53A7u16).wrapping_add(i as u16)));
        }
        black_box(x);
    });
    bench_region("gf16b mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf16>(
            black_box(dst16.as_mut_slice()),
            c16,
            black_box(src16.as_slice()),
        );
    });
    bench_region("gf16b mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf16>(
            black_box(dst16.as_mut_slice()),
            c16,
            black_box(src16.as_slice()),
        );
    });

    for &len in DOT_LENGTHS {
        bench_mul_into_gather(len);
    }

    for &nrows in ENCODE_ROWS {
        bench_encode(nrows);
    }

    // Last: the sweep allocates many buffers, and running it earlier shifts
    // the heap and cache state the rows above are recorded under.
    bench_peel_floors();
}
