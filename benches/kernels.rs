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
//! `SAMPLE\t{label}\t{logical_bytes}\t{ns_per_call:.9}` for region shapes.
//! Process entry prints `FIELD_BACKEND\t{label}\t{backend_name}` for every
//! measured field — each field's own routing, not the process backend —
//! plus the host's actual CPU affinity and scheduler policy. Each process
//! times an unchanged byte-XOR control at start and end
//! (`control byte xor start`, `control byte xor end`), so campaign drift is
//! separable from field speed. `--blocked-diagnostic` runs the
//! internals-only blocked-versus-AXPY probe; the direct prime panels never
//! invoke it. Stable public panels live in the `operations` benchmark.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::binary::AES;
use fgf::binary::{Polynomial, RS};
use fgf::extension::QuadMersenne31;
use fgf::internals::kernel::ResidueStrategy;
use fgf::prime::{Goldilocks, Mersenne31};
use fgf::{Elem, FieldBuffer, FiniteField, backend, backend_for, ops};

type Gf16 = fgf::FiniteField<2, 16, fgf::extension::Extension<fgf::binary::Rijndael16, 2>>;
type Gf32 = fgf::FiniteField<2, 32, fgf::extension::Extension<fgf::binary::Rijndael32, 2>>;
type Gf64 = fgf::FiniteField<2, 64, fgf::extension::Extension<fgf::binary::Rijndael64, 2>>;
type FanPaar16Field = fgf::FiniteField<2, 16, fgf::extension::Extension<fgf::binary::FanPaar16, 2>>;
type FanPaar32Field = fgf::FiniteField<2, 32, fgf::extension::Extension<fgf::binary::FanPaar32, 2>>;
type FanPaar64Field = fgf::FiniteField<2, 64, fgf::extension::Extension<fgf::binary::FanPaar64, 2>>;
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
    fn field<F: fgf::FieldBuffer>(label: &str) {
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    F: FieldBuffer,
{
    // The direct column calls the kernel's own body wherever its
    // preconditions allow (four-byte rows at any length), independent of
    // the routing selector. `takes` below only selects the oracle path.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let takes = F::BYTES == 4;
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let _ = p;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let takes = {
        let k = ((1u128 << 32) / P as u128) as u64;
        k >= 2
    };
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let _ = coeff_raw;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let flat: Vec<fgf::internals::kernel::x86::residue::PreparedResidue> = coeff_raw
        .iter()
        .map(|&c| {
            let q = ((c as u128 * (1u128 << 32)) / P as u128) as u64;
            fgf::internals::kernel::x86::residue::PreparedResidue::from_word(c | (q << 32))
        })
        .collect();
    // Rows the kernel rejects (delay depth below two) print no direct
    // cell; the production and gathers columns still time.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let kernel_accepts = {
        let k = ((1u128 << 32) / P as u128) as u64;
        k >= 2
    };
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    F: FieldBuffer,
{
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    use fgf::internals::kernel::routing as selector;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let bytes = F::BYTES;
    // The direct column calls the vector body for every kernel-covered
    // class at every measured length, independent of the routing floors.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let takes = selector::routes_vector(bytes, p);
    // Prepared word parts for the direct vector call, derived exactly as
    // dispatch derives them: Shoup pair halves on four-byte rows,
    // Montgomery word on eight-byte odd rows, canonical byte on byte rows.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let (c32, q32, c_mont, c8) = direct_coefficient_parts(c, p, bytes);
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let _ = (c, p);
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let params = direct_u32_params(p);
    // Rows the kernel rejects print no direct cell.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let kernel_accepts = selector::routes_vector(F::BYTES, p);
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn direct_u32_params(p: u64) -> fgf::internals::kernel::x86::residue::U32Params {
    fgf::internals::kernel::x86::residue::U32Params::new(p as u32)
}

/// Split the reduced coefficient into the direct call's word parts.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    F: FieldBuffer,
{
    use fgf::internals::kernel::prime as portable;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
    F: FieldBuffer,
{
    let bytes = F::BYTES;
    println!("prime row {row} — backend {}:", backend_for::<F>().name());
    println!("PRIME_ROW\t{row}\t{}", backend_for::<F>().name());
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use fgf::internals::kernel::routing as selector_oracle;
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            use fgf::internals::kernel::routing as selector;
            use fgf::internals::kernel::routing::VectorOp;
            let into_backend = selector::single_row_backend(bytes, p, len, VectorOp::Into);
            let add_backend = selector::single_row_backend(bytes, p, len, VectorOp::Add);
            let assign_backend = selector::single_row_backend(bytes, p, len, VectorOp::Assign);
            let takes_into = into_backend != ResidueStrategy::Scalar;
            let takes_add = add_backend != ResidueStrategy::Scalar;
            let takes_assign = assign_backend != ResidueStrategy::Scalar;
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
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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

/// Elementwise panels for one four-byte row: `mul_elementwise_into` and
/// `mul_elementwise_assign` at short lengths, with the same
/// region, placement, and oracle discipline as the single-row panels.
#[allow(clippy::cast_possible_truncation)]
fn prime_elementwise_panel<F>(row: &str, p: u64, seed: u64)
where
    F: FieldBuffer,
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
        ops::mul_elementwise_into::<F>(&mut probe, &a0, &b0);
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
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use fgf::internals::kernel::routing as selector_oracle;
            if !selector_oracle::routes_vector(bytes, p) || bytes != 4 {
                // Kernel rejects this class; no direct cell exists.
            } else {
                let params = direct_u32_params(p);
                let mut direct = vec![0u8; len];
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            use fgf::internals::kernel::routing as selector;
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            use fgf::internals::kernel::routing::VectorOp;
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            let operation_backend =
                selector::single_row_backend(bytes, p, len, VectorOp::Elementwise);
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            let assign_backend =
                selector::single_row_backend(bytes, p, len, VectorOp::ElementwiseAssign);
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let operation_backend = ResidueStrategy::Scalar;
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            let assign_backend = ResidueStrategy::Scalar;
            let takes = operation_backend != ResidueStrategy::Scalar;
            let takes_assign = assign_backend != ResidueStrategy::Scalar;
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
                ops::mul_elementwise_into::<F>(dst, a, b);
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
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    use fgf::internals::kernel::routing as selector;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let k = selector::delay_depth::<P>();
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
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
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                let matrix_backend = selector::delayed_matrix_backend(4, k, row_len, nsources, add);
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                let matrix_backend = ResidueStrategy::Scalar;
                let routes = matrix_backend != ResidueStrategy::Scalar;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                let operation_backend = if routes {
                    matrix_backend
                } else {
                    selector::single_row_backend(4, P, row_len, selector::VectorOp::Add)
                };
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                let operation_backend = ResidueStrategy::Scalar;
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
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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
                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
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

/// Remaining panels: the direct-kernel prime comparisons and the
/// internals-only blocked diagnostic. The stable public panels now live in
/// the `operations` benchmark.
fn run_panels() {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|arg| arg == "--blocked-diagnostic") {
            bench_blocked_vs_axpy();
            return;
        }
    }
    bench_prime_single_rows();
    bench_prime_elementwise();
    bench_prime_matrix();
}
