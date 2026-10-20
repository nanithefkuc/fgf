//! Public-API surface benchmark: every stable callable in `fgf` gets a
//! registered measurement with an independent correctness oracle, a stable
//! case string, and a machine record.
//!
//! ```sh
//! cargo bench --features std,simd,internals --bench surfaces
//! cargo bench --features std,simd,internals --bench surfaces -- --inventory
//! cargo bench --features std,simd,internals --bench surfaces -- --check
//! cargo bench --features std,simd,internals --bench surfaces -- --smoke
//! ```
//!
//! Every mode prints the same `SURFACE_INVENTORY\t{...}` stream — one JSON
//! record per registered measurement, without timing — so inventories match
//! byte-for-byte across rounds, hosts, and modes. A run that measures adds
//! one `SURFACE\t{...}` record per case carrying the same fields plus
//! `ns` (per-call median nanoseconds, nine decimal places), `backend`
//! (the field's actual backend name, `scalar` for maps), `samples`, and
//! `p10_ns`/`p90_ns` when enough samples exist. `--check` runs every
//! fixture's correctness oracle once and times nothing; `--smoke` keeps
//! every case and shortens repetition only. The process times an unchanged
//! byte-XOR control at entry and exit (`control: true`, `phase` entry/exit)
//! so campaign drift is separable from field work, and prints
//! `FIELD_BACKEND` routing lines for every registered field.

// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;
use std::time::Instant;

#[path = "surfaces/bits_cases.rs"]
mod bits_cases;
#[path = "surfaces/bulk.rs"]
mod bulk;
#[path = "surfaces/convert.rs"]
mod convert;
#[path = "surfaces/details_maps.rs"]
mod details_maps;
#[path = "surfaces/details_math.rs"]
mod details_math;
#[path = "surfaces/fields.rs"]
mod fields;
#[path = "surfaces/fixture.rs"]
mod fixture;
#[path = "surfaces/inherent_calls.rs"]
mod inherent_calls;
#[path = "surfaces/maps.rs"]
mod maps;
#[path = "surfaces/missing_calls.rs"]
mod missing_calls;
#[path = "surfaces/queries.rs"]
mod queries;
#[path = "surfaces/registry.rs"]
mod registry;
#[path = "surfaces/scalar.rs"]
mod scalar;
#[path = "surfaces/spread.rs"]
mod spread;

use registry::{Case, CaseCtx};

/// Logical byte count of the drift control's XOR pass.
const CONTROL_BYTES: usize = 256 * 1024;

/// The drift control: a plain byte XOR over black-boxed buffers. The body
/// names no field code, so start/end movement measures the measurement
/// environment.
fn control_body(dst: &mut [u8], src: &[u8]) {
    let dst = black_box(dst);
    let src = black_box(src);
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// One control measurement: `phase` is `entry` or `exit`.
fn control_case(phase: &'static str) -> Case {
    Case {
        case: format!("control/byte-xor/{phase}"),
        api: &["unchanged byte XOR control"],
        field: "bytes".to_string(),
        operation: "control byte xor".to_string(),
        scope: registry::Scope::Steady,
        bytes: CONTROL_BYTES,
        lanes: CONTROL_BYTES,
        sources: 1,
        destinations: 1,
        control: true,
        phase: Some(phase),
        run: |case, ctx| {
            let mut seed = scalar::seed_for(&case.case);
            let src = fixture::noise(CONTROL_BYTES, fixture::lcg(&mut seed));
            let mut dst = fixture::noise(CONTROL_BYTES, fixture::lcg(&mut seed));
            {
                let mut expected = dst.clone();
                control_body(&mut expected, &src);
                control_body(&mut dst, &src);
                assert_eq!(dst, expected, "{}: control oracle", case.case);
            }
            registry::time(case, ctx, "scalar", || {
                control_body(&mut dst, &src);
                fixture::observe(dst.len());
            });
        },
    }
}

/// The host's actual CPU affinity and scheduling policy, from the kernel.
fn print_host_lines() {
    let status = std::fs::read_to_string("/proc/self/status").ok();
    let cpus = status
        .as_deref()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
        })
        .map(|line| line["Cpus_allowed_list:".len()..].trim().to_string());
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

/// Deterministic registration order: the control entry first, the control
/// exit last, and every panel between them in module order.
fn build_registry() {
    registry::add(control_case("entry"));
    queries::register();
    scalar::register();
    inherent_calls::register();
    convert::register();
    bulk::register();
    spread::register();
    bits_cases::register();
    maps::register();
    details_math::register();
    details_maps::register();
    missing_calls::register();
    registry::add(control_case("exit"));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inventory_only = args.iter().any(|arg| arg == "--inventory");
    let check = args.iter().any(|arg| arg == "--check");
    let smoke = args.iter().any(|arg| arg == "--smoke");

    println!(
        "fgf surfaces benchmark — backend: {}",
        fgf::backend().name()
    );
    let mode = if inventory_only {
        "inventory"
    } else if check {
        "check"
    } else if smoke {
        "smoke"
    } else {
        "full"
    };
    println!("BENCHMARK_MODE\t{mode}");
    print_host_lines();
    fields::print_field_backends();
    build_registry();

    let cases = registry::cases();
    for case in &cases {
        registry::print_inventory(case);
    }
    if inventory_only {
        return;
    }

    let ctx = CaseCtx { check, smoke };
    if !check {
        let (ns, samples, _, _) = ctx.measure(|| {
            black_box(0_u64);
        });
        println!("TIMER_FLOOR\t{{\"ns\":{ns:.9},\"samples\":{samples}}}");
    }
    let started = Instant::now();
    for case in &cases {
        (case.run)(case, &ctx);
        if check {
            println!("SURFACE_CHECKED\t{}", case.case);
        }
    }
    if check {
        println!("SURFACES_CHECK_COMPLETE\tcases={}", cases.len());
    } else {
        println!(
            "SURFACES\tcases={}\telapsed_s={:.3}",
            cases.len(),
            started.elapsed().as_secs_f32()
        );
    }
}
