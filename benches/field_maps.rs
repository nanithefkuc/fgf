//! Field-map measurement baseline: the compact-map control every future
//! kernel PR must beat.
//!
//! Times the public compact-map surface — [`Embedding`] (`embed`,
//! `restrict`, `contains`, `frobenius`, `trace`, `norm`) and
//! [`Isomorphism`] (`apply`, `unapply`) — over the contracted pair set, as
//! scalar element chains and as region loops of `FieldBuffer::decode` into a
//! compact-map method into `FieldBuffer::encode`.
//!
//! ```sh
//! cargo bench --bench field_maps [-- --smoke]
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
//! separable from field speed. `--smoke` runs one panel per class with tiny
//! iteration counts; unknown flags pass through harmlessly.
// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::poly::{AES, RS};
use fgf::{
    BinaryField, Components, Elem, Embedding, Extension, ExtensionSpec, Field, FieldBuffer,
    FieldElem, FiniteField, Frobenius, Goldilocks, Isomorphism, Mersenne31, Normal, Polynomial,
    PreparedEmbedding, PreparedIsomorphism, PrimeCharacteristic, PrimeField, QuadMersenne31,
    Relation, Residue, backend, backend_for, ops,
};
use fgf::{FanPaar8, FanPaar16, FanPaar64, Rijndael16, Rijndael32, Rijndael64};

type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Rs8 = FiniteField<2, 8, Polynomial<RS>>;
type Normal8 = FiniteField<2, 8, Normal<0x11B, 0x20>>;
type Fp8 = FiniteField<2, 8, Extension<FanPaar8, 2>>;
type Fp16 = FiniteField<2, 16, Extension<FanPaar16, 2>>;
type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
type Gf32 = FiniteField<2, 32, Extension<Rijndael32, 2>>;
type Gf64 = FiniteField<2, 64, Extension<Rijndael64, 2>>;
type Fp64 = FiniteField<2, 64, Extension<FanPaar64, 2>>;
type Gf1 = FiniteField<2, 1, Polynomial<3>>;
type Ref2 = FiniteField<2, 2, Polynomial<0x7>>;
type Ref8 = Aes8;
type Res2 = FiniteField<2, 1, Residue>;
type R3 = FiniteField<3, 1, Residue>;
type R7 = FiniteField<7, 1, Residue>;

// ---------------------------------------------------------------------------
// Local inventory rows (copied shapes from `tests/relations.rs`)
// ---------------------------------------------------------------------------

struct G9AltSpec;
impl ExtensionSpec<2> for G9AltSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as a^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type G9Alt = FiniteField<3, 2, Extension<G9AltSpec, 2>>;

struct G81QuarticSpec;
impl ExtensionSpec<4> for G81QuarticSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as x^4 + 2x^3 + 2";
    const RELATION: Relation = Relation::from_coordinates(&[2, 0, 2, 0]);
}
type G81Quartic = FiniteField<3, 4, Extension<G81QuarticSpec, 4>>;

struct G81ExtSpec;
impl ExtensionSpec<2> for G81ExtSpec {
    type Base = G9Alt;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as t^2 - (1 + a)";
    // `-(1 + a)`: constant `2 + 2a`, linear zero.
    const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
}
type G81Ext = FiniteField<3, 4, Extension<G81ExtSpec, 2>>;

struct G343Spec;
impl ExtensionSpec<3> for G343Spec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "GF(343) as x^3 - 2";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}
type G343 = FiniteField<7, 3, Extension<G343Spec, 3>>;

struct OverRes2Spec;
impl ExtensionSpec<2> for OverRes2Spec {
    type Base = Res2;
    type Storage = Components;
    const NAME: &'static str = "GF(4) as t^2 + t + 1 over residues";
    const RELATION: Relation = Relation::from_coordinates(&[1, 1]);
}
type OverRes2 = FiniteField<2, 2, Extension<OverRes2Spec, 2>>;

struct OverAesSpec;
impl ExtensionSpec<2> for OverAesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) as t^2 + 0x53*t + 1";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[1, 0x53]);
}
type OverAes = FiniteField<2, 16, Extension<OverAesSpec, 2>>;

struct G320Spec;
impl ExtensionSpec<20> for G320Spec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^20) as t^20 + 2t^16 + t^14 + 2t^10 + t^2 + t + 2";
    // Constant first: `2 + t + t^2 + 2t^10 + t^14 + 2t^16`. Rabin-verified
    // irreducible (independent checker: trivial gcds at every proper divisor
    // of 20, `x^(3^20) == x`).
    #[rustfmt::skip]
    const RELATION: Relation = Relation::from_coordinates(&[
        2, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 1, 0, 2, 0, 0, 0,
    ]);
}
type G320 = FiniteField<3, 20, Extension<G320Spec, 20>>;

struct G340Spec;
impl ExtensionSpec<2> for G340Spec {
    type Base = G320;
    type Storage = Components;
    const NAME: &'static str = "GF(3^40) as t^2 - (1 + s)";
    // `-(1 + s)`: constant `2 + 2s`, linear zero, in the base's own order.
    // `(1 + s)` is a nonsquare in GF(3^20) (`(1+s)^((3^20-1)/2) != 1` in the
    // independent checker), so the quadratic is irreducible.
    const RELATION: Relation = Relation::from_coordinates(&[
        2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
}
type G340 = FiniteField<3, 40, Extension<G340Spec, 2>>;

struct GldSquareSpec;
impl ExtensionSpec<2> for GldSquareSpec {
    type Base = Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(g^2) as t^2 - 7";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}
type GldSquare = FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GldSquareSpec, 2>>;

// ---------------------------------------------------------------------------
// Harness conventions (sibling of `compare.rs` / `kernels.rs`)
// ---------------------------------------------------------------------------

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
/// Target wall-clock duration of one scalar sample batch.
const SCALAR_TARGET_NS: f64 = 1_000_000.0;
/// Upper bound on element operations in one scalar sample batch.
const MAX_SCALAR_OPS: usize = 1 << 20;
/// Staging length of every scalar panel's deterministic input cycle.
const SCALAR_STAGE: usize = 256;

/// Inventory mode for the publisher regression (`just bench-maps-publisher`):
/// print every `SCALAR`/`SAMPLE` label with a placeholder value and return,
/// so the full producer inventory feeds the grammar without timing. Staging,
/// validation, and headings still run; unknown flags pass through harmlessly.
static LIST_LABELS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::args().any(|arg| arg == "--list-labels"));

static SMOKE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::args().any(|arg| arg == "--smoke"));

/// Whether this process runs the smoke subset: one panel per class with
/// tiny iteration counts. Unknown flags pass through harmlessly.
fn smoke() -> bool {
    *SMOKE
}

/// Element counts of the region panels; the smoke subset keeps the two
/// smallest.
fn region_counts() -> &'static [usize] {
    if smoke() {
        &[1, 8]
    } else {
        &[1, 8, 64, 1024, 16384]
    }
}

fn max_samples() -> usize {
    if smoke() { 4 } else { 64 }
}

fn sample_deadline() -> Duration {
    if smoke() {
        Duration::from_millis(10)
    } else {
        Duration::from_millis(250)
    }
}

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
    field::<Aes8>("aes8");
    field::<Rs8>("rs8");
    field::<Normal8>("normal8");
    field::<Fp8>("fp8");
    field::<Ref8>("ref8");
    field::<Gf16>("gf16");
    field::<Gf32>("gf32");
    field::<Gf64>("gf64");
    field::<Fp16>("fp16");
    field::<Fp64>("fp64");
    field::<Res2>("res2");
    field::<OverRes2>("res2q");
    field::<OverAes>("aesq");
    field::<R3>("r3");
    field::<G9Alt>("g9");
    field::<G81Ext>("g81");
    field::<G81Quartic>("g81q");
    field::<R7>("r7");
    field::<G343>("g343");
    field::<G340>("g340");
    field::<Mersenne31>("m31");
    field::<QuadMersenne31>("qm31");
    field::<Goldilocks>("gld");
    field::<GldSquare>("gld2");
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
    if *LIST_LABELS {
        println!("SAMPLE\t{label}\t{bytes}\t0.000000001");
        return;
    }
    let warmup = if smoke() { 2 } else { 16 };
    for _ in 0..warmup {
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
    let mut samples = Vec::with_capacity(max_samples());
    let deadline = Instant::now() + sample_deadline();
    while Instant::now() < deadline && samples.len() < max_samples() {
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

/// Time a scalar element chain: the body takes an operation count sized by
/// calibration and returns the operations it performed, and the helper
/// reports one timed call per sample. Emits a `SCALAR` machine record
/// instead of `SAMPLE`; a chain has no logical byte count.
fn bench_scalar(label: &str, mut body: impl FnMut(usize) -> u64) {
    if *LIST_LABELS {
        println!("SCALAR\t{label}\t0.000000001");
        return;
    }
    let warmup = if smoke() { 1 } else { 4 };
    for _ in 0..warmup {
        body(8);
    }
    // One calibration call sizes every sample batch: the batch grows to
    // `SCALAR_TARGET_NS` for cheap operations and never drops below one.
    let est_ns = {
        let start = Instant::now();
        let ops = body(1).max(1);
        start.elapsed().as_secs_f64() * 1e9 / ops as f64
    };
    let ops = ((SCALAR_TARGET_NS / est_ns.max(1.0)).ceil() as usize).clamp(1, MAX_SCALAR_OPS);
    let mut samples = Vec::with_capacity(max_samples());
    let deadline = Instant::now() + sample_deadline();
    while Instant::now() < deadline && samples.len() < max_samples() {
        let start = Instant::now();
        let performed = body(ops).max(1);
        let elapsed = start.elapsed();
        samples.push((elapsed, performed));
    }
    samples.sort_unstable_by(|a, b| {
        (a.0.as_secs_f64() / a.1 as f64).total_cmp(&(b.0.as_secs_f64() / b.1 as f64))
    });
    let median = samples[samples.len() / 2];
    let ns = median.0.as_secs_f64() * 1e9 / median.1 as f64;
    let mops = median.1 as f64 / median.0.as_secs_f64() / 1e6;
    println!("  {label:<44} {ns:>9.2} ns/op  {mops:>7.2} Mops/s");
    println!("SCALAR\t{label}\t{ns:.9}");
}

// ---------------------------------------------------------------------------
// Deterministic inputs and pre-timing validation
// ---------------------------------------------------------------------------

/// Deterministic sample indices below `order`: boundaries first, then the
/// `compare.rs` LCG stream.
fn sample_indices(order: u128, salt: u64, n: usize) -> Vec<u128> {
    let mut out = Vec::with_capacity(n);
    for boundary in [0u128, 1, order - 1] {
        if out.len() < n {
            out.push(boundary);
        }
    }
    let mut state = salt | 1;
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let wide =
            ((u128::from(state >> 33) << 31) | u128::from((state >> 2) & 0x7fff_ffff)) % order;
        out.push(wide);
    }
    out
}

fn field_samples<F: Field>(n: usize, salt: u64) -> Vec<Elem<F>> {
    sample_indices(F::ORDER, salt, n)
        .into_iter()
        .map(|index| {
            Elem::<F>::from_index(index).expect("a sampled index sits below the field order")
        })
        .collect()
}

/// Synthesize `n` target elements outside the embedding image by flipping
/// high index bits of member images. Every candidate is verified against
/// [`Embedding::contains`] before use; failure to find enough outsiders
/// aborts the run naming the pair.
fn synthesize_outsiders<S: Field, T: Field<Characteristic = S::Characteristic>>(
    pair: &str,
    map: &Embedding<S, T>,
    members: &[Elem<T>],
    n: usize,
) -> Vec<Elem<T>> {
    let mut outsiders = Vec::with_capacity(n);
    for seed in members {
        for k in (0..128u32).rev() {
            if outsiders.len() == n {
                break;
            }
            let index = seed.to_index() ^ (1u128 << k);
            if index >= T::ORDER {
                continue;
            }
            let candidate =
                Elem::<T>::from_index(index).expect("a flipped index sits below the target order");
            if !map.contains(candidate) {
                outsiders.push(candidate);
            }
        }
        if outsiders.len() == n {
            break;
        }
    }
    assert!(
        outsiders.len() == n,
        "field-maps {pair}: could not synthesize {n} nonmembers"
    );
    outsiders
}

/// Validate every timed arm's results before timing: round-trips,
/// membership against the Frobenius fixed-point test, the relative
/// Frobenius identity, trace against conjugate sums, and norm against
/// conjugate products, on deterministic sampled inputs. Panics name the
/// pair and operation.
fn validate_embedding<S: Field, T: Field<Characteristic = S::Characteristic>>(
    pair: &str,
    map: &Embedding<S, T>,
) {
    let relative = T::DEGREE / S::DEGREE;
    assert!(
        relative >= 2 && S::DEGREE * relative == T::DEGREE,
        "field-maps {pair}: expected a proper inclusion"
    );
    let sources: Vec<Elem<S>> = field_samples(48, 0xE4B1);
    for (i, x) in sources.iter().enumerate() {
        let image = map.embed(*x);
        assert!(
            map.contains(image),
            "field-maps {pair} embed: image {i} is not contained"
        );
        assert_eq!(
            map.restrict(image),
            Some(*x),
            "field-maps {pair} embed/restrict round-trip {i}"
        );
    }
    let members: Vec<Elem<T>> = sources.iter().map(|x| map.embed(*x)).collect();
    let targets: Vec<Elem<T>> = field_samples(48, 0x7A60);
    for (i, y) in targets.iter().enumerate() {
        let fixed = y.pow(S::ORDER) == *y;
        assert_eq!(
            map.contains(*y),
            fixed,
            "field-maps {pair} contains against the Frobenius test {i}"
        );
        assert_eq!(
            map.restrict(*y).is_some(),
            fixed,
            "field-maps {pair} restrict against the Frobenius test {i}"
        );
    }
    let outsiders = synthesize_outsiders(pair, map, &members, 16);
    for (i, y) in outsiders.iter().enumerate() {
        assert!(
            !map.contains(*y),
            "field-maps {pair} contains admits nonmember {i}"
        );
        assert_eq!(
            map.restrict(*y),
            None,
            "field-maps {pair} restrict admits nonmember {i}"
        );
    }
    for (i, y) in targets.iter().enumerate() {
        let mut power = *y;
        for _ in 0..relative {
            power = map.frobenius(power);
        }
        assert_eq!(power, *y, "field-maps {pair} frobenius identity {i}");
        assert_eq!(
            map.contains(map.frobenius(*y)),
            map.contains(*y),
            "field-maps {pair} frobenius membership {i}"
        );
    }
    for (i, y) in targets.iter().enumerate() {
        let mut sum = *y;
        let mut product = *y;
        let mut power = *y;
        for _ in 1..relative {
            power = power.pow(S::ORDER);
            sum = sum.add(power);
            product = product.mul(power);
        }
        let expected_trace = map
            .restrict(sum)
            .unwrap_or_else(|| panic!("field-maps {pair} trace oracle outside the image {i}"));
        let expected_norm = map
            .restrict(product)
            .unwrap_or_else(|| panic!("field-maps {pair} norm oracle outside the image {i}"));
        assert_eq!(
            map.trace(*y),
            expected_trace,
            "field-maps {pair} trace against the conjugate sum {i}"
        );
        assert_eq!(
            map.norm(*y),
            expected_norm,
            "field-maps {pair} norm against the conjugate product {i}"
        );
    }
}

/// Validate both isomorphism directions before timing: round-trips on
/// sampled inputs and the homomorphism laws on samples. Panics name the
/// pair and operation.
fn validate_isomorphism<S: Field, T: Field<Characteristic = S::Characteristic>>(
    pair: &str,
    map: &Isomorphism<S, T>,
) {
    let sources: Vec<Elem<S>> = field_samples(48, 0x150);
    for (i, x) in sources.iter().enumerate() {
        assert_eq!(
            map.unapply(map.apply(*x)),
            *x,
            "field-maps {pair} apply/unapply round-trip {i}"
        );
    }
    let targets: Vec<Elem<T>> = field_samples(48, 0xA99);
    for (i, y) in targets.iter().enumerate() {
        assert_eq!(
            map.apply(map.unapply(*y)),
            *y,
            "field-maps {pair} unapply/apply round-trip {i}"
        );
    }
    for (i, (x, y)) in sources.iter().zip(targets.iter()).enumerate().take(16) {
        let a = sources[(i + 5) % sources.len()];
        assert_eq!(
            map.apply(x.add(a)),
            map.apply(*x).add(map.apply(a)),
            "field-maps {pair} apply preserves addition {i}"
        );
        assert_eq!(
            map.apply(x.mul(a)),
            map.apply(*x).mul(map.apply(a)),
            "field-maps {pair} apply preserves multiplication {i}"
        );
        let b = targets[(i + 5) % targets.len()];
        assert_eq!(
            map.unapply(y.add(b)),
            map.unapply(*y).add(map.unapply(b)),
            "field-maps {pair} unapply preserves addition {i}"
        );
        assert_eq!(
            map.unapply(y.mul(b)),
            map.unapply(*y).mul(map.unapply(b)),
            "field-maps {pair} unapply preserves multiplication {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Buffer staging for the region panels
// ---------------------------------------------------------------------------

/// Verify one prepared restriction row before timing: run the prepared
/// row body over `count` lanes, then check every lane, bitmap bit, and tail
/// bit against the compact map's own `restrict`/`contains` answers.
#[allow(clippy::too_many_arguments)]
fn check_prepared_restrict_row<
    S: FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
>(
    pair: &str,
    operation: &str,
    map: &Embedding<S, T>,
    prepared: &PreparedEmbedding<S, T>,
    src: &[u8],
    dst: &mut [u8],
    valid: &mut [u8],
    count: usize,
) {
    let mask_len = fgf::bits::bytes_for(count);
    ops::restrict_into_with(
        &mut dst[..count * S::BYTES],
        &mut valid[..mask_len],
        prepared,
        &src[..count * T::BYTES],
    );
    for (i, lane_src) in src.chunks_exact(T::BYTES).enumerate().take(count) {
        let value = T::decode(lane_src);
        let lane = &dst[i * S::BYTES..(i + 1) * S::BYTES];
        match map.restrict(value) {
            Some(preimage) => {
                let mut expect = vec![0u8; S::BYTES];
                S::encode(&mut expect, preimage);
                assert_eq!(
                    lane,
                    expect.as_slice(),
                    "field-maps {pair} {operation} lane {i}"
                );
                assert_eq!(
                    valid[i / 8] & (1 << (i % 8)),
                    1 << (i % 8),
                    "field-maps {pair} {operation} bit {i}"
                );
            }
            None => {
                let mut expect = vec![0u8; S::BYTES];
                S::encode(&mut expect, Elem::<S>::ZERO);
                assert_eq!(
                    lane,
                    expect.as_slice(),
                    "field-maps {pair} {operation} zero lane {i}"
                );
                assert_eq!(
                    valid[i / 8] & (1 << (i % 8)),
                    0,
                    "field-maps {pair} {operation} cleared bit {i}"
                );
                assert!(
                    !map.contains(value),
                    "field-maps {pair} {operation} member {i}"
                );
            }
        }
    }
    if mask_len * 8 > count {
        let tail = valid[mask_len - 1] >> (count % 8);
        assert_eq!(tail, 0, "field-maps {pair} {operation} tail bits");
    }
}

/// Verify one prepared membership row before timing: run the prepared
/// bitmap body over `count` lanes, then check every bit and tail bit
/// against the compact map's `contains` answers.
fn check_prepared_contains_row<
    S: FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
>(
    pair: &str,
    operation: &str,
    map: &Embedding<S, T>,
    prepared: &PreparedEmbedding<S, T>,
    src: &[u8],
    valid: &mut [u8],
    count: usize,
) {
    let mask_len = fgf::bits::bytes_for(count);
    ops::contains_into_with(&mut valid[..mask_len], prepared, &src[..count * T::BYTES]);
    for (i, lane_src) in src.chunks_exact(T::BYTES).enumerate().take(count) {
        let expect = map.contains(T::decode(lane_src));
        assert_eq!(
            valid[i / 8] & (1 << (i % 8)),
            if expect { 1 << (i % 8) } else { 0 },
            "field-maps {pair} {operation} bit {i}"
        );
    }
    if mask_len * 8 > count {
        let tail = valid[mask_len - 1] >> (count % 8);
        assert_eq!(tail, 0, "field-maps {pair} {operation} tail bits");
    }
}

/// Verify one prepared element row before timing: run the prepared row
/// body over `count` lanes, then check every lane against the compact
/// scalar method applied to the same decoded lane.
fn check_prepared_element_row<
    S: FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
>(
    pair: &str,
    operation: &str,
    row: impl Fn(&mut [u8], &[u8]),
    scalar: impl Fn(Elem<T>) -> Elem<S>,
    src: &[u8],
    dst: &mut [u8],
    count: usize,
) {
    row(&mut dst[..count * S::BYTES], &src[..count * T::BYTES]);
    for (i, (lane_src, lane_dst)) in src
        .chunks_exact(T::BYTES)
        .zip(dst.chunks_exact(S::BYTES))
        .enumerate()
        .take(count)
    {
        let mut expect = vec![0u8; S::BYTES];
        S::encode(&mut expect, scalar(T::decode(lane_src)));
        assert_eq!(
            lane_dst,
            expect.as_slice(),
            "field-maps {pair} {operation} lane {i}"
        );
    }
}

/// Encode staged elements into bytes, verifying the decode round-trip so a
/// staging bug aborts before timing.
fn encode_staged<F: FieldBuffer>(pair: &str, operation: &str, elems: &[Elem<F>]) -> Vec<u8> {
    let mut bytes = vec![0u8; elems.len() * F::BYTES];
    for (i, elem) in elems.iter().enumerate() {
        F::encode(&mut bytes[i * F::BYTES..(i + 1) * F::BYTES], *elem);
    }
    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            F::decode(&bytes[i * F::BYTES..(i + 1) * F::BYTES]),
            *elem,
            "field-maps {pair} {operation}: staged buffer does not round-trip"
        );
    }
    bytes
}

// ---------------------------------------------------------------------------
// Embedding panels
// ---------------------------------------------------------------------------

/// Scalar independent-input throughput for one embedding pair: each timed
/// iteration maps one staged input, and the reported count is one operation
/// per iteration. Only the Frobenius panel is a dependent chain (its seed
/// sits outside the relative fixed field).
fn emb_scalar_panels<S: Field, T: Field<Characteristic = S::Characteristic>>(
    pair: &str,
    map: &Embedding<S, T>,
    with_mixed: bool,
) {
    let sources: Vec<Elem<S>> = field_samples(SCALAR_STAGE, 0x501);
    let members: Vec<Elem<T>> = sources.iter().map(|x| map.embed(*x)).collect();
    let outsiders = synthesize_outsiders(pair, map, &members, SCALAR_STAGE);
    let mixed: Vec<Elem<T>> = (0..SCALAR_STAGE)
        .map(|i| {
            if i % 2 == 0 {
                members[i / 2]
            } else {
                outsiders[i / 2]
            }
        })
        .collect();
    let stage = SCALAR_STAGE - 1;

    bench_scalar(&format!("field-maps/{pair}/embed/scalar-chain"), |ops| {
        let mut acc = Elem::<T>::ZERO;
        for i in 0..ops {
            acc = black_box(map.embed(sources[i & stage]));
        }
        black_box(acc);
        ops as u64
    });
    bench_scalar(
        &format!("field-maps/{pair}/restrict-member/scalar-chain"),
        |ops| {
            let mut acc = Elem::<S>::ZERO;
            for i in 0..ops {
                acc = black_box(
                    map.restrict(members[i & stage])
                        .expect("an image member restricts"),
                );
            }
            black_box(acc);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/restrict-nonmember/scalar-chain"),
        |ops| {
            let mut misses = 0u64;
            for i in 0..ops {
                misses += map.restrict(outsiders[i & stage]).is_none() as u64;
            }
            black_box(misses);
            ops as u64
        },
    );
    // The publisher grammar accepts mixed scalar chains only from the
    // scalar-only pairs; byte-form pairs emit region rows only.
    if with_mixed {
        bench_scalar(
            &format!("field-maps/{pair}/restrict-mixed/scalar-chain"),
            |ops| {
                let mut hits = 0u64;
                let mut acc = Elem::<S>::ZERO;
                for i in 0..ops {
                    // Alternate member/outsider with the region mask's rotation.
                    let y = if (i + 0x5A) % 2 == 0 {
                        members[(i / 2) & (SCALAR_STAGE / 2 - 1)]
                    } else {
                        outsiders[(i / 2) & (SCALAR_STAGE / 2 - 1)]
                    };
                    match map.restrict(y) {
                        Some(preimage) => {
                            acc = black_box(preimage);
                            hits += 1;
                        }
                        None => acc = black_box(Elem::<S>::ZERO),
                    }
                }
                black_box((acc, hits));
                ops as u64
            },
        );
    }
    bench_scalar(&format!("field-maps/{pair}/contains/scalar-chain"), |ops| {
        let mut hits = 0u64;
        for i in 0..ops {
            hits += map.contains(mixed[i & stage]) as u64;
        }
        black_box(hits);
        ops as u64
    });
    // The chain seed sits outside the relative fixed field, so every step
    // does full work: a fixed seed would time the identity forever.
    let chain_seed = *mixed
        .iter()
        .find(|y| !map.contains(**y))
        .unwrap_or(&mixed[1]);
    bench_scalar(
        &format!("field-maps/{pair}/frobenius/scalar-chain"),
        |ops| {
            let mut value = chain_seed;
            for _ in 0..ops {
                value = black_box(map.frobenius(value));
            }
            black_box(value);
            ops as u64
        },
    );
    {
        // Same map as the compact `frobenius` row beside it: the relative
        // σ_m with m = S::DEGREE, prepared once outside the timed chain.
        let prepared = Frobenius::<T>::new(S::DEGREE);
        bench_scalar(
            &format!("field-maps/{pair}/frobenius-prepared/scalar-chain"),
            |ops| {
                let mut value = chain_seed;
                for _ in 0..ops {
                    value = black_box(value.frobenius_with(&prepared));
                }
                black_box(value);
                ops as u64
            },
        );
    }
    bench_scalar(&format!("field-maps/{pair}/trace/scalar-chain"), |ops| {
        let mut acc = Elem::<S>::ZERO;
        for i in 0..ops {
            acc = black_box(map.trace(mixed[i & stage]));
        }
        black_box(acc);
        ops as u64
    });
    bench_scalar(&format!("field-maps/{pair}/norm/scalar-chain"), |ops| {
        let mut acc = Elem::<S>::ZERO;
        for i in 0..ops {
            acc = black_box(map.norm(mixed[i & stage]));
        }
        black_box(acc);
        ops as u64
    });
    // Prepared scalar application beside every compact chain above: the
    // same staged inputs through the compiled map, preparation hoisted
    // out of the timed chain. Seeds match the compact panels lane for
    // lane, so each prepared chain times the same workload.
    let prepared = PreparedEmbedding::<S, T>::new(map);
    bench_scalar(
        &format!("field-maps/{pair}/embed-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<T>::ZERO;
            for i in 0..ops {
                acc = black_box(prepared.embed(sources[i & stage]));
            }
            black_box(acc);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/restrict-member-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<S>::ZERO;
            for i in 0..ops {
                acc = black_box(
                    prepared
                        .restrict(members[i & stage])
                        .expect("an image member restricts"),
                );
            }
            black_box(acc);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/restrict-nonmember-prepared/scalar-chain"),
        |ops| {
            let mut misses = 0u64;
            for i in 0..ops {
                misses += prepared.restrict(outsiders[i & stage]).is_none() as u64;
            }
            black_box(misses);
            ops as u64
        },
    );
    if with_mixed {
        bench_scalar(
            &format!("field-maps/{pair}/restrict-mixed-prepared/scalar-chain"),
            |ops| {
                let mut hits = 0u64;
                let mut acc = Elem::<S>::ZERO;
                for i in 0..ops {
                    let y = if (i + 0x5A) % 2 == 0 {
                        members[(i / 2) & (SCALAR_STAGE / 2 - 1)]
                    } else {
                        outsiders[(i / 2) & (SCALAR_STAGE / 2 - 1)]
                    };
                    match prepared.restrict(y) {
                        Some(preimage) => {
                            acc = black_box(preimage);
                            hits += 1;
                        }
                        None => acc = black_box(Elem::<S>::ZERO),
                    }
                }
                black_box((acc, hits));
                ops as u64
            },
        );
    }
    bench_scalar(
        &format!("field-maps/{pair}/contains-prepared/scalar-chain"),
        |ops| {
            let mut hits = 0u64;
            for i in 0..ops {
                hits += prepared.contains(mixed[i & stage]) as u64;
            }
            black_box(hits);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/trace-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<S>::ZERO;
            for i in 0..ops {
                acc = black_box(prepared.trace(mixed[i & stage]));
            }
            black_box(acc);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/norm-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<S>::ZERO;
            for i in 0..ops {
                acc = black_box(prepared.norm(mixed[i & stage]));
            }
            black_box(acc);
            ops as u64
        },
    );
}

/// Region panels for one embedding pair: the compact-map control loops of
/// `FieldBuffer::decode` into a compact-map method into preallocated
/// `FieldBuffer::encode` lanes (plus an LSB-first validity bitmap for the
/// membership-reporting operations, per the `bits.rs` convention), the
/// baseline every future kernel PR must beat. All buffers are allocated
/// once outside timing; the timed bodies borrow black-boxed slices only.
/// The `frobenius-prepared` rows time the prepared row entry point instead,
/// with preparation hoisted out of the timed region.
#[allow(clippy::too_many_lines)]
fn emb_region_panels<S: FieldBuffer, T: Field<Characteristic = S::Characteristic> + FieldBuffer>(
    pair: &str,
    map: &Embedding<S, T>,
) {
    let top = region_counts().iter().max().copied().unwrap_or(1);
    let sources: Vec<Elem<S>> = field_samples(top, 0x860);
    let members: Vec<Elem<T>> = sources.iter().map(|x| map.embed(*x)).collect();
    let outsiders = synthesize_outsiders(pair, map, &members, top);
    let mixed: Vec<Elem<T>> = (0..top)
        .map(|i| {
            if i % 2 == 0 {
                members[i / 2]
            } else {
                outsiders[i / 2]
            }
        })
        .collect();
    let source_bytes = encode_staged::<S>(pair, "embed", &sources);
    let member_bytes = encode_staged::<T>(pair, "restrict-member", &members);
    let outsider_bytes = encode_staged::<T>(pair, "restrict-nonmember", &outsiders);
    let mixed_bytes = encode_staged::<T>(pair, "contains", &mixed);
    // Deterministic mixed mask: alternate member/outsider with a seed-varied
    // rotation, so both outcomes interleave at every count.
    let mixed_restrict: Vec<Elem<T>> = (0..top)
        .map(|i| {
            if (i + 0x5A) % 2 == 0 {
                members[i % members.len()]
            } else {
                outsiders[i % outsiders.len()]
            }
        })
        .collect();
    let mixed_restrict_bytes = encode_staged::<T>(pair, "restrict-mixed", &mixed_restrict);
    let elem_traffic = S::BYTES + T::BYTES;
    let mask_bytes = fgf::bits::bytes_for(top);

    for &count in region_counts() {
        let label = format!("field-maps/{pair}/embed/row-{count}");
        let bytes = count * elem_traffic;
        let src = &source_bytes[..count * S::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            for (s, d) in src
                .chunks_exact(S::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, map.embed(S::decode(s)));
            }
            black_box(dst);
        });
    }
    let prepared = PreparedEmbedding::<S, T>::new(map);
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/embed-prepared/row-{count}");
        let bytes = count * elem_traffic;
        let src = &source_bytes[..count * S::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        for (i, (x, lane)) in src
            .chunks_exact(S::BYTES)
            .zip(dst.chunks_exact_mut(T::BYTES))
            .enumerate()
            .take(count)
        {
            let mut expect = vec![0u8; T::BYTES];
            T::encode(&mut expect, map.embed(S::decode(x)));
            ops::embed_into_with(lane, &prepared, x);
            assert_eq!(
                lane,
                expect.as_slice(),
                "field-maps {pair} embed-prepared lane {i}"
            );
        }
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            ops::embed_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-member/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &member_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut mask = vec![0u8; mask_bytes];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let mask = black_box(&mut mask[..fgf::bits::bytes_for(count)]);
            mask.fill(0);
            for (i, (s, d)) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
                .enumerate()
            {
                match map.restrict(T::decode(s)) {
                    Some(preimage) => {
                        S::encode(d, preimage);
                        mask[i / 8] |= 1 << (i % 8);
                    }
                    None => S::encode(d, Elem::<S>::ZERO),
                }
            }
            black_box((dst, mask));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-member-prepared/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &member_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut valid = vec![0u8; mask_bytes];
        check_prepared_restrict_row(
            pair,
            "restrict-member-prepared",
            map,
            &prepared,
            src,
            &mut dst,
            &mut valid,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let valid = black_box(&mut valid[..fgf::bits::bytes_for(count)]);
            ops::restrict_into_with(dst, valid, black_box(&prepared), src);
            black_box((dst, valid));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-nonmember/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &outsider_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut mask = vec![0u8; mask_bytes];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let mask = black_box(&mut mask[..fgf::bits::bytes_for(count)]);
            mask.fill(0);
            for (i, (s, d)) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
                .enumerate()
            {
                match map.restrict(T::decode(s)) {
                    Some(preimage) => {
                        S::encode(d, preimage);
                        mask[i / 8] |= 1 << (i % 8);
                    }
                    None => S::encode(d, Elem::<S>::ZERO),
                }
            }
            black_box((dst, mask));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-mixed/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &mixed_restrict_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut mask = vec![0u8; mask_bytes];
        // The exact row body the timing loop runs, shared so validation
        // executes the measured outputs rather than a parallel oracle.
        let run_mixed_row = |map: &Embedding<S, T>, dst: &mut [u8], mask: &mut [u8]| {
            mask.fill(0);
            for (i, (s, d)) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
                .enumerate()
            {
                match map.restrict(T::decode(s)) {
                    Some(preimage) => {
                        S::encode(d, preimage);
                        mask[i / 8] |= 1 << (i % 8);
                    }
                    None => S::encode(d, Elem::<S>::ZERO),
                }
            }
        };
        // Validate the measured lanes and bitmap against independently
        // computed expectations: canonical preimage-or-zero lanes,
        // LSB-first membership bits, cleared tail bits.
        {
            let mask_len = fgf::bits::bytes_for(count);
            run_mixed_row(map, &mut dst[..count * S::BYTES], &mut mask[..mask_len]);
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                let value = T::decode(s);
                let lane = &dst[i * S::BYTES..(i + 1) * S::BYTES];
                match map.restrict(value) {
                    Some(preimage) => {
                        let mut expect = vec![0u8; S::BYTES];
                        S::encode(&mut expect, preimage);
                        assert_eq!(
                            lane,
                            expect.as_slice(),
                            "field-maps {pair} restrict-mixed lane {i}"
                        );
                        assert_eq!(
                            mask[i / 8] & (1 << (i % 8)),
                            1 << (i % 8),
                            "field-maps {pair} restrict-mixed bit {i}"
                        );
                    }
                    None => {
                        let mut expect = vec![0u8; S::BYTES];
                        S::encode(&mut expect, Elem::<S>::ZERO);
                        assert_eq!(
                            lane,
                            expect.as_slice(),
                            "field-maps {pair} restrict-mixed zero lane {i}"
                        );
                        assert_eq!(
                            mask[i / 8] & (1 << (i % 8)),
                            0,
                            "field-maps {pair} restrict-mixed cleared bit {i}"
                        );
                        assert!(
                            !map.contains(value),
                            "field-maps {pair} restrict-mixed member {i}"
                        );
                    }
                }
            }
            assert_eq!(
                fgf::bits::weight(&mask[..mask_len], count),
                mixed_restrict[..count]
                    .iter()
                    .filter(|y| map.contains(**y))
                    .count(),
                "field-maps {pair} restrict-mixed validity weight"
            );
            if mask_len * 8 > count {
                let tail = mask[mask_len - 1] >> (count % 8);
                assert_eq!(tail, 0, "field-maps {pair} restrict-mixed tail bits");
            }
        }
        bench_region(&label, bytes, || {
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let mask = black_box(&mut mask[..fgf::bits::bytes_for(count)]);
            run_mixed_row(map, dst, mask);
            black_box((dst, mask));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-nonmember-prepared/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &outsider_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut valid = vec![0u8; mask_bytes];
        check_prepared_restrict_row(
            pair,
            "restrict-nonmember-prepared",
            map,
            &prepared,
            src,
            &mut dst,
            &mut valid,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let valid = black_box(&mut valid[..fgf::bits::bytes_for(count)]);
            ops::restrict_into_with(dst, valid, black_box(&prepared), src);
            black_box((dst, valid));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/restrict-mixed-prepared/row-{count}");
        let bytes = count * elem_traffic + fgf::bits::bytes_for(count);
        let src = &mixed_restrict_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        let mut valid = vec![0u8; mask_bytes];
        check_prepared_restrict_row(
            pair,
            "restrict-mixed-prepared",
            map,
            &prepared,
            src,
            &mut dst,
            &mut valid,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            let valid = black_box(&mut valid[..fgf::bits::bytes_for(count)]);
            ops::restrict_into_with(dst, valid, black_box(&prepared), src);
            black_box((dst, valid));
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/contains/row-{count}");
        let bytes = count * T::BYTES + fgf::bits::bytes_for(count);
        let src = &mixed_bytes[..count * T::BYTES];
        let mut mask = vec![0u8; mask_bytes];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let mask = black_box(&mut mask[..fgf::bits::bytes_for(count)]);
            mask.fill(0);
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                if map.contains(T::decode(s)) {
                    mask[i / 8] |= 1 << (i % 8);
                }
            }
            black_box(mask);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/contains-prepared/row-{count}");
        let bytes = count * T::BYTES + fgf::bits::bytes_for(count);
        let src = &mixed_bytes[..count * T::BYTES];
        let mut valid = vec![0u8; mask_bytes];
        check_prepared_contains_row(
            pair,
            "contains-prepared",
            map,
            &prepared,
            src,
            &mut valid,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let valid = black_box(&mut valid[..fgf::bits::bytes_for(count)]);
            ops::contains_into_with(valid, black_box(&prepared), src);
            black_box(valid);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/frobenius/row-{count}");
        let bytes = 2 * count * T::BYTES;
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            for (s, d) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, map.frobenius(T::decode(s)));
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/frobenius-prepared/row-{count}");
        // Frobenius is T → T: one target element each way, unlike the
        // S → T traffic of the embedding panels beside it.
        let bytes = count * 2 * T::BYTES;
        // Same relative map as the compact row, prepared once outside the
        // timed region; the destination is reused across batches.
        let prepared = Frobenius::<T>::new(S::DEGREE);
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            ops::frobenius_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/trace/row-{count}");
        let bytes = count * elem_traffic;
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            for (s, d) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
            {
                S::encode(d, map.trace(T::decode(s)));
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/trace-prepared/row-{count}");
        let bytes = count * elem_traffic;
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        check_prepared_element_row(
            pair,
            "trace-prepared",
            |d, x| ops::trace_into_with(d, &prepared, x),
            |y| map.trace(y),
            src,
            &mut dst,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            ops::trace_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/norm/row-{count}");
        let bytes = count * elem_traffic;
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            for (s, d) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
            {
                S::encode(d, map.norm(T::decode(s)));
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/norm-prepared/row-{count}");
        let bytes = count * elem_traffic;
        let src = &mixed_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        check_prepared_element_row(
            pair,
            "norm-prepared",
            |d, x| ops::norm_into_with(d, &prepared, x),
            |y| map.norm(y),
            src,
            &mut dst,
            count,
        );
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            ops::norm_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
}

/// Map construction timed separately, plus construction with 16384
/// applications as one amortized row.
fn emb_construct_panels<
    S: FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
>(
    pair: &str,
    sources: &[Elem<S>],
) {
    bench_scalar(&format!("field-maps/{pair}/construct/once"), |ops| {
        for _ in 0..ops {
            black_box(Embedding::<S, T>::new().expect("a validated pair constructs"));
        }
        ops as u64
    });
    // Isolated preparation cost for the PR-4 amortization decision: the
    // compact map is built once outside timing, so the row times
    // `PreparedEmbedding::new` alone, not combined construction.
    let compact = Embedding::<S, T>::new().expect("a validated pair constructs");
    bench_scalar(
        &format!("field-maps/{pair}/construct-embedding/once"),
        |ops| {
            for _ in 0..ops {
                black_box(PreparedEmbedding::<S, T>::new(black_box(&compact)));
            }
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/frobenius-prepared/once"),
        |ops| {
            for _ in 0..ops {
                black_box(Frobenius::<T>::new(S::DEGREE));
            }
            ops as u64
        },
    );
    if smoke() {
        return;
    }
    let count = 16_384;
    let staged: Vec<Elem<S>> = sources.iter().copied().take(count).collect();
    let source_bytes = encode_staged::<S>(pair, "construct", &staged);
    let bytes = count * (S::BYTES + T::BYTES);
    let mut dst = vec![0u8; count * T::BYTES];
    bench_region(
        &format!("field-maps/{pair}/construct/row-16384"),
        bytes,
        || {
            let map = black_box(Embedding::<S, T>::new()).expect("a validated pair constructs");
            let src = black_box(source_bytes.as_slice());
            let dst = black_box(dst.as_mut_slice());
            for (s, d) in src
                .chunks_exact(S::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, map.embed(S::decode(s)));
            }
            black_box(dst);
        },
    );
}

/// Scalar-only construction panel for pairs with no byte form.
fn emb_construct_once<S: Field, T: Field<Characteristic = S::Characteristic>>(pair: &str) {
    bench_scalar(&format!("field-maps/{pair}/construct/once"), |ops| {
        for _ in 0..ops {
            black_box(Embedding::<S, T>::new().expect("a validated pair constructs"));
        }
        ops as u64
    });
    let compact = Embedding::<S, T>::new().expect("a validated pair constructs");
    bench_scalar(
        &format!("field-maps/{pair}/construct-embedding/once"),
        |ops| {
            for _ in 0..ops {
                black_box(PreparedEmbedding::<S, T>::new(black_box(&compact)));
            }
            ops as u64
        },
    );
}

/// Scalar-only Frobenius preparation panel for pairs with no byte form.
fn frob_construct_once<T: Field>(pair: &str, m: u32) {
    bench_scalar(
        &format!("field-maps/{pair}/frobenius-prepared/once"),
        |ops| {
            for _ in 0..ops {
                black_box(Frobenius::<T>::new(m));
            }
            ops as u64
        },
    );
}

// ---------------------------------------------------------------------------
// Isomorphism panels
// ---------------------------------------------------------------------------

/// Scalar independent-input throughput for one isomorphism pair: each timed
/// iteration applies one direction to one pre-staged input, and the reported
/// count is one operation per iteration.
fn iso_scalar_panels<S: Field, T: Field<Characteristic = S::Characteristic>>(
    pair: &str,
    map: &Isomorphism<S, T>,
) {
    // Nonzero staged inputs on each side, so neither direction times the
    // zero map.
    let staged_sources: Vec<Elem<S>> = field_samples(SCALAR_STAGE, 0x150);
    let staged_targets: Vec<Elem<T>> = field_samples(SCALAR_STAGE, 0xA99);
    let stage = SCALAR_STAGE - 1;
    bench_scalar(&format!("field-maps/{pair}/apply/scalar-chain"), |ops| {
        let mut acc = Elem::<T>::ZERO;
        for i in 0..ops {
            acc = black_box(map.apply(staged_sources[i & stage]));
        }
        black_box(acc);
        ops as u64
    });
    bench_scalar(&format!("field-maps/{pair}/unapply/scalar-chain"), |ops| {
        let mut acc = Elem::<S>::ZERO;
        for i in 0..ops {
            acc = black_box(map.unapply(staged_targets[i & stage]));
        }
        black_box(acc);
        ops as u64
    });
    let prepared = PreparedIsomorphism::<S, T>::new(map);
    bench_scalar(
        &format!("field-maps/{pair}/apply-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<T>::ZERO;
            for i in 0..ops {
                acc = black_box(prepared.apply(staged_sources[i & stage]));
            }
            black_box(acc);
            ops as u64
        },
    );
    bench_scalar(
        &format!("field-maps/{pair}/unapply-prepared/scalar-chain"),
        |ops| {
            let mut acc = Elem::<S>::ZERO;
            for i in 0..ops {
                acc = black_box(prepared.unapply(staged_targets[i & stage]));
            }
            black_box(acc);
            ops as u64
        },
    );
}

/// Region panels for one isomorphism pair: the same compact-map control
/// loop as the embedding panels, over `apply` and `unapply`.
fn iso_region_panels<S: FieldBuffer, T: Field<Characteristic = S::Characteristic> + FieldBuffer>(
    pair: &str,
    map: &Isomorphism<S, T>,
) {
    let top = region_counts().iter().max().copied().unwrap_or(1);
    let sources: Vec<Elem<S>> = field_samples(top, 0xA90);
    let images: Vec<Elem<T>> = sources.iter().map(|x| map.apply(*x)).collect();
    let source_bytes = encode_staged::<S>(pair, "apply", &sources);
    let image_bytes = encode_staged::<T>(pair, "unapply", &images);
    let traffic = S::BYTES + T::BYTES;
    let prepared = PreparedIsomorphism::<S, T>::new(map);

    for &count in region_counts() {
        let label = format!("field-maps/{pair}/apply/row-{count}");
        let bytes = count * traffic;
        let src = &source_bytes[..count * S::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            for (s, d) in src
                .chunks_exact(S::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, map.apply(S::decode(s)));
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/unapply/row-{count}");
        let bytes = count * traffic;
        let src = &image_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            for (s, d) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(S::BYTES))
            {
                S::encode(d, map.unapply(T::decode(s)));
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/apply-prepared/row-{count}");
        let bytes = count * traffic;
        let src = &source_bytes[..count * S::BYTES];
        let mut dst = vec![0u8; top * T::BYTES];
        ops::apply_into_with(&mut dst[..count * T::BYTES], &prepared, src);
        for (i, (x, lane)) in src
            .chunks_exact(S::BYTES)
            .zip(dst.chunks_exact(T::BYTES))
            .enumerate()
            .take(count)
        {
            let mut expect = vec![0u8; T::BYTES];
            T::encode(&mut expect, map.apply(S::decode(x)));
            assert_eq!(
                lane,
                expect.as_slice(),
                "field-maps {pair} apply-prepared lane {i}"
            );
        }
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            ops::apply_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("field-maps/{pair}/unapply-prepared/row-{count}");
        let bytes = count * traffic;
        let src = &image_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; top * S::BYTES];
        ops::unapply_into_with(&mut dst[..count * S::BYTES], &prepared, src);
        for (i, (y, lane)) in src
            .chunks_exact(T::BYTES)
            .zip(dst.chunks_exact(S::BYTES))
            .enumerate()
            .take(count)
        {
            let mut expect = vec![0u8; S::BYTES];
            S::encode(&mut expect, map.unapply(T::decode(y)));
            assert_eq!(
                lane,
                expect.as_slice(),
                "field-maps {pair} unapply-prepared lane {i}"
            );
        }
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * S::BYTES]);
            ops::unapply_into_with(dst, black_box(&prepared), src);
            black_box(dst);
        });
    }
}

/// Isomorphism construction timed separately, plus construction with 16384
/// applications as one amortized row.
fn iso_construct_panels<
    S: FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
>(
    pair: &str,
    sources: &[Elem<S>],
) {
    bench_scalar(&format!("field-maps/{pair}/construct/once"), |ops| {
        for _ in 0..ops {
            black_box(Isomorphism::<S, T>::new().expect("a validated pair constructs"));
        }
        ops as u64
    });
    let compact = Isomorphism::<S, T>::new().expect("a validated pair constructs");
    bench_scalar(
        &format!("field-maps/{pair}/construct-isomorphism/once"),
        |ops| {
            for _ in 0..ops {
                black_box(PreparedIsomorphism::<S, T>::new(black_box(&compact)));
            }
            ops as u64
        },
    );
    if smoke() {
        return;
    }
    let count = 16_384;
    let staged: Vec<Elem<S>> = sources.iter().copied().take(count).collect();
    let source_bytes = encode_staged::<S>(pair, "construct", &staged);
    let bytes = count * (S::BYTES + T::BYTES);
    let mut dst = vec![0u8; count * T::BYTES];
    bench_region(
        &format!("field-maps/{pair}/construct/row-16384"),
        bytes,
        || {
            let map = black_box(Isomorphism::<S, T>::new()).expect("a validated pair constructs");
            let src = black_box(source_bytes.as_slice());
            let dst = black_box(dst.as_mut_slice());
            for (s, d) in src
                .chunks_exact(S::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, map.apply(S::decode(s)));
            }
            black_box(dst);
        },
    );
}

// ---------------------------------------------------------------------------
// Pair runners
// ---------------------------------------------------------------------------

/// One embedding pair with a byte form: construct, validate, then run the
/// scalar, region, and construction panels.
fn run_emb_pair<S: FieldBuffer, T: Field<Characteristic = S::Characteristic> + FieldBuffer>(
    pair: &str,
) {
    let map = Embedding::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_embedding(pair, &map);
    println!(
        "field-maps panel: {pair} (embedding) endpoint bytes {} -> {}",
        S::BYTES,
        T::BYTES,
    );
    emb_scalar_panels(pair, &map, false);
    emb_region_panels(pair, &map);
    let top = region_counts().iter().max().copied().unwrap_or(1);
    let control_targets: Vec<Elem<T>> = field_samples(top, 0xF21);
    pow_control_panels::<S, T>(
        pair,
        &encode_staged::<T>(pair, "pow-control", &control_targets),
    );
    let sources: Vec<Elem<S>> = field_samples(16_384, 0xC0);
    emb_construct_panels::<S, T>(pair, &sources);
}

/// One embedding pair without a byte form: scalar chains and construction
/// only.
fn run_emb_pair_scalar<S: Field, T: Field<Characteristic = S::Characteristic>>(pair: &str) {
    let map = Embedding::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_embedding(pair, &map);
    println!("field-maps panel: {pair} (embedding)");
    emb_scalar_panels(pair, &map, true);
    emb_construct_once::<S, T>(pair);
    frob_construct_once::<T>(pair, S::DEGREE);
}

/// One GF(2)-sourced bit/lane interop pair: the compact scalar chains and
/// construction panels, then the `ops::map_bits` packed-bit rows beside
/// them. The bits rows read packed GF(2) bits LSB-first on one side and
/// stable target lanes on the other, per the `ops::map_bits`
/// representation contract. The panel heading declares the bit-width
/// source (`endpoint bits 1 -> B`); the scalar chains underneath time the
/// same compact methods as every other pair.
#[allow(clippy::too_many_lines)]
fn run_emb_pair_bits<
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
>(
    pair: &str,
) {
    let map = Embedding::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_embedding(pair, &map);
    println!(
        "field-maps panel: {pair} (embedding) endpoint bits 1 -> {}",
        T::BYTES,
    );
    emb_scalar_panels(pair, &map, true);
    emb_construct_once::<S, T>(pair);
    frob_construct_once::<T>(pair, S::DEGREE);
    emb_bits_panels(pair, &map);
}

/// Bits rows appended to a pair the scalar runner already emitted: the
/// same `emb_bits_panels` rows under a second bits-form heading, without
/// repeating the scalar chains. Used for `gf1-gf64`, whose scalar-only
/// pair stays intact.
fn run_emb_pair_bits_append<
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
>(
    pair: &str,
) {
    let map = Embedding::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_embedding(pair, &map);
    println!(
        "field-maps panel: {pair} (embedding) endpoint bits 1 -> {}",
        T::BYTES,
    );
    emb_bits_panels(pair, &map);
}

/// Packed-bit rows for one GF(2)-sourced pair: each `*-bits` arm times an
/// `ops::map_bits` row, and each `*-bits-control` arm times the matching
/// per-bit read plus prepared scalar method plus write loop. The row must
/// equal the control before timing (full packed outputs including tails)
/// and beat it. Construction (`Embedding::new`,
/// `PreparedEmbedding::new`), staging, and allocation stay outside the
/// timed regions.
#[allow(clippy::too_many_lines)]
fn emb_bits_panels<
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
>(
    pair: &str,
    map: &Embedding<S, T>,
) {
    let prepared = PreparedEmbedding::<S, T>::new(map);
    let top = region_counts().iter().max().copied().unwrap_or(1);
    let pack_len = fgf::bits::bytes_for(top);
    // Staged target lanes: the mixed member/outsider rotation the
    // byte-form panels use, so every op sees both outcomes at each count.
    let sources: Vec<Elem<S>> = field_samples(top, 0x860);
    let members: Vec<Elem<T>> = sources.iter().map(|x| map.embed(*x)).collect();
    let outsiders = synthesize_outsiders(pair, map, &members, top);
    let mixed: Vec<Elem<T>> = (0..top)
        .map(|i| {
            if (i + 0x5A) % 2 == 0 {
                members[i % members.len()]
            } else {
                outsiders[i % outsiders.len()]
            }
        })
        .collect();
    let lane_bytes = encode_staged::<T>(pair, "bits", &mixed);
    let member_bytes = encode_staged::<T>(pair, "bits-member", &members);
    // Staged packed source bits: member preimages when the lane is in the
    // image, zero otherwise (a valid zero keeps its membership; every
    // embed row validates against the per-bit oracle below).
    let mut pack_src = vec![0u8; pack_len];
    fgf::bits::clear_range(&mut pack_src, top, 0, top);
    for (i, y) in mixed.iter().enumerate() {
        if let Some(x) = map.restrict(*y)
            && x.is_one()
        {
            fgf::bits::set_range(&mut pack_src, top, i, i + 1);
        }
    }
    let packed_traffic = |count: usize| count * T::BYTES + fgf::bits::bytes_for(count);
    // Decode one packed bit the way the control loop reads it: the scalar
    // oracle each `*-bits` row must equal.
    let read_bit = |packed: &[u8], i: usize| {
        if packed[i / 8] & (1 << (i % 8)) == 0 {
            Elem::<S>::ZERO
        } else {
            Elem::<S>::ONE
        }
    };
    for &count in region_counts() {
        let mask_len = fgf::bits::bytes_for(count);
        // --- embed-bits: packed bits into target lanes. ---
        {
            let label = format!("field-maps/{pair}/embed-bits/row-{count}");
            let bytes = packed_traffic(count);
            let src = &pack_src[..mask_len];
            let mut dst = vec![0u8; top * T::BYTES];
            let mut control = vec![0u8; top * T::BYTES];
            ops::map_bits::embed_into_with(&mut dst[..count * T::BYTES], &prepared, src, count);
            for (i, d) in control.chunks_exact_mut(T::BYTES).enumerate().take(count) {
                T::encode(d, prepared.embed(read_bit(src, i)));
            }
            assert_eq!(
                &dst[..count * T::BYTES],
                &control[..count * T::BYTES],
                "field-maps {pair} embed-bits row differs from its control"
            );
            bench_region(&label, bytes, || {
                let src = black_box(src);
                let dst = black_box(&mut dst[..count * T::BYTES]);
                ops::map_bits::embed_into_with(dst, black_box(&prepared), src, count);
                black_box(dst);
            });
        }
        {
            let label = format!("field-maps/{pair}/embed-bits-control/row-{count}");
            let bytes = packed_traffic(count);
            let src = &pack_src[..mask_len];
            let mut dst = vec![0u8; top * T::BYTES];
            bench_region(&label, bytes, || {
                let src = black_box(src);
                let dst = black_box(&mut dst[..count * T::BYTES]);
                for (i, d) in dst.chunks_exact_mut(T::BYTES).enumerate() {
                    T::encode(d, black_box(&prepared).embed(read_bit(black_box(src), i)));
                }
                black_box(dst);
            });
        }
        // --- trace-bits / norm-bits: lanes into packed bits. ---
        for (op, is_trace) in [("trace-bits", true), ("norm-bits", false)] {
            let label = format!("field-maps/{pair}/{op}/row-{count}");
            let bytes = packed_traffic(count);
            let src = &lane_bytes[..count * T::BYTES];
            let mut dst = vec![0u8; pack_len];
            let mut control = vec![0u8; pack_len];
            if is_trace {
                ops::map_bits::trace_into_with(&mut dst[..mask_len], &prepared, src, count);
            } else {
                ops::map_bits::norm_into_with(&mut dst[..mask_len], &prepared, src, count);
            }
            fgf::bits::clear_range(&mut control[..mask_len], count, 0, count);
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                let bit = if is_trace {
                    prepared.trace(T::decode(s))
                } else {
                    prepared.norm(T::decode(s))
                };
                if bit.is_one() {
                    fgf::bits::set_range(&mut control[..mask_len], count, i, i + 1);
                }
            }
            assert_eq!(
                &dst[..mask_len],
                &control[..mask_len],
                "field-maps {pair} {op} row differs from its control"
            );
            bench_region(&label, bytes, || {
                let src = black_box(src);
                let dst = black_box(&mut dst[..mask_len]);
                if is_trace {
                    ops::map_bits::trace_into_with(
                        dst,
                        black_box(&prepared),
                        black_box(src),
                        count,
                    );
                } else {
                    ops::map_bits::norm_into_with(dst, black_box(&prepared), black_box(src), count);
                }
                black_box(dst);
            });
            let control_label = format!("field-maps/{pair}/{op}-control/row-{count}");
            let mut control_dst = vec![0u8; pack_len];
            bench_region(&control_label, bytes, || {
                let src = black_box(src);
                let dst = black_box(&mut control_dst[..mask_len]);
                fgf::bits::clear_range(dst, count, 0, count);
                for (i, s) in black_box(src).chunks_exact(T::BYTES).enumerate() {
                    let bit = if is_trace {
                        black_box(&prepared).trace(T::decode(s))
                    } else {
                        black_box(&prepared).norm(T::decode(s))
                    };
                    if bit.is_one() {
                        fgf::bits::set_range(dst, count, i, i + 1);
                    }
                }
                black_box(dst);
            });
        }
        // --- restrict-bits-member / restrict-bits-mixed / contains-bits. ---
        for (op, staged) in [
            ("restrict-bits-member", member_bytes.as_slice()),
            ("restrict-bits-mixed", lane_bytes.as_slice()),
            ("contains-bits", lane_bytes.as_slice()),
        ] {
            let label = format!("field-maps/{pair}/{op}/row-{count}");
            let src = &staged[..count * T::BYTES];
            if op == "contains-bits" {
                let bytes = packed_traffic(count);
                let mut dst = vec![0u8; pack_len];
                let mut control = vec![0u8; pack_len];
                ops::map_bits::contains_into_with(&mut dst[..mask_len], &prepared, src, count);
                fgf::bits::clear_range(&mut control[..mask_len], count, 0, count);
                for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                    if prepared.contains(T::decode(s)) {
                        fgf::bits::set_range(&mut control[..mask_len], count, i, i + 1);
                    }
                }
                assert_eq!(
                    &dst[..mask_len],
                    &control[..mask_len],
                    "field-maps {pair} {op} row differs from its control"
                );
                bench_region(&label, bytes, || {
                    let src = black_box(src);
                    let dst = black_box(&mut dst[..mask_len]);
                    ops::map_bits::contains_into_with(dst, black_box(&prepared), src, count);
                    black_box(dst);
                });
                let control_label = format!("field-maps/{pair}/{op}-control/row-{count}");
                let mut control_dst = vec![0u8; pack_len];
                bench_region(&control_label, bytes, || {
                    let src = black_box(src);
                    let dst = black_box(&mut control_dst[..mask_len]);
                    fgf::bits::clear_range(dst, count, 0, count);
                    for (i, s) in black_box(src).chunks_exact(T::BYTES).enumerate() {
                        if black_box(&prepared).contains(T::decode(s)) {
                            fgf::bits::set_range(dst, count, i, i + 1);
                        }
                    }
                    black_box(dst);
                });
            } else {
                let bytes = packed_traffic(count) + fgf::bits::bytes_for(count);
                let mut dst = vec![0u8; pack_len];
                let mut valid = vec![0u8; pack_len];
                let mut control_dst = vec![0u8; pack_len];
                let mut control_valid = vec![0u8; pack_len];
                ops::map_bits::restrict_into_with(
                    &mut dst[..mask_len],
                    &mut valid[..mask_len],
                    &prepared,
                    src,
                    count,
                );
                fgf::bits::clear_range(&mut control_dst[..mask_len], count, 0, count);
                fgf::bits::clear_range(&mut control_valid[..mask_len], count, 0, count);
                for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                    if let Some(x) = prepared.restrict(T::decode(s)) {
                        if x.is_one() {
                            fgf::bits::set_range(&mut control_dst[..mask_len], count, i, i + 1);
                        }
                        fgf::bits::set_range(&mut control_valid[..mask_len], count, i, i + 1);
                    }
                }
                assert_eq!(
                    &dst[..mask_len],
                    &control_dst[..mask_len],
                    "field-maps {pair} {op} row differs from its control"
                );
                assert_eq!(
                    &valid[..mask_len],
                    &control_valid[..mask_len],
                    "field-maps {pair} {op} validity differs from its control"
                );
                bench_region(&label, bytes, || {
                    let src = black_box(src);
                    let dst = black_box(&mut dst[..mask_len]);
                    let valid = black_box(&mut valid[..mask_len]);
                    ops::map_bits::restrict_into_with(dst, valid, black_box(&prepared), src, count);
                    black_box((dst, valid));
                });
                let control_label = format!("field-maps/{pair}/{op}-control/row-{count}");
                let mut control_timed_dst = vec![0u8; pack_len];
                let mut control_timed_valid = vec![0u8; pack_len];
                bench_region(&control_label, bytes, || {
                    let src = black_box(src);
                    let dst = black_box(&mut control_timed_dst[..mask_len]);
                    let valid = black_box(&mut control_timed_valid[..mask_len]);
                    fgf::bits::clear_range(dst, count, 0, count);
                    fgf::bits::clear_range(valid, count, 0, count);
                    for (i, s) in black_box(src).chunks_exact(T::BYTES).enumerate() {
                        if let Some(x) = black_box(&prepared).restrict(T::decode(s)) {
                            if x.is_one() {
                                fgf::bits::set_range(dst, count, i, i + 1);
                            }
                            fgf::bits::set_range(valid, count, i, i + 1);
                        }
                    }
                    black_box((dst, valid));
                });
            }
        }
    }
}

/// Validation-only arm over the normal-basis target: the `gf1-normal8`
/// pair runs the same bits kernel through the oracle checks and times
/// nothing. It prints no `field-maps panel` line: `gf1-normal8` is not a
/// measured pair in the publisher grammar, so a bare panel line would
/// fail closed on full runs.
fn run_emb_pair_bits_validate_only<
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
>(
    pair: &str,
) {
    let map = Embedding::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_embedding(pair, &map);
    let prepared = PreparedEmbedding::<S, T>::new(&map);
    for &count in region_counts() {
        let pack_len = fgf::bits::bytes_for(count);
        let targets: Vec<Elem<T>> = field_samples(count, 0xB17);
        let lanes = encode_staged::<T>(pair, "bits-validate", &targets);
        let mut packed = vec![0u8; pack_len.max(1)];
        let mut control = vec![0u8; pack_len.max(1)];
        let slice_len = pack_len;
        ops::map_bits::trace_into_with(&mut packed[..slice_len], &prepared, &lanes, count);
        fgf::bits::clear_range(&mut control[..slice_len], count, 0, count);
        for (i, s) in lanes.chunks_exact(T::BYTES).enumerate() {
            if prepared.trace(T::decode(s)).is_one() {
                fgf::bits::set_range(&mut control[..slice_len], count, i, i + 1);
            }
        }
        assert_eq!(
            &packed[..slice_len],
            &control[..slice_len],
            "field-maps {pair} validation-only trace row differs from its oracle"
        );
    }
}

/// One isomorphism pair: construct, validate, then run the apply/unapply
/// scalar, region, and construction panels.
fn run_iso_pair<S: FieldBuffer, T: Field<Characteristic = S::Characteristic> + FieldBuffer>(
    pair: &str,
) {
    let map = Isomorphism::<S, T>::new()
        .unwrap_or_else(|error| panic!("field-maps {pair}: construction failed: {error:?}"));
    validate_isomorphism(pair, &map);
    println!(
        "field-maps panel: {pair} (isomorphism) endpoint bytes {} -> {}",
        S::BYTES,
        T::BYTES,
    );
    iso_scalar_panels(pair, &map);
    iso_region_panels(pair, &map);
    let sources: Vec<Elem<S>> = field_samples(16_384, 0xC0);
    iso_construct_panels::<S, T>(pair, &sources);
}

/// D7 arithmetic control: the direct `x.pow(S::ORDER)` power the relative
/// Frobenius names, timed as scalar chains and region rows beside the map's
/// own `frobenius` panels.
fn pow_control_panels<S: Field, T: Field<Characteristic = S::Characteristic> + FieldBuffer>(
    pair: &str,
    target_bytes: &[u8],
) {
    let stage_targets: Vec<Elem<T>> = field_samples(SCALAR_STAGE, 0xF20);
    let stage = SCALAR_STAGE - 1;
    bench_scalar(&format!("control pow {pair}/scalar-chain"), |ops| {
        let mut value = stage_targets[1];
        for i in 0..ops {
            value = black_box(value.pow(S::ORDER));
            value = value.add(stage_targets[i & stage]);
        }
        black_box(value);
        ops as u64
    });
    for &count in region_counts() {
        let label = format!("control pow {pair}/row-{count}");
        let bytes = 2 * count * T::BYTES;
        let src = &target_bytes[..count * T::BYTES];
        let mut dst = vec![0u8; target_bytes.len()];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * T::BYTES]);
            for (s, d) in src
                .chunks_exact(T::BYTES)
                .zip(dst.chunks_exact_mut(T::BYTES))
            {
                T::encode(d, T::decode(s).pow(S::ORDER));
            }
            black_box(dst);
        });
    }
}

/// D7 QM31 controls: the inherent `conjugate` and `norm` beside the map's
/// own panels.
fn qm31_control_panels(target_bytes: &[u8]) {
    let stage_targets: Vec<Elem<QuadMersenne31>> = field_samples(SCALAR_STAGE, 0xF23);
    let stage = SCALAR_STAGE - 1;
    bench_scalar("control qm31 conjugate/scalar-chain", |ops| {
        let mut value = stage_targets[1];
        for i in 0..ops {
            value = black_box(value.conjugate());
            value = value.add(stage_targets[i & stage]);
        }
        black_box(value);
        ops as u64
    });
    bench_scalar("control qm31 norm/scalar-chain", |ops| {
        let mut acc = Elem::<Mersenne31>::ZERO;
        for i in 0..ops {
            acc = black_box(stage_targets[i & stage].norm());
        }
        black_box(acc);
        ops as u64
    });
    for &count in region_counts() {
        let label = format!("control qm31 conjugate/row-{count}");
        let bytes = 2 * count * QuadMersenne31::BYTES;
        let src = &target_bytes[..count * QuadMersenne31::BYTES];
        let mut dst = vec![0u8; target_bytes.len()];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * QuadMersenne31::BYTES]);
            for (s, d) in src
                .chunks_exact(QuadMersenne31::BYTES)
                .zip(dst.chunks_exact_mut(QuadMersenne31::BYTES))
            {
                QuadMersenne31::encode(d, QuadMersenne31::decode(s).conjugate());
            }
            black_box(dst);
        });
    }
    for &count in region_counts() {
        let label = format!("control qm31 norm/row-{count}");
        let bytes = count * (QuadMersenne31::BYTES + Mersenne31::BYTES);
        let src = &target_bytes[..count * QuadMersenne31::BYTES];
        let mut dst = vec![0u8; target_bytes.len() / QuadMersenne31::BYTES * Mersenne31::BYTES];
        bench_region(&label, bytes, || {
            let src = black_box(src);
            let dst = black_box(&mut dst[..count * Mersenne31::BYTES]);
            for (s, d) in src
                .chunks_exact(QuadMersenne31::BYTES)
                .zip(dst.chunks_exact_mut(Mersenne31::BYTES))
            {
                Mersenne31::encode(d, QuadMersenne31::decode(s).norm());
            }
            black_box(dst);
        });
    }
}

fn main() {
    println!("fgf field-maps benchmark — backend: {}", backend().name());
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

/// Every panel, behind `main`'s drift controls. `--smoke` runs one panel
/// per class with tiny iteration counts and returns early; every other flag
/// passes through harmlessly, and the end control always runs.
fn run_panels() {
    if smoke() {
        run_emb_pair::<Gf16, Gf64>("gf16-gf64");
        run_emb_pair_bits::<Gf1, Gf16>("gf1-gf16");
        run_iso_pair::<Fp64, Gf64>("fp64-gf64-iso");
        return;
    }
    println!("  (override with SIMD_BACKEND=v3_gfni_crypto|v3|v2|neon|scalar)\n");

    run_emb_pair::<Gf32, Gf64>("gf32-gf64");
    run_emb_pair::<Gf16, Gf64>("gf16-gf64");
    run_emb_pair::<Aes8, Gf64>("aes8-gf64");
    run_emb_pair::<Fp8, Fp16>("fp8-fp16");
    run_emb_pair::<Rs8, Gf16>("rs8-gf16");
    run_emb_pair::<Normal8, Gf16>("normal8-gf16");
    run_emb_pair::<Res2, Gf64>("res2-gf64");
    run_emb_pair::<Res2, OverRes2>("res2-res2q");
    run_emb_pair::<Aes8, OverAes>("aes8-aesq");
    run_emb_pair::<Mersenne31, QuadMersenne31>("m31-qm31");
    let qm_top = region_counts().iter().max().copied().unwrap_or(1);
    let qm_targets: Vec<Elem<QuadMersenne31>> = field_samples(qm_top, 0x31);
    qm31_control_panels(&encode_staged::<QuadMersenne31>(
        "m31-qm31",
        "qm31-control",
        &qm_targets,
    ));
    run_emb_pair::<Goldilocks, GldSquare>("gdl2");
    run_emb_pair::<G9Alt, G81Ext>("g9-g81");
    run_emb_pair::<R3, G81Quartic>("r3-g81q");
    run_emb_pair::<R7, G343>("r7-g343");
    run_emb_pair::<R3, G340>("r3-g3-40");
    run_emb_pair_scalar::<Ref2, Ref8>("ref2-ref8");
    run_emb_pair_scalar::<Gf1, Gf64>("gf1-gf64");
    run_emb_pair_bits::<Gf1, Aes8>("gf1-aes8");
    run_emb_pair_bits::<Gf1, Gf16>("gf1-gf16");
    run_emb_pair_bits::<Gf1, Gf32>("gf1-gf32");
    run_emb_pair_bits_append::<Gf1, Gf64>("gf1-gf64");
    run_emb_pair_bits_validate_only::<Gf1, Normal8>("gf1-normal8");
    run_iso_pair::<Fp64, Gf64>("fp64-gf64-iso");
    run_iso_pair::<Normal8, Aes8>("normal8-aes8-iso");
}
