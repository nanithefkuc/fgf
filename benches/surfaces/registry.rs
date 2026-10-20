//! Case registry, machine records, and the measurement engine for the
//! public-API surface benchmark.
//!
//! Every benchmarked callable is one registered [`Case`]. Registration is
//! deterministic, so the `SURFACE_INVENTORY` stream is identical across
//! `--inventory`, `--check`, `--smoke`, and a full run; only `SURFACE`
//! timing records are added when cases actually run. The measurement engine
//! follows the warm-calibrate-median convention of `benches/kernels.rs`,
//! with a calibration cap so expensive one-shot calls (map restriction,
//! embedding build) stay bounded instead of paying a full sample campaign.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Where a case's timed work sits in the caller's workflow.
///
/// `setup` times public preparation consumed by steady cases; `steady` times
/// repeated hot-loop work over preallocated fixtures; `query` times trivial
/// capability or accessor reads; `end_to_end` times one-shot producing calls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    Setup,
    Steady,
    Query,
    EndToEnd,
}

impl Scope {
    fn label(self) -> &'static str {
        match self {
            Scope::Setup => "setup",
            Scope::Steady => "steady",
            Scope::Query => "query",
            Scope::EndToEnd => "end_to_end",
        }
    }
}

/// One registered measurement: the metadata the publisher sees and the entry
/// point that runs it.
#[derive(Clone)]
pub struct Case {
    /// Stable unique case string across rounds, hosts, and inventories.
    pub case: String,
    /// Qualified public symbols this measurement exercises.
    pub api: &'static [&'static str],
    /// Exact registered presentation label of the measured field.
    pub field: String,
    /// Operation name, matching the human-readable convention of the panel.
    pub operation: String,
    /// Scope of the timed work.
    pub scope: Scope,
    /// Logical throughput numerator in bytes; `0` for scalar/query work.
    pub bytes: usize,
    /// Logical elements in flight for one call.
    pub lanes: usize,
    /// Input buffers or sources for one call.
    pub sources: usize,
    /// Output buffers or destination rows for one call.
    pub destinations: usize,
    /// Drift-control marker; control cases also carry a `phase`.
    pub control: bool,
    /// `entry`/`exit` for the two control measurements, `None` elsewhere.
    pub phase: Option<&'static str>,
    /// The case body: builds fixtures, verifies once, then times.
    pub run: fn(&Case, &CaseCtx),
}

/// Per-run flags handed to every case body.
#[derive(Clone, Copy)]
pub struct CaseCtx {
    /// `--check`: run each fixture correctness check once, print the
    /// inventory, and time nothing.
    pub check: bool,
    /// `--smoke`: keep every case, shorten repetitions.
    pub smoke: bool,
}

impl CaseCtx {
    /// Warm-loop repetitions inside one sample batch for a body whose
    /// calibration pass took `est_ns`.
    fn batch_reps(&self, est_ns: f64) -> u64 {
        let target = if self.smoke { 15_000.0 } else { 100_000.0 };
        let min_reps = if self.smoke { 1 } else { 8 };
        let reps = (target / est_ns.max(1.0)).ceil() as u64;
        // Expensive one-shot calls keep a single call per batch so the case
        // stays bounded; cheap workloads never drop below the minimum batch.
        if est_ns > 5_000_000.0 {
            1
        } else {
            reps.clamp(min_reps, 1 << 20)
        }
    }

    fn sample_deadline(&self) -> Duration {
        Duration::from_millis(if self.smoke { 60 } else { 250 })
    }

    fn max_samples(&self) -> usize {
        if self.smoke { 5 } else { 24 }
    }

    /// Time `body` with the capped warm-calibrate-median engine and return
    /// `(median_ns, samples, p10_ns, p90_ns)`.
    ///
    /// The helper is generic over the body, so the timed loop is
    /// monomorphized per case: the fn-pointer dispatch of the registry sits
    /// outside the measured region, and no dynamic dispatch is added to
    /// trivial query timings. Percentiles are nearest-rank over the sorted
    /// samples and present only when enough samples exist.
    pub fn measure(&self, mut body: impl FnMut()) -> (f64, usize, Option<f64>, Option<f64>) {
        // Warm caches, branch predictors, and the lazily resolved backend;
        // expensive bodies stop after one warm call.
        let start = Instant::now();
        body();
        if start.elapsed() < Duration::from_millis(20) {
            for _ in 0..3 {
                body();
            }
        }
        let est_ns = {
            let start = Instant::now();
            body();
            let elapsed = start.elapsed().as_secs_f64() * 1e9;
            black_box(elapsed)
        };
        let mut reps = self.batch_reps(est_ns);
        let target = Duration::from_micros(if self.smoke { 15 } else { 100 });
        // Calibrate a whole batch: a single-call clock reading overestimates
        // sub-nanosecond getters and otherwise leaves their samples too short.
        loop {
            let start = Instant::now();
            for _ in 0..reps {
                body();
            }
            let elapsed = start.elapsed();
            if elapsed >= target || reps == 1 << 20 {
                break;
            }
            let scale = (target.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)).ceil() as u64;
            reps = reps.saturating_mul(scale.max(2)).min(1 << 20);
        }
        let deadline = self.sample_deadline();
        let max_samples = self.max_samples();
        let started = Instant::now();
        let mut samples: Vec<f64> = Vec::with_capacity(max_samples);
        while samples.len() < max_samples {
            let start = Instant::now();
            for _ in 0..reps {
                body();
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / reps as f64);
            if Instant::now() - started > deadline {
                break;
            }
        }
        samples.sort_unstable_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p10 = if samples.len() >= 10 {
            Some(samples[(samples.len() as f64 * 0.1).ceil() as usize - 1])
        } else {
            None
        };
        let p90 = if samples.len() >= 10 {
            Some(samples[(samples.len() as f64 * 0.9).ceil() as usize - 1])
        } else {
            None
        };
        (median, samples.len(), p10, p90)
    }
}

struct RegistryState {
    cases: Vec<Case>,
}

thread_local! {
    static REGISTRY: RefCell<RegistryState> = const { RefCell::new(RegistryState {
        cases: Vec::new(),
    }) };
}

/// Register one case; panics on a duplicate case string so the inventory can
/// never contain ambiguous records.
pub fn add(case: Case) {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        assert!(
            registry
                .cases
                .iter()
                .all(|existing| existing.case != case.case),
            "duplicate surface case {}",
            case.case
        );
        registry.cases.push(case);
    });
}

/// The registered cases in deterministic registration order.
pub fn cases() -> Vec<Case> {
    REGISTRY.with(|registry| registry.borrow().cases.clone())
}

/// Escape a string for embedding in a JSON record.
fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// JSON string literal for one field of a record.
fn json_str(value: &str) -> String {
    format!("\"{}\"", json_escape(value))
}

/// JSON array literal for the `api` symbol list.
fn json_api(api: &[&str]) -> String {
    let items: Vec<String> = api.iter().map(|symbol| json_str(symbol)).collect();
    format!("[{}]", items.join(","))
}

/// The shared metadata prefix of every inventory and timing record.
///
/// Field order is frozen: the publisher matches inventories across rounds
/// byte-for-byte, so this function is the single spelling of the prefix.
fn record_prefix(case: &Case) -> String {
    let mut out = String::new();
    let _ = write!(out, "\"case\":{},", json_str(&case.case));
    let _ = write!(out, "\"api\":{},", json_api(case.api));
    let _ = write!(out, "\"field\":{},", json_str(&case.field));
    let _ = write!(out, "\"operation\":{},", json_str(&case.operation));
    let _ = write!(out, "\"scope\":{},", json_str(case.scope.label()));
    let _ = write!(out, "\"bytes\":{},", case.bytes);
    let _ = write!(out, "\"lanes\":{},", case.lanes);
    let _ = write!(out, "\"sources\":{},", case.sources);
    let _ = write!(out, "\"destinations\":{},", case.destinations);
    let _ = write!(out, "\"control\":{},", case.control);
    match case.phase {
        Some(phase) => {
            let _ = write!(out, "\"phase\":{},", json_str(phase));
        }
        None => {
            let _ = out.write_str("\"phase\":null,");
        }
    }
    out
}

/// Emit one `SURFACE_INVENTORY` record: every registered measurement, no
/// timing. Identical across modes and rounds.
pub fn print_inventory(case: &Case) {
    let line = format!(
        "SURFACE_INVENTORY\t{{{}}}",
        record_prefix(case).trim_end_matches(',')
    );
    println!("{line}");
}

/// Emit one `SURFACE` timing record beside a human-readable panel line.
///
/// `ns` is the finite positive per-call median in nanoseconds, printed with
/// nine decimal places; `backend` is the field's actual backend name or
/// `scalar` for map work.
pub fn print_record(
    case: &Case,
    backend: &str,
    median: f64,
    samples: usize,
    p10: Option<f64>,
    p90: Option<f64>,
) {
    let mut out = record_prefix(case);
    let _ = write!(out, "\"ns\":{median:.9},");
    let _ = write!(out, "\"backend\":{},", json_str(backend));
    let _ = write!(out, "\"samples\":{samples}");
    if let Some(p10) = p10 {
        let _ = write!(out, ",\"p10_ns\":{p10:.9}");
    }
    if let Some(p90) = p90 {
        let _ = write!(out, ",\"p90_ns\":{p90:.9}");
    }
    println!("SURFACE\t{{{out}}}");
}

/// Human panel line printed beside each timing record.
pub fn print_panel(case: &Case, median: f64) {
    let ns = if median < 1_000.0 {
        format!("{median:.1}ns")
    } else if median < 1_000_000.0 {
        format!("{:.2}µs", median / 1_000.0)
    } else {
        format!("{:.2}ms", median / 1_000_000.0)
    };
    let scope = case.scope.label();
    println!("  {:<52} {:>10}  {}", case.case, ns, scope);
}

/// Measure one timed body and emit its panel line and `SURFACE` record,
/// reporting the backend the case registered.
///
/// In `--check` mode the caller has already exercised the same call path
/// with assertions, so nothing is measured and nothing is emitted.
pub fn time(case: &Case, ctx: &CaseCtx, backend: &str, mut body: impl FnMut()) {
    if ctx.check {
        body();
        return;
    }
    let (median, samples, p10, p90) = ctx.measure(body);
    print_panel(case, median);
    print_record(case, backend, median, samples, p10, p90);
}
