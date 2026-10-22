//! Public-route acceptance for the generated SIMD ledger cases.
//!
//! Every case names a public operation, a fixture type, a geometry, and the
//! ledger cells for the feature configuration and backend request. This
//! module runs the case through the public [`crate::ops`], [`crate::bits`],
//! and [`crate::ops::map`] surface, judges the output against an
//! independent per-element reference, and judges the test-only trace
//! against the budget of each stage the cell requires.
//!
//! The case table in `cases` is generated from `benches/simd_ledger.json`;
//! the adapters beside this file implement one runner per callable. The
//! runner prints one `ROUTE_RECEIPT` line per case and one `ROUTE_DONE`
//! line after the last case, so `benches/simd_ledger.py accept` can audit a
//! log without trusting the runner's exit status.
//!
//! # Modes
//!
//! A raw `SIMD_BACKEND` selects the request cell and enforces routing.
//! Without one, or with `FGF_ROUTE_SURVEY=1`, the runner surveys: every
//! case still checks its output, but a routing mismatch never fails the
//! run. `FGF_ROUTE_ACCEPT=1` additionally requires the run identity
//! environment (`FGF_SOURCE_ID`, `FGF_HOST_ID`, `FGF_FEATURES`) and refuses
//! a feature set that is neither the default nor the all-features set.

extern crate std;

use std::fmt::Write as _;
use std::string::{String, ToString};
use std::vec::Vec;
use std::{format, panic, vec};

use crate::field::{Elem, FieldBuffer, FieldElem};
use crate::kernel::trace::{self, Entry};
use crate::kernel::{Backend, FieldKernels, backend, backend_for};

pub(super) use crate::kernel::trace::{Pass, Scope, Substage, Unit};

mod bits;
mod cases;
mod elementwise;
mod fans;
mod fixtures;
mod map_bits;
mod maps;
mod packing;
mod reductions;
mod rows;
mod scalars;

// ---------------------------------------------------------------------------
// Contract types (generated tables depend on these names)
// ---------------------------------------------------------------------------

/// One generated case: the public operation, its fixture, and its cells.
pub(super) struct Case {
    pub id: &'static str,
    pub obligation: &'static str,
    pub run: fn(&Case, &Cell) -> Outcome,
    pub geometry: Geometry,
    pub coeffs: &'static [Coeff],
    pub map: Option<MapSpec>,
    pub cells: &'static [Cell],
}

/// Construction data for a prepared map.
///
/// Each image is the target element's `FieldBuffer` little-endian byte
/// encoding; an empty list selects structural construction.
pub(super) struct MapSpec {
    pub power: u32,
    pub images: &'static [&'static [u8]],
}

/// Buffer geometry shared by every case of one shape.
///
/// `len` counts source elements (one bit per element for bit-packed GF(2));
/// `misalign` shifts both buffer starts past the guard; `pitch_pad` adds
/// padding elements after each row of an offset-addressed destination.
pub(super) struct Geometry {
    pub rows: usize,
    pub sources: usize,
    pub len: usize,
    pub misalign: usize,
    pub pitch_pad: usize,
}

/// One coefficient slot, row-major over `(row, source)` or over the single
/// list a fan shape uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Coeff {
    General,
    Zero,
    One,
}

/// Which feature configuration a cell describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Features {
    Default,
    All,
}

impl Features {
    fn name(self) -> &'static str {
        match self {
            Features::Default => "default",
            Features::All => "all",
        }
    }
}

/// The raw `SIMD_BACKEND` request a cell describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Request {
    V4x,
    V3GfniCrypto,
    V3,
    V2,
    Scalar,
}

impl Request {
    fn name(self) -> &'static str {
        match self {
            Request::V4x => "v4x",
            Request::V3GfniCrypto => "v3_gfni_crypto",
            Request::V3 => "v3",
            Request::V2 => "v2",
            Request::Scalar => "scalar",
        }
    }

    fn from_backend(backend: Backend) -> Request {
        match backend {
            Backend::V4x => Request::V4x,
            Backend::V3GfniCrypto => Request::V3GfniCrypto,
            Backend::V3 => Request::V3,
            _ => Request::Scalar,
        }
    }

    fn parse(raw: &str) -> Option<Request> {
        match raw {
            "v4x" => Some(Request::V4x),
            "v3_gfni_crypto" => Some(Request::V3GfniCrypto),
            "v3" => Some(Request::V3),
            "v2" => Some(Request::V2),
            "scalar" => Some(Request::Scalar),
            _ => None,
        }
    }
}

/// The ledger's expectation for one case in one configuration.
pub(super) struct Cell {
    pub features: Features,
    pub request: Request,
    pub process: Backend,
    pub field: Backend,
    pub required: bool,
    pub min_serving: Backend,
    pub work: &'static [Work],
}

/// One named stage the cell requires vector work to cover.
pub(super) struct Work {
    pub substage: Substage,
    pub scope: Option<(Scope, Pass)>,
    pub unit: Unit,
    pub basis: Basis,
    pub multiplicity: Multiplicity,
    pub lane: usize,
    pub loss: usize,
    pub iterations: usize,
}

/// Which side of a width-changing operation a budget measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Basis {
    Source,
    Target,
}

/// How many times one stage runs for a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Multiplicity {
    One,
    Rows,
    ActiveRows,
    ActivePairs,
    UnitAddPairs,
}

// ---------------------------------------------------------------------------
// Observed outcome
// ---------------------------------------------------------------------------

/// One whole-buffer comparison: guards, padding, and unselected rows
/// included, so an operation that writes outside its named region fails.
pub(super) struct BufferCheck {
    pub name: &'static str,
    pub actual: Vec<u8>,
    pub expected: Vec<u8>,
}

/// What one public call produced.
pub(super) struct Outcome {
    pub checks: Vec<BufferCheck>,
    /// Scalar results compared by value, e.g. a Hamming weight.
    pub values: Vec<(&'static str, u128, u128)>,
    pub trace: Vec<Entry>,
    pub process: Backend,
    pub field: Backend,
    pub source_bytes: usize,
    pub target_bytes: usize,
    /// Whether routing mismatches fail this case.
    pub enforce_routing: bool,
    /// Construction could not run; the receipt reports `skip`.
    pub skip: Option<String>,
}

impl Outcome {
    pub(super) fn new(process: Backend, field: Backend) -> Self {
        Self {
            checks: Vec::new(),
            values: Vec::new(),
            trace: Vec::new(),
            process,
            field,
            source_bytes: 1,
            target_bytes: 1,
            enforce_routing: false,
            skip: None,
        }
    }

    /// Snapshot the trace taken immediately after the public call.
    pub(super) fn with_trace(mut self, trace: Vec<Entry>) -> Self {
        self.trace = trace;
        self
    }

    pub(super) fn skipped(reason: String) -> Self {
        let mut outcome = Self::new(Backend::Scalar, Backend::Scalar);
        outcome.skip = Some(reason);
        outcome
    }
}

/// Observe one public call: clear the trace, run it, snapshot the trace.
pub(super) fn observe<T>(call: impl FnOnce() -> T) -> (T, Vec<Entry>) {
    trace::clear();
    let value = call();
    (value, trace::snapshot())
}

/// Retain the generated fixture type in a runner signature.
pub(super) fn fixture_marker<F>() -> core::marker::PhantomData<F> {
    core::marker::PhantomData
}

/// Observe one public call whose result is not compared by value.
pub(super) fn observe_unit(call: impl FnOnce()) -> Vec<Entry> {
    trace::clear();
    call();
    trace::snapshot()
}

/// Standard outcome for a field-generic case over one element width.
pub(super) fn outcome<F: FieldKernels>(checks: Vec<BufferCheck>, trace: Vec<Entry>) -> Outcome {
    let mut outcome = Outcome::new(backend(), backend_for::<F>()).with_trace(trace);
    outcome.source_bytes = F::BYTES;
    outcome.target_bytes = F::BYTES;
    outcome.checks = checks;
    outcome
}

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

/// Guard byte around every buffer, so an underrun or overrun is visible.
pub(super) const GUARD: u8 = 0xA5;
/// Guard width in bytes.
pub(super) const GUARD_BYTES: usize = 64;

fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Case-salted seed: distinct per case and per call site.
pub(super) fn seed(case: &Case, salt: u64) -> u64 {
    fnv1a(case.id) ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// `count` canonical elements from the crate's fixed-seed noise helper.
///
/// `nonzero` replaces zero lanes with the field's one, for destinations the
/// contract describes as pre-filled nonzero.
pub(super) fn elements<F: FieldBuffer>(
    case: &Case,
    salt: u64,
    count: usize,
    nonzero: bool,
) -> Vec<u8> {
    let raw = super::noise(count * F::BYTES, seed(case, salt) | 1);
    let mut out = vec![0u8; count * F::BYTES];
    for (dst, src) in out
        .chunks_exact_mut(F::BYTES)
        .zip(raw.chunks_exact(F::BYTES))
    {
        let mut value = F::decode(src);
        if nonzero && value.is_zero() {
            value = Elem::<F>::ONE;
        }
        F::encode(dst, value);
    }
    out
}

/// A field element selected by one coefficient slot.
pub(super) fn coefficient<F: FieldKernels>(case: &Case, slot: usize) -> Elem<F> {
    let choice = case.coeffs.get(slot).copied().unwrap_or(Coeff::Zero);
    match choice {
        Coeff::Zero => Elem::<F>::ZERO,
        Coeff::One => Elem::<F>::ONE,
        Coeff::General => general::<F>(),
    }
}

/// A nondegenerate element: neither zero nor one.
fn general<F: FieldKernels>() -> Elem<F> {
    let order = <F as crate::field::Field>::ORDER;
    assert!(
        order > 2,
        "GF(2) admits no element distinct from zero and one"
    );
    let index = if order > 3 { 3u128 } else { 2u128 };
    let value = Elem::<F>::from_index(index).expect("index below the field order");
    assert!(
        !value.is_zero() && !value.is_one(),
        "a General coefficient is neither zero nor one"
    );
    value
}

/// One coefficient per slot.
pub(super) fn coefficients<F: FieldKernels>(case: &Case) -> Vec<Elem<F>> {
    (0..case.coeffs.len())
        .map(|slot| coefficient::<F>(case, slot))
        .collect()
}

// ---------------------------------------------------------------------------
// Guarded layouts
// ---------------------------------------------------------------------------

/// Byte layout of a guarded region with rows, misalignment, and padding.
pub(super) struct Layout {
    /// Offset of the first row from the start of the allocation.
    pub start: usize,
    pub row_bytes: usize,
    /// Distance between consecutive row starts.
    pub pitch: usize,
    pub rows: usize,
    pub total: usize,
}

impl Layout {
    fn new(case: &Case, elem_bytes: usize) -> Self {
        let geometry = &case.geometry;
        let row_bytes = geometry.len * elem_bytes;
        let pitch = row_bytes + geometry.pitch_pad * elem_bytes;
        let start = GUARD_BYTES + geometry.misalign;
        let total = start + pitch * geometry.rows + GUARD_BYTES;
        Self {
            start,
            row_bytes,
            pitch,
            rows: geometry.rows,
            total,
        }
    }

    /// Byte offset of row `index` from the allocation start.
    pub(super) fn row(&self, index: usize) -> usize {
        self.start + index * self.pitch
    }

    /// Bytes covered by contiguous rows starting at row zero.
    pub(super) fn contiguous(&self) -> usize {
        self.rows * self.row_bytes
    }

    /// End of the last row, from the allocation start.
    pub(super) fn span(&self) -> usize {
        self.start + self.row_bytes + self.pitch * self.rows.saturating_sub(1)
    }
}

/// A guarded region for `case` at `elem_bytes` per element.
pub(super) fn guarded(case: &Case, elem_bytes: usize) -> (Vec<u8>, Layout) {
    let layout = Layout::new(case, elem_bytes);
    (vec![GUARD; layout.total], layout)
}

/// A guarded single buffer of `len` bytes.
pub(super) fn guarded_bytes(len: usize, misalign: usize) -> (Vec<u8>, usize) {
    let start = GUARD_BYTES + misalign;
    let mut buffer = vec![GUARD; start + len + GUARD_BYTES];
    for byte in &mut buffer[start..start + len] {
        *byte = 0;
    }
    (buffer, start)
}

/// Fill every row of `layout` with canonical elements.
pub(super) fn fill_rows<F: FieldBuffer>(
    buffer: &mut [u8],
    case: &Case,
    layout: &Layout,
    salt: u64,
    nonzero: bool,
) {
    let count = case.geometry.len;
    for index in 0..layout.rows {
        let value = elements::<F>(case, salt + index as u64, count, nonzero);
        let start = layout.row(index);
        buffer[start..start + layout.row_bytes].copy_from_slice(&value);
    }
}

// ---------------------------------------------------------------------------
// Element-wise oracles
// ---------------------------------------------------------------------------

/// Apply `op` element by element over whole lanes.
pub(super) fn unary<F: FieldBuffer>(dst: &mut [u8], src: &[u8], op: impl Fn(Elem<F>) -> Elem<F>) {
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, op(F::decode(s)));
    }
}

/// Apply `op` element by element over two sources.
pub(super) fn binary<F: FieldBuffer>(
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
    op: impl Fn(Elem<F>, Elem<F>) -> Elem<F>,
) {
    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::encode(d, op(F::decode(x), F::decode(y)));
    }
}

/// Accumulate `coeff * src` into `dst`, preserving destination bytes for a
/// zero coefficient and adding whole lanes for a unit coefficient.
pub(super) fn accumulate<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        for (d, s) in dst
            .chunks_exact_mut(F::BYTES)
            .zip(src.chunks_exact(F::BYTES))
        {
            F::encode(d, F::decode(d).add(F::decode(s)));
        }
        return;
    }
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        F::encode(d, F::decode(d).add(coeff.mul(F::decode(s))));
    }
}

/// Compute `coeff * src` into `dst`, with the documented zero-fill and
/// byte-preserving unit copy.
pub(super) fn scale_into<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>, src: &[u8]) {
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    if coeff.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    unary::<F>(dst, src, |value| coeff.mul(value));
}

/// Scale `dst` in place, with the documented unit no-op and zero fill.
pub(super) fn scale_assign<F: FieldBuffer>(dst: &mut [u8], coeff: Elem<F>) {
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    for lane in dst.chunks_exact_mut(F::BYTES) {
        F::encode(lane, coeff.mul(F::decode(lane)));
    }
}

// ---------------------------------------------------------------------------
// Judge
// ---------------------------------------------------------------------------

/// One stage's observed work against the cell's budget.
pub(super) struct StageReport {
    pub substage: &'static str,
    pub scope: Option<(&'static str, &'static str)>,
    pub unit: &'static str,
    pub tier: Backend,
    pub work: usize,
    pub budget: usize,
    pub met: bool,
}

/// The decision for one case.
pub(super) struct Verdict {
    pub skip: Option<String>,
    pub detail: String,
    pub stages: Vec<StageReport>,
}

impl Verdict {
    fn pass(stages: Vec<StageReport>) -> Self {
        Self {
            skip: None,
            detail: String::new(),
            stages,
        }
    }

    fn fail(detail: String, stages: Vec<StageReport>) -> Self {
        Self {
            skip: None,
            detail,
            stages,
        }
    }
}

fn unit_name(unit: Unit) -> &'static str {
    match unit {
        Unit::Elements => "Elements",
        Unit::Bytes => "Bytes",
        Unit::Bits => "Bits",
        Unit::Words => "Words",
    }
}

fn substage_name(substage: Substage) -> &'static str {
    match substage {
        Substage::Add => "Add",
        Substage::Sub => "Sub",
        Substage::Mul => "Mul",
        Substage::MulAdd => "MulAdd",
        Substage::MulElementwise => "MulElementwise",
        Substage::Inverse => "Inverse",
        Substage::Dot => "Dot",
        Substage::Linear => "Linear",
        Substage::Pack => "Pack",
        Substage::Unpack => "Unpack",
        Substage::BitAnd => "BitAnd",
        Substage::BitAndNot => "BitAndNot",
        Substage::BitXor => "BitXor",
        Substage::BitWeight => "BitWeight",
        Substage::BitParity => "BitParity",
    }
}

fn scope_name(scope: Scope) -> &'static str {
    match scope {
        Scope::Inverse => "Inverse",
        Scope::Division => "Division",
        Scope::Frobenius => "Frobenius",
        Scope::Norm => "Norm",
        Scope::Trace => "Trace",
        Scope::Restriction => "Restriction",
        Scope::Membership => "Membership",
    }
}

fn pass_name(pass: Pass) -> &'static str {
    match pass {
        Pass::Main => "Main",
        Pass::Prefix => "Prefix",
        Pass::Backward => "Backward",
        Pass::Output => "Output",
        Pass::Conjugate => "Conjugate",
        Pass::Accumulate => "Accumulate",
        Pass::Project => "Project",
    }
}

fn scope_names(scope: Option<(Scope, Pass)>) -> Option<(&'static str, &'static str)> {
    scope.map(|(scope, pass)| (scope_name(scope), pass_name(pass)))
}

fn multiplicity(case: &Case, work: &Work) -> usize {
    let geometry = &case.geometry;
    match work.multiplicity {
        Multiplicity::One => 1,
        Multiplicity::Rows => geometry.rows,
        Multiplicity::ActiveRows => {
            let slots = case.coeffs;
            let sources = geometry.sources;
            if sources == 0 || slots.len() != geometry.rows * sources {
                return 0;
            }
            (0..geometry.rows)
                .filter(|&row| {
                    slots[row * sources..(row + 1) * sources]
                        .iter()
                        .any(|slot| !matches!(slot, Coeff::Zero))
                })
                .count()
        }
        Multiplicity::ActivePairs => case
            .coeffs
            .iter()
            .filter(|slot| matches!(slot, Coeff::General))
            .count(),
        Multiplicity::UnitAddPairs => case
            .coeffs
            .iter()
            .filter(|slot| matches!(slot, Coeff::One))
            .count(),
    }
}

/// Budget for one stage: `iterations × multiplicity × (length − loss)`.
fn budget(case: &Case, work: &Work, outcome: &Outcome) -> usize {
    debug_assert!(work.lane > 0, "a work rule lane holds at least one unit");
    let width_bytes = match work.basis {
        Basis::Source => outcome.source_bytes,
        Basis::Target => outcome.target_bytes,
    };
    let elements = case.geometry.len;
    let length = match work.unit {
        Unit::Elements => elements,
        Unit::Bytes => elements.saturating_mul(width_bytes),
        Unit::Bits => elements,
        Unit::Words => (elements.saturating_mul(width_bytes) / 8).max(1),
    };
    let count = multiplicity(case, work);
    work.iterations
        .saturating_mul(count)
        .saturating_mul(length.saturating_sub(work.loss))
}

fn strongest(tier: Backend, candidate: Backend) -> Backend {
    if trace::strength(candidate) > trace::strength(tier) {
        candidate
    } else {
        tier
    }
}

fn entry_key(entry: &Entry) -> (Substage, Option<(Scope, Pass)>) {
    (entry.substage, entry.scope)
}

fn work_key(work: &Work) -> (Substage, Option<(Scope, Pass)>) {
    (work.substage, work.scope)
}

/// Decide one case: output equality always, routing when enforced.
#[allow(clippy::too_many_lines)]
pub(super) fn judge(case: &Case, cell: &Cell, outcome: &Outcome) -> Verdict {
    if let Some(reason) = &outcome.skip {
        return Verdict {
            skip: Some(reason.clone()),
            detail: String::new(),
            stages: Vec::new(),
        };
    }
    let mut detail = String::new();
    for check in &outcome.checks {
        if check.actual != check.expected {
            let at = check
                .actual
                .iter()
                .zip(&check.expected)
                .position(|(a, e)| a != e);
            let _ = write!(
                detail,
                "output {} differs at {:?} (actual {} bytes, expected {} bytes); ",
                check.name,
                at,
                check.actual.len(),
                check.expected.len(),
            );
        }
    }
    for &(name, actual, expected) in &outcome.values {
        if actual != expected {
            let _ = write!(detail, "value {name} is {actual} but expected {expected}; ");
        }
    }
    let output_ok = detail.is_empty();

    let enforce = outcome.enforce_routing && cell.required;
    let mut stages = Vec::new();
    let mut routing_ok = true;
    let mut routing_detail = String::new();
    if enforce {
        if outcome.process != cell.process {
            routing_ok = false;
            let _ = write!(
                routing_detail,
                "process is {} but the cell expects {}; ",
                outcome.process.name(),
                cell.process.name(),
            );
        }
        if outcome.field != cell.field {
            routing_ok = false;
            let _ = write!(
                routing_detail,
                "field backend is {} but the cell expects {}; ",
                outcome.field.name(),
                cell.field.name(),
            );
        }
    }

    for work in cell.work {
        let key = work_key(work);
        let mut served = 0usize;
        let mut tier = Backend::Scalar;
        let mut wrong_unit = false;
        for entry in &outcome.trace {
            if entry_key(entry) != key {
                continue;
            }
            if entry.unit != work.unit {
                wrong_unit = true;
                continue;
            }
            if trace::strength(entry.tier) >= trace::strength(cell.min_serving) {
                served = served.saturating_add(entry.work);
                tier = strongest(tier, entry.tier);
            }
        }
        let required = budget(case, work, outcome);
        let mut met = served >= required;
        if wrong_unit {
            met = false;
            let _ = write!(
                routing_detail,
                "stage {} ({:?}) emitted a unit other than {}; ",
                substage_name(work.substage),
                scope_names(work.scope),
                unit_name(work.unit),
            );
        }
        if !met && enforce {
            routing_ok = false;
            if !wrong_unit {
                let _ = write!(
                    routing_detail,
                    "stage {} ({:?}) served {served} < {required} {} at {} or stronger; ",
                    substage_name(work.substage),
                    scope_names(work.scope),
                    unit_name(work.unit),
                    cell.min_serving.name(),
                );
            }
        }
        stages.push(StageReport {
            substage: substage_name(work.substage),
            scope: scope_names(work.scope),
            unit: unit_name(work.unit),
            tier,
            work: served,
            budget: required,
            met,
        });
    }

    // Stages the cell does not budget still appear in the receipt, so a
    // survey shows every stretch of work the call performed.
    for entry in &outcome.trace {
        let key = entry_key(entry);
        if cell.work.iter().any(|work| work_key(work) == key) {
            continue;
        }
        let listed = stages.iter_mut().find(|stage| {
            stage.substage == substage_name(entry.substage)
                && stage.scope == scope_names(entry.scope)
                && stage.unit == unit_name(entry.unit)
        });
        match listed {
            Some(stage) => {
                stage.work = stage.work.saturating_add(entry.work);
                stage.tier = strongest(stage.tier, entry.tier);
            }
            None => stages.push(StageReport {
                substage: substage_name(entry.substage),
                scope: scope_names(entry.scope),
                unit: unit_name(entry.unit),
                tier: entry.tier,
                work: entry.work,
                budget: 0,
                met: true,
            }),
        }
    }

    if output_ok && (!enforce || routing_ok) {
        Verdict::pass(stages)
    } else {
        detail.push_str(&routing_detail);
        Verdict::fail(detail, stages)
    }
}

// ---------------------------------------------------------------------------
// Fault injection (test-only pipeline proof)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    CorruptOutput,
    SuppressTrace,
    HalveWork,
    LowerTier,
    DropReceipt,
}

fn fault_from_env() -> Option<Fault> {
    let raw = std::env::var("FGF_ROUTE_FAULT").ok()?;
    if raw.is_empty() {
        return None;
    }
    let fault = match raw.as_str() {
        "corrupt-output" => Fault::CorruptOutput,
        "suppress-trace" => Fault::SuppressTrace,
        "halve-work" => Fault::HalveWork,
        "lower-tier" => Fault::LowerTier,
        "drop-receipt" => Fault::DropReceipt,
        other => panic!("FGF_ROUTE_FAULT names no known fault: {other}"),
    };
    Some(fault)
}

fn apply_fault(fault: Fault, outcome: &mut Outcome) {
    match fault {
        Fault::CorruptOutput => {
            let check = outcome
                .checks
                .first_mut()
                .expect("a fault target produced no output buffer");
            if let Some(byte) = check.actual.first_mut() {
                *byte ^= 0xFF;
            }
        }
        Fault::SuppressTrace => outcome.trace.clear(),
        Fault::HalveWork => {
            for entry in &mut outcome.trace {
                entry.work /= 2;
            }
        }
        Fault::LowerTier => {
            for entry in &mut outcome.trace {
                entry.tier = match entry.tier {
                    Backend::V4x => Backend::V3GfniCrypto,
                    Backend::V3GfniCrypto => Backend::V3,
                    Backend::V3 => Backend::V2,
                    _ => Backend::Scalar,
                };
            }
        }
        Fault::DropReceipt => {}
    }
}

// ---------------------------------------------------------------------------
// Environment and receipts
// ---------------------------------------------------------------------------

/// The feature set a run was compiled with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fingerprint {
    Default,
    All,
    Other,
}

fn fingerprint() -> Fingerprint {
    let default = cfg!(feature = "std") && cfg!(feature = "simd");
    let all = default && cfg!(feature = "simd512") && cfg!(feature = "internals");
    if all {
        Fingerprint::All
    } else if default && !cfg!(feature = "simd512") && !cfg!(feature = "internals") {
        Fingerprint::Default
    } else {
        Fingerprint::Other
    }
}

fn fingerprint_name(fingerprint: Fingerprint) -> &'static str {
    match fingerprint {
        Fingerprint::Default => "default",
        Fingerprint::All => "all",
        Fingerprint::Other => "other",
    }
}

struct Env {
    fingerprint: Fingerprint,
    request: Option<Request>,
    request_raw: String,
    enforce: bool,
    source_id: String,
    host_id: String,
    fault: Option<Fault>,
    list: bool,
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

impl Env {
    fn load() -> Self {
        let fingerprint = fingerprint();
        let request_raw = env_string("SIMD_BACKEND");
        let request = request_raw.as_deref().and_then(Request::parse);
        if let Some(raw) = request_raw.as_deref()
            && request.is_none()
        {
            panic!("SIMD_BACKEND names no selectable backend: {raw}");
        }
        let accepting = env_string("FGF_ROUTE_ACCEPT").as_deref() == Some("1");
        let survey = env_string("FGF_ROUTE_SURVEY").as_deref() == Some("1");
        let source_id = env_string("FGF_SOURCE_ID").unwrap_or_else(|| "unset".to_string());
        let host_id = env_string("FGF_HOST_ID").unwrap_or_else(|| "unset".to_string());
        if accepting {
            assert!(
                !survey,
                "FGF_ROUTE_ACCEPT and FGF_ROUTE_SURVEY are mutually exclusive"
            );
            let declared = env_string("FGF_FEATURES")
                .unwrap_or_else(|| panic!("acceptance mode requires FGF_FEATURES"));
            assert_eq!(
                declared,
                fingerprint_name(fingerprint),
                "FGF_FEATURES does not match the compiled feature set",
            );
            assert_ne!(
                fingerprint,
                Fingerprint::Other,
                "acceptance mode requires the default or all-features set",
            );
            assert!(request.is_some(), "acceptance mode requires SIMD_BACKEND");
            assert_ne!(source_id, "unset", "acceptance mode requires FGF_SOURCE_ID");
            assert_ne!(host_id, "unset", "acceptance mode requires FGF_HOST_ID");
        }
        let enforce = accepting || (request.is_some() && !survey);
        Self {
            fingerprint,
            request,
            request_raw: request_raw.unwrap_or_else(|| "unset".to_string()),
            enforce,
            source_id,
            host_id,
            fault: fault_from_env(),
            list: env_string("FGF_ROUTE_LIST").as_deref() == Some("1"),
        }
    }
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
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

fn quote(text: &str) -> String {
    format!("\"{}\"", escape(text))
}

fn stages_json(stages: &[StageReport]) -> String {
    let mut out = String::from("[");
    for (index, stage) in stages.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let scope = match stage.scope {
            None => "null".to_string(),
            Some((scope, pass)) => format!("[{},{}]", quote(scope), quote(pass)),
        };
        let _ = write!(
            out,
            "{{\"substage\":{},\"scope\":{},\"unit\":{},\"tier\":{},\"work\":{},\"budget\":{},\"met\":{}}}",
            quote(stage.substage),
            scope,
            quote(stage.unit),
            quote(stage.tier.name()),
            stage.work,
            stage.budget,
            stage.met,
        );
    }
    out.push(']');
    out
}

struct Receipt<'a> {
    case: &'a Case,
    cell: &'a Cell,
    env: &'a Env,
    verdict: &'a Verdict,
    process: Backend,
    field: Backend,
}

fn receipt_json(receipt: &Receipt<'_>) -> String {
    let verdict = receipt.verdict;
    let (result, reason) = match (&verdict.skip, verdict.detail.is_empty()) {
        (Some(reason), _) => ("skip", reason.clone()),
        (None, true) => ("pass", String::new()),
        (None, false) => ("fail", verdict.detail.clone()),
    };
    format!(
        "ROUTE_RECEIPT\t{{\"id\":{},\"obligation\":{},\"features\":{},\"request_raw\":{},\
         \"process\":{},\"field\":{},\"stages\":{},\"required\":{},\"result\":{},\"reason\":{},\
         \"source_id\":{},\"host_id\":{}}}",
        quote(receipt.case.id),
        quote(receipt.case.obligation),
        quote(receipt.cell.features.name()),
        quote(&receipt.env.request_raw),
        quote(receipt.process.name()),
        quote(receipt.field.name()),
        stages_json(&verdict.stages),
        receipt.cell.required,
        quote(result),
        quote(&reason),
        quote(&receipt.env.source_id),
        quote(&receipt.env.host_id),
    )
}

fn cell_for(case: &Case, features: Features, request: Request) -> &Cell {
    case.cells
        .iter()
        .find(|cell| cell.features == features && cell.request == request)
        .unwrap_or_else(|| {
            panic!(
                "case {} has no cell for {} {}",
                case.id,
                features.name(),
                request.name()
            )
        })
}

fn all_cases() -> impl Iterator<Item = &'static Case> {
    let cases = cases::CASES.iter();
    #[cfg(feature = "simd512")]
    let cases = cases.chain(cases::CASES_SIMD512.iter());
    cases
}

fn list_cases() {
    for case in all_cases() {
        std::println!("ROUTE_CASE_LIST\t{}", case.id);
    }
}

/// Run every case for the selected configuration and print receipts.
#[test]
fn accept() {
    let env = Env::load();
    if env.list {
        list_cases();
        return;
    }
    let lookup = match env.fingerprint {
        Fingerprint::Default => Features::Default,
        Fingerprint::All => Features::All,
        Fingerprint::Other => {
            if cfg!(feature = "simd512") {
                Features::All
            } else {
                Features::Default
            }
        }
    };

    let mut drop_target: Option<&'static str> = None;
    if env.fault == Some(Fault::DropReceipt) {
        drop_target = all_cases()
            .find(|case| {
                let cell = cell_for(case, lookup, env.request.unwrap_or(Request::Scalar));
                cell.required
            })
            .map(|case| case.id);
    }

    let mut cases_run = 0usize;
    let mut dropped = false;
    let mut failures: Vec<String> = Vec::new();
    for case in all_cases() {
        cases_run += 1;
        let request = env.request.unwrap_or(Request::Scalar);
        let initial = cell_for(case, lookup, request);
        let mut outcome = (case.run)(case, initial);
        let request = env
            .request
            .unwrap_or_else(|| Request::from_backend(outcome.process));
        let cell = cell_for(case, lookup, request);
        outcome.enforce_routing = env.enforce;
        if let Some(fault) = env.fault
            && fault != Fault::DropReceipt
            && cell.required
        {
            apply_fault(fault, &mut outcome);
        }
        let verdict = judge(case, cell, &outcome);
        if verdict.skip.is_none() && !verdict.detail.is_empty() {
            failures.push(format!("{}: {}", case.id, verdict.detail));
        }
        if drop_target == Some(case.id) && !dropped {
            dropped = true;
            continue;
        }
        let receipt = Receipt {
            case,
            cell,
            env: &env,
            verdict: &verdict,
            process: outcome.process,
            field: outcome.field,
        };
        std::println!("{}", receipt_json(&receipt));
    }
    std::println!(
        "ROUTE_DONE\t{{\"cases\":{},\"source_id\":{},\"host_id\":{},\"features\":{},\"request\":{}}}",
        cases_run,
        quote(&env.source_id),
        quote(&env.host_id),
        quote(lookup_name(env.fingerprint, lookup)),
        quote(&env.request_raw),
    );
    assert!(
        failures.is_empty(),
        "{} route case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn lookup_name(fingerprint: Fingerprint, lookup: Features) -> &'static str {
    match fingerprint {
        Fingerprint::Default | Fingerprint::All => fingerprint_name(fingerprint),
        Fingerprint::Other => lookup.name(),
    }
}

/// One field element repeated across a row.
pub(super) fn constant_row<F: FieldBuffer>(value: Elem<F>, count: usize) -> Vec<u8> {
    let mut out = vec![0u8; count * F::BYTES];
    for lane in out.chunks_exact_mut(F::BYTES) {
        F::encode(lane, value);
    }
    out
}

/// Apply a width-changing map element by element: `dst` holds `D` lanes,
/// `src` holds `S` lanes.
pub(super) fn map_unary<D: FieldBuffer, S: FieldBuffer>(
    dst: &mut [u8],
    src: &[u8],
    op: impl Fn(Elem<S>) -> Elem<D>,
) {
    for (d, s) in dst
        .chunks_exact_mut(D::BYTES)
        .zip(src.chunks_exact(S::BYTES))
    {
        D::encode(d, op(S::decode(s)));
    }
}

// ---------------------------------------------------------------------------
// Tests of the judge
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const GEO: Geometry = Geometry {
        rows: 1,
        sources: 1,
        len: 64,
        misalign: 1,
        pitch_pad: 0,
    };
    const COEFFS: &[Coeff] = &[Coeff::General];
    const RUN: fn(&Case, &Cell) -> Outcome = dummy_run;
    const CHECK_CELL: Cell = Cell {
        features: Features::Default,
        request: Request::Scalar,
        process: Backend::Scalar,
        field: Backend::Scalar,
        required: true,
        min_serving: Backend::Scalar,
        work: &[],
    };
    const WORK: &[Work] = &[Work {
        substage: Substage::MulAdd,
        scope: None,
        unit: Unit::Elements,
        basis: Basis::Source,
        multiplicity: Multiplicity::One,
        lane: 1,
        loss: 0,
        iterations: 1,
    }];
    const CELLS: &[Cell] = &[
        CHECK_CELL,
        Cell {
            features: Features::Default,
            request: Request::Scalar,
            process: Backend::Scalar,
            field: Backend::Scalar,
            required: true,
            min_serving: Backend::Scalar,
            work: WORK,
        },
    ];
    const CELL_V2: Cell = Cell {
        features: Features::Default,
        request: Request::Scalar,
        process: Backend::Scalar,
        field: Backend::Scalar,
        required: true,
        min_serving: Backend::V2,
        work: WORK,
    };
    const CASE: Case = Case {
        id: "judge@test#fixture/geometry/_",
        obligation: "judge@test",
        run: RUN,
        geometry: GEO,
        coeffs: COEFFS,
        map: None,
        cells: CELLS,
    };

    fn dummy_run(_: &Case, _: &Cell) -> Outcome {
        Outcome::new(Backend::Scalar, Backend::Scalar)
    }

    fn entries(work: usize) -> Vec<Entry> {
        vec![Entry {
            kernel: "test",
            substage: Substage::MulAdd,
            scope: None,
            unit: Unit::Elements,
            work,
            tier: Backend::Scalar,
        }]
    }

    fn outcome_with(actual: u8, expected: u8, work: usize) -> Outcome {
        let mut outcome = Outcome::new(Backend::Scalar, Backend::Scalar).with_trace(entries(work));
        outcome.checks.push(BufferCheck {
            name: "dst",
            actual: vec![actual],
            expected: vec![expected],
        });
        outcome
    }

    /// A correct call with enough work passes.
    #[test]
    fn judge_accepts_correct_output_and_work() {
        let mut outcome = outcome_with(7, 7, 64);
        outcome.enforce_routing = true;
        let verdict = judge(&CASE, &CELLS[1], &outcome);
        assert!(verdict.detail.is_empty(), "{}", verdict.detail);
        assert!(verdict.skip.is_none());
        assert_eq!(verdict.stages.len(), 1);
        assert!(verdict.stages[0].met);
    }

    /// A wrong output byte fails even when routing is not enforced.
    #[test]
    fn judge_rejects_corrupt_output() {
        let outcome = outcome_with(7, 8, 64);
        let verdict = judge(&CASE, &CELLS[1], &outcome);
        assert!(!verdict.detail.is_empty());
    }

    /// A missing stage fails the budget when routing is enforced.
    #[test]
    fn judge_rejects_missing_work() {
        let mut outcome = outcome_with(7, 7, 0);
        outcome.enforce_routing = true;
        let verdict = judge(&CASE, &CELLS[1], &outcome);
        assert!(!verdict.detail.is_empty());
        assert!(!verdict.stages[0].met);
    }

    /// A wrong unit on the matching stage key fails the stage.
    #[test]
    fn judge_rejects_wrong_unit() {
        let mut outcome = outcome_with(7, 7, 64);
        outcome.trace[0].unit = Unit::Bytes;
        outcome.enforce_routing = true;
        let verdict = judge(&CASE, &CELLS[1], &outcome);
        assert!(!verdict.detail.is_empty());
    }

    /// A tier below `min_serving` contributes no work.
    #[test]
    fn judge_rejects_lower_tier() {
        let mut outcome = outcome_with(7, 7, 64);
        outcome.enforce_routing = true;
        let verdict = judge(&CASE, &CELL_V2, &outcome);
        assert!(!verdict.detail.is_empty());
    }

    const COEFFS_MIX: &[Coeff] = &[Coeff::General, Coeff::Zero, Coeff::One, Coeff::One];
    const GEO_MIX: Geometry = Geometry {
        rows: 2,
        sources: 2,
        len: 16,
        misalign: 0,
        pitch_pad: 1,
    };
    const CASE_MIX: Case = Case {
        id: "judge@mix#fixture/geometry/_",
        obligation: "judge@mix",
        run: RUN,
        geometry: GEO_MIX,
        coeffs: COEFFS_MIX,
        map: None,
        cells: CELLS,
    };

    fn work(multiplicity: Multiplicity, basis: Basis, unit: Unit) -> Work {
        Work {
            substage: Substage::MulAdd,
            scope: None,
            unit,
            basis,
            multiplicity,
            lane: 1,
            loss: 0,
            iterations: 1,
        }
    }

    /// Every multiplicity counts what its name says.
    #[test]
    fn judge_counts_every_multiplicity() {
        assert_eq!(
            multiplicity(
                &CASE_MIX,
                &work(Multiplicity::One, Basis::Source, Unit::Elements)
            ),
            1
        );
        assert_eq!(
            multiplicity(
                &CASE_MIX,
                &work(Multiplicity::Rows, Basis::Source, Unit::Elements)
            ),
            2
        );
        assert_eq!(
            multiplicity(
                &CASE_MIX,
                &work(Multiplicity::ActiveRows, Basis::Source, Unit::Elements)
            ),
            2
        );
        assert_eq!(
            multiplicity(
                &CASE_MIX,
                &work(Multiplicity::ActivePairs, Basis::Source, Unit::Elements)
            ),
            1
        );
        assert_eq!(
            multiplicity(
                &CASE_MIX,
                &work(Multiplicity::UnitAddPairs, Basis::Source, Unit::Elements)
            ),
            2
        );
    }

    /// A target-basis budget measures the target width.
    #[test]
    fn judge_target_basis_uses_target_width() {
        let mut outcome = Outcome::new(Backend::Scalar, Backend::Scalar);
        outcome.source_bytes = 1;
        outcome.target_bytes = 4;
        let work = work(Multiplicity::One, Basis::Target, Unit::Bytes);
        assert_eq!(budget(&CASE_MIX, &work, &outcome), GEO_MIX.len * 4);
    }

    /// A field mismatch fails only when routing is enforced.
    #[test]
    fn judge_enforces_process_and_field() {
        let mut outcome = outcome_with(7, 7, 64);
        outcome.field = Backend::V2;
        let surveyed = judge(&CASE, &CELLS[1], &outcome);
        assert!(surveyed.detail.is_empty());
        outcome.enforce_routing = true;
        let enforced = judge(&CASE, &CELLS[1], &outcome);
        assert!(!enforced.detail.is_empty());
    }
}
