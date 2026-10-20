//! Interleaved native GF16 field operations with exact representation validation.
//!
//! Preparation and conversion are excluded. Multi-row competitor arms compose
//! their native region primitives; unsupported primitives are not substituted.
//!
//! `--presentations` runs the separate registered-presentation campaign: every
//! public GF16 presentation (shipped towers, the custom Reed–Solomon
//! relation, transported basis-base relations, and flat competitor
//! polynomials) interleaves the same native arms under validated exact maps
//! and emits `SURFACE_INVENTORY` and `GF16_SURFACE` machine records. The
//! default legacy output is unchanged.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::{Elem, FieldElem, FieldKernels, ops};
use reed_solomon_erasure::{Field, galois_16};
use reed_solomon_simd::engine::{Avx2, Engine, tables};

mod native;
mod presentations;
mod representation;

use native::Native;
use representation::{Layout, Representations, tower_mul};

pub(crate) type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;

const ALIGN: usize = 4096;
pub(crate) const NAMES: [&str; 6] = [
    "fgf",
    "fgf-control",
    "gf-complete",
    "reed-solomon-erasure",
    "reed-solomon-simd",
    "leopard",
];

struct Buffer {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl Buffer {
    fn new(len: usize, offset: usize) -> Self {
        let storage = vec![0; len + ALIGN + offset];
        let start = storage.as_ptr().align_offset(ALIGN) + offset;
        Self {
            storage,
            start,
            len,
        }
    }
    fn bytes(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }
}

pub(crate) fn noise(len: usize, seed: u64) -> Vec<u8> {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Operation {
    Assign,
    Into,
    Add,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) operation: Operation,
    pub(crate) sources: usize,
    pub(crate) destinations: usize,
    pub(crate) bytes: usize,
    pub(crate) cycling: bool,
}

impl Shape {
    pub(crate) fn label(self) -> &'static str {
        match (self.operation, self.sources, self.destinations) {
            (Operation::Assign, _, _) => "mul_assign",
            (Operation::Into, 1, 1) => "mul_into",
            (Operation::Add, 1, 1) => "mul_add",
            (Operation::Into, _, 1) => "mul_into_gather_with",
            (Operation::Add, _, 1) => "mul_add_gather_with",
            (Operation::Into, _, _) => "mul_into_matrix_with",
            (Operation::Add, _, _) => "mul_add_matrix_with",
        }
    }
    pub(crate) fn schedule(self) -> &'static str {
        if self.cycling { "cycling16" } else { "fixed" }
    }
}

struct Bank<F: FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>> {
    /// Coefficient words in the canonical tower encoding: the oracle's
    /// arithmetic and every native arm's layout map read these.
    canonical: Vec<u16>,
    /// The same coefficients as presentation words: the fgf arms multiply
    /// by these, so every arm multiplies by one abstract field element.
    presentation: Vec<u16>,
    complete: Vec<u16>,
    erasure: Vec<u16>,
    logs: Vec<u16>,
    matrix: ops::CoeffMatrix<F>,
    gather: ops::CoeffVec<F>,
}

struct Plan<F: FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>> {
    shape: Shape,
    banks: Vec<Bank<F>>,
}

impl<F: FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>> Plan<F> {
    fn new(
        shape: Shape,
        maps: &Representations,
        map: Option<&[u16]>,
        special: Option<u16>,
    ) -> Self {
        let banks = (0..if shape.cycling { 16 } else { 1 })
            .map(|bank| {
                let canonical: Vec<_> = (0..shape.sources * shape.destinations)
                    .map(|index| {
                        let mut value = (0x1234usize + index * 0x2311 + bank * 0x19ab) as u16;
                        if value >> 8 == 0 {
                            value |= 0x100;
                        }
                        special.unwrap_or(value)
                    })
                    .collect();
                let word = |value: u16| match map {
                    Some(map) => map[usize::from(value)],
                    None => value,
                };
                let presentation: Vec<Elem<F>> = canonical
                    .iter()
                    .map(|&value| Elem::<F>::from_raw(word(value)))
                    .collect();
                let layout_words = |layout| {
                    canonical
                        .iter()
                        .map(|&value| maps.element(value, layout))
                        .collect::<Vec<_>>()
                };
                let logs = layout_words(Layout::Cantor)
                    .iter()
                    .map(|&c| tables::get_exp_log().log[usize::from(c)])
                    .collect();
                let matrix = ops::CoeffMatrix::from_source_major(
                    shape.sources,
                    shape.destinations,
                    &presentation,
                );
                let gather = ops::CoeffVec::new(&presentation);
                Bank {
                    complete: layout_words(Layout::Complete),
                    erasure: layout_words(Layout::Erasure),
                    logs,
                    matrix,
                    gather,
                    presentation: canonical.iter().map(|&value| word(value)).collect(),
                    canonical,
                }
            })
            .collect();
        Self { shape, banks }
    }
}

pub(crate) struct Arm<'a> {
    pub(crate) kind: usize,
    layout: Layout,
    source: Vec<&'a [u8]>,
    destination: Buffer,
}

impl Arm<'_> {
    fn apply<F: FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>>(
        &mut self,
        plan: &Plan<F>,
        bank_index: usize,
        native: &Native,
        simd: &Avx2,
    ) {
        let shape = plan.shape;
        let bank = &plan.banks[bank_index % plan.banks.len()];
        let dst = black_box(self.destination.bytes_mut());
        let sources = black_box(self.source.as_slice());
        if self.kind <= 1 {
            let coefficient = Elem::<F>::from_raw(bank.presentation[0]);
            match (shape.operation, shape.sources, shape.destinations) {
                (Operation::Assign, _, _) => ops::mul_assign::<F>(dst, black_box(coefficient)),
                (Operation::Into, 1, 1) => {
                    ops::mul_into::<F>(dst, black_box(coefficient), sources[0])
                }
                (Operation::Add, 1, 1) => {
                    ops::mul_add::<F>(dst, black_box(coefficient), sources[0])
                }
                (Operation::Into, _, 1) => {
                    ops::mul_into_gather_with::<F>(dst, bank.gather.as_ref(), sources)
                }
                (Operation::Add, _, 1) => {
                    ops::mul_add_gather_with::<F>(dst, bank.gather.as_ref(), sources)
                }
                (Operation::Into, _, _) => {
                    ops::mul_into_matrix_with::<F>(dst, shape.bytes, &bank.matrix, sources)
                }
                (Operation::Add, _, _) => {
                    ops::mul_add_matrix_with::<F>(dst, shape.bytes, &bank.matrix, sources)
                }
            }
            return;
        }
        if shape.operation == Operation::Assign {
            match self.kind {
                2 => native.complete_assign(dst, bank.complete[0]),
                4 => {
                    if bank.canonical[0] == 0 {
                        dst.fill(0);
                    } else {
                        simd.mul(dst.as_chunks_mut::<64>().0, black_box(bank.logs[0]));
                    }
                }
                _ => unreachable!("unsupported assign arm"),
            }
            return;
        }
        for (source_index, src) in sources.iter().enumerate() {
            for (row, output) in dst.chunks_exact_mut(shape.bytes).enumerate() {
                let index = source_index * shape.destinations + row;
                let add = shape.operation == Operation::Add || source_index != 0;
                match self.kind {
                    2 => native.complete_region(output, src, black_box(bank.complete[index]), add),
                    3 => {
                        let coefficient = black_box(bank.erasure[index].to_be_bytes());
                        if add {
                            galois_16::Field::mul_slice_add(
                                coefficient,
                                src.as_chunks::<2>().0,
                                output.as_chunks_mut::<2>().0,
                            );
                        } else {
                            galois_16::Field::mul_slice(
                                coefficient,
                                src.as_chunks::<2>().0,
                                output.as_chunks_mut::<2>().0,
                            );
                        }
                    }
                    5 => {
                        if bank.canonical[index] == 0 {
                            if !add {
                                output.fill(0);
                            }
                        } else {
                            native.leopard_region(output, src, black_box(bank.logs[index]), add);
                        }
                    }
                    _ => unreachable!("unsupported region arm"),
                }
            }
        }
    }
}

struct Oracle {
    products: HashMap<u16, Vec<u16>>,
}

impl Oracle {
    pub(crate) fn new() -> Self {
        Self {
            products: HashMap::new(),
        }
    }

    fn result(
        &mut self,
        plan: &Plan<impl FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>>,
        bank_index: usize,
        source: &[Vec<u8>],
        initial: &[u8],
    ) -> Vec<u8> {
        let shape = plan.shape;
        let coefficients = &plan.banks[bank_index].canonical;
        for c in coefficients {
            self.products
                .entry(*c)
                .or_insert_with(|| (0..=u16::MAX).map(|x| tower_mul(x, *c)).collect());
        }
        let mut output = if shape.operation == Operation::Into {
            vec![0; initial.len()]
        } else {
            initial.to_vec()
        };
        if shape.operation == Operation::Assign {
            let table = &self.products[&coefficients[0]];
            for pair in output.as_chunks_mut::<2>().0 {
                *pair = table[usize::from(u16::from_le_bytes(*pair))].to_le_bytes();
            }
            return output;
        }
        for (term, src) in source.iter().enumerate() {
            for row in 0..shape.destinations {
                let table = &self.products[&coefficients[term * shape.destinations + row]];
                for (dst, pair) in output[row * shape.bytes..(row + 1) * shape.bytes]
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(src.as_chunks::<2>().0)
                {
                    let product =
                        table[usize::from(u16::from_le_bytes([pair[0], pair[1]]))].to_le_bytes();
                    dst[0] ^= product[0];
                    dst[1] ^= product[1];
                }
            }
        }
        output
    }
}

pub(crate) fn quantile(samples: &[f64], q: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * q).round() as usize]
}

/// A Williams balanced Latin square: within one cycle every arm takes every
/// position equally often and directly follows every other arm equally often.
/// Even arm counts need `count` orders; odd counts add each order reversed.
fn arm_orders(count: usize) -> Vec<Vec<usize>> {
    let first: Vec<usize> = (0..count)
        .map(|j| {
            if j % 2 == 1 {
                j.div_ceil(2)
            } else {
                (count - j / 2) % count
            }
        })
        .collect();
    let mut result: Vec<Vec<usize>> = (0..count)
        .map(|shift| first.iter().map(|&arm| (arm + shift) % count).collect())
        .collect();
    if !count.is_multiple_of(2) {
        let reversed: Vec<Vec<usize>> = result
            .iter()
            .map(|order| order.iter().rev().copied().collect())
            .collect();
        result.extend(reversed);
    }
    result
}

pub(crate) struct RunOptions {
    pub(crate) timing: bool,
    pub(crate) smoke: bool,
    pub(crate) offset: usize,
    pub(crate) exhaustive: bool,
}

/// The native arms a shape interleaves: assign cases carry no region
/// primitive for reed-solomon-erasure or Leopard, and those absences stay
/// absent.
pub(crate) fn arm_kinds(operation: Operation) -> &'static [usize] {
    if operation == Operation::Assign {
        &[0, 1, 2, 4]
    } else {
        &[0, 1, 2, 3, 5]
    }
}

pub(crate) fn run_case<F: FieldKernels<Raw = u16, Characteristic = fgf::PrimeCharacteristic<2>>>(
    plan: Plan<F>,
    maps: &Representations,
    native: &Native,
    simd: &Avx2,
    oracle: &mut Oracle,
    options: RunOptions,
    presentation: Option<&presentations::Presentation>,
) {
    let RunOptions {
        timing,
        smoke,
        offset,
        exhaustive,
    } = options;
    let shape = plan.shape;
    let canonical_sources: Vec<_> = (0..shape.sources)
        .map(|term| {
            if exhaustive {
                (0..=u16::MAX).flat_map(u16::to_le_bytes).collect()
            } else {
                noise(shape.bytes, 0x800 + term as u64)
            }
        })
        .collect();
    let initial = noise(shape.bytes * shape.destinations, 0x900);
    let layouts = [
        Layout::Tower,
        Layout::Complete,
        Layout::Erasure,
        Layout::Cantor,
    ];
    let mut fixtures: Vec<Vec<Buffer>> = layouts
        .iter()
        .map(|&layout| {
            canonical_sources
                .iter()
                .map(|src| {
                    let mut buffer = Buffer::new(shape.bytes, offset);
                    maps.pack(buffer.bytes_mut(), src, layout);
                    buffer
                })
                .collect()
        })
        .collect();
    // The fgf arms multiply presentation words, so their sources pack
    // through the presentation's validated map, not the tower encoding.
    if let Some(presentation) = presentation {
        fixtures.push(
            canonical_sources
                .iter()
                .map(|src| {
                    let mut buffer = Buffer::new(shape.bytes, offset);
                    presentation.pack(buffer.bytes_mut(), src);
                    buffer
                })
                .collect(),
        );
    }
    let mut arms: Vec<_> = arm_kinds(shape.operation)
        .iter()
        .copied()
        .map(|kind| {
            // The fgf arms always carry the tower layout — their packing
            // goes through the presentation map when one is present — and
            // read the presentation-packed source fixture when it exists.
            let (fixture, layout) = match kind {
                0 | 1 => (if presentation.is_some() { 4 } else { 0 }, Layout::Tower),
                2 => (1, Layout::Complete),
                3 => (2, Layout::Erasure),
                4 | 5 => (3, Layout::Cantor),
                _ => unreachable!(),
            };
            Arm {
                kind,
                layout,
                source: fixtures[fixture].iter().map(Buffer::bytes).collect(),
                destination: Buffer::new(shape.bytes * shape.destinations, offset),
            }
        })
        .collect();
    // Presentation arms pack canonical words through the presentation's
    // validated map; every native arm keeps its own layout encoding.
    let pack = |kind: usize, layout: Layout, dst: &mut [u8], canonical: &[u8]| match presentation {
        Some(presentation) if kind <= 1 => presentation.pack(dst, canonical),
        _ => maps.pack(dst, canonical, layout),
    };
    for bank in 0..plan.banks.len() {
        let expected = oracle.result(&plan, bank, &canonical_sources, &initial);
        for arm in &mut arms {
            let destination = arm.destination.bytes_mut();
            pack(arm.kind, arm.layout, destination, &initial);
            arm.apply(&plan, bank, native, simd);
            let mut packed = vec![0; expected.len()];
            pack(arm.kind, arm.layout, &mut packed, &expected);
            assert_eq!(
                arm.destination.bytes(),
                packed,
                "{} {:?} bank {bank} offset {offset}",
                NAMES[arm.kind],
                shape
            );
            let repeated = oracle.result(&plan, bank, &canonical_sources, &expected);
            arm.apply(&plan, bank, native, simd);
            pack(arm.kind, arm.layout, &mut packed, &repeated);
            assert_eq!(
                arm.destination.bytes(),
                packed,
                "repeated {} {:?}",
                NAMES[arm.kind],
                shape
            );
        }
    }
    if !timing {
        return;
    }
    for arm in &mut arms {
        for call in 0..32 {
            arm.apply(&plan, call, native, simd);
        }
    }
    // Calibrate each arm: generic presentations and native kernels can differ
    // by orders of magnitude, but each timed batch must amortize the clock.
    let target = Duration::from_micros(if smoke { 15 } else { 100 });
    let mut reps = Vec::with_capacity(arms.len());
    for arm in &mut arms {
        if arm.kind == 1 {
            reps.push(reps[0]);
            continue;
        }
        let mut batch = plan.banks.len();
        loop {
            let begin = Instant::now();
            for call in 0..batch {
                arm.apply(&plan, call, native, simd);
            }
            black_box(arm.destination.bytes());
            if begin.elapsed() >= target || batch == 1 << 20 {
                break;
            }
            batch *= 2;
        }
        reps.push(batch);
    }
    let orders = arm_orders(arms.len());
    let period = orders.len();
    let budget = Duration::from_millis(if smoke { 50 } else { 500 });
    // At least 30 rounds (one cycle in smoke) and at most 1200, both rounded
    // up to whole cycles, so every arm keeps its balanced positions.
    let min_rounds = if smoke {
        period
    } else {
        30usize.div_ceil(period) * period
    };
    let max_rounds = 1200usize.div_ceil(period) * period;
    let mut samples = vec![Vec::new(); arms.len()];
    let start = Instant::now();
    let mut round = 0;
    loop {
        for &index in &orders[round % period] {
            let reps = reps[index];
            for call in 0..reps {
                arms[index].apply(&plan, round * reps + call, native, simd);
            }
            let begin = Instant::now();
            for call in 0..reps {
                arms[index].apply(&plan, round * reps + call, native, simd);
            }
            let nanos = begin.elapsed().as_secs_f64() * 1e9 / reps as f64;
            black_box(arms[index].destination.bytes());
            samples[index].push(nanos);
        }
        round += 1;
        if round % period == 0
            && (round >= max_rounds || (round >= min_rounds && start.elapsed() >= budget))
        {
            break;
        }
    }
    for list in &mut samples {
        list.sort_unstable_by(f64::total_cmp);
    }
    if let Some(presentation) = presentation {
        presentations::emit_case_records(presentation, shape, &arms, &samples, &reps);
        return;
    }
    let control = quantile(&samples[1], 0.5) / quantile(&samples[0], 0.5);
    println!(
        "CONTROL\t{}\t{}\t{}\t{}\t{}\t{control:.6}",
        shape.label(),
        shape.sources,
        shape.destinations,
        shape.bytes,
        shape.schedule()
    );
    for ((arm, samples), &batch_reps) in arms.iter().zip(&samples).zip(&reps) {
        println!(
            "CALIBRATION\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            shape.label(),
            shape.sources,
            shape.destinations,
            shape.bytes,
            shape.schedule(),
            NAMES[arm.kind],
            batch_reps
        );
        let ns = quantile(samples, 0.5);
        let gib = (shape.bytes * shape.sources) as f64 / ns * 1e9 / 1073741824.0;
        println!(
            "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{ns:.6}\t{gib:.6}\t{:.6}\t{:.6}\t{}",
            shape.label(),
            shape.sources,
            shape.destinations,
            shape.bytes,
            shape.schedule(),
            NAMES[arm.kind],
            quantile(samples, 0.1),
            quantile(samples, 0.9),
            samples.len()
        );
    }
}

fn shapes(smoke: bool) -> Vec<Shape> {
    let mut result = Vec::new();
    for &cycling in &[false, true] {
        for &bytes in if smoke {
            &[4096][..]
        } else {
            &[64, 4096, 16384, 65536, 262144, 1048576, 8388608][..]
        } {
            result.push(Shape {
                operation: Operation::Assign,
                sources: 1,
                destinations: 1,
                bytes,
                cycling,
            });
        }
        for &bytes in if smoke {
            &[4096][..]
        } else {
            &[4096, 16384, 65536, 1048576, 8388608][..]
        } {
            for operation in [Operation::Into, Operation::Add] {
                result.push(Shape {
                    operation,
                    sources: 1,
                    destinations: 1,
                    bytes,
                    cycling,
                });
            }
        }
    }
    for &bytes in if smoke {
        &[4096][..]
    } else {
        &[4096, 16384, 65536][..]
    } {
        for operation in [Operation::Into, Operation::Add] {
            result.push(Shape {
                operation,
                sources: 16,
                destinations: 1,
                bytes,
                cycling: false,
            });
            for &destinations in if smoke { &[2][..] } else { &[2, 4, 6][..] } {
                result.push(Shape {
                    operation,
                    sources: 10,
                    destinations,
                    bytes,
                    cycling: false,
                });
            }
        }
    }
    result
}

fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        arguments
            .iter()
            .all(|arg| arg == "--check" || arg == "--smoke" || arg == "--presentations"),
        "supported options: --check, --smoke, --presentations"
    );
    let check = arguments.iter().any(|arg| arg == "--check");
    let smoke = arguments.iter().any(|arg| arg == "--smoke");
    let native = Native::new();
    let simd = Avx2::new();
    if arguments.iter().any(|arg| arg == "--presentations") {
        presentations::run(check, smoke, &native, &simd);
        return;
    }
    let maps = Representations::new(&native);
    let mut oracle = Oracle {
        products: HashMap::new(),
    };
    println!(
        "GF16 native field comparison; fgf={} backend={:?} field_backend={:?}; rss=3.1.0 AVX2; rse=6.0.0 scalar GF16; GF-Complete=1.03 {} SSSE3 split-4/16 polynomial=0x1100b; Leopard-RS 1.x (pre-Leopard2)={} AVX2",
        env!("FGF_VERSION"),
        fgf::backend(),
        fgf::backend_for::<Gf16>(),
        env!("COMPLETE_REVISION"),
        env!("LEOPARD_REVISION")
    );
    let mode = if check {
        "check"
    } else if smoke {
        "smoke"
    } else {
        "full"
    };
    println!("BENCHMARK_MODE\t{mode}");
    println!(
        "affinity={} scheduler={}",
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find(|line| line.starts_with("Cpus_allowed_list:"))
            .unwrap(),
        std::fs::read_to_string("/proc/self/sched")
            .unwrap()
            .lines()
            .find(|line| line.starts_with("policy"))
            .unwrap_or("unknown")
    );
    println!(
        "Native layouts; page-aligned buffers; setup, maps, logarithms and allocation excluded; source-byte throughput; single-row raw coefficients, prepared fgf multi-row coefficients; other multi-row arms repeat native regions; fixed or cycling16 coefficients; duplicate fgf control per case."
    );
    for operation in [Operation::Assign, Operation::Into, Operation::Add] {
        for &coefficient in &[0, 1, 0x100, 0x1234, 0xffff] {
            let shape = Shape {
                operation,
                sources: 1,
                destinations: 1,
                bytes: 131072,
                cycling: false,
            };
            run_case(
                Plan::<Gf16>::new(shape, &maps, None, Some(coefficient)),
                &maps,
                &native,
                &simd,
                &mut oracle,
                RunOptions {
                    timing: false,
                    smoke,
                    offset: 0,
                    exhaustive: true,
                },
                None,
            );
        }
        for bytes in [64, 128, 192, 320] {
            let shape = Shape {
                operation,
                sources: 1,
                destinations: 1,
                bytes,
                cycling: false,
            };
            run_case(
                Plan::<Gf16>::new(shape, &maps, None, None),
                &maps,
                &native,
                &simd,
                &mut oracle,
                RunOptions {
                    timing: false,
                    smoke,
                    offset: 16,
                    exhaustive: false,
                },
                None,
            );
        }
    }
    println!(
        "CHECK field maps: all 65536 values, all 256 basis products; native scalar oracles; all-value regions with zero/unit/full-field coefficients; repeated calls; legal tile boundaries and offset alignment passed."
    );
    for shape in shapes(smoke) {
        run_case(
            Plan::<Gf16>::new(shape, &maps, None, None),
            &maps,
            &native,
            &simd,
            &mut oracle,
            RunOptions {
                timing: !check,
                smoke,
                offset: 0,
                exhaustive: false,
            },
            None,
        );
    }
    println!(
        "CHECK every measured geometry and coefficient schedule passed exact differential validation."
    );
}
