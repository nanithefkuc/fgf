//! Interleaved native GF16 field operations with exact representation validation.
//!
//! Preparation and conversion are excluded. Multi-row competitor arms compose
//! their native region primitives; unsupported primitives are not substituted.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::{Elem, Gf16, ops};
use reed_solomon_erasure::{Field, galois_16};
use reed_solomon_simd::engine::{Avx2, Engine, tables};

mod native;
mod representation;

use native::Native;
use representation::{Layout, Representations, tower_mul};

const ALIGN: usize = 4096;
const NAMES: [&str; 6] = [
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Assign,
    Into,
    Add,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    operation: Operation,
    sources: usize,
    destinations: usize,
    bytes: usize,
    cycling: bool,
}

impl Shape {
    fn label(self) -> &'static str {
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
    fn schedule(self) -> &'static str {
        if self.cycling { "cycling16" } else { "fixed" }
    }
}

struct Bank {
    canonical: Vec<Elem<Gf16>>,
    complete: Vec<u16>,
    erasure: Vec<u16>,
    logs: Vec<u16>,
    matrix: ops::CoeffMatrix<Gf16>,
    gather: ops::CoeffVec<Gf16>,
}

struct Plan {
    shape: Shape,
    banks: Vec<Bank>,
}

impl Plan {
    fn new(shape: Shape, maps: &Representations, special: Option<u16>) -> Self {
        let banks = (0..if shape.cycling { 16 } else { 1 })
            .map(|bank| {
                let canonical: Vec<_> = (0..shape.sources * shape.destinations)
                    .map(|index| {
                        let mut value = (0x1234usize + index * 0x2311 + bank * 0x19ab) as u16;
                        if value >> 8 == 0 {
                            value |= 0x100;
                        }
                        Elem::<Gf16>::from_raw(special.unwrap_or(value))
                    })
                    .collect();
                let map = |layout| {
                    canonical
                        .iter()
                        .map(|c| maps.element(c.to_raw(), layout))
                        .collect::<Vec<_>>()
                };
                let logs = map(Layout::Cantor)
                    .iter()
                    .map(|&c| tables::get_exp_log().log[usize::from(c)])
                    .collect();
                let matrix = ops::CoeffMatrix::from_source_major(
                    shape.sources,
                    shape.destinations,
                    &canonical,
                );
                let gather = ops::CoeffVec::new(&canonical);
                Bank {
                    complete: map(Layout::Complete),
                    erasure: map(Layout::Erasure),
                    logs,
                    matrix,
                    gather,
                    canonical,
                }
            })
            .collect();
        Self { shape, banks }
    }
}

struct Arm<'a> {
    kind: usize,
    layout: Layout,
    source: Vec<&'a [u8]>,
    destination: Buffer,
}

impl Arm<'_> {
    fn apply(&mut self, plan: &Plan, bank_index: usize, native: &Native, simd: &Avx2) {
        let shape = plan.shape;
        let bank = &plan.banks[bank_index % plan.banks.len()];
        let dst = black_box(self.destination.bytes_mut());
        let sources = black_box(self.source.as_slice());
        if self.kind <= 1 {
            let coefficient = bank.canonical[0];
            match (shape.operation, shape.sources, shape.destinations) {
                (Operation::Assign, _, _) => ops::mul_assign::<Gf16>(dst, black_box(coefficient)),
                (Operation::Into, 1, 1) => {
                    ops::mul_into::<Gf16>(dst, black_box(coefficient), sources[0])
                }
                (Operation::Add, 1, 1) => {
                    ops::mul_add::<Gf16>(dst, black_box(coefficient), sources[0])
                }
                (Operation::Into, _, 1) => {
                    ops::mul_into_gather_with::<Gf16>(dst, bank.gather.as_ref(), sources)
                }
                (Operation::Add, _, 1) => {
                    ops::mul_add_gather_with::<Gf16>(dst, bank.gather.as_ref(), sources)
                }
                (Operation::Into, _, _) => {
                    ops::mul_into_matrix_with::<Gf16>(dst, shape.bytes, &bank.matrix, sources)
                }
                (Operation::Add, _, _) => {
                    ops::mul_add_matrix_with::<Gf16>(dst, shape.bytes, &bank.matrix, sources)
                }
            }
            return;
        }
        if shape.operation == Operation::Assign {
            match self.kind {
                2 => native.complete_assign(dst, bank.complete[0]),
                4 => {
                    if bank.canonical[0] == Elem::<Gf16>::ZERO {
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
                        if bank.canonical[index] == Elem::<Gf16>::ZERO {
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
    fn result(
        &mut self,
        plan: &Plan,
        bank_index: usize,
        source: &[Vec<u8>],
        initial: &[u8],
    ) -> Vec<u8> {
        let shape = plan.shape;
        let coefficients = &plan.banks[bank_index].canonical;
        for c in coefficients {
            self.products
                .entry(c.to_raw())
                .or_insert_with(|| (0..=u16::MAX).map(|x| tower_mul(x, c.to_raw())).collect());
        }
        let mut output = if shape.operation == Operation::Into {
            vec![0; initial.len()]
        } else {
            initial.to_vec()
        };
        if shape.operation == Operation::Assign {
            let table = &self.products[&coefficients[0].to_raw()];
            for pair in output.as_chunks_mut::<2>().0 {
                *pair = table[usize::from(u16::from_le_bytes(*pair))].to_le_bytes();
            }
            return output;
        }
        for (term, src) in source.iter().enumerate() {
            for row in 0..shape.destinations {
                let table = &self.products[&coefficients[term * shape.destinations + row].to_raw()];
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

fn quantile(samples: &[f64], q: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * q).round() as usize]
}

fn arm_orders(count: usize) -> Vec<Vec<usize>> {
    fn visit(order: &mut [usize], position: usize, result: &mut Vec<Vec<usize>>) {
        if position == order.len() {
            result.push(order.to_vec());
            return;
        }
        for index in position..order.len() {
            order.swap(position, index);
            visit(order, position + 1, result);
            order.swap(position, index);
        }
    }
    let mut result = Vec::new();
    visit(&mut (0..count).collect::<Vec<_>>(), 0, &mut result);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for index in (1..result.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        result.swap(index, (state as usize) % (index + 1));
    }
    result
}

struct RunOptions {
    timing: bool,
    smoke: bool,
    offset: usize,
    exhaustive: bool,
}

fn run_case(
    plan: Plan,
    maps: &Representations,
    native: &Native,
    simd: &Avx2,
    oracle: &mut Oracle,
    options: RunOptions,
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
    let fixtures: Vec<Vec<Buffer>> = layouts
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
    let kinds: Vec<_> = if shape.operation == Operation::Assign {
        vec![0, 1, 2, 4]
    } else {
        vec![0, 1, 2, 3, 5]
    };
    let mut arms: Vec<_> = kinds
        .into_iter()
        .map(|kind| {
            let fixture = match kind {
                0 | 1 => 0,
                2 => 1,
                3 => 2,
                4 | 5 => 3,
                _ => unreachable!(),
            };
            Arm {
                kind,
                layout: layouts[fixture],
                source: fixtures[fixture].iter().map(Buffer::bytes).collect(),
                destination: Buffer::new(shape.bytes * shape.destinations, offset),
            }
        })
        .collect();
    for bank in 0..plan.banks.len() {
        let expected = oracle.result(&plan, bank, &canonical_sources, &initial);
        for arm in &mut arms {
            maps.pack(arm.destination.bytes_mut(), &initial, arm.layout);
            arm.apply(&plan, bank, native, simd);
            let mut packed = vec![0; expected.len()];
            maps.pack(&mut packed, &expected, arm.layout);
            assert_eq!(
                arm.destination.bytes(),
                packed,
                "{} {:?} bank {bank} offset {offset}",
                NAMES[arm.kind],
                shape
            );
            let repeated = oracle.result(&plan, bank, &canonical_sources, &expected);
            arm.apply(&plan, bank, native, simd);
            maps.pack(&mut packed, &repeated, arm.layout);
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
    let reps = if shape.bytes >= 1024 * 1024 {
        1
    } else if shape.bytes >= 64 * 1024 {
        8
    } else {
        32
    };
    let orders = arm_orders(arms.len());
    let period = orders.len();
    let budget = Duration::from_millis(if smoke { 50 } else { 500 });
    let mut samples = vec![Vec::new(); arms.len()];
    let start = Instant::now();
    let mut round = 0;
    loop {
        for &index in &orders[round % period] {
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
            && ((round >= period * 3 && start.elapsed() >= budget) || round >= 1200)
        {
            break;
        }
    }
    for list in &mut samples {
        list.sort_unstable_by(f64::total_cmp);
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
    for (arm, samples) in arms.iter().zip(&samples) {
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
            .all(|arg| arg == "--check" || arg == "--smoke"),
        "supported options: --check, --smoke"
    );
    let check = arguments.iter().any(|arg| arg == "--check");
    let smoke = arguments.iter().any(|arg| arg == "--smoke");
    let native = Native::new();
    let simd = Avx2::new();
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
                Plan::new(shape, &maps, Some(coefficient)),
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
                Plan::new(shape, &maps, None),
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
            );
        }
    }
    println!(
        "CHECK field maps: all 65536 values, all 256 basis products; native scalar oracles; all-value regions with zero/unit/full-field coefficients; repeated calls; legal tile boundaries and offset alignment passed."
    );
    for shape in shapes(smoke) {
        run_case(
            Plan::new(shape, &maps, None),
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
        );
    }
    println!(
        "CHECK every measured geometry and coefficient schedule passed exact differential validation."
    );
}
