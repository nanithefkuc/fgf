//! Paired-measurement harness for the stable vector API.
//!
//! Each registered cell is one public `fgf::ops` or `fgf::ops::bits` call
//! shape over deterministic fixtures. The harness never names
//! `fgf::internals`.
//!
//! ```sh
//! cargo bench --features std,simd --bench operations -- --inventory
//! cargo bench --features std,simd --bench operations -- --check
//! cargo bench --features std,simd --bench operations -- --time --rounds 5
//! cargo bench --features std,simd --bench operations -- --time --rounds 5 --cells cells.txt
//! ```
//!
//! Every mode prints `BACKEND` first and one `FIELD_BACKEND` line per
//! registered field label, then one `CELL` JSON record per cell in
//! `--inventory`, one `OUTPUT` digest per cell in `--check`, or a
//! `CONTROL`-bracketed round-robin stream of `ROUND` records in `--time`.
//! Setup, allocation, and coefficient construction happen outside the timed
//! region; a round's value is the median of an adaptive inner batch.
//!
//! Fixture placement is deterministic. Every allocation is page-aligned; a
//! destination starts 64 bytes into its page, the first source 512, a second
//! source 960, and an offset-addressed gather region 1408. The offsets are
//! multiples of 64 and distinct modulo 4096, so no destination and source
//! share a 4 KiB page offset. `offset{k}` adds `k` to the destination only.
//! Buffers of 2 MiB or more use 2 MiB-aligned storage. Library-owned
//! prepared coefficient tables are allocated page-aligned at offset 0,
//! distinct from every fixture role.

#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[path = "operations/cells.rs"]
mod cells;
#[path = "operations/fields.rs"]
mod fields;
#[path = "surfaces/fixture.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "operations/oracle.rs"]
mod oracle;
#[path = "operations/sha256.rs"]
mod sha256;

/// Every allocation in this bench is page-aligned: fixture buffers take
/// their role offsets inside one page, and library-owned prepared
/// coefficient tables begin at page offset 0, distinct from the fixture
/// destination and source offsets. Buffers at or above the streaming-store
/// size are aligned to the 2 MiB huge-page size instead of 4 KiB.
struct PageAligned;

unsafe impl GlobalAlloc for PageAligned {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.align() as *mut u8;
        }
        let align = if layout.size() >= cells::LARGE_ALIGN {
            cells::LARGE_ALIGN
        } else {
            cells::PAGE_BYTES
        }
        .max(layout.align());
        let layout = Layout::from_size_align(layout.size(), align)
            .expect("page-aligned benchmark allocation layout");
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let align = if layout.size() >= cells::LARGE_ALIGN {
            cells::LARGE_ALIGN
        } else {
            cells::PAGE_BYTES
        }
        .max(layout.align());
        let layout = Layout::from_size_align(layout.size(), align)
            .expect("page-aligned benchmark deallocation layout");
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static PAGE_ALLOCATOR: PageAligned = PageAligned;

use cells::Cell;

/// One adaptive inner batch targets this wall-clock length.
const BATCH_TARGET_NS: f64 = 100_000.0;
/// Upper bound on repetitions in one batch.
const MAX_BATCH_REPS: u64 = 1 << 20;
/// Minimum samples per cell per round.
const MIN_SAMPLES: usize = 5;
/// Maximum samples per cell per round.
const MAX_SAMPLES: usize = 9;
/// Time budget after the minimum sample count.
const SAMPLE_BUDGET: Duration = Duration::from_millis(20);
/// Logical byte count of the drift control's XOR pass.
const CONTROL_BYTES: usize = 256 * 1024;

/// Per-run mode handed to every cell body.
pub struct Ctx {
    /// `--check`: run the fixture once, verify it, and time nothing.
    pub check: bool,
}

impl Ctx {
    /// Time `body` with a warm-up, one calibration pass, and a median over
    /// an adaptive whole-batch sample series.
    pub fn measure<B: FnMut()>(&self, mut body: B) -> f64 {
        // Warm caches, branch predictors, and the lazily resolved backend.
        for _ in 0..2 {
            body();
        }
        let estimate = {
            let start = Instant::now();
            body();
            start.elapsed().as_secs_f64() * 1e9
        };
        let mut reps = if estimate > 5_000_000.0 {
            1
        } else {
            ((BATCH_TARGET_NS / estimate.max(1.0)).ceil() as u64).clamp(1, MAX_BATCH_REPS)
        };
        let target = Duration::from_micros(100);
        loop {
            let start = Instant::now();
            for _ in 0..reps {
                body();
            }
            let elapsed = start.elapsed();
            if elapsed >= target || reps == MAX_BATCH_REPS {
                break;
            }
            let scale = (target.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)).ceil() as u64;
            reps = reps.saturating_mul(scale.max(2)).min(MAX_BATCH_REPS);
        }

        let mut samples = Vec::with_capacity(MAX_SAMPLES);
        let pmu_before = pmu::read();
        let started = Instant::now();
        while samples.len() < MAX_SAMPLES {
            let start = Instant::now();
            for _ in 0..reps {
                body();
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / reps as f64);
            if samples.len() >= MIN_SAMPLES && started.elapsed() >= SAMPLE_BUDGET {
                break;
            }
        }
        pmu::report(
            &pmu_before,
            &pmu::read(),
            reps * samples.len() as u64,
            &samples,
        );
        samples.sort_unstable_by(f64::total_cmp);
        samples[samples.len() / 2]
    }
}

/// The drift control: a plain byte XOR over black-boxed buffers. The body
/// names no field code, so start/end movement measures the environment.
fn control_body(dst: &mut [u8], src: &[u8]) {
    let dst = black_box(dst);
    let src = black_box(src);
    for (left, &right) in dst.iter_mut().zip(src) {
        *left ^= right;
    }
}

fn control_ns(ctx: &Ctx) -> f64 {
    let src = fixture::noise(CONTROL_BYTES, 0x57a7);
    let mut dst = fixture::noise(CONTROL_BYTES, 0x57a8);
    ctx.measure(|| control_body(&mut dst, &src))
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

fn print_inventory(cells: &[Cell]) {
    for cell in cells {
        println!(
            "CELL\t{{\"id\":\"{}\",\"field\":\"{}\",\"operation\":\"{}\",\"prep\":\"{}\",\"row_len\":{},\"rows\":{},\"sources\":{},\"alignment\":\"{}\",\"bytes\":{}}}",
            json_escape(&cell.id),
            json_escape(cell.field),
            json_escape(cell.operation),
            json_escape(cell.prep),
            cell.row_len,
            cell.rows,
            cell.sources,
            json_escape(&cell.alignment),
            cell.bytes,
        );
    }
}

fn load_filter(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read cell filter {path}: {error}"));
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn apply_filter(cells: &mut Vec<Cell>, wanted: &[String]) {
    for id in wanted {
        assert!(
            cells.iter().any(|cell| &cell.id == id),
            "unknown cell id in --cells: {id}"
        );
    }
    cells.retain(|cell| wanted.iter().any(|id| id == &cell.id));
}

fn parse_args() -> (
    bool,
    bool,
    bool,
    Option<usize>,
    Option<String>,
    Option<String>,
) {
    let mut inventory = false;
    let mut check = false;
    let mut time = false;
    let mut rounds = None;
    let mut cells_path = None;
    let mut family = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--inventory" => inventory = true,
            "--check" => check = true,
            "--time" => time = true,
            // Cargo appends libtest-compatible flags; this harness is not
            // libtest, so accept and ignore them.
            "--bench" | "--test" => {}
            // The `just bench` wrappers add Criterion's baseline flags for
            // targets that use them; this harness has its own modes.
            "--baseline" | "--save-baseline" | "--load-baseline" | "--profile-time" => {
                args.next().expect("benchmark wrapper flag needs a value");
            }
            "--rounds" => {
                let value = args.next().expect("--rounds needs a value");
                rounds = Some(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("--rounds is not a count: {value}")),
                );
            }
            "--cells" => {
                cells_path = Some(args.next().expect("--cells needs a path"));
            }
            "--family" => {
                family = Some(args.next().expect("--family needs a value"));
            }
            other => panic!("unknown argument {other}"),
        }
    }
    if inventory && (check || time) || check && time {
        panic!("choose one of --inventory, --check, or --time");
    }
    if let Some(0) = rounds {
        panic!("--rounds must be at least one");
    }
    (inventory, check, time, rounds, cells_path, family)
}

fn main() {
    let (inventory, check, _time, rounds, cells_path, family) = parse_args();

    let mut cells = cells::registry();
    if let Some(family) = &family {
        cells.retain(|cell| cell.family == family);
        assert!(!cells.is_empty(), "no cells registered for family {family}");
    }
    if let Some(path) = &cells_path {
        let wanted = load_filter(path);
        apply_filter(&mut cells, &wanted);
    }

    println!("BACKEND\t{}", fgf::backend().name());
    fields::print_backends();

    if inventory {
        print_inventory(&cells);
        return;
    }

    let ctx = Ctx { check };
    if check {
        for cell in &cells {
            let _ = (cell.run)(cell, &ctx);
        }
        return;
    }

    let rounds = rounds.unwrap_or(1);
    println!("CONTROL\tstart\t{:.9}", control_ns(&ctx));
    for round in 0..rounds {
        for cell in &cells {
            let ns = (cell.run)(cell, &ctx).expect("timed cell returned no measurement");
            println!("ROUND\t{}\t{round}\t{ns:.9}", cell.id);
        }
    }
    println!("CONTROL\tend\t{:.9}", control_ns(&ctx));
}

/// Diagnostic per-round hardware counters (user space only), selected by
/// `FGF_PMU` as comma-separated raw `cpu_core` configs.
mod pmu {
    use std::sync::OnceLock;

    #[repr(C)]
    #[derive(Default)]
    struct Attr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup: u32,
        bp_type: u32,
        config1: u64,
        config2: u64,
        branch_sample_type: u64,
        sample_regs_user: u64,
        sample_stack_user: u32,
        clockid: i32,
        sample_regs_intr: u64,
        aux_watermark: u32,
        sample_max_stack: u16,
        reserved: u16,
        aux_sample_size: u32,
        reserved3: u32,
        sig_data: u64,
        config3: u64,
    }

    struct Counter {
        name: String,
        fd: i32,
    }

    fn counters() -> &'static [Counter] {
        static C: OnceLock<Vec<Counter>> = OnceLock::new();
        C.get_or_init(|| {
            let Ok(list) = std::env::var("FGF_PMU") else {
                return Vec::new();
            };
            let ty: u32 = std::fs::read_to_string("/sys/bus/event_source/devices/cpu_core/type")
                .or_else(|_| std::fs::read_to_string("/sys/bus/event_source/devices/cpu/type"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            list.split(',')
                .map(|item| {
                    let (name, hex) = item.split_once('=').unwrap();
                    let config = u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap();
                    let mut attr = Attr {
                        type_: ty,
                        size: std::mem::size_of::<Attr>() as u32,
                        config,
                        // TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING
                        read_format: 1 | 2,
                        // exclude_kernel (bit 5) | exclude_hv (bit 6)
                        flags: (1 << 5) | (1 << 6),
                        ..Attr::default()
                    };
                    let fd = unsafe {
                        libc::syscall(
                            libc::SYS_perf_event_open,
                            &mut attr as *mut Attr,
                            0,
                            -1,
                            -1,
                            0,
                        )
                    } as i32;
                    assert!(
                        fd >= 0,
                        "perf_event_open {name}: {}",
                        std::io::Error::last_os_error()
                    );
                    Counter {
                        name: name.to_string(),
                        fd,
                    }
                })
                .collect()
        })
    }

    pub fn read() -> Vec<[u64; 3]> {
        counters()
            .iter()
            .map(|c| {
                let mut v = [0u64; 3];
                let n = unsafe { libc::read(c.fd, v.as_mut_ptr().cast(), 24) };
                assert_eq!(n, 24);
                v
            })
            .collect()
    }

    pub fn report(before: &[[u64; 3]], after: &[[u64; 3]], calls: u64, samples: &[f64]) {
        if before.is_empty() {
            return;
        }
        let mut line = format!("PMU\tcalls={calls}");
        for ((c, b), a) in counters().iter().zip(before).zip(after) {
            let enabled = (a[1] - b[1]) as f64;
            let running = (a[2] - b[2]) as f64;
            let scaled = (a[0] - b[0]) as f64
                * if running > 0.0 {
                    enabled / running
                } else {
                    0.0
                };
            line.push_str(&format!("\t{}={:.3}", c.name, scaled / calls as f64));
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        line.push_str(&format!(
            "\tns_min={:.2}\tns_max={:.2}",
            sorted[0],
            sorted[sorted.len() - 1]
        ));
        println!("{line}");
    }
}
