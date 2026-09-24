//! In-process comparison of `fgf` against Plonky3 over the prime fields.
//!
//! Both libraries implement the same fields — GF(2^31 − 1),
//! GF(2^64 − 2^32 + 1), and their quadratic extension GF(p²) with
//! `i² = −1` — so every arm is a comparison of the same mathematical
//! result. Every shape is validated against the other library's output
//! before a single timing is taken.
//!
//! Each library runs on its own native layout: `fgf` on packed
//! little-endian byte buffers, Plonky3 on typed scalar slices packed
//! through `PackedValue`, and the quadratic extension on Plonky3's packed
//! extension representation. Layout conversion is excluded from every
//! timed region by construction, so the comparison is between kernels,
//! not encodings. Throughput counts one operand pair per element — twice
//! the region bytes — identically for both arms.
//!
//! The shapes are the recorded prime-field operations: scalar `a * b`,
//! scale overwrite `dst = c * src`, fused scale-accumulate
//! `dst += c * src`, elementwise product `dst = a * b`, and broadcast
//! `dst += v` / `dst -= v`.
//!
//! # Protocol
//!
//! The two implementations are interleaved inside one process against one
//! fixture set: each round takes a sample from both arms, and the order
//! alternates every round so that neither arm owns the warm side of a
//! thermal or frequency drift. A sample is the mean of 32 repetitions;
//! the reported figure is the median sample. Fixture construction,
//! buffer allocation, packing, and broadcast preparation all happen
//! outside the timed region — the timed closures allocate nothing.
//!
//! The first line of the table is a control: the same `fgf` body
//! registered as both arms, over two distinct destinations. Its ratio is
//! the harness's own noise floor and is expected at 1.00x; a control that
//! drifts invalidates every ratio below it.
//!
//! # Running
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release -- [gdl|m31]
//! ```
//!
//! Plonky3 selects its packed kernels with `cfg(target_feature)`, so the
//! package builds with `-C target-cpu=native`; the record states that
//! flag. Only the AVX2 tier is measured. The `gdl` selection runs the
//! Goldilocks rows, `m31` the Mersenne31 and QuadMersenne31 rows, and no
//! argument runs all three.

use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::Field as FgfField;
use fgf::{
    Goldilocks, Mersenne31, QuadMersenne31, backend, backend_for, goldilocks, mersenne31, ops,
    quad_mersenne31,
};
use p3_field::extension::Complex;
use p3_field::{
    ExtensionField, Field as P3Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing as _,
    PrimeField32 as _, PrimeField64 as _,
};
use p3_goldilocks::Goldilocks as P3Gld;
use p3_mersenne_31::Mersenne31 as P3M31;

/// The Plonky3 release this harness pins in `Cargo.toml`.
const P3_VERSION: &str = "0.7.0";

/// Every fixture payload starts on a page boundary. Equal page offsets
/// hold the 4 KiB store-to-load aliasing penalty identical for both arms;
/// with weaker alignment the control drifts depending on which buffers
/// the allocator placed opposite each other.
const ALIGN: usize = 4096;
const REGION_BYTES: usize = 64 * 1024;
const SCALAR_ITERS: usize = 1 << 20;

const WARMUP: usize = 16;
const REPS: u32 = 32;
const MAX_SAMPLES: usize = 64;
const BUDGET: Duration = Duration::from_millis(1500);

const LABELS: [&str; 5] = [
    "dst = c * src",
    "dst += c * src",
    "dst = a * b",
    "dst += v",
    "dst -= v",
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A buffer whose payload starts on a page boundary, built from an
/// over-allocated `Vec` so both arms share one alignment discipline.
struct Aligned<T> {
    backing: Vec<T>,
    start: usize,
    len: usize,
}

impl<T: Copy + Default> Aligned<T> {
    fn from_vec(data: Vec<T>) -> Self {
        let elem = size_of::<T>();
        assert!(
            elem.is_power_of_two() && elem <= ALIGN,
            "fixture element packs into one page",
        );
        assert_eq!(
            align_of::<T>(),
            elem,
            "fixture elements must be self-aligned to reach a page offset",
        );
        let mut backing = vec![T::default(); data.len() + ALIGN / elem + 2];
        let mut start = 0;
        while (backing.as_ptr() as usize + start * elem) % ALIGN != 0 {
            start += 1;
            assert!(start <= ALIGN / elem, "page offset is reachable in one page");
        }
        backing[start..start + data.len()].copy_from_slice(&data);
        Self {
            backing,
            start,
            len: data.len(),
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.backing[self.start..self.start + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.backing[self.start..self.start + self.len]
    }
}

/// One deterministic xorshift stream shared by every library.
fn seeds(n: usize, seed: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed | 1;
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push(s);
    }
    out
}

// ---------------------------------------------------------------------------
// Plonky3 scalar-slice shapes (Mersenne31, Goldilocks)
// ---------------------------------------------------------------------------

/// `dst = c * src` over `F`'s packed slices with a prepared broadcast.
fn p3_scale_into<F: P3Field>(dst: &mut [F], cb: F::Packing, src: &[F]) {
    for (d, s) in <F::Packing as PackedValue>::pack_slice_mut(dst)
        .iter_mut()
        .zip(<F::Packing as PackedValue>::pack_slice(src))
    {
        *d = *s * cb;
    }
}

/// `dst += c * src` as one fused packed loop.
fn p3_scale_add<F: P3Field>(dst: &mut [F], cb: F::Packing, src: &[F]) {
    for (d, s) in <F::Packing as PackedValue>::pack_slice_mut(dst)
        .iter_mut()
        .zip(<F::Packing as PackedValue>::pack_slice(src))
    {
        *d += *s * cb;
    }
}

/// `dst = a * b` over packed slices.
fn p3_elementwise<F: P3Field>(dst: &mut [F], a: &[F], b: &[F]) {
    for ((d, x), y) in <F::Packing as PackedValue>::pack_slice_mut(dst)
        .iter_mut()
        .zip(<F::Packing as PackedValue>::pack_slice(a))
        .zip(<F::Packing as PackedValue>::pack_slice(b))
    {
        *d = *x * *y;
    }
}

/// `dst += v` with one prepared broadcast.
fn p3_add_scalar<F: P3Field>(dst: &mut [F], cb: F::Packing) {
    for d in <F::Packing as PackedValue>::pack_slice_mut(dst) {
        *d += cb;
    }
}

/// `dst -= v` with one prepared broadcast.
fn p3_sub_scalar<F: P3Field>(dst: &mut [F], cb: F::Packing) {
    for d in <F::Packing as PackedValue>::pack_slice_mut(dst) {
        *d -= cb;
    }
}

// ---------------------------------------------------------------------------
// Plonky3 packed-extension shapes (GF(p²))
// ---------------------------------------------------------------------------

/// The packed representation of `GF(p²)` over `PackedMersenne31AVX2`:
/// `WIDTH` extension elements per value, coefficients transposed.
type Ext = Complex<P3M31>;
type PackedExt = <Ext as ExtensionField<P3M31>>::ExtensionPacking;

/// One packed extension value in a self-aligned slot.
///
/// `PackedExt` aligns to its coefficient packing, which is narrower than
/// its own size, so a `Vec` of it cannot be windowed to a page offset.
/// The wrapper raises alignment to the element size without changing
/// layout: field at offset zero, one packed value wide.
#[repr(align(64))]
#[derive(Clone, Copy, Default)]
struct PeSlot(PackedExt);

fn pe_scale_into(dst: &mut [PeSlot], cb: PackedExt, src: &[PeSlot]) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.0 = s.0 * cb;
    }
}

fn pe_scale_add(dst: &mut [PeSlot], cb: PackedExt, src: &[PeSlot]) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.0 += s.0 * cb;
    }
}

fn pe_elementwise(dst: &mut [PeSlot], a: &[PeSlot], b: &[PeSlot]) {
    for ((d, x), y) in dst.iter_mut().zip(a).zip(b) {
        d.0 = x.0 * y.0;
    }
}

fn pe_add_scalar(dst: &mut [PeSlot], v: Ext) {
    for d in dst.iter_mut() {
        d.0 += v;
    }
}

fn pe_sub_scalar(dst: &mut [PeSlot], v: Ext) {
    for d in dst.iter_mut() {
        d.0 -= v;
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Format a nanosecond count the way `Duration`'s debug output scales its
/// unit, but from an `f64`: `Duration` cannot hold sub-nanosecond values,
/// so dividing one by a repetition count quantizes every estimate to
/// whole nanoseconds.
fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.2}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

fn sample(body: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..REPS {
        body();
    }
    start.elapsed().as_secs_f64() * 1e9 / REPS as f64
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// The tenth and ninetieth percentile of the per-round paired ratios.
///
/// Each round times both arms back to back, so a paired ratio cancels the
/// drift the two samples share. The band is what says whether a ratio
/// near 1.00 is parity or noise.
fn spread(mut ratios: Vec<f64>) -> (f64, f64) {
    ratios.sort_unstable_by(f64::total_cmp);
    let last = ratios.len() - 1;
    (ratios[last / 10], ratios[last - last / 10])
}

fn throughput(bytes: usize, ns_per_call: f64) -> f64 {
    bytes as f64 / (ns_per_call / 1e9) / (1024.0 * 1024.0 * 1024.0)
}

/// Time two implementations of one shape, interleaved, and print the pair.
///
/// `bytes` is the logical volume one call processes over one operand pair,
/// so the two columns are comparable across shapes. The ratio is the
/// Plonky3 median divided by the `fgf` median: above 1.00 means `fgf` is
/// faster. `Scalar` rows report nanoseconds per element multiply instead
/// of throughput.
fn compare(
    label: &str,
    kind: RowKind,
    mut ours: impl FnMut(),
    mut theirs: impl FnMut(),
) {
    for _ in 0..WARMUP {
        ours();
        theirs();
    }

    let mut our_samples = Vec::with_capacity(MAX_SAMPLES);
    let mut their_samples = Vec::with_capacity(MAX_SAMPLES);
    let deadline = Instant::now() + BUDGET;
    let mut round = 0usize;
    while Instant::now() < deadline && our_samples.len() < MAX_SAMPLES {
        if round.is_multiple_of(2) {
            our_samples.push(sample(&mut ours));
            their_samples.push(sample(&mut theirs));
        } else {
            their_samples.push(sample(&mut theirs));
            our_samples.push(sample(&mut ours));
        }
        round += 1;
    }

    let paired: Vec<f64> = our_samples
        .iter()
        .zip(&their_samples)
        .map(|(ours, theirs)| theirs / ours)
        .collect();
    let our_median = median(our_samples);
    let their_median = median(their_samples);
    let ratio = their_median / our_median;
    let (low, high) = spread(paired);
    match kind {
        RowKind::Rate(bytes) => println!(
            "  {label:<34} {:>9} {:>7.2} GiB/s | {:>9} {:>7.2} GiB/s | \
             {ratio:>5.2}x [{low:.2}-{high:.2}]",
            fmt_ns(our_median),
            throughput(bytes, our_median),
            fmt_ns(their_median),
            throughput(bytes, their_median),
        ),
        RowKind::Scalar => {
            let rate = |ns: f64| SCALAR_ITERS as f64 * 1e3 / ns;
            println!(
                "  {label:<34} {:>8.2} ns/op {:>8.2} Mops/s | {:>8.2} ns/op {:>8.2} Mops/s | \
                 {ratio:>5.2}x [{low:.2}-{high:.2}]",
                our_median / SCALAR_ITERS as f64,
                rate(our_median),
                their_median / SCALAR_ITERS as f64,
                rate(their_median),
            )
        }
    }
}

/// What one timed call reports: operand-pair throughput, or the scalar
/// multiply loop's per-operation time.
#[derive(Clone, Copy)]
enum RowKind {
    Rate(usize),
    Scalar,
}

fn header() {
    println!(
        "fgf — process backend {}, m31 {}, gld {}, qm31 {}",
        backend().name(),
        backend_for::<Mersenne31>().name(),
        backend_for::<Goldilocks>().name(),
        backend_for::<QuadMersenne31>().name(),
    );
    println!(
        "Plonky3 {P3_VERSION} compile-time target features, built -C target-cpu=native, AVX2 tier only",
    );
    println!("ratio = Plonky3 median / fgf median; above 1.00x means fgf is faster");
    println!("[low-high] = 10th and 90th percentile of the per-round paired ratios");
    println!(
        "  {:<34} {:>23} | {:>23} | {:>6}",
        "shape", "fgf", "Plonky3", "ratio [band]"
    );
}

/// Fold an accumulator into one consumed `u64` so no scalar loop is dead.
fn sink(canon: u128) -> u64 {
    let folded = (canon ^ (canon >> 64)) as u64;
    black_box(folded)
}

/// The harness's own noise floor: one `fgf` body timed against itself
/// over two distinct destinations. Anything other than 1.00x is
/// interleaving bias.
fn control<Fg: fgf::FieldKernels>(
    c: Fg::Elem,
    src: &Aligned<u8>,
    first: &mut Aligned<u8>,
    second: &mut Aligned<u8>,
) {
    compare(
        "control: mul_into vs itself",
        RowKind::Rate(2 * REGION_BYTES),
        || {
            ops::mul_into::<Fg>(
                black_box(first.as_mut_slice()),
                c,
                black_box(src.as_slice()),
            );
        },
        || {
            ops::mul_into::<Fg>(
                black_box(second.as_mut_slice()),
                c,
                black_box(src.as_slice()),
            );
        },
    );
}

// ---------------------------------------------------------------------------
// Scalar-slice fields: Mersenne31 and Goldilocks
// ---------------------------------------------------------------------------

/// One scalar-slice field's rows: fixtures on both native layouts, every
/// arm validated against the other library's output, then the six paired
/// shapes.
///
/// `draw` maps one raw stream word to the same logical element on both
/// sides; `canon_fg` and `canon_p` map each library's element to one
/// comparable key.
fn run_slice_field<Fg, Fp>(
    name: &str,
    control_row: bool,
    draw: impl Fn(u64) -> (Fg::Elem, Fp),
    canon_fg: impl Fn(Fg::Elem) -> u128,
    canon_p: impl Fn(Fp) -> u128,
) where
    Fg: fgf::FieldKernels,
    Fg::Elem: Copy + core::ops::Add<Output = Fg::Elem> + core::ops::Mul<Output = Fg::Elem>,
    Fp: P3Field,
{
    let elem_bytes = <Fg as FgfField>::BYTES;
    let n = REGION_BYTES / elem_bytes;

    let raws = |count: usize, seed: u64| seeds(count, seed);
    let elems = |count: usize, seed: u64| -> (Vec<Fg::Elem>, Vec<Fp>) {
        raws(count, seed).into_iter().map(&draw).unzip()
    };

    let (src_fg, src_p) = elems(n, 0x600);
    let (a_fg, a_p) = elems(n, 0x610);
    let (b_fg, b_p) = elems(n, 0x611);
    let mut dst_slots = Vec::new();
    for slot in 0..LABELS.len() {
        dst_slots.push(elems(n, 0x620 + slot as u64));
    }
    let (c_fg, c_p) = draw(0x5555_5555_5555_5553);
    let (v_fg, v_p) = draw(0x2AAA_AAAA);

    // `fgf` native layout: packed little-endian bytes.
    let pack = |elems: &[Fg::Elem]| -> Aligned<u8> {
        let mut bytes = vec![0u8; n * elem_bytes];
        ops::pack::<Fg>(&mut bytes, elems);
        Aligned::from_vec(bytes)
    };
    let src_f = pack(&src_fg);
    let a_f = pack(&a_fg);
    let b_f = pack(&b_fg);
    let mut d_f: Vec<Aligned<u8>> = dst_slots.iter().map(|(e, _)| pack(e)).collect();

    // Plonky3 native layout: typed scalar slices.
    let src_v = Aligned::from_vec(src_p.clone());
    let a_v = Aligned::from_vec(a_p.clone());
    let b_v = Aligned::from_vec(b_p.clone());
    let mut d_v: Vec<Aligned<Fp>> = dst_slots.iter().map(|(_, e)| Aligned::from_vec(e.clone())).collect();

    // Prepared broadcasts, outside every timed region.
    let cb = <Fp::Packing as PackedValue>::broadcast(c_p);
    let vb = <Fp::Packing as PackedValue>::broadcast(v_p);

    // Untimed validation: both libraries' outputs must agree elementwise.
    let mut check_f = d_f[0].as_slice().to_vec();
    ops::mul_into::<Fg>(&mut check_f, c_fg, src_f.as_slice());
    let mut check_v = d_v[0].as_slice().to_vec();
    p3_scale_into::<Fp>(&mut check_v, cb, src_v.as_slice());
    assert_canon::<Fg, Fp>(&format!("{name} {}", LABELS[0]), &check_f, &check_v, &canon_fg, &canon_p, n, elem_bytes);

    let mut check_f = d_f[1].as_slice().to_vec();
    ops::mul_add::<Fg>(&mut check_f, c_fg, src_f.as_slice());
    let mut check_v = d_v[1].as_slice().to_vec();
    p3_scale_add::<Fp>(&mut check_v, cb, src_v.as_slice());
    assert_canon::<Fg, Fp>(&format!("{name} {}", LABELS[1]), &check_f, &check_v, &canon_fg, &canon_p, n, elem_bytes);

    let mut check_f = vec![0u8; n * elem_bytes];
    ops::mul_elementwise::<Fg>(&mut check_f, a_f.as_slice(), b_f.as_slice());
    let mut check_v = vec![Fp::ZERO; n];
    p3_elementwise::<Fp>(&mut check_v, a_v.as_slice(), b_v.as_slice());
    assert_canon::<Fg, Fp>(&format!("{name} {}", LABELS[2]), &check_f, &check_v, &canon_fg, &canon_p, n, elem_bytes);

    let mut check_f = d_f[3].as_slice().to_vec();
    ops::add_assign_scalar::<Fg>(&mut check_f, v_fg);
    let mut check_v = d_v[3].as_slice().to_vec();
    p3_add_scalar::<Fp>(&mut check_v, vb);
    assert_canon::<Fg, Fp>(&format!("{name} {}", LABELS[3]), &check_f, &check_v, &canon_fg, &canon_p, n, elem_bytes);

    let mut check_f = d_f[4].as_slice().to_vec();
    ops::sub_assign_scalar::<Fg>(&mut check_f, v_fg);
    let mut check_v = d_v[4].as_slice().to_vec();
    p3_sub_scalar::<Fp>(&mut check_v, vb);
    assert_canon::<Fg, Fp>(&format!("{name} {}", LABELS[4]), &check_f, &check_v, &canon_fg, &canon_p, n, elem_bytes);

    if control_row {
        let mut second = pack(&dst_slots[0].0);
        control::<Fg>(c_fg, &src_f, &mut d_f[0], &mut second);
    }

    println!("{name} — {REGION_BYTES} B regions, {elem_bytes} B lanes");
    println!(
        "[Plonky3 packing width: {}]",
        <Fp::Packing as PackedValue>::WIDTH
    );

    compare("scalar a * b", RowKind::Scalar, || {
        let mut acc = Fg::Elem::default();
        for i in 0..SCALAR_ITERS {
            let x = src_fg[(i.wrapping_mul(7919)) & (n - 1)];
            let y = src_fg[(i.wrapping_mul(104729)) & (n - 1)];
            acc = acc + x * y;
        }
        sink(canon_fg(acc));
    }, || {
        let mut acc = Fp::ZERO;
        for i in 0..SCALAR_ITERS {
            let x = src_p[(i.wrapping_mul(7919)) & (n - 1)];
            let y = src_p[(i.wrapping_mul(104729)) & (n - 1)];
            acc += x * y;
        }
        sink(canon_p(acc));
    });

    let bytes = RowKind::Rate(2 * REGION_BYTES);
    compare(LABELS[0], bytes, || {
        ops::mul_into::<Fg>(black_box(d_f[0].as_mut_slice()), c_fg, black_box(src_f.as_slice()));
    }, || {
        p3_scale_into::<Fp>(black_box(d_v[0].as_mut_slice()), cb, black_box(src_v.as_slice()));
    });
    compare(LABELS[1], bytes, || {
        ops::mul_add::<Fg>(black_box(d_f[1].as_mut_slice()), c_fg, black_box(src_f.as_slice()));
    }, || {
        p3_scale_add::<Fp>(black_box(d_v[1].as_mut_slice()), cb, black_box(src_v.as_slice()));
    });
    compare(LABELS[2], bytes, || {
        ops::mul_elementwise::<Fg>(
            black_box(d_f[2].as_mut_slice()),
            black_box(a_f.as_slice()),
            black_box(b_f.as_slice()),
        );
    }, || {
        p3_elementwise::<Fp>(
            black_box(d_v[2].as_mut_slice()),
            black_box(a_v.as_slice()),
            black_box(b_v.as_slice()),
        );
    });
    compare(LABELS[3], bytes, || {
        ops::add_assign_scalar::<Fg>(black_box(d_f[3].as_mut_slice()), v_fg);
    }, || {
        p3_add_scalar::<Fp>(black_box(d_v[3].as_mut_slice()), vb);
    });
    compare(LABELS[4], bytes, || {
        ops::sub_assign_scalar::<Fg>(black_box(d_f[4].as_mut_slice()), v_fg);
    }, || {
        p3_sub_scalar::<Fp>(black_box(d_v[4].as_mut_slice()), vb);
    });
}

/// Compare one shape's `fgf` bytes against Plonky3's elements as
/// canonical keys, panicking on the first disagreement.
fn assert_canon<Fg, Fp>(
    label: &str,
    fg_bytes: &[u8],
    p3_elems: &[Fp],
    canon_fg: &impl Fn(Fg::Elem) -> u128,
    canon_p: &impl Fn(Fp) -> u128,
    n: usize,
    elem_bytes: usize,
) where
    Fg: fgf::FieldKernels,
    Fg::Elem: Copy + Default,
    Fp: Copy,
{
    let mut elems = vec![Fg::Elem::default(); n];
    ops::unpack::<Fg>(&mut elems, fg_bytes);
    assert_eq!(fg_bytes.len(), n * elem_bytes, "whole elements");
    for (index, (fg, p3)) in elems.iter().zip(p3_elems).enumerate() {
        assert_eq!(
            canon_fg(*fg),
            canon_p(*p3),
            "{label}: libraries disagree at element {index}",
        );
    }
}

fn run_mersenne31(control_row: bool) {
    run_slice_field::<Mersenne31, P3M31>(
        "Mersenne31",
        control_row,
        |raw| {
            let lane = (raw % mersenne31::MODULUS as u64) as u32;
            (
                mersenne31::Elem::from_raw(lane),
                P3M31::new(lane),
            )
        },
        |e| u128::from(e.canonical().to_raw()),
        |e| u128::from(e.as_canonical_u32()),
    );
}

fn run_goldilocks(control_row: bool) {
    run_slice_field::<Goldilocks, P3Gld>(
        "Goldilocks",
        control_row,
        |raw| {
            let lane = raw % goldilocks::MODULUS;
            (
                goldilocks::Elem::from_raw(lane),
                P3Gld::new(lane),
            )
        },
        |e| u128::from(e.canonical().to_raw()),
        |e| u128::from(e.to_unique_u64()),
    );
}

// ---------------------------------------------------------------------------
// GF(p²): QuadMersenne31 and Plonky3's packed complex
// ---------------------------------------------------------------------------

fn run_quadratic(control_row: bool) {
    let elem_bytes = <QuadMersenne31 as FgfField>::BYTES;
    let n = REGION_BYTES / elem_bytes;
    let width = <P3M31 as P3Field>::Packing::WIDTH;
    assert_eq!(n % width, 0, "whole packed extension groups");

    let elems = |count: usize, seed: u64| -> (Vec<quad_mersenne31::Elem>, Vec<Ext>) {
        let raw = seeds(2 * count, seed);
        (0..count)
            .map(|i| {
                let re = (raw[2 * i] % mersenne31::MODULUS as u64) as u32;
                let im = (raw[2 * i + 1] % mersenne31::MODULUS as u64) as u32;
                (
                    quad_mersenne31::Elem::from_raw(re, im),
                    Complex::<P3M31>::new_complex(P3M31::new(re), P3M31::new(im)),
                )
            })
            .unzip()
    };

    let (src_fg, src_p) = elems(n, 0x700);
    let (a_fg, a_p) = elems(n, 0x710);
    let (b_fg, b_p) = elems(n, 0x711);
    let mut dst_slots = Vec::new();
    for slot in 0..LABELS.len() {
        dst_slots.push(elems(n, 0x720 + slot as u64));
    }
    let (c_fg, c_p) = {
        let raw = seeds(2, 0x7F0);
        let re = (raw[0] % mersenne31::MODULUS as u64) as u32;
        let im = (raw[1] % mersenne31::MODULUS as u64) as u32;
        (
            quad_mersenne31::Elem::from_raw(re, im),
            Complex::<P3M31>::new_complex(P3M31::new(re), P3M31::new(im)),
        )
    };
    let (v_fg, v_p) = {
        let raw = seeds(2, 0x7F1);
        let re = (raw[0] % mersenne31::MODULUS as u64) as u32;
        let im = (raw[1] % mersenne31::MODULUS as u64) as u32;
        (
            quad_mersenne31::Elem::from_raw(re, im),
            Complex::<P3M31>::new_complex(P3M31::new(re), P3M31::new(im)),
        )
    };

    // `fgf` native layout: interleaved little-endian limb pairs.
    let pack = |elems: &[quad_mersenne31::Elem]| -> Aligned<u8> {
        let mut bytes = vec![0u8; n * elem_bytes];
        ops::pack::<QuadMersenne31>(&mut bytes, elems);
        Aligned::from_vec(bytes)
    };
    let src_f = pack(&src_fg);
    let a_f = pack(&a_fg);
    let b_f = pack(&b_fg);
    let mut d_f: Vec<Aligned<u8>> = dst_slots.iter().map(|(e, _)| pack(e)).collect();

    // Plonky3 native layout: the packed extension representation its
    // transforms keep — `WIDTH` extension elements per value.
    let to_packed = |elems: &[Ext]| -> Vec<PeSlot> {
        elems
            .chunks_exact(width)
            .map(|chunk| PeSlot(PackedExt::from_ext_slice(chunk)))
            .collect()
    };
    let src_v = Aligned::from_vec(to_packed(&src_p));
    let a_v = Aligned::from_vec(to_packed(&a_p));
    let b_v = Aligned::from_vec(to_packed(&b_p));
    let mut d_v: Vec<Aligned<PeSlot>> = dst_slots
        .iter()
        .map(|(_, e)| Aligned::from_vec(to_packed(e)))
        .collect();

    let cb = PackedExt::from(c_p);

    let canon_fg = |e: quad_mersenne31::Elem| -> u128 {
        let (re, im) = e.canonical().to_raw();
        (u128::from(im) << 32) | u128::from(re)
    };
    let canon_p = |e: Ext| -> u128 {
        (u128::from(e.imag().as_canonical_u32()) << 32) | u128::from(e.real().as_canonical_u32())
    };
    let unpack_packed = |region: &[PeSlot]| -> Vec<Ext> {
        region
            .iter()
            .flat_map(|pe| (0..width).map(move |lane| pe.0.extract(lane)))
            .collect()
    };

    let validate = |slot: usize, label: &str, fg_body: &dyn Fn(&mut [u8]), p3_body: &dyn Fn(&mut [PeSlot])| {
        let mut check_f = d_f[slot].as_slice().to_vec();
        fg_body(&mut check_f);
        let mut check_v = d_v[slot].as_slice().to_vec();
        p3_body(&mut check_v);
        assert_canon::<QuadMersenne31, Ext>(
            &format!("QuadMersenne31 {label}"),
            &check_f,
            &unpack_packed(&check_v),
            &canon_fg,
            &canon_p,
            n,
            elem_bytes,
        );
    };

    validate(0, LABELS[0], &|d| ops::mul_into::<QuadMersenne31>(d, c_fg, src_f.as_slice()), &|d| {
        pe_scale_into(d, cb, src_v.as_slice())
    });
    validate(1, LABELS[1], &|d| ops::mul_add::<QuadMersenne31>(d, c_fg, src_f.as_slice()), &|d| {
        pe_scale_add(d, cb, src_v.as_slice())
    });
    {
        let mut check_f = vec![0u8; n * elem_bytes];
        ops::mul_elementwise::<QuadMersenne31>(&mut check_f, a_f.as_slice(), b_f.as_slice());
        let mut check_v = vec![PeSlot::default(); d_v[2].as_slice().len()];
        pe_elementwise(&mut check_v, a_v.as_slice(), b_v.as_slice());
        assert_canon::<QuadMersenne31, Ext>(
            &format!("QuadMersenne31 {}", LABELS[2]),
            &check_f,
            &unpack_packed(&check_v),
            &canon_fg,
            &canon_p,
            n,
            elem_bytes,
        );
    }
    validate(3, LABELS[3], &|d| ops::add_assign_scalar::<QuadMersenne31>(d, v_fg), &|d| {
        pe_add_scalar(d, v_p)
    });
    validate(4, LABELS[4], &|d| ops::sub_assign_scalar::<QuadMersenne31>(d, v_fg), &|d| {
        pe_sub_scalar(d, v_p)
    });

    if control_row {
        let mut second = pack(&dst_slots[0].0);
        control::<QuadMersenne31>(c_fg, &src_f, &mut d_f[0], &mut second);
    }

    println!("QuadMersenne31 — {REGION_BYTES} B regions, {elem_bytes} B lanes");
    println!(
        "[Plonky3 packing width: {} extension elements]",
        width
    );

    compare("scalar a * b", RowKind::Scalar, || {
        let mut acc = quad_mersenne31::Elem::ZERO;
        for i in 0..SCALAR_ITERS {
            let x = src_fg[(i.wrapping_mul(7919)) & (n - 1)];
            let y = src_fg[(i.wrapping_mul(104729)) & (n - 1)];
            acc = acc + x * y;
        }
        sink(canon_fg(acc));
    }, || {
        let mut acc = Ext::ZERO;
        for i in 0..SCALAR_ITERS {
            let x = src_p[(i.wrapping_mul(7919)) & (n - 1)];
            let y = src_p[(i.wrapping_mul(104729)) & (n - 1)];
            acc += x * y;
        }
        sink(canon_p(acc));
    });

    let bytes = RowKind::Rate(2 * REGION_BYTES);
    compare(LABELS[0], bytes, || {
        ops::mul_into::<QuadMersenne31>(
            black_box(d_f[0].as_mut_slice()),
            c_fg,
            black_box(src_f.as_slice()),
        );
    }, || {
        pe_scale_into(black_box(d_v[0].as_mut_slice()), cb, black_box(src_v.as_slice()));
    });
    compare(LABELS[1], bytes, || {
        ops::mul_add::<QuadMersenne31>(
            black_box(d_f[1].as_mut_slice()),
            c_fg,
            black_box(src_f.as_slice()),
        );
    }, || {
        pe_scale_add(black_box(d_v[1].as_mut_slice()), cb, black_box(src_v.as_slice()));
    });
    compare(LABELS[2], bytes, || {
        ops::mul_elementwise::<QuadMersenne31>(
            black_box(d_f[2].as_mut_slice()),
            black_box(a_f.as_slice()),
            black_box(b_f.as_slice()),
        );
    }, || {
        pe_elementwise(
            black_box(d_v[2].as_mut_slice()),
            black_box(a_v.as_slice()),
            black_box(b_v.as_slice()),
        );
    });
    compare(LABELS[3], bytes, || {
        ops::add_assign_scalar::<QuadMersenne31>(black_box(d_f[3].as_mut_slice()), v_fg);
    }, || {
        pe_add_scalar(black_box(d_v[3].as_mut_slice()), v_p);
    });
    compare(LABELS[4], bytes, || {
        ops::sub_assign_scalar::<QuadMersenne31>(black_box(d_f[4].as_mut_slice()), v_fg);
    }, || {
        pe_sub_scalar(black_box(d_v[4].as_mut_slice()), v_p);
    });
}

fn main() {
    let family = std::env::args().nth(1);
    let selection = match family.as_deref() {
        None | Some("all") => "all",
        Some("gdl") => "gdl",
        Some("m31") => "m31",
        Some(other) => {
            eprintln!("unknown family {other:?}; expected gdl, m31, or no argument");
            std::process::exit(2);
        }
    };

    header();
    match selection {
        "gdl" => run_goldilocks(true),
        "m31" => {
            run_mersenne31(true);
            run_quadratic(false);
        }
        _ => {
            run_mersenne31(true);
            run_goldilocks(false);
            run_quadratic(false);
        }
    }
}
