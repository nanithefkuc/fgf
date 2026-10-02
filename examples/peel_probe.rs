//! Offset-sweep and overwrite-gather probe for the half-lane peel and
//! gather-dispatch records in `benchmarks/gf8.md` and `benchmarks/gf16.md`.
//!
//! ```sh
//! taskset -c <core> cargo run --release --all-features --example peel_probe
//! ```
//!
//! The `Gf16` sweep places destination and source at the same base offset
//! modulo the cache-line width, so the 16-mod-64 rows exercise the
//! half-lane peel and the 0-mod-64 rows are the aligned control. The
//! gather section times the raw-coefficient and prepared `CoeffVec`
//! overwrite forms over sixteen sources, counting source bytes once per
//! call. Timing is the median of five calibrated rounds after warm-up.

use std::hint::black_box;
use std::time::Instant;

use fgf::{FieldKernels, Gf8B, Gf8D, Gf16, backend, gf8b, gf8d, gf16, ops};

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

/// Median per-iteration nanoseconds over five rounds after warm-up, with a
/// calibrated round length so every shape spends comparable wall time.
fn time_ns(mut body: impl FnMut()) -> f64 {
    for _ in 0..16 {
        body();
    }
    let t = Instant::now();
    body();
    let single = t.elapsed().as_secs_f64().max(1e-9);
    let iters = ((0.03 / single) as u32).clamp(10, 200_000);
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            body();
        }
        samples.push(t.elapsed().as_secs_f64() / f64::from(iters));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[2] * 1e9
}

fn gib_per_s(bytes: usize, ns: f64) -> f64 {
    bytes as f64 / ns / 1.073_741_824
}

/// Buffer whose `[start, start + len)` window begins at `off` modulo 64.
struct OffBuf {
    data: Vec<u8>,
    start: usize,
    len: usize,
}

impl OffBuf {
    fn new(len: usize, off: usize, seed: u64) -> Self {
        let data = noise(len + 128, seed);
        let base = data.as_ptr() as usize;
        let start = (off + 64 - (base % 64)) % 64;
        OffBuf { data, start, len }
    }
    fn src(&self) -> &[u8] {
        &self.data[self.start..self.start + self.len]
    }
    fn dst(&mut self) -> &mut [u8] {
        &mut self.data[self.start..self.start + self.len]
    }
}

fn gf16_sweep() {
    println!("\n== Gf16 single-row offset sweep (dst and src at the same base offset) ==");
    let coeff = gf16::Elem::from_raw(0x53a7);
    for &len in &[1536usize, 2048, 3072, 3584, 4096, 8192, 16384, 65536] {
        let mut row = String::new();
        for &off in &[0usize, 16] {
            let mut d = OffBuf::new(len, off, 1);
            let s = OffBuf::new(len, off, 2);
            let ns_add = time_ns(|| ops::mul_add::<Gf16>(d.dst(), coeff, s.src()));
            let mut d2 = OffBuf::new(len, off, 3);
            let s2 = OffBuf::new(len, off, 4);
            let ns_into = time_ns(|| ops::mul_into::<Gf16>(d2.dst(), coeff, s2.src()));
            let mut d3 = OffBuf::new(len, off, 5);
            let ns_assign = time_ns(|| ops::mul_assign::<Gf16>(d3.dst(), coeff));
            row.push_str(&format!(
                "  off={off:>2}: mul_add {ns_add:8.1}ns {:6.1} GiB/s | mul_into {ns_into:8.1}ns {:6.1} GiB/s | mul_assign {ns_assign:8.1}ns {:6.1} GiB/s\n",
                gib_per_s(len, ns_add),
                gib_per_s(len, ns_into),
                gib_per_s(len, ns_assign),
            ));
        }
        println!("{len:>6} B\n{row}");
    }
}

fn gather_probe<F: FieldKernels>(name: &str, coeff: impl Fn(usize) -> F::Elem) {
    println!("\n== {name} overwrite gather, 16 sources: raw coefficients vs prepared CoeffVec ==");
    for &len in &[4096usize, 16384] {
        let n = 16usize;
        let sources: Vec<Vec<u8>> = (0..n)
            .map(|i| noise(len, 0x1000 + i as u64 * 17 + len as u64))
            .collect();
        let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        let coeffs: Vec<F::Elem> = (0..n).map(&coeff).collect();
        let prepared = ops::CoeffVec::<F>::new(&coeffs);
        let mut dst_raw = noise(len, 0x2000 + len as u64);
        let mut dst_prep = noise(len, 0x3000 + len as u64);
        let ns_raw = time_ns(|| {
            ops::mul_into_gather::<F>(
                black_box(dst_raw.as_mut_slice()),
                black_box(&coeffs),
                black_box(&srcs),
            )
        });
        let ns_prep = time_ns(|| {
            ops::mul_into_gather_with::<F>(
                black_box(dst_prep.as_mut_slice()),
                prepared.as_ref(),
                black_box(&srcs),
            )
        });
        let logical = len * n;
        println!(
            "  {len:>6} B x {n}: raw {:6.1} GiB/s | prepared {:6.1} GiB/s | prep/raw {:.2}",
            gib_per_s(logical, ns_raw),
            gib_per_s(logical, ns_prep),
            ns_raw / ns_prep,
        );
    }
}

fn main() {
    println!("peel/gather probe — backend: {}", backend().name());
    gf16_sweep();
    gather_probe::<Gf8D>("Gf8D", |i| {
        gf8d::Elem::from_raw(2 + ((i * 73 + 19) % 254) as u8)
    });
    gather_probe::<Gf8B>("Gf8B", |i| {
        gf8b::Elem::from_raw(2 + ((i * 73 + 19) % 254) as u8)
    });
}
