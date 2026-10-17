//! Pinned interleaved floor probe for the GF(2) bit-interop tiers.
//!
//! ```sh
//! FEC_GOLDEN_CORE=<core> cargo run --release --all-features --example gf1_floors
//! ```
//!
//! Times each interop op (`expand`, `compress`, `restrict`) at floor
//! boundaries and widths through the public prepared row entry point under
//! the process-selected tier. Run it twice on the pinned core — once at
//! the host tier and once with `SIMD_BACKEND=scalar` as the unchanged
//! portable control — interleaving the two invocations round by round; the
//! ratio is portable/tier. Environment, command, and aggregation are
//! printed in the header for the benchmark record.

use std::hint::black_box;
use std::time::Instant;

use fgf::{
    BinaryField, Elem, Embedding, Field, FieldBuffer, FieldElem, FiniteField,
    Polynomial, PreparedEmbedding, PrimeCharacteristic, PrimeField, backend, ops,
};

type Gf1 = FiniteField<2, 1, Polynomial<3>>;
type Gf64 = FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;

const ROUNDS: usize = 5;
/// Lane counts straddling each tier floor in row bytes (lanes are 8 bytes
/// for the `Gf64` target: 15/16, 31/32, and 63/64 byte rows) plus one wide
/// control.
const LANES: &[usize] = &[1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 128, 1024];

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

/// Median per-call nanoseconds of `body` over one calibrated round.
fn round_ns(calls: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..4 {
        for _ in 0..calls {
            body();
        }
    }
    let start = Instant::now();
    for _ in 0..calls {
        body();
    }
    start.elapsed().as_secs_f64() * 1e9 / calls as f64
}

fn main() {
    let map = Embedding::<Gf1, Gf64>::new().expect("gf1-gf64 constructs");
    let prepared = PreparedEmbedding::<Gf1, Gf64>::new(&map);
    println!("gf1 floors: process backend {}", backend().name());
    println!(
        "command: FEC_GOLDEN_CORE=<core> cargo run --release --all-features --example gf1_floors"
    );
    println!("aggregation: median of {ROUNDS} per-round medians per case");
    println!();
    println!("| op | row bytes | ns/call |");
    println!("| --- | --- | --- |");
    for &lanes in LANES {
        let bits = lanes;
        let bytes = lanes * 8;
        let src = noise(fgf::bits::bytes_for(bits), 0x50 + lanes as u64);
        let lane_bytes = noise(bytes, 0x60 + lanes as u64);
        let calls = (1_000_000usize / bytes.max(1)).clamp(16, 4096);
        let mut dst = vec![0u8; bytes];
        let mut samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            samples.push(round_ns(calls, || {
                black_box(ops::map_bits::embed_into_with(
                    black_box(&mut dst),
                    &prepared,
                    black_box(&src),
                    black_box(bits),
                ));
            }));
        }
        samples.sort_by(f64::total_cmp);
        println!("| expand | {bytes} | {:.2} |", samples[ROUNDS / 2]);
        let mut packed = vec![0u8; fgf::bits::bytes_for(bits)];
        let mut samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            samples.push(round_ns(calls, || {
                black_box(ops::map_bits::trace_into_with(
                    black_box(&mut packed),
                    &prepared,
                    black_box(&lane_bytes),
                    black_box(bits),
                ));
            }));
        }
        samples.sort_by(f64::total_cmp);
        println!("| compress | {bytes} | {:.2} |", samples[ROUNDS / 2]);
        let mut dst = vec![0u8; fgf::bits::bytes_for(bits)];
        let mut valid = vec![0u8; fgf::bits::bytes_for(bits)];
        let mut samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            samples.push(round_ns(calls, || {
                black_box(ops::map_bits::restrict_into_with(
                    black_box(&mut dst),
                    black_box(&mut valid),
                    &prepared,
                    black_box(&lane_bytes),
                    black_box(bits),
                ));
            }));
        }
        samples.sort_by(f64::total_cmp);
        println!("| restrict | {bytes} | {:.2} |", samples[ROUNDS / 2]);
    }
    // Silence unused-import warnings for the explicitly named trait bounds.
    fn _assert_bounds<T>()
    where
        T: BinaryField + FieldBuffer + Field,
        Elem<T>: FieldElem,
    {
    }
    fn _assert_prime<S>()
    where
        S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    {
    }
    _assert_bounds::<Gf64>();
    _assert_prime::<Gf1>();
    let _ = Elem::<Gf64>::ZERO;
}
