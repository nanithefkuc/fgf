//! `fgf` side of the external-library comparison (see external-bench/PLAN.md).
//!
//! Measures scalar and 64 KiB single-source region operations plus prepared
//! 16-source overwrite dot products at 4 and 16 KiB. Buffers are 64-byte
//! aligned to match the ISA-L harness.
//!
//! ```sh
//! cargo bench --bench compare
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::{Gf8B, Gf8D, Gf16, backend, gf8b, gf8d, gf16, ops};

const BYTES: usize = 64 * 1024;
const SCALAR_ITERS: usize = 1 << 20;
const DOT_SOURCES: usize = 16;
const DOT_LENGTHS: &[usize] = &[4 * 1024, 16 * 1024];

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

struct AlignedBuf {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    fn noise(len: usize, seed: u64) -> Self {
        let mut storage = vec![0; len + 63];
        let base = storage.as_ptr() as usize;
        let start = (64 - (base & 63)) & 63;
        storage[start..start + len].copy_from_slice(&noise(len, seed));
        let result = Self {
            storage,
            start,
            len,
        };
        assert_eq!((result.as_slice().as_ptr() as usize) & 63, 0);
        result
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }
}

fn bench_dot_product(len: usize) {
    let sources: Vec<AlignedBuf> = (0..DOT_SOURCES)
        .map(|index| AlignedBuf::noise(len, 0x800 + index as u64))
        .collect();
    let srcs: Vec<&[u8]> = sources.iter().map(AlignedBuf::as_slice).collect();
    let coefficient_bytes: Vec<u8> = (0..DOT_SOURCES)
        .map(|index| 2 + ((index * 73 + 19) % 254) as u8)
        .collect();
    let coeffs_8b: Vec<_> = coefficient_bytes.iter().copied().map(gf8b::Elem).collect();
    let coeffs_8d: Vec<_> = coefficient_bytes.iter().copied().map(gf8d::Elem).collect();
    let plan_8b = ops::Plan::<Gf8B>::new(&coeffs_8b);
    let plan_8d = ops::Plan::<Gf8D>::new(&coeffs_8d);
    let mut dst_8b = AlignedBuf::noise(len, 0x900);
    let mut dst_8d = AlignedBuf::noise(len, 0x901);
    let mut rse_dst = AlignedBuf::noise(len, 0x902);

    ops::dot_product_with::<Gf8D>(dst_8d.as_mut_slice(), &plan_8d, &srcs);
    rse_dst.as_mut_slice().fill(0);
    for (&coefficient, &source) in coefficient_bytes.iter().zip(&srcs) {
        reed_solomon_erasure::galois_8::mul_slice_xor(coefficient, source, rse_dst.as_mut_slice());
    }
    assert_eq!(
        dst_8d.as_slice(),
        rse_dst.as_slice(),
        "Gf8D and reed-solomon-erasure dot products differ"
    );

    println!("{len} B x {DOT_SOURCES} sources overwrite dot product");
    let logical_bytes = len * DOT_SOURCES;
    bench_region("fgf Gf8B dot_product_with", logical_bytes, || {
        ops::dot_product_with::<Gf8B>(black_box(dst_8b.as_mut_slice()), &plan_8b, black_box(&srcs));
    });
    bench_region("fgf Gf8D dot_product_with", logical_bytes, || {
        ops::dot_product_with::<Gf8D>(black_box(dst_8d.as_mut_slice()), &plan_8d, black_box(&srcs));
    });
    bench_region("RSE zero + 16 x mul_slice_xor", logical_bytes, || {
        rse_dst.as_mut_slice().fill(0);
        for (&coefficient, &source) in coefficient_bytes.iter().zip(&srcs) {
            reed_solomon_erasure::galois_8::mul_slice_xor(
                coefficient,
                black_box(source),
                black_box(rse_dst.as_mut_slice()),
            );
        }
    });
}

fn bench_region(label: &str, bytes: usize, mut body: impl FnMut()) {
    for _ in 0..16 {
        body();
    }
    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        for _ in 0..32 {
            body();
        }
        samples.push(start.elapsed() / 32);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let gib = bytes as f64 / median.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("  {label:<44} {:>9.2?}  {gib:>7.2} GiB/s", median);
}

fn bench_scalar(label: &str, mut body: impl FnMut()) {
    for _ in 0..4 {
        body();
    }
    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        body();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let ns = median.as_secs_f64() * 1e9 / SCALAR_ITERS as f64;
    let mops = SCALAR_ITERS as f64 / median.as_secs_f64() / 1e6;
    println!("  {label:<44} {ns:>9.2} ns/op  {mops:>7.2} Mops/s");
}

fn main() {
    println!("fgf — backend: {}", backend().name());

    let src8 = AlignedBuf::noise(BYTES, 0x600);
    let mut dst8 = AlignedBuf::noise(BYTES, 0x601);
    let src16 = AlignedBuf::noise(BYTES, 0x602);
    let mut dst16 = AlignedBuf::noise(BYTES, 0x603);

    // w8
    let c8 = gf8b::Elem(0x53);
    bench_scalar("w8 scalar mul", || {
        let mut x = gf8b::Elem(0xA5);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(gf8b::Elem((0x53u8).wrapping_add(i as u8)));
        }
        black_box(x);
    });
    bench_region("w8 mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf8B>(
            black_box(dst8.as_mut_slice()),
            c8,
            black_box(src8.as_slice()),
        );
    });
    bench_region("w8 mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf8B>(
            black_box(dst8.as_mut_slice()),
            c8,
            black_box(src8.as_slice()),
        );
    });

    // Bit-compatible GF(2^8)/0x11D used by ISA-L and RSE.
    let c8d = gf8d::Elem(0x53);
    bench_scalar("w8d scalar mul", || {
        let mut x = gf8d::Elem(0xA5);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(gf8d::Elem((0x53u8).wrapping_add(i as u8)));
        }
        black_box(x);
    });
    bench_region("w8d mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf8D>(
            black_box(dst8.as_mut_slice()),
            c8d,
            black_box(src8.as_slice()),
        );
    });
    bench_region("w8d mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf8D>(
            black_box(dst8.as_mut_slice()),
            c8d,
            black_box(src8.as_slice()),
        );
    });

    // w16
    let c16 = gf16::Elem(0x53A7);
    bench_scalar("w16 scalar mul", || {
        let mut x = gf16::Elem(0x1234);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(gf16::Elem((0x53A7u16).wrapping_add(i as u16)));
        }
        black_box(x);
    });
    bench_region("w16 mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf16>(
            black_box(dst16.as_mut_slice()),
            c16,
            black_box(src16.as_slice()),
        );
    });
    bench_region("w16 mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf16>(
            black_box(dst16.as_mut_slice()),
            c16,
            black_box(src16.as_slice()),
        );
    });

    // reed-solomon-erasure (simd-accel: SSSE3/AVX2). GF(2^8), poly 0x11D.
    // mul_slice_xor(c, input, out) == dst ^= c * src, the exact mul_add analog.
    // The C `simd-accel` path has no wasm sysroot, so the dependency — and
    // this section — exist on native targets only.
    #[cfg(not(target_family = "wasm"))]
    {
        println!("reed-solomon-erasure 6 (simd-accel)");
        let mut rse_dst = AlignedBuf::noise(BYTES, 0x701);
        let rse_src = AlignedBuf::noise(BYTES, 0x702);
        bench_region("w8 mul_slice_xor (dst ^= c*src)", BYTES, || {
            reed_solomon_erasure::galois_8::mul_slice_xor(
                0x53,
                black_box(rse_src.as_slice()),
                black_box(rse_dst.as_mut_slice()),
            );
        });
        bench_region("w8 mul_slice (dst = c*src)", BYTES, || {
            reed_solomon_erasure::galois_8::mul_slice(
                0x53,
                black_box(rse_src.as_slice()),
                black_box(rse_dst.as_mut_slice()),
            );
        });
    }

    for &len in DOT_LENGTHS {
        bench_dot_product(len);
    }
}
