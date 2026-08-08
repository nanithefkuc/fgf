> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# fgf - Faster Galois Fields

`fgf` provides safe APIs for scalar arithmetic and runtime-dispatched vector kernels for binary finite fields. 

## Usage

The MSRV is Rust 1.89.

`fgf` is distributed through git only; it is not published to [crates.io](https://crates.io).

```toml
[dependencies]
fgf = { git = "https://github.com/nanithefkuc/fgf" }
```

Portable `no_std` builds are also available:

```toml
[dependencies]
fgf = { git = "https://github.com/nanithefkuc/fgf", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | runtime CPU detection and vector kernels |
| `std` without `simd` | portable kernels with allocation-backed plans |
| `--no-default-features` | `no_std`, portable kernels, allocation-free API |

### Platforms

| Platform | Result |
| --- | --- |
| x86/x86_64 | GFNI/AVX2/SSSE3 runtime dispatch |
| AArch64 | NEON runtime dispatch, optional PMULL |
| wasm32 + `simd128` | WebAssembly vector kernels |
| other targets | portable scalar kernels |

### Fields

| Field | Marker | Element | Construction | Vector backend |
| --- | --- | --- | --- | --- |
| GF(2^8) | `Gf8` | `gf8::Elem` | AES polynomial `0x11B` | GFNI x86 |
| GF(2^16) | `Gf16` | `gf16::Elem` | quadratic tower over `Gf8` | GFNI x86 |
| GF(2^32) | `Gf32` | `gf32::Elem` | quadratic tower over `Gf16` | GFNI x86 |
| GF(2^64) | `Gf64` | `gf64::Elem` | quadratic tower over `Gf32` | GFNI x86 |
| Fan–Paar GF(2^8) | `FanPaar8` | `fan_paar::fp8::Elem` | canonical recursive tower | portable |
| Fan–Paar GF(2^16) | `FanPaar16` | `fan_paar::fp16::Elem` | canonical recursive tower | x86 AVX2/SSSE3 |
| Fan–Paar GF(2^32) | `FanPaar32` | `fan_paar::fp32::Elem` | canonical recursive tower | x86 AVX2 |
| Fan–Paar GF(2^64) | `FanPaar64` | `fan_paar::fp64::Elem` | canonical recursive tower | x86 AVX2 |

### Scalar arithmetic

Concrete methods are `const fn` so coefficients and coding matrices can be built at compile time.
Import `fgf::field::Elem` when generic code needs the trait methods in scope.

```rust
use fgf::gf16;

const A: gf16::Elem = gf16::Elem(0x1234);
const B: gf16::Elem = gf16::Elem(0x0108);
const PRODUCT: gf16::Elem = A.mul(B);

assert_eq!(PRODUCT.div(B), A);
assert_eq!(A + B, A.sub(B));
```

By library-wide convention `inv(0) == 0` and `x / 0 == 0`. All element families implement `Add`, `Sub`, `Mul`, `Div`, their assignment
forms, `Sum`, `Product`, `Display`, and representation-order `Ord`. `Ord` is
for map keys only and has no field-theoretic meaning.

### Packed vector operations

Buffers contain consecutive stable little-endian element encodings. Their
length must be a multiple of `F::BYTES`.

```rust
use fgf::{Gf8, gf8, ops};

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

ops::mul_add::<Gf8>(&mut dst, gf8::Elem(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

| Shape | Function | Typical use |
| --- | --- | --- |
| `dst ^= src` | `add_assign` / `sub_assign` | XOR parity |
| `dst ^= c * src` | `mul_add` / `mul_add_with` | AXPY |
| `dst = c * src` | `mul_into` / `mul_into_with` | scale a row |
| `dst *= c` | `mul_assign` / `mul_assign_with` | in-place scale |
| one source, many rows | `mul_add_scatter` / `mul_add_scatter_with` | systematic encode |
| many sources, one row | `mul_add_gather` / `mul_add_gather_with` | recover one symbol |
| many sources, many rows | `mul_add_matrix` / `mul_add_matrix_with` | reconstruction |
| varying pair per lane | `mul_elementwise` | pointwise products |

Prefer the widest shape that matches the operation. The blocked kernels can
retain destination tiles in registers across sources.

### Reusing coefficients

`Coeff<F>` prepares one coefficient. With `std`, `Plan<F>` prepares a vector or
row-major matrix once and drives every multi-row `_with` operation directly.
A matrix plan has dimensions `(sources, destination_rows)`:

```rust
use fgf::{Gf16, gf16, ops};

let coeffs = [
    gf16::Elem(1), gf16::Elem(2),
    gf16::Elem(3), gf16::Elem(4),
];
let plan = ops::Plan::<Gf16>::matrix(2, 2, &coeffs);
let a = [1u8, 0, 2, 0];
let b = [3u8, 0, 4, 0];
let sources = [&a[..], &b[..]];
let mut rows = [0u8; 8];

ops::mul_add_matrix_with(&mut rows, 4, 2, &plan, &sources);
```

Use `ops::pack`, `ops::unpack`, or `ops::pack_to_vec` at element/buffer
boundaries instead of writing chunk loops by hand.

## Building

`fgf` builds on stable Rust (edition 2024, MSRV 1.89) with no extra tooling or
target-feature flags — SIMD kernels are selected at runtime:

```sh
cargo build                        # default: std + simd
cargo build --no-default-features  # portable no_std
cargo test --all-features
```

## Backends

`backend()` reports the process-wide SIMD selection. `backend_for::<F>()`
reports what a particular field actually uses; the GF(2^32)/GF(2^64) towers
report the GFNI backend on x86 GFNI hosts, the canonical Fan–Paar GF(2^16)/
GF(2^32)/GF(2^64) report AVX2/SSSE3 on x86, and the remaining fields report
`scalar`. `has_vector_elementwise::<F>()` exposes the notable performance
boundary of `mul_elementwise`.

| Identifier | Target and requirements | Lane width |
| --- | --- | --- |
| `v3_gfni_crypto` | x86 AVX2 + GFNI + crypto | 32 bytes |
| `v3` | x86 AVX2 shuffle | 32 bytes |
| `v2` | x86 SSSE3/SSE4.2 shuffle | 16 bytes |
| `neon_aes` | AArch64 NEON + AES (the AES feature proves PMULL) | 16 bytes |
| `neon` | AArch64 NEON split-nibble shuffle | 16 bytes |
| `wasm128` | WebAssembly `simd128` | 16 bytes |
| `scalar` | portable fallback | scalar |

`SIMD_BACKEND=v3_gfni_crypto|v3|v2|neon_aes|neon|wasm128|scalar` requests a
backend at process startup. It is downgrade-only: an unsupported upgrade is
ignored. Backends are a re-export of `simdispatch::Backend`; `name`,
`from_name`, `Display`, and `FromStr` support diagnostics and CLI wiring.

The 64-byte AVX-512 GFNI tier is written (`kernel::x86::avx512`, under the
`internals` feature) but deferred: it is cross-compile-only and stays out of
the dispatch ladder until validated on executing hardware, so AVX-512 hosts
resolve to the 32-byte GFNI tier.

## Benchmarks

`cargo bench --bench kernels` measures the operation shapes;
`cargo bench --bench compare` compares against `reed-solomon-erasure`.
Measurement and reproduction notes are in [BENCHMARKS.md](BENCHMARKS.md).

## License

MIT - see [LICENSE](LICENSE)
