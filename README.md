> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

# fgf — Faster Galois Fields

`fgf` provides finite-field arithmetic and fast bulk operations over packed
byte buffers. It is the arithmetic layer beneath erasure codes, proof systems,
and other workloads that repeatedly scale, combine, and reconstruct rows of
field elements.

- **Fields:** binary fields from GF(2) to GF(2^64) — polynomial bases,
  ordered (normal and Cantor) bases, and quadratic towers, including custom
  ones — plus the Mersenne31, Goldilocks, and QM31 prime fields, and
  bit-packed GF(2).
- **One element type:** `Elem<F>` is the element of every field, with `const`
  arithmetic, so coefficients and matrices can be built at compile time.
- **Bulk operations:** `ops` provides checked AXPY, scaling, scatter, gather,
  matrix, and elementwise operations over unaligned little-endian buffers,
  with prepared coefficients for allocation-free repeated use.
- **Runtime SIMD:** the best available backend is selected once per process,
  with a portable fallback everywhere.
- **Embeddings:** `Embedding<S, T>` maps one binary field into a larger one,
  with Frobenius, trace, and norm.

`fgf` is an arithmetic library, not a codec: it does not build coding
matrices, manage shards, or implement a recovery protocol.

## Usage

```toml
[dependencies]
fgf = "3.0.0"
```

Scalar arithmetic is `const`:

```rust
use fgf::{Binary, Elem, Polynomial, poly::AES};

type Gf2P8 = Binary<8, Polynomial<AES>>;

const A: Elem<Gf2P8> = Elem::<Gf2P8>::from_raw(0x53);
const B: Elem<Gf2P8> = Elem::<Gf2P8>::from_raw(0xCA);

assert_eq!(A.mul(B), Elem::<Gf2P8>::ONE);
assert_eq!(A.mul(B).div(B), A);
```

Bulk operations take the field as a type parameter and work on packed bytes:

```rust
use fgf::{Binary, Elem, Polynomial, poly::AES, ops};

type Gf2P8 = Binary<8, Polynomial<AES>>;

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

// dst += 0x03 * src
ops::mul_add::<Gf2P8>(&mut dst, Elem::<Gf2P8>::from_raw(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

The [`examples/`](https://github.com/nanithefkuc/fgf/tree/main/examples)
directory has self-contained programs covering field arithmetic, shard
recovery, streaming encoding, GF(2) syndromes, extension fields, embeddings,
and prime-field vectors:

```sh
cargo run --example shard_recovery
```

The API reference is on [docs.rs](https://docs.rs/fgf); guides and background
are on the [wiki](https://github.com/nanithefkuc/fgf/wiki). Changes and
migration from 2.x are in
[`CHANGELOG.md`](https://github.com/nanithefkuc/fgf/blob/main/CHANGELOG.md).

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | yes | runtime backend detection and shared lookup tables; implies `alloc` |
| `simd` | yes | runtime-dispatched SIMD kernels; implies `std` |
| `alloc` | via `std` | prepared coefficient collections and `pack_to_vec` |
| `simd512` | no | AVX-512 kernels; implies `simd` |
| `internals` | no | direct access to kernels and tables; unstable, no compatibility guarantee |

With `default-features = false`, `fgf` is `no_std` and runs the portable
implementation.

## Platforms

| Target | Accelerated backends |
| --- | --- |
| `x86` / `x86_64` | SSSE3/SSE4.2, AVX2, GFNI; AVX-512 with `simd512` |
| `aarch64` | NEON, NEON with AES/PMULL |
| `wasm32` | `simd128` |
| any other | portable scalar |

`fgf::backend()` reports the selected backend. Setting `SIMD_BACKEND` (for
example `SIMD_BACKEND=scalar`) at process start requests a weaker backend;
requests the host cannot support are ignored.

## Safety

- **Not constant-time.** Table lookups and data-dependent work leak timing.
  Do not use `fgf` with secret elements, coefficients, or buffers without
  your own audit.
- The public API is safe Rust. Operations validate lengths and row geometry
  before running and panic on invalid input.
- Unsafe code is confined to SIMD kernels, with each use documented at its
  site.
- Inversion and division are total: `inv(0) == 0` and `x / 0 == 0`.

## Building

The minimum supported Rust version is 1.93.

```sh
cargo build
cargo test --all-features
```

The repository uses [`just`](https://github.com/casey/just) for its full
check suite; `just validate` runs it.

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/fgf/blob/main/LICENSE).
