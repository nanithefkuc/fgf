> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

# fgf — Faster Galois Fields

`fgf` provides finite-field arithmetic and fast bulk operations over packed
byte buffers. It is the arithmetic layer beneath erasure codes, proof systems,
and other workloads that repeatedly scale, combine, and reconstruct rows of
field elements. It is not a codec: it does not build coding matrices, manage
shards, or implement a recovery protocol.

## One model for every field

`FiniteField<P, N, Representation>` selects a field's characteristic, extension
degree, and coordinates. `Elem<F>` holds its values through one scalar API.
Fields with byte-lane storage also share the packed-operation API.

| Presentation | Covers |
| --- | --- |
| `Residue` (default) | Prime fields `FiniteField<P, 1>` for any `u64` prime |
| `Polynomial<POLY>` | Binary polynomial bases, including AES and Reed–Solomon conventions |
| `Normal<POLY, E>`, `Cantor<POLY, SEED>` | Ordered binary bases |
| `Extension<Spec, D>` | Extensions over supported bases, with a chosen relation and storage representation |

`Mersenne31` and `Goldilocks` select specialized prime representations;
`QuadMersenne31` selects a quadratic extension. They use the same model, not
separate element systems. Shipped binary towers reach GF(2^64); custom
extensions use `ExtensionSpec`. [The wiki](https://github.com/nanithefkuc/fgf/wiki)
explains supported degrees, relations, storage choices, and embeddings.

## Usage

```toml
[dependencies]
fgf = "3.0.1"
```

Choose the field, then construct its elements. Shipped scalar arithmetic is
`const`; operators follow field arithmetic, not ordinary integer arithmetic.

```rust
use fgf::{Elem, FiniteField, Polynomial, AES};

type Prime = FiniteField<17, 1>;
type Binary = FiniteField<2, 8, Polynomial<AES>>;

const X: Elem<Prime> = Elem::<Prime>::from_raw(7);
assert_eq!(X.mul(X).to_raw(), 15);
assert_eq!(
    Elem::<Binary>::from_raw(0x53) * Elem::<Binary>::from_raw(0xca),
    Elem::<Binary>::ONE,
);
```

`ops` scales and combines rows of packed elements, with prepared coefficients
for repeated calls. Prepared Frobenius, embedding, and isomorphism maps apply
one compiled map across scalar elements and byte rows alike, with a validity
bitmap for partial restriction; `ops::map_bits` carries the GF(2) bit/lane
interop rows over packed bits. The prepared maps, `ops::map` and
`ops::map_bits`, and the prepared `_with` forms require the `alloc` feature.
Byte lanes are fixed-width and
little-endian; their size depends on the field representation. Bit-packed
GF(2) vectors use `bits`.

```rust
use fgf::{FiniteField, Elem, Polynomial, poly::AES, ops};

type Gf2P8 = FiniteField<2, 8, Polynomial<AES>>;

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

// dst += 0x03 * src
ops::mul_add::<Gf2P8>(&mut dst, Elem::<Gf2P8>::from_raw(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

The [`examples/`](https://github.com/nanithefkuc/fgf/tree/main/examples)
directory has self-contained programs covering field arithmetic, shard
recovery, streaming encoding, GF(2) syndromes, extension fields, embeddings,
custom extensions and basis maps, prime-field vectors, and packed operations
without an allocator:

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
| `alloc` | via `std` | prepared maps (`Frobenius`, `PreparedEmbedding`, `PreparedIsomorphism`), `ops::map` and `ops::map_bits`, prepared coefficient collections (`CoeffVec`, `CoeffMatrix`), the prepared scatter/gather/matrix `_with` forms, and `pack_to_vec` |
| `simd512` | no | AVX-512 kernels; implies `simd` |
| `internals` | no | direct access to kernels and tables; unstable, no compatibility guarantee |

With `default-features = false`, `fgf` is `no_std` and runs the portable
implementation.

## Platforms

SIMD kernels are x86-64 only (`v2` SSSE3/SSE4.2, `v3` AVX2, `v3_gfni_crypto` GFNI, `v4x` AVX-512 with `simd512`); other targets, including 32-bit `x86`, compile and run the portable scalar bodies, including with default features.

The best available backend is selected once per process, with a portable
fallback everywhere. `fgf::backend()` reports the selection, and setting
`SIMD_BACKEND` (for example `SIMD_BACKEND=scalar`) at process start requests
a weaker backend; requests the host cannot support are ignored.

Which operations accelerate depends on the field, the operation, and the
buffer geometry; `backend_for::<F>()` reports the backend implemented for a
field. Release acceptance runs the per-tier differentials on dedicated Willow
Cove (`v4x`) and Golden Cove (`v3_gfni_crypto`) hosts; GitHub-hosted runners
only cover tiers they can prove. Routing notes are in the
[deployment guide](https://github.com/nanithefkuc/fgf/wiki/Features-and-deployment),
and measured results live in
[`BENCHMARKS.md`](https://github.com/nanithefkuc/fgf/blob/main/BENCHMARKS.md).

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

The repository uses [`just`](https://github.com/casey/just) for its check
suites: `just validate` runs the complete pull-request gate, and
`just release-accept` runs the crate-specific release acceptance.

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/fgf/blob/main/LICENSE).
