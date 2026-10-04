> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

> [!WARNING]
> `fgf` is variable-time throughout and guarantees no constant-time operation.
> Security applications are not a supported use: any handling of secret field
> elements, coefficients, or buffers must be audited separately.

# fgf — Faster Galois Fields

`fgf` provides finite-field elements and high-throughput operations over packed
buffers. It targets the arithmetic underneath erasure coding, proof systems,
and other workloads that repeatedly scale, combine, or reconstruct rows.

| Main property | What it provides |
| --- | --- |
| Checked public API | Safe operations validate element widths, slice lengths, coefficient counts, and row geometry before dispatch. |
| Runtime SIMD | One process-wide selection chooses GFNI, AVX2, SSSE3/SSE4.2, NEON, Wasm SIMD, or the portable fallback. |
| Broad field set | Binary polynomial fields, binary towers, canonical Fan–Paar towers, Mersenne31, Goldilocks, QM31, and bit-packed GF(2). |
| Packed operations | Single-row AXPY, scatter, gather, matrix, overwrite, elementwise, and bit-range kernels. |
| Reusable preparation | Prepared coefficients remove repeated table construction while keeping the steady-state operation allocation-free. |
| Const scalar arithmetic | Concrete element operations are `const`, allowing coefficients and coding matrices to be built at compile time. |
| Stable representation | Fixed-width, little-endian encodings do not require aligned buffers. |

`fgf` is an arithmetic library, not a codec. It does not construct coding
matrices, own shards, invert matrices, or implement a recovery protocol.

## Installation

The minimum supported Rust version is 1.93.

```toml
[dependencies]
fgf = "2.1.0"
```

For the portable `no_std` surface without allocation:

```toml
[dependencies]
fgf = { version = "2.1.0", default-features = false }
```

## Quick start

Scalar elements expose inherent `const` arithmetic:

```rust
use fgf::gf16;

const A: gf16::Elem = gf16::Elem::from_raw(0x1234);
const B: gf16::Elem = gf16::Elem::from_raw(0x0108);
const PRODUCT: gf16::Elem = A.mul(B);

assert_eq!(PRODUCT.div(B), A);
```

Packed operations use the field marker as their generic parameter. Buffers
contain consecutive little-endian elements and may be unaligned.

```rust
use fgf::{Gf, Gf8, Elem, Poly, poly::AES, ops};

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

ops::mul_add::<Gf8<Poly<AES>>>(&mut dst, Elem::<Gf<8, Poly<AES>>>::from_raw(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

## Runnable examples

The [`examples/`](https://github.com/nanithefkuc/fgf/tree/main/examples)
programs use the stable API, print their results, and assert the demonstrated
identities or recovered data. Each file is self-contained.

| Example | Field coverage | Workflow |
| --- | --- | --- |
| `field_arithmetic` | Every field | Const coefficients, generic `Field`/`Elem` arithmetic, scalar encoding, and the difference between `0x11B` and `0x11D`. |
| `shard_recovery` | `Gf8<Poly<REED_SOLOMON>>` | Apply a fixed parity matrix, recover one erased shard with a gather, and recover both data shards with a prepared matrix. |
| `streaming_encode` | `Gf16` | Pack tower elements, scatter arriving sources, update parity from a source delta, and write disjoint output slots without overwriting frame metadata. |
| `binary_syndrome` | Bit-packed GF(2) | Encode a Hamming codeword by subset XOR, locate a single error with parity-check dot products, and handle logical lengths and padding. |
| `extension_fields` | AES-rooted and Fan–Paar towers, `QuadMersenne31` | Subfield embeddings, Frobenius conjugation, relative trace and norm, and quadratic-extension arithmetic. |
| `prime_vectors` | `Mersenne31`, `Goldilocks`, `QuadMersenne31` | Canonicalize raw lanes, pack evaluation vectors, compute pointwise products, and apply a prepared linear combination. |

Run one example with Cargo:

```sh
cargo run --example shard_recovery
```

From a repository checkout, the recipes run one example or the complete
educational set:

```sh
just example shard_recovery
just examples
just examples --all-features
just examples --no-default-features --features alloc
just example field_arithmetic --no-default-features
SIMD_BACKEND=scalar just examples
```

`shard_recovery`, `streaming_encode`, and `prime_vectors` require `alloc`;
the other examples also work with the library's features disabled. The
executables themselves use the host's standard library. The scalar override
exercises the portable backend; packed examples report the backend for their
field.

The coding examples supply fixed coefficients and cover only their stated
error or erasure patterns. They illustrate field operations, not production
codecs or general matrix solvers. Element storage is canonical, so packed
buffers produced by `pack` satisfy the packed-operation contract directly.
These examples do not make the library suitable for secret inputs.

`peel_probe` is a timing diagnostic, not a usage tutorial, and is excluded
from `just examples`.

## Supported fields

| Field | Marker and element | Construction | Accelerated backends |
| --- | --- | --- | --- |
| GF(2^8) | `Gf8<Poly<AES>>`, `Elem<Gf<8, Poly<AES>>>` | AES polynomial `0x11B` | x86 GFNI and shuffle; `AArch64` NEON/PMULL; Wasm SIMD |
| GF(2^8) | `Gf8<Poly<REED_SOLOMON>>`, `Elem<Gf<8, Poly<REED_SOLOMON>>>` | polynomial `0x11D` | x86 GFNI affine and shuffle; `AArch64` and Wasm single-row shuffle |
| GF(2^8) | `Gf8<Poly<POLY>>`, `Elem<Gf<8, Poly<POLY>>>` | any irreducible degree-eight polynomial | x86 GFNI affine and shuffle; `AArch64` and Wasm single-row shuffle |
| GF(2^16) | `Gf16`, `gf16::Elem` | quadratic tower over `Gf8<Poly<AES>>` | x86 GFNI and shuffle |
| GF(2^32) | `Gf32`, `gf32::Elem` | quadratic tower over `Gf16` | x86 GFNI |
| GF(2^64) | `Gf64`, `gf64::Elem` | quadratic tower over `Gf32` | x86 GFNI |
| Fan–Paar GF(2^8) | `FanPaar8`, `fan_paar::fp8::Elem` | canonical recursive tower | portable |
| Fan–Paar GF(2^16) | `FanPaar16`, `fan_paar::fp16::Elem` | canonical recursive tower | x86 AVX2 and SSSE3 |
| Fan–Paar GF(2^32) | `FanPaar32`, `fan_paar::fp32::Elem` | canonical recursive tower | x86 AVX2 |
| Fan–Paar GF(2^64) | `FanPaar64`, `fan_paar::fp64::Elem` | canonical recursive tower | x86 AVX2 |
| GF(2^31 − 1) | `Mersenne31`, `Elem<Mersenne31>` | Mersenne prime in `u32` lanes | x86 AVX2 and SSE4.2 |
| GF(2^64 − 2^32 + 1) | `Goldilocks`, `Elem<Goldilocks>` | Goldilocks prime in `u64` lanes | x86 AVX2 and SSE4.2 |
| GF((2^31 − 1)²) | `QuadMersenne31`, `Elem<QuadMersenne31>` | `i² = −1` over Mersenne31 | x86 AVX2 |
| GF(2) | `Gf1`, `Elem<Gf1>` | one element per bit | dispatched XOR; portable word kernels |

`AES` and `REED_SOLOMON` name the two standard polynomial constants in
`fgf::poly`. Other conventions use the complete polynomial directly:
`Gf8<Poly<0x12D>>` for Data Matrix and `Gf8<Poly<0x187>>` for the CCSDS
polynomial basis. `POLY` includes the leading `x^8` bit and is checked for
irreducibility at compile time. Elements and prepared coefficients retain the
representation in their type; copying identical raw bytes between polynomial
bases is not a field conversion. CCSDS dual-basis wire conversion is outside
this library.

All element families support arithmetic operators including unary negation,
assignment operators, `Sum`, `Product`, ordering, formatting, and named raw
conversions. In characteristic two negation is the identity. Import
`fgf::field::FieldElem` when generic code needs the trait methods.

By convention, inversion and division are total: `inv(0) == 0` and
`x / 0 == 0`.

## Packed operations

| Operation | Contract | Typical use |
| --- | --- | --- |
| `add_assign` | `dst += src` | parity or modular row addition |
| `add_assign_rows` | row-shaped `dst += src` | batched row addition |
| `add_gather_offsets` | add selected rows from one region | compact back-substitution |
| `mul_add` | `dst += coefficient × src` | finite-field AXPY |
| `mul_into` | `dst = coefficient × src` | overwrite scaling |
| `mul_assign` | `dst *= coefficient` | in-place scaling |
| `mul_add_scatter` | one source accumulates into many rows | systematic encoding |
| `mul_add_gather` | many sources accumulate into one row | reconstruction |
| `mul_into_gather` | many sources overwrite one row | fresh recovered symbol |
| `mul_add_matrix` | many sources accumulate into many rows | matrix application |
| `mul_into_matrix` | many sources overwrite many rows | fresh encoded rows |
| `mul_add_matrix_at` | many sources accumulate into offset rows | in-place reconstruction |
| `mul_elementwise` | `dst[i] = a[i] × b[i]` | pointwise products |

The `_with` forms consume prepared coefficients. Preparation does not weaken
validation: prepared and one-shot operations enforce the same public geometry.

```rust
use fgf::{Gf16, gf16, ops};

let coeff = ops::Coeff::<Gf16>::new(gf16::Elem::from_raw(0x0108));
let src = 0x1234u16.to_le_bytes();
let mut dst = [0u8; 2];

for _ in 0..3 {
    ops::mul_add_with::<Gf16>(&mut dst, &coeff, &src);
}

let once = gf16::Elem::from_raw(0x1234).mul(gf16::Elem::from_raw(0x0108));
assert_eq!(dst, once.to_bytes());
```

`Coeff<F>` prepares one coefficient without allocation. With `alloc`,
`CoeffVec<F>` and `CoeffMatrix<F>` own reusable prepared collections; a single
entry borrows as `CoeffRef<'_, F>`, which the single-coefficient `_with`
operations accept directly. Prepared values are process-backend-specific and
should be rebuilt rather than serialized.

`ops::pack`, `ops::unpack`, and `ops::pack_to_vec` convert between typed
elements and packed byte buffers.

## Bit-packed GF(2)

The `bits` module stores one GF(2) element per bit, least-significant bit first.
Operations that depend on the logical element count take it explicitly.

```rust
use fgf::bits;

let mut row = [0u8; 1];
bits::set_range(&mut row, 7, 0, 7);
bits::clear_range(&mut row, 7, 1, 3);

assert_eq!(row[0], 0b0111_1001);
assert_eq!(bits::weight(&row, 7), 5);
```

Range operations leave padding bits untouched. Whole-buffer operations process
the complete supplied slices.

## Features

| Feature | Effect |
| --- | --- |
| default | enables `std` and runtime-dispatched `simd` |
| `alloc` | prepared coefficient collections and `pack_to_vec` |
| `std` | runtime support and lazily initialized shared tables; implies `alloc` |
| `simd` | runtime-dispatched architecture kernels; implies `std` |
| `simd512` | 64-byte AVX-512 kernels for `Gf8<Poly<POLY>>` and `Gf16` single-row, elementwise, and matrix operations, `Mersenne31` and `Goldilocks` arithmetic, `QuadMersenne31` multiplication, binary-field XOR, and `bits::weight`; enables `V4x` dispatch and implies `simd` |
| `internals` | re-export-only facade of direct kernel and table surfaces |

Nothing behind `internals` is a compatibility promise; the facade groups the
implementation modules under `fgf::internals` and may change shape in any
release. Normal users should use `ops`, which handles backend selection and
capability proofs internally.

## Platforms and backends

| Backend | Target |
| --- | --- |
| `v4x` | `x86_64` with AVX-512 and GFNI (`simd512` feature) |
| `v3_gfni_crypto` | `x86`/`x86_64` with AVX2 and GFNI |
| `v3` | `x86`/`x86_64` AVX2 shuffle kernels |
| `v2` | `x86`/`x86_64` SSSE3/SSE4.2 kernels |
| `neon_aes` | `AArch64` NEON with AES/PMULL capability |
| `neon` | `AArch64` NEON shuffle kernels |
| `wasm128` | WebAssembly `simd128` |
| `scalar` | portable fallback |

`backend()` reports the process-wide selection. `backend_for::<F>()` reports the
backend implemented by a particular field. `SIMD_BACKEND` may request a weaker
backend at process startup; unsupported upgrades are ignored.

## Frozen encodings

These are the wire conventions a consumer above this crate depends on. Each is
stated in full by the module that owns it.

| Surface | Convention |
| --- | --- |
| every field | fixed-width little-endian element encoding of `FieldBuffer::BYTES`, packed without alignment or padding |
| `Gf8<Poly<AES>>` | reduction polynomial `0x11B`, generator `0x03` |
| `Gf8<Poly<REED_SOLOMON>>` | reduction polynomial `0x11D`, generator `0x02`, the Reed–Solomon interop field |
| `field::tower` | component order `[a, b]`, constant component first, over `Gf8<Poly<AES>>` |
| `fan_paar` | canonical Wiedemann tower basis, a different basis from `field::tower` and not interoperable with it |
| `bits` | one GF(2) element per bit, LSB-first within each byte, padding bits caller-owned |
| prime fields | canonical lanes on packed buffer boundaries |

## Safety and scope

The stable API is safe, but the implementation is variable-time. Table lookups,
cache behavior, and data-dependent work make `fgf` unsuitable for secret field
elements or coefficients.

Packed prime-field operations accept every lane bit pattern and produce
canonical lanes on output. Scalar prime-field operations accept any raw lane
and reduce arithmetic results canonically.

## Performance

[BENCHMARKS.md](https://github.com/nanithefkuc/fgf/blob/main/BENCHMARKS.md)
indexes current public API timings and interleaved competitor comparisons,
one page per field family under
[`benchmarks/`](https://github.com/nanithefkuc/fgf/tree/main/benchmarks).
Every result cell reports Tiger Lake / Golden Cove.
The complete snapshot campaign runs on each host's isolated CPU:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
FEC_GOLDEN_CORE=<cpu> just bench-gdl-comp
FEC_GOLDEN_CORE=<cpu> just bench-m31-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf2-comp
```

The GF(2^16) field-kernel comparison also runs independently:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-gf16-comp
```

The harness validates equivalent field elements across each library's native
basis and byte layout before timing. It excludes representation conversion and
does not substitute codec encoding or decoding for field operations.
The native competitor build requires Git, C/C++ compilers and an x86-64 host
with AVX2, GFNI and crypto instruction support.

## Building

```sh
cargo build
cargo build --no-default-features
cargo test --all-features
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/fgf/blob/main/LICENSE).
