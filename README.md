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
fgf = "1.0.0"
```

For the portable `no_std` surface without allocation:

```toml
[dependencies]
fgf = { version = "1.0.0", default-features = false }
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
use fgf::{Gf8B, gf8b, ops};

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

ops::mul_add::<Gf8B>(&mut dst, gf8b::Elem::from_raw(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

## Supported fields

| Field | Marker and element | Construction | Accelerated backends |
| --- | --- | --- | --- |
| GF(2^8) | `Gf8B`, `gf8b::Elem` | AES polynomial `0x11B` | x86 GFNI |
| GF(2^8) | `Gf8D`, `gf8d::Elem` | polynomial `0x11D` | x86 GFNI affine; x86, `AArch64`, and Wasm shuffle |
| GF(2^16) | `Gf16`, `gf16::Elem` | quadratic tower over `Gf8B` | x86 GFNI and shuffle |
| GF(2^32) | `Gf32`, `gf32::Elem` | quadratic tower over `Gf16` | x86 GFNI |
| GF(2^64) | `Gf64`, `gf64::Elem` | quadratic tower over `Gf32` | x86 GFNI |
| Fan–Paar GF(2^8) | `FanPaar8`, `fan_paar::fp8::Elem` | canonical recursive tower | portable |
| Fan–Paar GF(2^16) | `FanPaar16`, `fan_paar::fp16::Elem` | canonical recursive tower | x86 AVX2 and SSSE3 |
| Fan–Paar GF(2^32) | `FanPaar32`, `fan_paar::fp32::Elem` | canonical recursive tower | x86 AVX2 |
| Fan–Paar GF(2^64) | `FanPaar64`, `fan_paar::fp64::Elem` | canonical recursive tower | x86 AVX2 |
| GF(2^31 − 1) | `Mersenne31`, `mersenne31::Elem` | Mersenne prime in `u32` lanes | x86 AVX2 and SSE4.2 |
| GF(2^64 − 2^32 + 1) | `Goldilocks`, `goldilocks::Elem` | Goldilocks prime in `u64` lanes | x86 AVX2 and SSE4.2 |
| GF((2^31 − 1)²) | `QuadMersenne31`, `quad_mersenne31::Elem` | `i² = −1` over Mersenne31 | x86 AVX2 |
| GF(2) | `Gf2`, `gf2::Elem` | one element per bit | dispatched XOR; portable word kernels |

All element families support arithmetic operators including unary negation,
assignment operators, `Sum`, `Product`, ordering, formatting, and named raw
conversions. In characteristic two negation is the identity. Import
`fgf::field::Elem` when generic code needs the trait methods.

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
| `internals` | unstable direct kernel and experimentation surface |

Nothing behind `internals` is a compatibility promise. Normal users should use
`ops`, which handles backend selection and capability proofs internally.

## Platforms and backends

| Backend | Target |
| --- | --- |
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
| every field | fixed-width little-endian element encoding of `Field::BYTES`, packed without alignment or padding |
| `gf8b` | reduction polynomial `0x11B`, generator `0x03` |
| `gf8d` | reduction polynomial `0x11D`, generator `0x02`, the Reed-Solomon interop field |
| `field::tower` | component order `[a, b]`, constant component first, over `Gf8B` |
| `fan_paar` | canonical Wiedemann tower basis, a different basis from `field::tower` and not interoperable with it |
| `bits` | one GF(2) element per bit, LSB-first within each byte, padding bits caller-owned |
| prime fields | canonical lanes on packed buffer boundaries |

## Safety and scope

The stable API is safe, but the implementation is variable-time. Table lookups,
cache behavior, and data-dependent work make `fgf` unsuitable for secret field
elements or coefficients.

Packed prime-field operations expect canonical input lanes and preserve
canonical output. Scalar prime-field operations accept any raw lane and reduce
arithmetic results canonically.

## Performance

`BENCHMARKS.md` records pinned measurements for the public operation shapes on
multiple x86 hosts. The benchmark targets are:

```sh
cargo bench --bench kernels
cargo bench --bench compare
```

## Building

```sh
cargo build
cargo build --no-default-features
cargo test --all-features
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/fgf/blob/main/LICENSE).
