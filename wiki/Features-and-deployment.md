# Features and deployment

Feature selection determines whether `fgf` can allocate, use standard-library support, and dispatch architecture kernels. Representation and operation support determine what a particular field can execute on that host.

This guide targets fgf 3.0.1 and Rust 1.93 or later. The [README](https://github.com/nanithefkuc/fgf/blob/v3.0.1/README.md) and [kernel reference](https://docs.rs/fgf/3.0.1/fgf/kernel/index.html) own the supported configuration and query contracts.

## Select the runtime requirements

| Configuration | Library environment | Prepared collections | Arithmetic routing |
| --- | --- | --- | --- |
| Defaults | `std` and allocation | Available | Runtime SIMD where implemented, otherwise scalar |
| No default features | `no_std`, no allocation requirement | Unavailable | Portable scalar |
| No defaults plus `alloc` | `no_std` with an allocator supplied by the application | Available | Portable scalar |
| `std` without `simd` | Standard-library support and allocation | Available | Scalar |
| `simd512` | Implies `simd`, `std`, and `alloc` | Available | Adds AVX-512 capability selection on suitable x86 hosts |

`internals` only exposes an unstable facade. It does not enable SIMD or provide compatibility guarantees. Normal applications use `ops` rather than direct architecture entries.

For a portable library dependency:

```toml
[dependencies]
fgf = { version = "=3.0.1", default-features = false }
```

For portable prepared collections:

```toml
[dependencies]
fgf = { version = "=3.0.1", default-features = false, features = ["alloc"] }
```

Cargo unifies features across dependents. One dependency disabling defaults does not force the whole dependency graph to disable another dependent's request for `std` or `simd`. A constrained application must check its resolved feature graph as well as its own declaration.

## Process selection and field selection differ

`backend()` reports the process-wide choice. `backend_for::<F>()` reports the backend implemented for that field. Custom presentations may use scalar arithmetic on a host whose byte-polynomial field uses GFNI.

```rust
use fgf::{AES, FiniteField, Normal, Polynomial, backend, backend_for, has_vector_elementwise};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type NormalField = FiniteField<2, 8, Normal<AES, 0x20>>;
println!("process: {:?}", backend());
println!("AES polynomial field: {:?}", backend_for::<PolynomialField>());
println!("normal presentation: {:?}", backend_for::<NormalField>());
println!("AES vector elementwise capability: {}", has_vector_elementwise::<PolynomialField>());
assert_eq!(backend_for::<NormalField>(), fgf::Backend::Scalar);
```

A field backend report is not an instruction-level trace for every operation or length. An operation can use a narrower kernel or a scalar tail. `has_vector_elementwise` asks about that operation family, not all field operations; short rows can still take another path. The minimum-length query in the kernel reference provides the corresponding elementwise threshold.

## Overrides request a downgrade

`SIMD_BACKEND` is read at process startup when backend selection first occurs. The selection is cached, so changing the environment after first use is not a per-call backend control. Unsupported upgrades are ignored; a forced request is not proof that a requested ISA executed.

From a checkout:

```sh
SIMD_BACKEND=scalar just example shard_recovery
SIMD_BACKEND=scalar just examples
```

The complete packed examples print their field backends. Those reports should be inspected rather than treating process exit success as backend evidence. A scalar override is useful for arithmetic comparison, not a replacement for hardware capable of exercising an architecture kernel.

## Targets and fallback behavior

| Target | Architecture support |
| --- | --- |
| x86 / x86-64 | SSSE3/SSE4.2, AVX2, and GFNI; AVX-512 with `simd512` |
| AArch64 | NEON and NEON with AES/PMULL capability |
| Wasm | `simd128` when enabled in the target configuration |
| Other targets | Portable scalar |

The table describes available architecture families, not uniform acceleration of every presentation or operation. Custom extension rows whose base, relation, and storage differ from every shipped instance use the typed scalar fallback. Ordered-basis byte presentations also report scalar.

Wasm SIMD needs a module built with the explicit `+simd128` target feature and an execution environment supporting it. The repository's `just cross-check` compiles both the portable Wasm path and that SIMD configuration, together with the AArch64 library path. Compilation does not prove that either target executed.

## Allocation and first use

Scalar elements and caller-owned byte slices do not require `alloc`. `Coeff` is inline preparation; `CoeffVec`, `CoeffMatrix`, and `pack_to_vec` require allocation. Their construction belongs outside a repeated allocation-free application loop.

Feature-enabled dispatch and shared tables can have startup work. A steady-state claim refers to the warmed, prepared path documented by the operation, not every conceivable first call. [Prepared coefficients](Prepared-coefficients) explains state ownership and links the counting-allocator evidence.

## Exercise the selected surface

The example programs use the host standard library even when the dependency is built with `no_std`. These commands exercise library feature configurations, not a complete embedded application:

```sh
just doctor
just example field_arithmetic --no-default-features
just example embeddings --no-default-features
just examples --no-default-features --features alloc
just examples --all-features
```

`just doc` builds rustdoc with warnings denied. `just validate` is the repository's full submission gate; routine checks use recipes from the crate directory.

## Safety and measurements

Safe APIs validate buffer geometry, but invalid inputs can panic. A higher-level multi-call workflow can have changed state before a later call panics. Rust's memory guarantees do not imply a recovery protocol, transactional updates, or correct coefficients.

Arithmetic is variable-time. Table lookups, cache behavior, and data-dependent work make secret elements, coefficients, and buffers an unsupported security use without an independent audit.

[BENCHMARKS.md](https://github.com/nanithefkuc/fgf/blob/v3.0.1/BENCHMARKS.md) owns measured public-operation results and reproduction details. A capability report alone is not a performance measurement.

Next: [Home](Home), or [Migrating from 2.x](Migrating-from-2.x).
