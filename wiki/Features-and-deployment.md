# Features and deployment

Choose features for the execution environment. They do not change field
coordinates or guarantee SIMD for every representation and operation.
Rust 1.93 or later is required.

## Choose the dependency configuration

| Configuration | Environment | Prepared forms | Routing |
| --- | --- | --- | --- |
| Defaults | `std` and allocation | Available | Runtime SIMD where implemented |
| No defaults | `no_std`, no allocator required | Unavailable | Portable scalar |
| No defaults plus `alloc` | `no_std` with an application allocator | Available | Portable scalar |
| `std` without `simd` | Standard library and allocation | Available | Scalar |
| `simd512` | Implies `simd`, `std`, and `alloc` | Available | Adds AVX-512 on capable x86 hosts |

`Prepared forms` covers the prepared maps (`Frobenius`, `PreparedEmbedding`,
`PreparedIsomorphism`), the `ops::map` and `ops::map_bits` modules, the
prepared coefficient collections (`CoeffVec`, `CoeffMatrix`), the prepared
scatter/gather/matrix `_with` forms, and `pack_to_vec`.

For portable prepared operations:

```toml
[dependencies]
fgf = { version = "=3.0.1", default-features = false, features = ["alloc"] }
```

Omit `features = ["alloc"]` for scalar arithmetic and caller-owned storage
without an allocator. Cargo unifies features: another dependent can still
enable `std` or `simd`, so constrained applications must inspect the resolved
dependency graph.

`internals` exposes an unstable facade, not additional runtime capability or a
compatibility promise. Applications normally use `ops`, not direct kernels.

## Inspect the field, not only the process

```rust
use fgf::{AES, FiniteField, Normal, Polynomial, backend, backend_for};

type PolynomialField = FiniteField<2, 8, Polynomial<AES>>;
type NormalField = FiniteField<2, 8, Normal<AES, 0x20>>;
println!("process: {:?}", backend());
println!("polynomial: {:?}", backend_for::<PolynomialField>());
println!("normal: {:?}", backend_for::<NormalField>());
assert_eq!(backend_for::<NormalField>(), fgf::Backend::Scalar);
```

`backend()` reports process selection; `backend_for::<F>()` reports the field's
implemented tier. Neither traces every call: short rows can use a narrower
path, vector loops can finish with scalar tails, and elementwise support differs
from fixed-coefficient support. Ordered-basis rows and custom extensions not
matching a shipped instance use scalar fallback.

`SIMD_BACKEND` requests a downgrade when selection first occurs. The cached
selection does not change per call; unsupported upgrades are ignored. For a
portable comparison from a checkout:

```sh
SIMD_BACKEND=scalar just examples
```

Inspect reported field backends. A successful forced-tier command alone does
not prove that ISA executed.

## Reuse state and verify the target

`Coeff` is inline preparation. `CoeffVec`, `CoeffMatrix`, and `pack_to_vec`
require `alloc`, as do the prepared maps and the `ops::map` / `ops::map_bits`
modules; construct owned preparation outside a repeated application
loop. Warmed steady-state behavior is distinct from first-use dispatch or shared
table initialization. [Prepared coefficients](Prepared-coefficients) explains
ownership and allocation evidence.

The [README](https://github.com/nanithefkuc/fgf/blob/main/README.md#platforms)
lists targets: SIMD kernels are x86-64 only and every other target runs the
portable scalar bodies. Cross-compilation is not execution proof.

```sh
just doctor
just examples --no-default-features --features alloc
just examples --all-features
just cross-check
```

Examples use a host standard library even when the dependency is `no_std`;
they do not constitute an embedded application. `just doc` checks rustdoc;
`just validate` is the repository submission gate.

## Keep safety and measurement claims separate

Checked APIs can panic on invalid geometry. A multi-call workflow can have
written output before a later call fails; it is not transactional. Arithmetic
is variable-time and requires an independent audit before handling secrets.

The [kernel reference](https://docs.rs/fgf/3.0.1/fgf/kernel/index.html) owns backend
query contracts. [BENCHMARKS.md](https://github.com/nanithefkuc/fgf/blob/main/BENCHMARKS.md)
owns measured results and reproduction, including per-operation residue routes.
