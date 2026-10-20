# fgf guides

Define a field, compute with `Elem<F>`, then apply the same arithmetic to packed
buffers. These guides use 3.0.2 and require Rust 1.93 or later.

> This library is AI-assisted and requires independent audit. Arithmetic is
> variable-time; safe Rust does not make it suitable for secret data.

## Start here

| Step | Guide |
| --- | --- |
| Choose a field and construct values | [Fields and scalar arithmetic](Type-model-and-generic-algorithms) |
| Match a protocol's coordinates | [Presentations and bases](Choosing-a-field-and-basis) |
| Encode values for bulk work | [Coordinates and packed bytes](Coordinates-and-packed-bytes) |
| Express a linear combination | [Row operations and geometry](Linear-operations-and-row-geometry) |
| Reuse coefficients across blocks | [Prepared coefficients](Prepared-coefficients) |
| Choose runtime requirements | [Features and deployment](Features-and-deployment) |

[Operation suffixes](Operation-suffixes) is the naming reference. Exact
signatures, panics, and invariants belong to the
[API reference](https://docs.rs/fgf/3.0.2/fgf/).

## A linear combination

Here `out[i] = a[i] + 2*b[i]` in the AES binary field. The overwrite call needs
no pre-zeroing:

```rust
use fgf::{AES, Elem, FiniteField, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let a = [1u8, 2, 3];
let b = [4u8, 5, 6];
let coefficients = [Elem::<F>::ONE, Elem::<F>::from_raw(2)];
let mut out = [0xa5u8; 3];
ops::mul_into_gather::<F>(&mut out, &coefficients, &[&a, &b]);
assert_eq!(out, [9, 8, 15]);
```

The application supplies coefficients and recovery decisions; `fgf` is not a codec.

## Follow the application's needs

| Need | Guide |
| --- | --- |
| A prime modulus or specialized prime representation | [Prime fields](Prime-fields) |
| One bit per GF(2) vector element | [Bit-packed GF(2)](Bit-packed-GF2) |
| Larger fields from base components | [Quadratic towers](Quadratic-towers) |
| Value-preserving field conversions, trace, or norm | [Embeddings](Embeddings-and-presentation-changes) |
| A custom extension relation or storage | [Custom fields](Downstream-scalar-fields) |
| Upgrade an existing application | [Migrating from 2.x](Migrating-from-2.x) |

Rust blocks are self-contained unless marked as historical text; pages identify
`alloc` requirements. Local type aliases are examples, not extra crate exports.
The [example programs](https://github.com/nanithefkuc/fgf/tree/v3.0.2/examples)
run from a checkout with `just example NAME` or `just examples`.

[Installation](https://github.com/nanithefkuc/fgf/blob/main/README.md),
[release changes](https://github.com/nanithefkuc/fgf/blob/main/CHANGELOG.md), and
[measurements](https://github.com/nanithefkuc/fgf/blob/main/BENCHMARKS.md)
each have their own reference page.
