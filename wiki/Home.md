# fgf guides

`fgf` supplies finite-field values and arithmetic over application-owned packed buffers. These guides explain field presentations, generic algorithms, and the mapping from equations to bulk operations. They target **fgf 3.0.1** and Rust **1.93 or later**.

> This library is AI-assisted. Its code requires independent audit. Arithmetic is variable-time; safe Rust does not make operations suitable for secret elements, coefficients, or buffers.

## Using fgf

1. [The type model and generic algorithms](Type-model-and-generic-algorithms) — distinguish a field from its elements and choose the necessary trait bounds.
2. [Choosing a field and basis](Choosing-a-field-and-basis) — match the mathematical field and coordinate convention to an application.
3. [Coordinates and packed bytes](Coordinates-and-packed-bytes) — encode values without confusing representation conversion with serialization.
4. [Linear operations and row geometry](Linear-operations-and-row-geometry) — map scaling and linear combinations to scatter, gather, and matrix calls.
5. [Prepared coefficients](Prepared-coefficients) — reuse coefficients across blocks and incremental updates.

Specialized buffer guides cover [prime fields](Prime-fields) and [bit-packed GF(2)](Bit-packed-GF2). [Features and deployment](Features-and-deployment) covers portable builds, allocation, and runtime backends.

## Working across presentations

[Choosing a field and basis](Choosing-a-field-and-basis) establishes what raw coordinates mean. [Embeddings and presentation changes](Embeddings-and-presentation-changes) maps values between bases and into subfields. [Quadratic towers](Quadratic-towers) explains components, defining relations, and custom extension specifications.

## Extending or upgrading

- [Downstream scalar fields](Downstream-scalar-fields) explains choosing a crate-defined row, the `Residue` row for a custom prime modulus, and an `ExtensionSpec` for a custom extension; `Field` itself is sealed.
- [Migrating from 2.x](Migrating-from-2.x) gives replacements for removed names and explains changed construction and buffer behavior.
- [Migrating from 3.0.0-rc.1](Migrating-from-3.0.0-rc.1) covers the canonical family cutover: named rows, binary spellings, extension instances, residue and index conversions, and the field-level embedding surface.

## A first useful computation

A supplied pair of coefficients defines a linear combination at every byte position. In the AES field, multiplication is polynomial arithmetic, not integer multiplication modulo 256.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let a = [1u8, 2, 3];
let b = [4u8, 5, 6];
let coefficients = [Elem::<F>::ONE, Elem::<F>::from_raw(2)];
let mut out = [0xa5u8; 3];
ops::mul_into_gather::<F>(&mut out, &coefficients, &[&a, &b]);
assert_eq!(out, [9, 8, 15]);
```

The overwrite operation ignores the previous destination bytes. A production encoder or decoder supplies its own coefficients, matrix construction, and recovery decisions: `fgf` applies arithmetic, not a coding protocol.

## Reference and executable examples

The [versioned API reference](https://docs.rs/fgf/3.0.1/fgf/) owns exact signatures, panics, and invariants. The [README](https://github.com/nanithefkuc/fgf/blob/v3.0.1/README.md) covers installation and scope. The [changelog](https://github.com/nanithefkuc/fgf/blob/v3.0.1/CHANGELOG.md) records release changes; [BENCHMARKS.md](https://github.com/nanithefkuc/fgf/blob/v3.0.1/BENCHMARKS.md) owns measurements.

The [examples directory](https://github.com/nanithefkuc/fgf/tree/v3.0.1/examples) contains complete programs with assertions and printed results. From a crate checkout:

```sh
just example shard_recovery
just examples
```

The Rust blocks in these guides are self-contained unless marked as historical text. Blocks using prepared collections require `alloc`; their pages state that requirement. The guides use local type aliases for readability, not additional public names supplied by the crate.
