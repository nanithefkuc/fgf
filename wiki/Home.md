# fgf guides

`fgf` provides finite-field arithmetic for individual values and packed buffers. These guides start with choosing a field and constructing its elements, then work through arithmetic, storage, and conversions between fields. **3.0.1 is the base v3 release** used throughout, with Rust **1.93 or later**.

> This library is AI-assisted. Its code requires independent audit. Arithmetic is variable-time; safe Rust does not make operations suitable for secret elements, coefficients, or buffers.

## Start with a field and a few values

Begin with [Fields, elements, and scalar arithmetic](Type-model-and-generic-algorithms). It introduces `FiniteField` and `Elem<F>` with a small computation before explaining how to write an algorithm that works over several fields.

## Apply arithmetic to buffers

[Operation suffixes](Operation-suffixes) explains what names such as `_into`, `_add`, `_with`, and `_at` say about a call. [Linear operations and row geometry](Linear-operations-and-row-geometry) then shows how to express scaling and linear combinations, and [Prepared coefficients](Prepared-coefficients) shows how to reuse the same coefficients across blocks.

For layouts other than byte-sized binary elements, use [Prime fields](Prime-fields) or [Bit-packed GF(2)](Bit-packed-GF2). The buffer examples introduce their layouts where they are needed; the presentation guides below explain what the stored coordinates mean.

## Understand the presentation

[Field presentations and bases](Choosing-a-field-and-basis) explains polynomial, normal, and Cantor coordinates. [Coordinates and packed bytes](Coordinates-and-packed-bytes) connects those coordinates to integer conversions and serialization. [Quadratic towers](Quadratic-towers) builds larger fields from base-field components.

## Relate fields and define new ones

[Embeddings and presentation changes](Embeddings-and-presentation-changes) covers value-preserving conversions, subfield membership, trace, and norm. [Custom prime fields and extensions](Downstream-scalar-fields) explains how to choose a new prime modulus or supply an extension relation without implementing the sealed field traits.

## Build or upgrade an application

[Features and deployment](Features-and-deployment) covers `no_std`, allocation, and backend selection. Existing 2.x applications can use [Migrating from 2.x to 3.0.1](Migrating-from-2.x); new applications can start directly with the examples here.

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
