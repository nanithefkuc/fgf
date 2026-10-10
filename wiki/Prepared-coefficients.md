# Prepared coefficients

When the same coefficient is used across several buffers, prepare it once and reuse it. A prepared coefficient keeps the backend-specific multiplier state; an `_with` call borrows that state and applies it to new data.

The examples use fgf 3.0.1 and the row shapes from [Linear operations and row geometry](Linear-operations-and-row-geometry). The [ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) describes the prepared types.

## Separate values, preparation, and destination storage

`Elem<F>` is the coefficient's field value. `Coeff<F>` prepares that value for the process backend. Neither preparation nor reuse takes ownership of the source or destination buffers, and preparation retains no buffer contents.

```rust
use fgf::{Elem, Extension, FiniteField, Rijndael16, ops};

type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
let value = Elem::<F>::from_raw(0x0108);
let prepared = ops::Coeff::<F>::new(value);
for raw in [0x1234u16, 0xabcd] {
    let source = raw.to_le_bytes();
    let mut destination = [0xa5u8; 2];
    ops::mul_into_with::<F>(&mut destination, &prepared, &source);
    assert_eq!(destination, (Elem::<F>::from_raw(raw) * value).to_bytes());
}
assert_eq!(prepared.value(), value);
```

`Coeff` construction allocates nothing. It is still computation, and its size and contents depend on the field and backend. Preparation is appropriate when a coefficient repeats; it is not a blanket recommendation to prepare every one-shot call.

## Collections encode a coefficient axis

| Type | Meaning | Construction storage |
| --- | --- | --- |
| `Coeff<F>` | One coefficient | Inline, no allocation |
| `CoeffVec<F>` | A sequence indexed by the consuming operation | Owned allocation; requires `alloc` |
| `CoeffMatrix<F>` | A source-major `(sources, outputs)` matrix | Owned allocation; requires `alloc` |
| `CoeffRef<'_, F>` | One borrowed prepared entry | Borrows its owner |
| `CoeffVecRef<'_, F>` | A borrowed coefficient sequence | Borrows its owner |

A vector position denotes an output for scatter and a source for gather. The vector does not silently supply either axis. `CoeffMatrix::from_source_major` makes both dimensions explicit, with coefficient index `source * outputs + output`.

## Apply a matrix repeatedly

The following block requires the `alloc` feature, enabled by default. One matrix supplies two source terms and three output rows for each block. A borrowed source view also drives incremental scatter without cloning prepared entries.

<!-- requires: alloc -->
```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let coefficients = [1u8, 2, 3, 4, 5, 6].map(Elem::<F>::from_raw);
let matrix = ops::CoeffMatrix::<F>::from_source_major(2, 3, &coefficients);
let blocks = [([1u8, 2], [3u8, 4]), ([5u8, 6], [7u8, 8])];
for (a, b) in blocks {
    let mut block_output = [0xa5u8; 6];
    ops::mul_into_matrix_with::<F>(&mut block_output, 2, &matrix, &[&a, &b]);
    let mut incremental = [0u8; 6];
    ops::mul_add_scatter_with::<F>(&mut incremental, 2, matrix.source(0).unwrap(), &a);
    ops::mul_add_scatter_with::<F>(&mut incremental, 2, matrix.source(1).unwrap(), &b);
    assert_eq!(incremental, block_output);
    for output in 0..3 {
        for lane in 0..2 {
            let expected = coefficients[output] * Elem::<F>::from_raw(a[lane])
                + coefficients[3 + output] * Elem::<F>::from_raw(b[lane]);
            assert_eq!(block_output[output * 2 + lane], expected.to_raw());
        }
    }
}
let borrowed = matrix.get_at(0, 1).unwrap();
let mut scaled = [0xa5u8; 2];
ops::mul_into_with::<F>(&mut scaled, &borrowed, &[1, 2]);
assert_eq!(scaled, [2, 4]);
```

The same source view works for incremental scatter and block-final matrix application. The scalar calculation in the example also checks the coefficient ordering. Reuse needs neither a second matrix nor a cloned coefficient table.

## Gather views and lifetimes

A `CoeffVec::as_ref()` view feeds gather or scatter `_with` operations. `CoeffMatrix::source(s)` yields the same view for source `s` across every output. `get` and `get_at` return `Option<CoeffRef>` for individual positions.

Views cannot outlive their owning collection. Moving or dropping the owner while a view is still borrowed is rejected by Rust. A copied borrowed view does not copy the backend tables; cloning an owning preparation is a different operation.

## Allocation and startup

Owned collections allocate during construction. Their documented applications reuse the prepared state without allocation. That steady-state contract does not mean an application has no first-use setup: feature-enabled dispatch or shared table initialization can have separate startup work. Buffers and source-slice arrays must also be supplied without allocating inside the repeated region if the application needs an allocation-free loop.

The crate's [counting-allocator tests](https://github.com/nanithefkuc/fgf/blob/v3.0.1/tests/zero_alloc.rs) defend its steady-state promises. These examples check arithmetic and reuse; they are not allocation measurements.

Preparation hoists coefficient work, not validation. A wrong source count, destination span, or row length still panics through a prepared operation. `CoeffVec` and `CoeffMatrix` require `alloc`, not `std`; [deployment](Features-and-deployment) covers that feature combination.

## Persist the scalar values, not the prepared form

Prepared state is bound to the process backend and deliberately opaque. Persist the coefficient values in their agreed field encoding and reconstruct preparation after loading. An application also persists matrix dimensions and source-major ordering; raw coefficient bytes alone do not describe geometry.

