# Prepared coefficients

Prepare coefficients when they repeat across buffers. A `_with` call borrows
the backend-specific state without retaining source or destination contents.
Preparation does not relax [row validation](Linear-operations-and-row-geometry).

## Prepare one coefficient

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

`Coeff` is inline and does not allocate. Preparation is useful for reuse, not
a requirement for every one-shot call.

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

This example requires `alloc` (enabled by default). One matrix drives both
block application and incremental scatter without cloning its entries:

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

The scalar checks make source-major ordering explicit.

## Gather views and lifetimes

`CoeffVec::as_ref()` feeds gather or scatter calls. `CoeffMatrix::source(s)`
borrows source `s` across all outputs; `get` and `get_at` borrow individual
entries. Views cannot outlive their owner and do not copy its tables.

## Keep allocation outside the repeated loop

Owned collections allocate during construction; their prepared applications
reuse that state. Source-slice arrays and destination buffers are also
caller-owned. First-use dispatch and table initialization are separate from
the warmed steady-state contract.

The [counting-allocator tests](https://github.com/nanithefkuc/fgf/blob/v3.0.1/tests/zero_alloc.rs)
defend allocation promises; these examples check arithmetic, not allocation.
`CoeffVec` and `CoeffMatrix` require `alloc`, not `std`.

## Persist the scalar values, not the prepared form

Serialize scalar coefficients, matrix dimensions, and source-major ordering.
Rebuild preparation after loading; its opaque backend state is not a wire
format. See [deployment](Features-and-deployment) for feature choices and the
[ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) for prepared contracts.

