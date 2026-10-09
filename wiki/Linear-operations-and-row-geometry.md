# Linear operations and row geometry

Packed linear operations apply field coefficients independently at every element position of a row. Their shape names describe how sources feed outputs, while their suffixes describe what happens to the destination.

This guide targets fgf 3.0.0. [Coordinates and packed bytes](Coordinates-and-packed-bytes) establishes the lane format. The [ops reference](https://docs.rs/fgf/3.0.0/fgf/ops/index.html) owns each call's validation and mutation contract.

## Match the equation to the operation

| Equation at each lane | Shape | Operation |
| --- | --- | --- |
| `dst += src` | One source, one output | `add_assign` |
| `dst += c * src` | One source, one output | `mul_add` |
| `dst = c * src` | One source, one output | `mul_into` |
| `dst *= c` | Destination is also the source | `mul_assign` |
| `out[o] += c[o] * src` | One source, many outputs | `mul_add_scatter` |
| `dst = sum(c[s] * src[s])` | Many sources, one output | `mul_into_gather` |
| `out[o] = sum(c[s,o] * src[s])` | Many sources, many outputs | `mul_into_matrix` |
| `dst[i] = a[i] * b[i]` | Two varying operands | `mul_elementwise` |

`_into` ignores previous destination contents; `_add` accumulates into them; `_assign` makes the destination an operand. Accumulation needs an intentional initial value, often zero. Overwrite needs no pre-zeroing. `_with` selects reusable prepared state, not a different mathematical operation.

Scatter can update all parity rows when one source arrives. Gather can reconstruct one output from several inputs. Matrix application performs several such linear combinations together. None of these operations chooses coefficients, solves a system, or validates a code's erasure-correction capability.

## A rectangular example exposes coefficient axes

Let two input rows `a` and `b` produce three outputs:

```text
out[0] = 1*a + 4*b
out[1] = 2*a + 5*b
out[2] = 3*a + 6*b
```

The source-major coefficient order is `[1, 2, 3, 4, 5, 6]`: all output coefficients for `a`, then all output coefficients for `b`. Its index is `source * outputs + output`. The application equations can be written as a conventional output-by-source matrix, but `fgf`'s coefficient storage uses the opposite axis order.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let a = [1u8, 2];
let b = [3u8, 4];
let from_a = [1u8, 2, 3].map(Elem::<F>::from_raw);
let from_b = [4u8, 5, 6].map(Elem::<F>::from_raw);
let terms = [
    (from_a.as_slice(), a.as_slice()),
    (from_b.as_slice(), b.as_slice()),
];
let mut outputs = [0xa5u8; 6];
ops::mul_into_matrix::<F>(&mut outputs, 2, 3, &terms);
assert_eq!(outputs, [13, 18, 13, 16, 9, 30]);
```

Each output row occupies two bytes. In the first lane, for example, `1*1 + 4*3` is `1 XOR 12 = 13`. The initialized `0xa5` bytes disappear because this is an overwrite call.

## Scatter is the same matrix applied incrementally

The same terms can accumulate as inputs arrive. The result agrees with the block-final overwrite result when accumulation begins at zero.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let a = [1u8, 2];
let b = [3u8, 4];
let from_a = [1u8, 2, 3].map(Elem::<F>::from_raw);
let from_b = [4u8, 5, 6].map(Elem::<F>::from_raw);
let mut outputs = [0u8; 6];
ops::mul_add_scatter::<F>(&mut outputs, 2, &from_a, &a);
ops::mul_add_scatter::<F>(&mut outputs, 2, &from_b, &b);
assert_eq!(outputs, [13, 18, 13, 16, 9, 30]);
```

A source replacement can use the delta `new_source - old_source`: scattering that delta updates the existing outputs. Subtraction is XOR for binary fields, but not for prime fields. The [streaming example](https://github.com/nanithefkuc/fgf/blob/v3.0.0/examples/streaming_encode.rs) demonstrates this workflow over degree-sixteen lanes.

## Offset destinations preserve unrelated frame bytes

`_at` addresses destination rows by byte offsets within one mutable region. Named rows must be in bounds, start on field-element boundaries relative to that region, and be pairwise disjoint. Unnamed bytes remain outside the operation.

```rust
use fgf::{AES, FiniteField, Elem, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let coefficients = [Elem::<F>::ONE, Elem::<F>::from_raw(2)];
let source = [1u8, 2];
let terms = [(coefficients.as_slice(), source.as_slice())];
let mut frame = [0xeeu8; 8];
frame[1..3].fill(0);
frame[5..7].fill(0);
ops::mul_add_matrix_at::<F>(&mut frame, 2, &[1, 5], &terms);
assert_eq!(frame, [0xee, 1, 2, 0xee, 0xee, 2, 4, 0xee]);
```

This is accumulation, so the example initializes only the output slots to zero. Offset lists do not relax borrowing: sources still borrow memory disjoint from the destination. `split_at_mut` can separate source and destination regions of one allocation before constructing their views.

`add_gather_offsets` addresses sources in a shared region instead. `_offsets` names source addressing; `_at` names destination addressing. They are not interchangeable suffixes.

## Geometry and errors

`row_len` is measured in bytes and must describe whole lanes. Sources must have the documented row length, every term must supply the output coefficient count, and destination storage must cover the used row prefix or every named offset span. Prepared matrix dimensions are `(sources, outputs)`, not implicitly square.

Invalid lengths, counts, offsets, or overlapping destination rows panic through the checked surface. Valid zero-work shapes still undergo structural validation. An empty source sum overwrites with zero in an overwrite operation; an empty sum contributes nothing to an accumulation operation. Consult each operation's reference rather than treating all empty geometries alike.

The [shard recovery example](https://github.com/nanithefkuc/fgf/blob/v3.0.0/examples/shard_recovery.rs) derives a fixed recovery matrix. Its supplied equations cover the stated erasure patterns, not arbitrary decoding.

Next: [Prepared coefficients](Prepared-coefficients).
