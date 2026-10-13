# Linear operations and row geometry

Choose the equation and output shape, then supply the coefficients and packed
rows. `ops` applies arithmetic; it does not choose coding coefficients or solve
a recovery problem. Examples use one-byte AES elements; wider fields follow
the same flow with their [lane encoding](Coordinates-and-packed-bytes).

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

Overwrite calls need no pre-zeroing. Accumulation needs a deliberate starting
value; `_assign` uses the old destination as an operand. See
[operation suffixes](Operation-suffixes) for the naming reference.

## Apply a rectangular matrix

Two sources produce three outputs. Coefficients are source-major:
`[1, 2, 3, 4, 5, 6]` lists the coefficients for the first source, then the
second. The index is `source * outputs + output`, not the output-major
order often written on paper.

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

Each output occupies a complete row. Prior destination bytes are ignored.

## Scatter is the same matrix applied incrementally

Start at zero and scatter each source as it arrives to obtain the same result:

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

For a source replacement, scatter `new_source - old_source` into existing
outputs. Subtraction is XOR only in binary fields. The
[streaming example](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/streaming_encode.rs)
uses this update with wider lanes.

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

`row_len` counts bytes, not elements. Rows must contain whole lanes; term
coefficient counts and prepared `(sources, outputs)` dimensions must match
the declared geometry. Invalid spans, counts, or overlapping outputs panic.

Empty overwrite sums produce zero; empty accumulation sums contribute nothing.
Zero-work calls still validate structure. A multi-call workflow is not
transactional: validate it before writing if later failure must leave output
unchanged.

For repeated coefficients, continue with [preparation](Prepared-coefficients).
The [ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) owns each call's
exact validation and empty-input rules.
