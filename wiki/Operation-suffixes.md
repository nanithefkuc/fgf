# Operation suffixes

Read destination behavior, row shape, preparation, and addressing separately.
For example, `mul_add_matrix_with` accumulates matrix results using prepared
coefficients. Suffixes do not imply every combination exists.

## What happens to the destination?

| Suffix | Meaning | Example |
| --- | --- | --- |
| `_into` | Writes the result without using the destination's previous contents. No pre-zeroing is needed. | `mul_into`: `dst = c * src` |
| `_assign` | Uses the old destination as an operand and replaces it with the result. | `mul_assign`: `dst = c * old_dst` |
| `_add` | Adds the result to the destination's previous contents. | `mul_add`: `dst = old_dst + c * src` |

Start accumulation at zero for a new sum, or at an existing value for an update.
Addition folds this into the verb: `add_assign` adds a source; `add_gather_offsets`
adds sources addressed in one region.

```rust
use fgf::{AES, Elem, FiniteField, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let source = [1u8, 2, 3];
let coefficient = Elem::<F>::from_raw(2);
let mut overwritten = [0xa5u8; 3];
ops::mul_into::<F>(&mut overwritten, coefficient, &source);
assert_eq!(overwritten, [2, 4, 6]);
let mut accumulated = [1u8; 3];
ops::mul_add::<F>(&mut accumulated, coefficient, &source);
assert_eq!(accumulated, [3, 5, 7]);
let mut scaled = source;
ops::mul_assign::<F>(&mut scaled, coefficient);
assert_eq!(scaled, overwritten);
```

These are field equations, not integer equations. In this binary field, adding one means XOR with one. In a prime field, addition is modular integer addition.

## What does `_with` reuse?

`_with` borrows reusable preparation without changing destination behavior.
`mul_into_with` overwrites; `mul_add_with` accumulates. Preparation does not
skip validation or imply allocation-free construction.

[Prepared coefficients](Prepared-coefficients) covers coefficient, vector,
and matrix ownership. Bit operations use the same convention:
`bits::xor_range_with` borrows an `XorRange` with cached masks.

## Which rows does the call address?

`_at` means destination rows are addressed by explicit offsets within a mutable region, instead of being consecutive rows. In byte-lane `ops`, offsets are measured in bytes. Destination spans must fit, begin on lane boundaries relative to the region, and be disjoint. Bytes outside those spans are not outputs of the call.

`_offsets` names offset-addressed sources instead. For example, `add_gather_offsets` gathers sources from a shared immutable region into one destination. It does not change destination addressing. [Linear operations and row geometry](Linear-operations-and-row-geometry) includes an `_at` example with untouched frame bytes.

The shape words answer a different question:

| Shape | Source and output relationship |
| --- | --- |
| `scatter` | One source contributes to several outputs. |
| `gather` | Several sources contribute to one output. |
| `matrix` | Several sources contribute to several outputs; the matrix need not be square. |
| `elementwise` | Each position combines the corresponding elements of two buffers. |

`mul_add_matrix_with` takes destination, row length, prepared matrix, then
sources. `mul_add_matrix_at` instead takes destination offsets and ordinary
coefficient terms. There is no prepared `_at_with` matrix form.

## Allocation

`_to_vec` allocates output; `pack` instead fills caller-owned storage.
`_with` means reusable state, not scratch workspace: `fgf` has no public
`_scratch` operation.

Use [row geometry](Linear-operations-and-row-geometry) to select the call and
the [ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) for exact sizes,
empty-input behavior, and errors. Names are not a substitute for the contract:
`pack` and `mul_elementwise` overwrite without an `_into` suffix.
