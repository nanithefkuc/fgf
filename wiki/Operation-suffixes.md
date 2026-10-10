# Operation suffixes

An operation name tells how it uses the destination, whether it takes prepared state, and how rows are addressed. Read those parts separately: `mul_add_matrix_with` accumulates matrix results using prepared coefficients, while `mul_add_matrix_at` accumulates into destination rows at explicit offsets.

This guide uses fgf 3.0.1. The [ops reference](https://docs.rs/fgf/3.0.1/fgf/ops/index.html) gives the exact signatures and validation rules. Suffixes describe a contract; they do not imply that every possible combination exists.

## What happens to the destination?

| Suffix | Meaning | Example |
| --- | --- | --- |
| `_into` | Writes the result without using the destination's previous contents. No pre-zeroing is needed. | `mul_into`: `dst = c * src` |
| `_assign` | Uses the old destination as an operand and replaces it with the result. | `mul_assign`: `dst = c * old_dst` |
| `_add` | Adds the result to the destination's previous contents. | `mul_add`: `dst = old_dst + c * src` |

Accumulation needs a deliberate initial value. Zero is appropriate for building a new sum; an existing result is appropriate for an incremental update. The addition operations fold that behavior into the verb: `add_assign` adds a source to the destination, and `add_gather` adds several sources. They do not need a separate multiplication coefficient.

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

`_with` means that a call takes reusable prepared state. In `ops`, that is usually a prepared coefficient, vector, or matrix. The destination behavior does not change: `mul_into_with` overwrites, and `mul_add_with` accumulates.

```rust
use fgf::{AES, Elem, FiniteField, Polynomial, ops};

type F = FiniteField<2, 8, Polynomial<AES>>;
let coefficient = ops::Coeff::<F>::new(Elem::<F>::from_raw(2));
let mut destination = [0xa5u8; 3];
ops::mul_into_with::<F>(&mut destination, &coefficient, &[1, 2, 3]);
assert_eq!(destination, [2, 4, 6]);
ops::mul_add_with::<F>(&mut destination, &coefficient, &[1, 2, 3]);
assert_eq!(destination, [0, 0, 0]);
```

Prepared state can also describe geometry. For example, `bits::xor_range_with` borrows an `XorRange` containing a bit range and its masks. `_with` does not mean an arbitrary extra argument, caller scratch, or permission to skip validation. [Prepared coefficients](Prepared-coefficients) explains the ownership and allocation requirements of the coefficient forms.

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

For example, `mul_add_matrix_with` accumulates several linear combinations using a prepared matrix. Its argument order is destination, row length, prepared matrix, then sources. Preparing the matrix does not relax row-length or source-count checks. `mul_add_matrix_at` takes offset-addressed destinations and ordinary coefficient terms; fgf 3.0.1 has no prepared `_at_with` matrix form.

## Allocation and scratch are separate questions

`_to_vec` marks an allocating conversion, such as `pack_to_vec`. The corresponding `pack` call fills caller-owned storage. `_with` alone makes no allocation promise: a single `Coeff` is inline, while owned coefficient collections require `alloc` during construction.

The ecosystem's `_scratch` suffix means a call borrows caller-provided workspace instead of allocating that workspace itself. Scratch carries no reusable mathematical state between calls, unlike preparation. fgf 3.0.1 does not expose a public `_scratch` operation; there is no scratch argument to add to its `_with` calls.

Names are a useful starting point, but the operation's reference decides its exact buffer sizes, empty-input behavior, and failure conditions. Not every output-writing operation includes `_into`: `pack` and `mul_elementwise` write their results to the destination under their documented contracts.
