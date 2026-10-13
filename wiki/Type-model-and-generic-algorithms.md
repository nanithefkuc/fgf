# Fields, elements, and scalar arithmetic

Choose a field with `FiniteField<P, N, Representation>`, then construct its
values as `Elem<F>`. `P` is the characteristic; `N` is the extension degree.
The representation fixes the meaning of the stored coordinates.

## Choose a field and compute

The default representation is a prime residue. Binary fields select a
polynomial, ordered basis, or extension explicitly:

```rust
use fgf::{AES, Elem, FiniteField, Polynomial};

type Prime = FiniteField<7, 1>;
type Binary = FiniteField<2, 8, Polynomial<AES>>;

let a = Elem::<Prime>::from_raw(12);
assert_eq!(a.to_raw(), 5);
assert_eq!((a * a).to_raw(), 4);

const X: Elem<Binary> = Elem::<Binary>::from_raw(0x53);
const Y: Elem<Binary> = Elem::<Binary>::from_raw(0xca);
assert_eq!(X.mul(Y), Elem::<Binary>::ONE);
```

`from_raw` reads coordinates, not an integer modulo the field's order. Prime
residues reduce integers; binary polynomial coordinates read coefficient bits.
Use `ZERO` and `ONE` rather than assuming raw `0` and `1` encode both identities
in every basis. [Choose a presentation](Choosing-a-field-and-basis) when the
application needs a specific convention.

Arithmetic uses `+`, `-`, `*`, `/`, `square`, `pow`, and `inv`. Division is total:
`inv(0) == 0` and `x / 0 == 0`. Algorithms requiring a nonzero inverse must
check that condition themselves.

Different representations are different types, even at the same field order.
Use an [embedding or presentation change](Embeddings-and-presentation-changes),
not a raw-coordinate copy, to preserve a value.

## Reuse an algorithm across fields

Horner evaluation needs only scalar arithmetic. Coefficients are ordered
constant-first; the reversed fold evaluates `c0 + c1*x + ...`:

```rust
use fgf::{AES, Elem, Field, FiniteField, Polynomial};

fn evaluate<F: Field>(coefficients: &[Elem<F>], x: Elem<F>) -> Elem<F> {
    coefficients.iter().rev().fold(Elem::<F>::ZERO, |sum, &c| sum * x + c)
}

type Binary = FiniteField<2, 8, Polynomial<AES>>;
type Prime = FiniteField<19, 1>;
let binary = [1u8, 2, 3].map(Elem::<Binary>::from_raw);
let prime = [1u64, 2, 3].map(Elem::<Prime>::from_raw);
assert_eq!(evaluate(&binary, Elem::<Binary>::from_raw(2)).to_raw(), 9);
assert_eq!(evaluate(&prime, Elem::<Prime>::from_raw(2)).to_raw(), 17);
assert_eq!(evaluate::<Prime>(&[], Elem::<Prime>::ONE), Elem::<Prime>::ZERO);
```

The function is unchanged; binary addition is XOR and prime addition is
modular. Choose only the capabilities an algorithm needs:

| Bound | Use |
| --- | --- |
| `Field` | Scalar arithmetic and field identity |
| `FieldBuffer` | Fixed-width byte encoding |
| `FieldKernels` | Checked packed arithmetic through `ops` |
| `HasGenerator` | A selected multiplicative generator |
| `BinaryField` / `BinaryDegree<N>` | Binary coordinate operations / an exact degree |

Import `FieldElem` for scalar trait methods on a generic `Elem<F>`. These field
traits are sealed: custom fields select supported representations or supply an
[extension specification](Downstream-scalar-fields), not a new arithmetic provider.

## Keep algebra and storage separate

`Field::DEGREE` describes the extension over the characteristic;
`Field::ORDER` counts elements. `FieldBuffer::STORAGE_BITS` describes the
storage lane, which need not equal the logarithm of the order.

The polynomial GF(2) row and binary degrees two and four are scalar-only.
GF(2) vectors use [bit-packed operations](Bit-packed-GF2); byte-buffer workflows
require `FieldBuffer` and `FieldKernels`.

Concrete shipped element families have inherent `const` arithmetic. Custom
extensions provide `const` identities and `to_raw`, but arithmetic uses runtime
trait methods and operators. An `F: Field` bound does not make a generic call
const-evaluable.

Continue with [coordinates and packing](Coordinates-and-packed-bytes), then
[choose a row operation](Linear-operations-and-row-geometry). The
[field reference](https://docs.rs/fgf/3.0.1/fgf/field/index.html) owns exact contracts.
