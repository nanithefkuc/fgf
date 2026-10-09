# Migrating from 3.0.0-rc.1 to 3.0.1

fgf 3.0.1 completes the canonical family cutover: named prime fields are rows of `FiniteField`, binary presentations are rows of `FiniteField<2, N, R>`, quadratic towers are `Extension` instances, and residue/index conversions plus embeddings are field-level capabilities. Every breaking entry below names its replacement. Historical rc.1 fragments are fenced as text because they do not compile against 3.0.1; all Rust-fenced replacements target 3.0.1.

The [3.0.1 changelog](https://github.com/nanithefkuc/fgf/blob/v3.0.1/CHANGELOG.md) is the authoritative release record. [Migrating from 2.x](Migrating-from-2.x) covers the earlier element-wrapper migration; this page covers only the rc.1-to-3.0.1 cutover.

## Named prime fields are family rows

`Mersenne31`, `Goldilocks`, and `QuadMersenne31` are type aliases of canonical family rows instead of marker structs. Moduli, raw widths, bytes, generators, arithmetic, and kernel routing are unchanged. The aliases are no longer value expressions.

```text
let field = Mersenne31;
let limbs = quad_mersenne31::Elem::from_raw(re, im);
```

```rust
use fgf::{Elem, Mersenne31, QuadMersenne31};

let field = Mersenne31::default();
let limbs = Elem::<QuadMersenne31>::from_raw([re, im]);
```

`QuadMersenne31` limb pairs are arrays: `from_raw` takes one `[re, im]` array, `from_components` takes `[re, im]`, and `to_raw` and `to_components` return `[re, im]`. Tuple destructuring becomes array destructuring, and element `Display` prints `[re, im]` in the base field's form.

## Binary fields use the family spelling

The `Binary<N, R>` marker is removed. Binary fields are the canonical family rows `FiniteField<2, N, R>` with `u32` degree parameters.

```text
use fgf::Binary;
type F = Binary<8, Polynomial<AES>>;
```

```rust
use fgf::{FiniteField, Polynomial, AES};
type F = FiniteField<2, 8, Polynomial<AES>>;
```

`BinaryRepr` extends `FiniteFieldRepr<2, N>`, which owns `NAME` and `VALID`; `DESCRIPTION` and `LOG_EXP` stay on `BinaryRepr`. Qualified paths change accordingly, and marker `Debug` prints the family spelling with the presentation label.

## Towers are extension instances

Shipped towers are extension rows `FiniteField<2, N, Extension<S, 2>>`, with `S` an `ExtensionSpec<2>`. `Tower`, `TowerSpec`, and `TowerGeneratorSpec` are removed. Raw words, bytes, coordinate words, generators, embeddings, and kernel routing are unchanged, and each instance keeps its `const` element surface and `mul_alpha`.

```text
use fgf::Tower;
type F = FiniteField<2, 16, Tower<Rijndael16>>;
```

```rust
use fgf::{Extension, FiniteField, Rijndael16};
type F = FiniteField<2, 16, Extension<Rijndael16, 2>>;
```

A custom tower becomes an `ExtensionSpec<2>`: keep the same `Base` and `NAME`, use `Packed<W>` storage at the field's width (`PackedPadded<u8>` below one byte), and build the relation constant-first with `Relation::from_binary_words::<Base>(&[B, A])`, the reverse of the old `LINEAR_COEFFICIENT = A`, `CONSTANT_COEFFICIENT = B` field order. Generic bounds `R: BinaryRepr<N>` that reached towers become field-level bounds (`F: BinaryDegree<N>` or `F: BinaryField`). Wide shipped instances' component conversions take and return arrays matching `ExtensionField<2>`. `TowerGeneratorSpec` has no replacement; custom extension rows carry no generator. Custom rows also lose the `const` element surface; `Elem::ZERO`, `Elem::ONE`, and `to_raw` stay `const`.

## Residue and index conversions move to `FieldElem`

`FieldElem` owns `from_residue`/`to_residue` and `from_index`/`to_index`, shared by every row. `BinaryField::to_coordinates`/`from_coordinates` are removed; `to_index`/`from_index` carry identical words below degree 64 widened to `u128`. `CoordinateError` is replaced by `fgf::RangeError`.

```text
let word: u64 = value.to_coordinates();
```

```rust
let word: u128 = value.to_index();
```

Residues and indices are distinct: the index of one is the presentation's coordinate encoding of one.

## Embeddings move to the crate root

The `field::binary` embedding paths are removed. The characteristic-independent surface lives in `field::embedding`, re-exported at the root, over every pair of rows of equal characteristic whose degrees divide. Binary embedding evaluation no longer uses precomputed linear columns; results are unchanged.

```text
use fgf::field::binary::embedding::{Embedding, EmbeddingError};
```

```rust
use fgf::{Embedding, EmbeddingError};
```

## Sealed capabilities

`Field`, `HasGenerator`, and `FieldBuffer` are sealed. A custom prime field becomes the family's degree-one row `FiniteField<P, 1>` at any checked `u64` prime; a custom extension becomes an `ExtensionSpec` over a crate row.

## Check the application boundary

Migration checks compare canonical wire fixtures, generator-dependent schedules, rectangular coefficient geometry, and prime inputs near the modulus. A passing type check alone does not establish that a raw-coordinate reinterpretation is correct. From a crate checkout, `just examples` exercises complete workflows and `just doc` builds the API reference.
