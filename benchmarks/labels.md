# Canonical field labels

Labels identify field presentations independently of their API spellings.
Field size alone does not identify the polynomial, basis, tower relation,
or packing used by a benchmark.

The numeric part of a binary-field label is its extension degree over $GF(2)$.
Thus `gf1` denotes $GF(2)$ independently of the v2 spelling `Gf2`.

## API mapping

The v2 column describes released `v2.1.0`. The v3 column gives field markers;
v3 elements use `Elem<F>`. `Rs16Tower` is a benchmark-owned `TowerSpec`, not a
production alias. A missing v2 marker has no public version counterpart.

| Label | v2 Canonical | v3 Canonical | Definition |
| --- | --- | --- | --- |
| `gf8b` | `Gf8B` | `Binary<8, Polynomial<0x11B>>` | $GF(2^8)$ in the AES polynomial basis, $x^8+x^4+x^3+x+1$ |
| `gf8d` | `Gf8D` | `Binary<8, Polynomial<0x11D>>` | $GF(2^8)$ in the Reed–Solomon polynomial basis, $x^8+x^4+x^3+x^2+1$ |
| `gf16b` | `Gf16` | `Binary<16, Tower<Rijndael16>>` | Quadratic extension of `gf8b`, $t^2+t+\texttt{0x20}=0$ |
| `gf16d` | - | `Binary<16, Tower<Rs16Tower>>` | Quadratic extension of `gf8d`, $t^2+\texttt{0x02}t+\texttt{0x80}=0$ |
| `gf32b` | `Gf32` | `Binary<32, Tower<Rijndael32>>` | Quadratic extension of `gf16b`, $t^2+t+\texttt{0x2000}=0$ |
| `gf64b` | `Gf64` | `Binary<64, Tower<Rijndael64>>` | Quadratic extension of `gf32b`, $t^2+t+\texttt{0x2000\_0000}=0$ |
| `fp8` | `FanPaar8` | `Binary<8, Tower<FanPaar8>>` | Fan–Paar tower from GF(2), through degrees 2 and 4 |
| `fp16` | `FanPaar16` | `Binary<16, Tower<FanPaar16>>` | Fan–Paar quadratic extension of `fp8` |
| `fp32` | `FanPaar32` | `Binary<32, Tower<FanPaar32>>` | Fan–Paar quadratic extension of `fp16` |
| `fp64` | `FanPaar64` | `Binary<64, Tower<FanPaar64>>` | Fan–Paar quadratic extension of `fp32` |
| `gf1` | `Gf2` and `bits` | `Binary<1, Polynomial<0x3>>` and `bits` | $GF(2)=GF(2^1)$; bulk cases use bit-packed `bits` storage, not one byte per element |
| `m31` | `Mersenne31` | `Mersenne31` | Prime field $GF(2^{31}-1)$ |
| `qm31` | `QuadMersenne31` | `QuadMersenne31` | $GF((2^{31}-1)^2)$, coordinates $a+bi$ with $i^2=-1$ |
| `gld` | `Goldilocks` | `Goldilocks` | Prime field $GF(2^{64}-2^{32}+1)$ |

`fp8` identifies the base of the wider Fan–Paar panels; those panels time
`fp16`, `fp32`, and `fp64`. Normal and Cantor presentations require separate
labels including their polynomial and basis seed. They are not included
under a polynomial-field or Fan–Paar label.

## Recursive tower definitions

The Rust API uses `Tower<S>` with `S: TowerSpec`, not `Tower<Base, A, B>`.
Each row specifies the exact base marker and the raw base-coordinate
coefficients of $t^2+At+B=0$: `LINEAR_COEFFICIENT` supplies $A$ and
`CONSTANT_COEFFICIENT` supplies $B$. Recursing through the base column gives
the complete presentation without omitted parameters or deeply nested names.

| Spec | Exact base marker | Linear coefficient (raw) | Constant coefficient (raw) | Generator (raw) |
| --- | --- | --- | --- | --- |
| `Rijndael16` | `Binary<8, Polynomial<0x11B>>` | `0x1` | `0x20` | `0x0108` |
| `Rijndael32` | `Binary<16, Tower<Rijndael16>>` | `0x1` | `0x2000` | `0x0001_0002` |
| `Rijndael64` | `Binary<32, Tower<Rijndael32>>` | `0x1` | `0x2000_0000` | `0x0000_0001_0000_0004` |
| `Rs16Tower` | `Binary<8, Polynomial<0x11D>>` | `0x2` | `0x80` | - |
| `FanPaar2` | `Binary<1, Polynomial<0x3>>` | `0x1` | `0x1` | `0x2` |
| `FanPaar4` | `Binary<2, Tower<FanPaar2>>` | `0x2` | `0x1` | `0x7` |
| `FanPaar8` | `Binary<4, Tower<FanPaar4>>` | `0x4` | `0x1` | `0x2D` |
| `FanPaar16` | `Binary<8, Tower<FanPaar8>>` | `0x10` | `0x1` | `0xE2DE` |
| `FanPaar32` | `Binary<16, Tower<FanPaar16>>` | `0x100` | `0x1` | `0x03E2_1CEA` |
| `FanPaar64` | `Binary<32, Tower<FanPaar32>>` | `0x10000` | `0x1` | `0x070F_870D_CD9C_1D88` |

Tower words place the constant component in the low half and the extension
component in the high half, recursively. Packed elements use little-endian
words with adjacent components, not separate planes. A native competitor
with another basis or component order uses a validated isomorphism; its raw
words are not treated as interchangeable with fgf coordinates.

`gf16d` freezes the registered relation and packing, not every tower rooted
at `0x11D`. Another relation requires another label. No multiplicative
generator is selected for the `Rs16Tower` fixture. Its custom-tower scalar route
is distinct from the host's requested SIMD tier.

## Case identities

A paired case identifies canonical label, semantic operation, preparation,
coefficient schedule, row bytes, source/destination counts, row pitch,
alignment, base offsets, and timing scope. Host and version identify the
measurement arm, not the workload. Changed geometry or duplicate identities
reject a pair.

The suffixes `b` and `d` distinguish the registered AES- and RS-rooted
presentations. `fp`, `m31`, `qm31`, and `gld` retain established harness
labels. API spelling changes do not change these field identities.
