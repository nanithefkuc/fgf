# Canonical field labels

Labels identify field presentations independently of their API spellings.
Field size alone does not identify the polynomial, basis, tower relation,
or packing used by a benchmark.

The numeric part of a binary-field label is its extension degree over $GF(2)$.
Thus `gf1` denotes $GF(2)$ independently of the v2 spelling `Gf2`.

## API mapping

The v2 column describes released `v2.1.0`; the v3 column describes the
measured working tree based on `3.0.1`. `Rs16Spec` is a benchmark-owned
extension specification, not a production alias. A missing v2 marker has no
public version counterpart.

| Label | v2 Canonical | v3 Canonical | Definition |
| --- | --- | --- | --- |
| `gf8b` | `Gf8B` | `FiniteField<2, 8, Polynomial<AES>>` | $GF(2^8)$ in the AES polynomial basis, $x^8+x^4+x^3+x+1$ |
| `gf8d` | `Gf8D` | `FiniteField<2, 8, Polynomial<RS>>` | $GF(2^8)$ in the Reed–Solomon polynomial basis, $x^8+x^4+x^3+x^2+1$ |
| `gf16b` | `Gf16` | `FiniteField<2, 16, Extension<Rijndael16, 2>>` | Quadratic extension of `gf8b`, $t^2+t+\texttt{0x20}=0$ |
| `gf16d` | - | `FiniteField<2, 16, Extension<Rs16Spec, 2>>` | Quadratic extension of `gf8d`, $t^2+\texttt{0x02}t+\texttt{0x80}=0$ |
| `gf32b` | `Gf32` | `FiniteField<2, 32, Extension<Rijndael32, 2>>` | Quadratic extension of `gf16b`, $t^2+t+\texttt{0x2000}=0$ |
| `gf64b` | `Gf64` | `FiniteField<2, 64, Extension<Rijndael64, 2>>` | Quadratic extension of `gf32b`, $t^2+t+\texttt{0x2000\_0000}=0$ |
| `fp8` | `FanPaar8` | `FiniteField<2, 8, Extension<FanPaar8, 2>>` | Fan–Paar tower from GF(2), through degrees 2 and 4 |
| `fp16` | `FanPaar16` | `FiniteField<2, 16, Extension<FanPaar16, 2>>` | Fan–Paar quadratic extension of `fp8` |
| `fp32` | `FanPaar32` | `FiniteField<2, 32, Extension<FanPaar32, 2>>` | Fan–Paar quadratic extension of `fp16` |
| `fp64` | `FanPaar64` | `FiniteField<2, 64, Extension<FanPaar64, 2>>` | Fan–Paar quadratic extension of `fp32` |
| `gf1` | `Gf2` and `bits` | `FiniteField<2, 1, Polynomial<0x3>>` and `bits` | $GF(2)=GF(2^1)$; bulk cases use bit-packed `bits` storage, not one byte per element |
| `m31` | `Mersenne31` | `Mersenne31` | Prime field $GF(2^{31}-1)$ |
| `qm31` | `QuadMersenne31` | `QuadMersenne31` | $GF((2^{31}-1)^2)$, coordinates $a+bi$ with $i^2=-1$ |
| `gld` | `Goldilocks` | `Goldilocks` | Prime field $GF(2^{64}-2^{32}+1)$ |
| `res101` | - | `FiniteField<101, 1>` | Prime field $GF(101)$, one-byte lanes |
| `res131` | - | `FiniteField<131, 1>` | Prime field $GF(131)$, one-byte lanes |
| `res65521` | - | `FiniteField<65_521, 1>` | Prime field $GF(65521)$, two-byte lanes |
| `res65537` | - | `FiniteField<65_537, 1>` | Prime field $GF(65537)$, four-byte lanes |
| `res858m` | - | `FiniteField<858_993_433, 1>` | Prime field $GF(858993433)$, four-byte lanes |
| `res536m` | - | `FiniteField<536_870_909, 1>` | Prime field $GF(536870909)$, four-byte lanes |
| `res268m` | - | `FiniteField<268_435_399, 1>` | Prime field $GF(268435399)$, four-byte lanes |
| `res67m` | - | `FiniteField<67_108_859, 1>` | Prime field $GF(67108859)$, four-byte lanes |
| `res1e9p7` | - | `FiniteField<1_000_000_007, 1>` | Prime field $GF(1000000007)$, four-byte lanes |
| `res1280000003` | - | `FiniteField<1_280_000_003, 1>` | Prime field $GF(1280000003)$, four-byte lanes |
| `reswordmax` | - | `FiniteField<4_294_967_291, 1>` | Prime field $GF(4294967291)$, four-byte lanes |
| `reswide` | - | `FiniteField<18_446_744_073_709_551_557, 1>` | Prime field $GF(18446744073709551557)$, eight-byte lanes |
| `res2` | - | `FiniteField<2, 1>` | Prime field $GF(2)$, eight-byte lanes |

`fp8` identifies the base of the wider Fan–Paar panels; those panels time
`fp16`, `fp32`, and `fp64`. Normal and Cantor presentations require separate
labels including their polynomial and basis seed. They are not included
under a polynomial-field or Fan–Paar label.

## Recursive tower definitions

Each `Extension<S, 2>` specifies its base and the constant-first
`S::RELATION` for $t^2+At+B=0$. The table separates the linear and constant
coefficients for readability. Recursing through the base column gives the
complete presentation.

| Spec | Exact base marker | Linear coefficient (raw) | Constant coefficient (raw) | Generator (raw) |
| --- | --- | --- | --- | --- |
| `Rijndael16` | `FiniteField<2, 8, Polynomial<AES>>` | `0x1` | `0x20` | `0x0108` |
| `Rijndael32` | `FiniteField<2, 16, Extension<Rijndael16, 2>>` | `0x1` | `0x2000` | `0x0001_0002` |
| `Rijndael64` | `FiniteField<2, 32, Extension<Rijndael32, 2>>` | `0x1` | `0x2000_0000` | `0x0000_0001_0000_0004` |
| `Rs16Spec` | `FiniteField<2, 8, Polynomial<RS>>` | `0x2` | `0x80` | - |
| `FanPaar2` | `FiniteField<2, 1, Polynomial<0x3>>` | `0x1` | `0x1` | `0x2` |
| `FanPaar4` | `FiniteField<2, 2, Extension<FanPaar2, 2>>` | `0x2` | `0x1` | `0x7` |
| `FanPaar8` | `FiniteField<2, 4, Extension<FanPaar4, 2>>` | `0x4` | `0x1` | `0x2D` |
| `FanPaar16` | `FiniteField<2, 8, Extension<FanPaar8, 2>>` | `0x10` | `0x1` | `0xE2DE` |
| `FanPaar32` | `FiniteField<2, 16, Extension<FanPaar16, 2>>` | `0x100` | `0x1` | `0x03E2_1CEA` |
| `FanPaar64` | `FiniteField<2, 32, Extension<FanPaar32, 2>>` | `0x10000` | `0x1` | `0x070F_870D_CD9C_1D88` |

Tower words place the constant component in the low half and the extension
component in the high half, recursively. Packed elements use little-endian
words with adjacent components, not separate planes. A native competitor
with another basis or component order uses a validated isomorphism; its raw
words are not treated as interchangeable with fgf coordinates.

`gf16d` freezes the registered relation and packing, not every tower rooted
at `0x11D`. Another relation requires another label. No multiplicative
generator is selected for the `Rs16Spec` fixture. Its custom-extension scalar
route is distinct from the host's requested SIMD tier.

## Case identities

A paired case identifies canonical label, semantic operation, preparation,
coefficient schedule, row bytes, source/destination counts, row pitch,
alignment, base offsets, and timing scope. Host and version identify the
measurement arm, not the workload. Changed geometry or duplicate identities
reject a pair.

The suffixes `b` and `d` distinguish the registered AES- and RS-rooted
presentations. `fp`, `m31`, `qm31`, and `gld` retain established harness
labels. API spelling changes do not change these field identities.
