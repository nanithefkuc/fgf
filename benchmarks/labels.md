# Canonical field labels

Labels identify field presentations independently of their API spellings.
Field size alone does not identify the polynomial, basis, tower relation,
or packing used by a benchmark.

The numeric part of a binary-field label is its extension degree over $GF(2)$.
Thus `gf1` denotes $GF(2)$ independently of the v2 spelling `Gf2`.

## API mapping

The v2 column describes released `v2.1.0`; the v3 column describes the
measured `3.0.1` source archive. `Rs16Spec` is a benchmark-owned extension
specification, not a production alias. A missing
v2 marker has no public version counterpart.

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

## Public-surface field labels

The `surfaces` panels register forty-five representative field presentations
in `benches/surfaces/fields.rs`. The labels below are the exact `field`
strings of every record, frozen across rounds and hosts; they cover
representative classes, not every generic parameter combination. The custom
specs are benchmark-owned `ExtensionSpec` implementations exercising the
open extension surface, not public aliases. `Packed<uN>` packs little-endian
`uN` words with adjacent components; `PackedPadded<u8>` gives every element
its own padded byte; `Components` stores one base-field coordinate per lane.

### Polynomial rows

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p1-poly` | `FiniteField<2, 1, Polynomial<3>>` | $GF(2)$ under $x^2+x+1$ |
| `gf2p2-poly` | `FiniteField<2, 2, Polynomial<7>>` | $GF(2^2)$, $x^2+x+1$ |
| `gf2p4-poly` | `FiniteField<2, 4, Polynomial<19>>` | $GF(2^4)$, $x^4+x+1$ |
| `gf2p8-aes` | `FiniteField<2, 8, Polynomial<AES>>` | $GF(2^8)$, AES $x^8+x^4+x^3+x+1$ (`0x11B`) |
| `gf2p8-rs` | `FiniteField<2, 8, Polynomial<RS>>` | $GF(2^8)$, Reed–Solomon $x^8+x^4+x^3+x^2+1$ (`0x11D`) |

### Normal-basis rows

Bit $i$ of the coordinates selects $E^{2^i}$ over `Polynomial<P>`; the
multiplicative one is the all-ones word.

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p2-normal` | `FiniteField<2, 2, Normal<0x7, 2>>` | $GF(2^2)$, $P=\texttt{0x7}$, $E=\texttt{2}$ |
| `gf2p4-normal` | `FiniteField<2, 4, Normal<0x13, 8>>` | $GF(2^4)$, $P=\texttt{0x13}$, $E=\texttt{8}$ |
| `gf2p8-normal` | `FiniteField<2, 8, Normal<AES, 0x20>>` | $GF(2^8)$, AES field, $E=\texttt{0x20}$ |

### Cantor-basis rows

The chain runs $c_{N-1} = $ seed down through $c_i = c_{i+1}^2 + c_{i+1}$;
bit $i$ selects $c_i$, and the one is word `0x01`.

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p2-cantor` | `FiniteField<2, 2, Cantor<0x7, 2>>` | $GF(2^2)$, $P=\texttt{0x7}$, seed `2` |
| `gf2p4-cantor` | `FiniteField<2, 4, Cantor<0x13, 8>>` | $GF(2^4)$, $P=\texttt{0x13}$, seed `8` |
| `gf2p8-cantor` | `FiniteField<2, 8, Cantor<AES, 0x20>>` | $GF(2^8)$, AES field, seed `0x20` |

### Fan–Paar towers

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p2-fanpaar` | `FiniteField<2, 2, Extension<FanPaar2, 2>>` | $t^2+t+1$ over $GF(2)$, `PackedPadded<u8>` |
| `gf2p4-fanpaar` | `FiniteField<2, 4, Extension<FanPaar4, 2>>` | $t^2+2t+1$ over `gf2p2-fanpaar`, `PackedPadded<u8>` |
| `gf2p8-fanpaar` | `FiniteField<2, 8, Extension<FanPaar8, 2>>` | $t^2+4t+1$ over `gf2p4-fanpaar`, `Packed<u8>` |
| `gf2p16-fanpaar` | `FiniteField<2, 16, Extension<FanPaar16, 2>>` | $t^2+\texttt{0x10}t+1$ over `gf2p8-fanpaar`, `Packed<u16>` |
| `gf2p32-fanpaar` | `FiniteField<2, 32, Extension<FanPaar32, 2>>` | $t^2+\texttt{0x100}t+1$ over `gf2p16-fanpaar`, `Packed<u32>` |
| `gf2p64-fanpaar` | `FiniteField<2, 64, Extension<FanPaar64, 2>>` | $t^2+\texttt{0x10000}t+1$ over `gf2p32-fanpaar`, `Packed<u64>` |

### Rijndael towers

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p16-rijndael` | `FiniteField<2, 16, Extension<Rijndael16, 2>>` | $t^2+t+\texttt{0x20}$ over `gf2p8-aes`, `Packed<u16>` |
| `gf2p32-rijndael` | `FiniteField<2, 32, Extension<Rijndael32, 2>>` | $t^2+t+\texttt{0x2000}$ over `gf2p16-rijndael`, `Packed<u32>` |
| `gf2p64-rijndael` | `FiniteField<2, 64, Extension<Rijndael64, 2>>` | $t^2+t+\texttt{0x2000\_0000}$ over `gf2p32-rijndael`, `Packed<u64>` |

### Custom packed extensions

Benchmark-owned `ExtensionSpec` rows; `RELATION` arrays are constant-first
raw base-field words.

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p16-custom-rs` | `FiniteField<2, 16, Extension<CustomRs16, 2>>` | $t^2+t+\texttt{0x21}$ over `gf2p8-rs`, raw `[0x21, 0x01]`, `Packed<u16>` |
| `gf2p16-custom-rijndael` | `FiniteField<2, 16, Extension<CustomRijndael16, 2>>` | $t^2+t+\texttt{0x2C}$ over `gf2p8-aes`, raw `[0x2C, 0x01]`, `Packed<u16>` |
| `gf2p16-custom-normal` | `FiniteField<2, 16, Extension<CustomNormal16, 2>>` | $t^2+\beta t+\beta$ over `gf2p8-normal`, raw `[0x01, 0x01]`, `Packed<u16>`, where $\beta$ is basis vector zero. The linear coefficient is the basis word `0x01`, not the normal-basis one `0xFF`: this is not $t^2+t+\beta$ |
| `gf2p16-custom-cantor` | `FiniteField<2, 16, Extension<CustomCantor16, 2>>` | $t^2+t+\texttt{0x80}$ over `gf2p8-cantor`, raw `[0x80, 0x01]`, `Packed<u16>`; word `0x80` is the top Cantor basis vector, the only one with binary trace one |
| `gf2p5-padded` | `FiniteField<2, 5, Extension<OddPadded5, 5>>` | $t^5+t^2+1$ over `gf2p1-poly`, raw `[1, 0, 1, 0, 0]`, `PackedPadded<u8>`; five-bit lanes have no vector route, so bulk cases run the portable path |

### Specialized prime and residue rows

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf2p1-residue` | `FiniteField<2, 1>` | $GF(2)$ through the generic residue row, eight-byte lane |
| `gf101` | `FiniteField<101, 1>` | $GF(101)$, one-byte lanes |
| `gf127` | `FiniteField<127, 1>` | $GF(127)$, one-byte lanes |
| `gf131` | `FiniteField<131, 1>` | $GF(131)$, one-byte lanes |
| `gf65521` | `FiniteField<65_521, 1>` | $GF(65521)$, two-byte lanes |
| `gf65537` | `FiniteField<65_537, 1>` | $GF(65537)$, four-byte lanes |
| `gf1e9p7` | `FiniteField<1_000_000_007, 1>` | $GF(1000000007)$, four-byte lanes |
| `gfu32max` | `FiniteField<4_294_967_291, 1>` | $GF(4294967291)$, the largest prime under $2^{32}$, four-byte lanes |
| `gfmaxp` | `FiniteField<18_446_744_073_709_551_557, 1>` | $GF(18446744073709551557)$, the largest `u64` prime, eight-byte lanes |
| `m31` | `Mersenne31` | $GF(2^{31}-1)$ |
| `m31-residue` | `FiniteField<0x7FFF_FFFF, 1, Residue>` | the same modulus through the generic residue row, four-byte lanes |
| `goldilocks` | `Goldilocks` | $GF(2^{64}-2^{32}+1)$ |
| `goldilocks-residue` | `FiniteField<0xFFFF_FFFF_0000_0001, 1, Residue>` | the same modulus through the generic residue row, eight-byte lanes |
| `qm31` | `QuadMersenne31` | $GF((2^{31}-1)^2)$, coordinates $a+bi$ with $i^2=-1$ |

### Odd-characteristic and wide prime extensions

`Components` rows; `RELATION` arrays are constant-first base coordinates.

| Label | API spelling | Definition |
| --- | --- | --- |
| `gf9` | `FiniteField<3, 2, Extension<Gf9Spec, 2>>` | $t^2+1$ over $GF(3)$ |
| `gf81` | `FiniteField<3, 4, Extension<Gf81Spec, 2>>` | $t^2+(1+u)$ over `gf9`, where $u^2=-1$ |
| `gf49` | `FiniteField<7, 2, Extension<Gf49Spec, 2>>` | $t^2+t+3$ over $GF(7)$ |
| `gf125` | `FiniteField<5, 3, Extension<Gf125Spec, 3>>` | $t^3+2t+1$ over $GF(5)$ |
| `goldilocks-squared` | `FiniteField<0xFFFF_FFFF_0000_0001, 2, Extension<GoldilocksSquareSpec, 2>>` | $t^2+7$ over `goldilocks`; two 64-bit lanes |
| `maxprime-squared` | `FiniteField<18_446_744_073_709_551_557, 2, Extension<MaxPrimeSquareSpec, 2>>` | $t^2+2$ over `gfmaxp`; two 64-bit lanes |

## Native GF16 presentation labels

The native harness's `--presentations` mode registers seven public
$GF(2^{16})$ presentations in `external/gf16-bench`. These labels are the
exact `field` strings of its `SURFACE_INVENTORY` and `GF16_SURFACE` records.
Every non-canonical presentation ships a caller-owned exact map into the
canonical tower encoding, validated outside timing over all $65536$ values
and all $256$ basis products against an independent scalar oracle, so raw
words are never read as values across bases. Native competitor arms consume
their own encodings through the same validated layout maps as the legacy
harness; group coefficient banks are canonical tower words transported
through each arm's map, so competing arms multiply matched abstract
elements, not the same raw words.

| Label | Row | Definition |
| --- | --- | --- |
| `gf16-rijndael16` | `FiniteField<2, 16, Extension<Rijndael16, 2>>` | Shipped Rijndael tower, $t^2+t+\texttt{0x20}$ over $GF(2^8)/\texttt{0x11B}$, packed; the canonical encoding with the identity map |
| `gf16-fanpaar16` | `FiniteField<2, 16, Extension<FanPaar16, 2>>` | Shipped Fan–Paar tower $t^2+\texttt{0x10}t+1$ over $t^2+4t+1$ over $t^2+2t+1$ over $t^2+t+1$, packed |
| `gf16-rs-quad` | benchmark-owned `RsQuadSpec` | $t^2+2t+\texttt{0x80}$ over $GF(2^8)/\texttt{0x11D}$, raw `[0x80, 0x02]`, packed; the relation the kernel tests and the paired `gf16d` panel exercise, distinct from the public-surface `gf2p16-custom-rs` tower |
| `gf16-cantor-base` | benchmark-owned `CantorBaseSpec` over `Cantor<AES, 0x20>` | The Rijndael relation transported to the Cantor basis: raw constant `0x80`, the word of the basis vector $c_7$ and the image of `0x20`; linear coefficient `0x01`, the Cantor one |
| `gf16-normal-base` | benchmark-owned `NormalBaseSpec` over `Normal<AES, 0x20>` | The Rijndael relation transported to the normal basis: raw constant `0x01`, the word of $E$ itself and the image of `0x20`; linear coefficient `0xFF`, the normal one |
| `gf16-flat-0x1100b` | benchmark-owned `FlatCompleteSpec`, flat degree 16 | $GF(2^{16})$ over $GF(2)$ under the GF-Complete polynomial `0x1100B` = $t^{16}+t^{12}+t^3+t+1$, packed |
| `gf16-flat-0x1002d` | benchmark-owned `FlatRssSpec`, flat degree 16 | $GF(2^{16})$ over $GF(2)$ under the reed-solomon-simd polynomial `0x1002D` = $t^{16}+t^5+t^3+t^2+1$, packed |

The transported constants are pinned against the library's own maps before
any case runs. The `cycling-nonzero` scalar chain walks each presentation's
own raw words `1..=65535`, so the abstract coefficients it traverses differ
across presentations while the chain shape stays fixed; native and `fgf`
arms compare matched abstract elements under each arm's validated encoding.

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

## Surface cases

The current-only surfaces family registers every measurement twice, as an
inventory entry without timing and as a timed record, under one stable
workload identity: case string, public callable paths, registered presentation
label, operation, timing scope, byte numerator, lanes, sources, destinations,
and the arm for GF16 presentations. Surfaces inventories also carry the
control flag and its phase. Workload identities match across rounds and hosts.
The resolved backend is measurement metadata: it matches the host's field
route and cannot change between that host's rounds, but is not part of the
cross-host workload identity. The `maps/` case prefix is reserved for map work.

Timing scopes are `setup`, `steady`, `query`, and `end_to_end` for public-API
surfaces and `native`, `end_to_end`, and `setup` for GF16 presentations.
A case with a nonzero byte numerator publishes GiB/s; scalar, query, and
getter and no-op cases publish per-call latency. Map work uses scalar public
calls and caller loops; composite `S->T` labels stay outside the field-banner
list. Both map-surface cases and presentation conversion records report the
scalar route and appear under `v3/maps.md`. Other operation records identify
the field's implemented backend; bit-range and gather interiors identify the
process XOR backend. Query tags identify the queried field or the scalar
metadata route, not an ISA-execution claim. Public-surface and presentation
matrix numerators count `row bytes × sources × destinations`, one field
multiplication operand pair per lane; legacy competitor panels retain their
documented payload convention. Presentation arms are `fgf`, the duplicate
`fgf-control` that only native groups run, and the native competitor labels.
The presentation
labels are the registered `gf16-*` rows tabulated under the native GF16
presentation labels above; the published inventories and `evidence.json`
fix the exact set per capture.
