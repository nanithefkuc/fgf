# Public operations

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../BENCHMARKS.md) · [Field definitions](../labels.md)

Choose the work first, then find the field in its operation table. Each measured case appears once. Preparation and compositions remain separate from steady-state calls.

[Field maps](maps.md) · [GF16 presentations](gf16-presentations.md)

| Task | Operations | Cases |
| --- | --- | --- |
| [Buffer addition and subtraction](public-api/buffer-addition.md) | 7 | 187 |
| [Elementwise buffer products](public-api/buffer-products.md) | 2 | 70 |
| [Single-row multiplication](public-api/row-multiplication.md) | 19 | 243 |
| [Scatter, gather, and matrix](public-api/multirow.md) | 24 | 354 |
| [Scalar arithmetic](public-api/scalar-arithmetic.md) | 19 | 459 |
| [Inherent scalar arithmetic](public-api/scalar-inherent.md) | 14 | 146 |
| [Encoding and coordinate conversion](public-api/encodings.md) | 13 | 302 |
| [Queries and prepared-state access](public-api/queries.md) | 9 | 92 |
| [Preparation and construction](public-api/preparation.md) | 3 | 15 |
| [Field-specific arithmetic](public-api/field-math.md) | 16 | 24 |
| [Bit-packed operations](public-api/bits.md) | 15 | 17 |

## Reading geometry

**Counted bytes** is the throughput numerator, not necessarily one row's size. **Elements**, **Sources**, and **Destinations** are the registered fixture counts. Shared counts appear above a table; only varying counts occupy columns. **S → T** names a prepared map's field pair, not every method's value direction.

## Measurement details

<details>
<summary>Setup, resolved field backends, timer floor, and validation</summary>

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (current-only family `surfaces`) |
| Revision | `63b8d168a7e4b81043fafe4db1ffb9918dc1c231` plus the measured working-tree archive; campaign evidence fixes its digest. |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| `gf101` backend | `v4x` / `v3` |
| `gf125` backend | `scalar` / `scalar` |
| `gf127` backend | `v4x` / `v3` |
| `gf131` backend | `scalar` / `scalar` |
| `gf1e9p7` backend | `v4x` / `v3` |
| `gf2p1-poly` backend | `scalar` / `scalar` |
| `gf2p1-residue` backend | `scalar` / `scalar` |
| `gf2p16-custom-cantor` backend | `scalar` / `scalar` |
| `gf2p16-custom-normal` backend | `scalar` / `scalar` |
| `gf2p16-custom-rijndael` backend | `scalar` / `scalar` |
| `gf2p16-custom-rs` backend | `scalar` / `scalar` |
| `gf2p16-fanpaar` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p16-rijndael` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p2-cantor` backend | `scalar` / `scalar` |
| `gf2p2-fanpaar` backend | `scalar` / `scalar` |
| `gf2p2-normal` backend | `scalar` / `scalar` |
| `gf2p2-poly` backend | `scalar` / `scalar` |
| `gf2p32-fanpaar` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p32-rijndael` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p4-cantor` backend | `scalar` / `scalar` |
| `gf2p4-fanpaar` backend | `scalar` / `scalar` |
| `gf2p4-normal` backend | `scalar` / `scalar` |
| `gf2p4-poly` backend | `scalar` / `scalar` |
| `gf2p5-padded` backend | `scalar` / `scalar` |
| `gf2p64-fanpaar` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p64-rijndael` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p8-aes` backend | `v4x` / `v3_gfni_crypto` |
| `gf2p8-cantor` backend | `scalar` / `scalar` |
| `gf2p8-fanpaar` backend | `scalar` / `scalar` |
| `gf2p8-normal` backend | `scalar` / `scalar` |
| `gf2p8-rs` backend | `v4x` / `v3_gfni_crypto` |
| `gf49` backend | `scalar` / `scalar` |
| `gf65521` backend | `scalar` / `scalar` |
| `gf65537` backend | `v4x` / `v3` |
| `gf81` backend | `scalar` / `scalar` |
| `gf9` backend | `scalar` / `scalar` |
| `gfmaxp` backend | `v4x` / `v3` |
| `gfu32max` backend | `v4x` / `v3` |
| `goldilocks` backend | `v4x` / `v3` |
| `goldilocks-residue` backend | `v4x` / `v3` |
| `goldilocks-squared` backend | `scalar` / `scalar` |
| `m31` backend | `v4x` / `v3` |
| `m31-residue` backend | `v4x` / `v3` |
| `maxprime-squared` backend | `scalar` / `scalar` |
| `qm31` backend | `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs the public-API surfaces artifact and the GF16 presentations harness into one log; inventories must match across rounds and hosts. |
| Aggregation | Median of five per-run medians. |
| Metrics | Cases with a logical byte numerator report GiB/s; scalar, query, and getter cases report per-call latency. |
| Timer floor | Each public-API process measures a black-boxed u64 loop control; per-run latency and sample count remain in the campaign evidence. Query timings include loop overhead; values near this floor do not distinguish work from measurement overhead. The floor-limited-round count identifies medians at or below each run's control, in the declared host order. |
| Validation | Fixture correctness is proven by `--check` captures validated with `benches/check_surfaces.py`; campaign logs carry measurement records only. |

</details>

Timer-floor counts remain beside affected latency rows. A count of five means all five rounds are at or below the loop control; such figures do not distinguish call cost from measurement overhead.

Raw case identifiers, exact per-case API paths, controls, and every round remain in the campaign evidence.
