# Bit-packed GF(2): `gf1` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v2](../v2/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |

## Self-timings

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits [N/4, 7N/8); `bits::XorRange` is the prepared geometry that `bits::xor_range_with` applies.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| bits::XorRange 5/8 | 58.9 / 82.9 | 66.0 / 71.5 | 21.4 / 38.2 | 9.84 / 24.1 |
| bits::and_into | 19.3 / 66.0 | 19.3 / 43.0 | 5.89 / 18.9 | 4.64 / 10.8 |
| bits::dot_product | 20.3 / 38.2 | 18.3 / 41.7 | 11.4 / 37.9 | 8.71 / 18.2 |
| bits::weight | 115 / 14.1 | 67.5 / 14.4 | 38.7 / 14.4 | 17.9 / 13.9 |
| bits::xor_assign | 43.4 / 101 | 41.4 / 63.4 | 8.70 / 24.1 | 6.11 / 14.7 |
| bits::xor_range 5/8 | 54.2 / 74.3 | 65.9 / 71.4 | 21.3 / 38.2 | 9.77 / 23.7 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.40 / 2.16 |
| bits::xor_range | 11.7 / 5.57 |


### Bit interop tier floors

`expand`, `compress`, and `restrict` cross the bit-packed GF(2) seam through
the public prepared entry points. The probe times 1–1024 lanes (8-bit lanes,
so 8–8192-byte rows, straddling the 16/32/64-byte floors) under each tier
request and under `scalar`, pinned to an isolated core at `SCHED_IDLE`.
Each cell reports the Willow Cove / Golden Cove median of five per-round
medians, in ns/call; the faster side is bold. Tier requests run
sequentially within each round (tiers loop inside the round, rounds do not
interleave); the session still brackets every tier against the same scalar
control on both hosts.

| Op | Row bytes | scalar | v2 | v3 | v4x |
| --- | --- | --- | --- | --- | --- |
| expand | 8 | **5.58** / **12.78** | 6.04 / 13.77 | 6.20 / 13.88 | - / 14.17 |
| expand | 16 | **7.89** / **14.83** | 7.83 / 16.91 | 8.32 / 16.08 | - / 17.22 |
| expand | 32 | 11.95 / 19.80 | 8.45 / 18.81 | **8.02** / **16.76** | - / 16.69 |
| expand | 64 | 19.20 / 30.91 | 9.65 / 22.13 | 8.65 / 18.75 | - / **16.59** |
| expand | 128 | 41.25 / 67.19 | 12.07 / 28.34 | 9.98 / 22.80 | - / **17.74** |
| compress | 8 | **9.08** / **17.97** | 9.66 / 18.90 | 9.64 / 18.93 | - / 19.33 |
| compress | 16 | 12.70 / 25.58 | **11.20** / **24.00** | 13.16 / 26.42 | - / 23.56 |
| compress | 32 | 18.56 / 38.15 | 12.96 / 27.81 | **12.05** / 24.86 | - / 24.87 |
| compress | 64 | 30.04 / 61.99 | 16.43 / 33.47 | 15.61 / 29.33 | - / **21.91** |
| compress | 128 | 56.42 / 116.31 | 25.89 / 54.82 | 24.30 / 46.19 | - / **31.72** |
| restrict | 8 | **10.26** / **21.43** | 10.88 / 22.74 | 10.98 / 22.68 | - / 22.98 |
| restrict | 16 | **13.06** / **27.55** | 13.30 / 28.05 | 13.74 / 28.32 | - / 27.85 |
| restrict | 32 | 18.66 / 38.56 | 17.28 / 32.30 | **14.16** / **28.48** | - / 28.70 |
| restrict | 64 | 36.87 / 62.06 | 23.53 / 42.68 | 18.11 / 34.03 | - / **30.20** |
| restrict | 128 | 59.46 / 106.80 | 42.02 / 71.24 | 26.05 / 45.61 | - / **36.55** |

The floors hold on both hosts: below one vector unit the portable body wins
(8-byte rows at every tier; 16-byte rows at `V3`/`V3GfniCrypto`/`V4x`), and
from one unit up every tier wins (16 bytes at `V2`, 32 at `V3` and
`V3GfniCrypto`, 64 at `V4x`). Golden Cove has no live AVX-512 capability, so
it contributes no `V4x` column; on Willow Cove the `V4x` arm matches `V3`
within noise on 16–63-byte rows (the fall-through region) and leads clearly
from 64 bytes up. Campaign: `examples/gf1_floors.rs` at `f10e59d`, Willow
Cove core 3 (`v4x`/`v3`/`v2`/`scalar`, requested `v4x`) and Golden Cove core
8 (`v3_gfni_crypto`/`v3`/`v2`/`scalar`, requested `v3_gfni_crypto`), Rust
1.98.1, five rounds per tier per host, `chrt --idle 0 taskset`, median of
five per-round medians. The dispatch floors in
`src/kernel/gf1/dispatch.rs` (16 bytes at `V2`, 32 at `V3`/`V3GfniCrypto`,
64 at `V4x`) are unchanged: the data confirm them, so no threshold moved
and no differential re-run was needed.

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
