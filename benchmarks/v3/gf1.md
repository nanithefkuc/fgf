# Bit-packed GF(2): `gf1` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v2](../v2/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `388cbb4a43d5d20e8a663e89c3b44380c7c4f8b7` |
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
| bits::XorRange 5/8 | 59.1 / 82.9 | 66.0 / 71.9 | 21.2 / 38.2 | 9.86 / 24.0 |
| bits::and_into | 19.4 / 64.2 | 19.3 / 45.9 | 5.95 / 18.1 | 4.64 / 10.8 |
| bits::dot_product | 20.3 / 38.6 | 18.3 / 37.9 | 11.4 / 35.4 | 8.78 / 18.9 |
| bits::weight | 114 / 14.1 | 67.9 / 14.4 | 38.6 / 14.4 | 18.0 / 13.9 |
| bits::xor_assign | 43.0 / 99.5 | 41.4 / 63.4 | 8.68 / 24.5 | 6.13 / 14.6 |
| bits::xor_range 5/8 | 54.1 / 74.4 | 65.9 / 71.4 | 21.2 / 38.2 | 9.78 / 23.6 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.41 / 2.18 |
| bits::xor_range | 11.8 / 5.59 |


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

| Op | Row bytes | v4x | v3 | v2 | scalar |
| --- | --- | --- | --- | --- | --- |
| expand | 8 | - / 14.17 | 6.20 / 13.88 | 6.04 / 13.77 | **5.58** / **12.78** |
| expand | 16 | - / 17.22 | 8.32 / 16.08 | 7.83 / 16.91 | **7.89** / **14.83** |
| expand | 32 | - / 16.69 | **8.02** / **16.76** | 8.45 / 18.81 | 11.95 / 19.80 |
| expand | 64 | - / **16.59** | 8.65 / 18.75 | 9.65 / 22.13 | 19.20 / 30.91 |
| expand | 128 | - / **17.74** | 9.98 / 22.80 | 12.07 / 28.34 | 41.25 / 67.19 |
| compress | 8 | - / 19.33 | 9.64 / 18.93 | 9.66 / 18.90 | **9.08** / **17.97** |
| compress | 16 | - / 23.56 | 13.16 / 26.42 | **11.20** / **24.00** | 12.70 / 25.58 |
| compress | 32 | - / 24.87 | **12.05** / 24.86 | 12.96 / 27.81 | 18.56 / 38.15 |
| compress | 64 | - / **21.91** | 15.61 / 29.33 | 16.43 / 33.47 | 30.04 / 61.99 |
| compress | 128 | - / **31.72** | 24.30 / 46.19 | 25.89 / 54.82 | 56.42 / 116.31 |
| restrict | 8 | - / 22.98 | 10.98 / 22.68 | 10.88 / 22.74 | **10.26** / **21.43** |
| restrict | 16 | - / 27.85 | 13.74 / 28.32 | 13.30 / 28.05 | **13.06** / **27.55** |
| restrict | 32 | - / 28.70 | **14.16** / **28.48** | 17.28 / 32.30 | 18.66 / 38.56 |
| restrict | 64 | - / **30.20** | 18.11 / 34.03 | 23.53 / 42.68 | 36.87 / 62.06 |
| restrict | 128 | - / **36.55** | 26.05 / 45.61 | 42.02 / 71.24 | 59.46 / 106.80 |

The floors hold on both hosts: below one vector unit the portable body wins
(8-byte rows at every tier; 16-byte rows at `V3`/`V3GfniCrypto`/`V4x`), and
from one unit up every tier wins (16 bytes at `V2`, 32 at `V3` and
`V3GfniCrypto`, 64 at `V4x`). The `v4x` column is Willow Cove only: Golden
Cove has no live AVX-512 capability, so it contributes no `v4x` cell there,
and each tier column is the process-backend run for that tier (one process
per requested tier, `SIMD_BACKEND=<tier>`, resolving down on request only
when the host cannot prove the tier — so the `v4x` cells are the `V4x` arm
where the host proves it and the fall-through narrower-tier arm inside the
same process where the row is below the 64-byte floor). On Willow Cove the
`v4x` process matches `v3` within noise on 16–63-byte rows (the fall-through
region) and leads clearly from 64 bytes up. Campaign:
`examples/gf1_floors.rs` at `f10e59d`, Willow Cove core 3
(`v4x`/`v3`/`v2`/`scalar`, requested `v4x`) and Golden Cove core 8
(`v3_gfni_crypto`/`v3`/`v2`/`scalar`, requested `v3_gfni_crypto`), Rust
1.98.1, five rounds per tier per host, `chrt --idle 0 taskset`, median of
five per-round medians. The dispatch floors in
`src/kernel/gf1/dispatch.rs` (16 bytes at `V2`, 32 at `V3`/`V3GfniCrypto`,
64 at `V4x`) are unchanged: the data confirm them, so no threshold moved
and no differential re-run was needed.

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
