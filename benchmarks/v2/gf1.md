# Bit-packed GF(2): `gf1` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
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
| bits::XorRange 5/8 | 58.7 / 84.7 | 66.0 / 74.4 | 20.6 / 39.9 | 9.85 / 24.3 |
| bits::and_into | 19.3 / 67.3 | 19.3 / 47.7 | 5.89 / 17.4 | 4.61 / 10.9 |
| bits::dot_product | 16.4 / 37.7 | 17.8 / 43.5 | 11.3 / 34.2 | 8.76 / 19.2 |
| bits::weight | 121 / 14.4 | 67.9 / 15.0 | 35.0 / 15.0 | 18.0 / 14.4 |
| bits::xor_assign | 43.2 / 105 | 41.4 / 64.8 | 8.75 / 24.8 | 6.12 / 14.8 |
| bits::xor_range 5/8 | 54.1 / 76.9 | 65.9 / 74.2 | 20.6 / 39.9 | 9.78 / 23.9 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.74 / 2.07 |
| bits::xor_range | 12.0 / 5.55 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
