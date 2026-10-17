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
| bits::XorRange 5/8 | 58.6 / 82.7 | 66.0 / 71.5 | 20.9 / 38.2 | 9.84 / 24.2 |
| bits::and_into | 19.4 / 64.5 | 19.3 / 44.9 | 5.83 / 18.2 | 4.65 / 10.8 |
| bits::dot_product | 18.1 / 36.7 | 18.2 / 41.7 | 11.1 / 34.5 | 8.83 / 18.6 |
| bits::weight | 117 / 14.1 | 67.9 / 14.4 | 35.5 / 14.4 | 18.0 / 13.9 |
| bits::xor_assign | 43.4 / 101 | 41.4 / 63.4 | 8.60 / 24.5 | 6.11 / 14.6 |
| bits::xor_range 5/8 | 54.1 / 74.3 | 65.9 / 71.3 | 20.8 / 38.2 | 9.79 / 23.8 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.74 / 2.19 |
| bits::xor_range | 12.0 / 5.79 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
