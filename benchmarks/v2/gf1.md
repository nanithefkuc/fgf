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
| bits::XorRange 5/8 | 58.7 / 82.9 | 66.0 / 71.5 | 20.7 / 38.2 | 9.78 / 24.4 |
| bits::and_into | 19.4 / 64.5 | 19.3 / 44.4 | 5.82 / 17.8 | 4.62 / 10.8 |
| bits::dot_product | 18.1 / 36.7 | 18.2 / 40.1 | 11.3 / 34.9 | 8.73 / 18.0 |
| bits::weight | 120 / 14.1 | 67.9 / 14.4 | 38.6 / 14.4 | 17.9 / 13.9 |
| bits::xor_assign | 43.4 / 101 | 41.4 / 63.4 | 8.63 / 24.1 | 6.10 / 14.7 |
| bits::xor_range 5/8 | 54.0 / 74.5 | 65.9 / 71.4 | 20.7 / 38.2 | 9.73 / 24.0 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.74 / 2.16 |
| bits::xor_range | 12.0 / 5.79 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
