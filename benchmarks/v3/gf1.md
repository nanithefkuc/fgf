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

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
