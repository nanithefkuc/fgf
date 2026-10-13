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

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
