# Bit-packed GF(2): `gf1` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v2](../v2/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| bits::XorRange 5/8 | 58.3 / 82.7 | 66.0 / 71.5 | 20.9 / 38.2 | 9.83 / 24.1 |
| bits::and_into | 19.4 / 66.2 | 19.3 / 43.7 | 5.93 / 16.8 | 4.63 / 10.8 |
| bits::dot_product | 20.4 / 34.0 | 18.3 / 36.2 | 11.1 / 35.0 | 8.81 / 18.4 |
| bits::weight | 116 / 13.9 | 67.5 / 14.4 | 38.5 / 14.4 | 17.9 / 13.9 |
| bits::xor_assign | 43.3 / 100 | 41.4 / 63.4 | 8.59 / 24.1 | 6.09 / 14.6 |
| bits::xor_range 5/8 | 54.2 / 74.3 | 65.9 / 71.4 | 20.9 / 38.2 | 9.75 / 23.7 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.39 / 2.16 |
| bits::xor_range | 11.8 / 5.59 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
