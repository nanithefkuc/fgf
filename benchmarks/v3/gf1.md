# Bit-packed GF(2): `gf1` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v2](../v2/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `-` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |

## Self-timings

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits [N/4, 7N/8); `bits::XorRange` is the prepared geometry that `bits::xor_range_with` applies.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| bits::XorRange 5/8 | 58.9 / - | 66.0 / - | 21.4 / - | 9.84 / - |
| bits::and_into | 19.3 / - | 19.3 / - | 5.89 / - | 4.64 / - |
| bits::dot_product | 20.3 / - | 18.3 / - | 11.4 / - | 8.71 / - |
| bits::weight | 115 / - | 67.5 / - | 38.7 / - | 17.9 / - |
| bits::xor_assign | 43.4 / - | 41.4 / - | 8.70 / - | 6.11 / - |
| bits::xor_range 5/8 | 54.2 / - | 65.9 / - | 21.3 / - | 9.77 / - |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.40 / - |
| bits::xor_range | 11.7 / - |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
