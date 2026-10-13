# Bit-packed GF(2): `gf1` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
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
| bits::XorRange 5/8 | 59.0 / - | 66.0 / - | 20.7 / - | 9.83 / - |
| bits::and_into | 19.4 / - | 19.3 / - | 5.86 / - | 4.65 / - |
| bits::dot_product | 18.1 / - | 18.2 / - | 11.3 / - | 8.85 / - |
| bits::weight | 111 / - | 67.9 / - | 38.7 / - | 18.0 / - |
| bits::xor_assign | 43.3 / - | 41.4 / - | 8.57 / - | 6.10 / - |
| bits::xor_range 5/8 | 54.1 / - | 65.9 / - | 20.7 / - | 9.77 / - |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.74 / - |
| bits::xor_range | 12.0 / - |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
