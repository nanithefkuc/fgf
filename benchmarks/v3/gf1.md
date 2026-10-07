# Bit-packed GF(2): `gf1` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf1.md); the other snapshot at [v2](../v2/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| Revision | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| bits::XorRange 5/8 | 58.9 / 86.5 | 66.0 / 74.5 | 21.0 / 39.9 | 9.88 / 24.7 |
| bits::and_into | 19.3 / 66.9 | 19.3 / 45.9 | 5.86 / 19.6 | 4.64 / 11.0 |
| bits::dot_product | 18.1 / 38.5 | 18.2 / 43.4 | 11.1 / 39.3 | 8.89 / 18.9 |
| bits::weight | 122 / 14.7 | 67.9 / 15.0 | 38.8 / 15.0 | 18.0 / 14.4 |
| bits::xor_assign | 43.1 / 105 | 41.4 / 64.8 | 8.65 / 25.2 | 6.10 / 14.8 |
| bits::xor_range 5/8 | 53.9 / 77.4 | 65.9 / 74.5 | 21.0 / 39.9 | 9.80 / 24.3 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | Latency (ns/row pair) |
| --- | --- |
| bits::XorRange | 4.47 / 2.07 |
| bits::xor_range | 11.9 / 5.54 |

## Competitors

No matched competitor measurements are available.

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
