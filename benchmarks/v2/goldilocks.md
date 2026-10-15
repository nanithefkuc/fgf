# Goldilocks: `gld` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v3](../v3/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `gld`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K): Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## Self-timings

### Packed operations

`gld`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.4 / 21.6 | 27.3 / 21.6 | 7.98 / 20.1 |
| add_assign_scalar | 44.6 / 37.8 | 44.4 / 39.1 | 25.8 / 32.3 |
| mul_elementwise | 10.8 / 10.2 | 11.3 / 10.8 | 5.23 / 10.1 |
| mul_elementwise_assign | 10.9 / 10.7 | 11.3 / 10.8 | 7.13 / 9.39 |
| sub_assign_scalar | 60.4 / 39.1 | 52.7 / 38.7 | 27.7 / 32.5 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.740 / 1.696 | 2.574 / 1.157 |
| m31 | scalar a * b | 4.210 / 1.615 | 2.104 / 0.837 |
| qm31 | scalar a * b | 9.003 / 4.591 | 6.964 / 3.038 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.6 |
| gld | dst += v | 97.5 / 78.3 | 103 / 142 |
| gld | dst -= v | 103 / 77.6 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.4 |
| gld | dst = c * src | 25.1 / 22.8 | 27.4 / 26.0 |
| m31 | dst += c * src | 55.5 / 58.7 | 49.8 / 54.8 |
| m31 | dst += v | 87.5 / 103 | 103 / 147 |
| m31 | dst -= v | 70.3 / 82.9 | 103 / 147 |
| m31 | dst = a * b | 59.5 / 65.4 | 59.9 / 66.5 |
| m31 | dst = c * src | 73.2 / 73.7 | 65.4 / 69.9 |
| qm31 | dst += c * src | 22.9 / 27.2 | 27.0 / 30.4 |
| qm31 | dst += v | 62.5 / 103 | 103 / 147 |
| qm31 | dst -= v | 49.9 / 82.9 | 103 / 147 |
| qm31 | dst = a * b | 27.0 / 29.8 | 24.1 / 25.1 |
| qm31 | dst = c * src | 32.1 / 36.1 | 30.6 / 34.6 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
