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
| add_assign | 29.6 / 22.0 | 27.3 / 22.5 | 8.11 / 20.9 |
| add_assign_scalar | 43.9 / 39.4 | 44.4 / 40.8 | 26.0 / 33.7 |
| mul_elementwise | 10.8 / 10.6 | 11.3 / 11.2 | 5.18 / 10.4 |
| mul_elementwise_assign | 10.9 / 11.1 | 11.3 / 11.3 | 7.08 / 9.77 |
| sub_assign_scalar | 58.9 / 40.8 | 52.7 / 40.2 | 27.9 / 33.9 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.740 / 1.640 | 2.570 / 1.110 |
| m31 | scalar a * b | 4.219 / 1.546 | 2.107 / 0.801 |
| qm31 | scalar a * b | 9.003 / 4.365 | 6.963 / 2.908 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.9 | 22.5 / 20.6 |
| gld | dst += v | 97.5 / 81.6 | 103 / 148 |
| gld | dst -= v | 103 / 80.9 | 103 / 147 |
| gld | dst = a * b | 23.8 / 23.3 | 26.9 / 26.4 |
| gld | dst = c * src | 25.1 / 23.2 | 27.4 / 27.1 |
| m31 | dst += c * src | 55.6 / 61.1 | 49.8 / 57.1 |
| m31 | dst += v | 87.6 / 108 | 103 / 153 |
| m31 | dst -= v | 70.3 / 86.4 | 103 / 153 |
| m31 | dst = a * b | 59.5 / 68.2 | 59.8 / 69.3 |
| m31 | dst = c * src | 73.1 / 76.8 | 65.4 / 72.9 |
| qm31 | dst += c * src | 22.9 / 28.3 | 27.0 / 31.7 |
| qm31 | dst += v | 62.6 / 108 | 103 / 153 |
| qm31 | dst -= v | 49.9 / 86.4 | 103 / 153 |
| qm31 | dst = a * b | 27.0 / 31.1 | 24.1 / 26.2 |
| qm31 | dst = c * src | 32.1 / 37.6 | 30.5 / 36.1 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
