# Goldilocks: `gld` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v3](../v3/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `-` |
| Resolved field backends | `gld`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16} |

## Self-timings

### Packed operations

`gld`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.5 / - | 27.3 / - | 8.14 / - |
| add_assign_scalar | 44.6 / - | 44.4 / - | 25.8 / - |
| mul_elementwise | 10.8 / - | 11.3 / - | 5.25 / - |
| mul_elementwise_assign | 11.0 / - | 11.3 / - | 7.13 / - |
| sub_assign_scalar | 60.6 / - | 52.7 / - | 27.8 / - |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.74 / - | 2.57 / - |
| m31 | scalar a * b | 4.20 / - | 2.10 / - |
| qm31 | scalar a * b | 9.00 / - | 6.96 / - |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / - | 22.5 / - |
| gld | dst += v | 97.5 / - | 103 / - |
| gld | dst -= v | 103 / - | 103 / - |
| gld | dst = a * b | 23.8 / - | 26.9 / - |
| gld | dst = c * src | 25.1 / - | 27.4 / - |
| m31 | dst += c * src | 55.6 / - | 49.8 / - |
| m31 | dst += v | 87.6 / - | 103 / - |
| m31 | dst -= v | 70.3 / - | 103 / - |
| m31 | dst = a * b | 59.5 / - | 59.9 / - |
| m31 | dst = c * src | 73.1 / - | 65.4 / - |
| qm31 | dst += c * src | 22.9 / - | 27.0 / - |
| qm31 | dst += v | 62.5 / - | 103 / - |
| qm31 | dst -= v | 49.9 / - | 103 / - |
| qm31 | dst = a * b | 27.0 / - | 24.1 / - |
| qm31 | dst = c * src | 32.1 / - | 30.6 / - |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
