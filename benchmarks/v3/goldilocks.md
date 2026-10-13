# Goldilocks: `gld` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v2](../v2/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| add_assign | 29.4 / - | 27.3 / - | 7.99 / - |
| add_assign_scalar | 44.8 / - | 44.4 / - | 25.5 / - |
| mul_elementwise | 10.8 / - | 11.3 / - | 5.20 / - |
| mul_elementwise_assign | 11.0 / - | 11.3 / - | 7.23 / - |
| sub_assign_scalar | 59.3 / - | 52.7 / - | 27.4 / - |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.48 / - | 2.56 / - |
| m31 | scalar a * b | 4.18 / - | 2.09 / - |
| qm31 | scalar a * b | 16.1 / - | 7.05 / - |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / - | 22.5 / - |
| gld | dst += v | 97.5 / - | 103 / - |
| gld | dst -= v | 103 / - | 103 / - |
| gld | dst = a * b | 23.8 / - | 26.9 / - |
| gld | dst = c * src | 25.1 / - | 27.5 / - |
| m31 | dst += c * src | 32.9 / - | 49.8 / - |
| m31 | dst += v | 98.5 / - | 103 / - |
| m31 | dst -= v | 98.5 / - | 103 / - |
| m31 | dst = a * b | 34.8 / - | 59.9 / - |
| m31 | dst = c * src | 50.4 / - | 65.4 / - |
| qm31 | dst += c * src | 21.4 / - | 27.0 / - |
| qm31 | dst += v | 68.1 / - | 103 / - |
| qm31 | dst -= v | 67.7 / - | 103 / - |
| qm31 | dst = a * b | 23.9 / - | 24.1 / - |
| qm31 | dst = c * src | 29.2 / - | 30.6 / - |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
