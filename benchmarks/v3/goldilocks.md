# Goldilocks: `gld` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v2](../v2/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| add_assign | 29.4 / 21.1 | 27.3 / 21.6 | 7.99 / 20.9 |
| add_assign_scalar | 44.8 / 39.7 | 44.4 / 39.1 | 25.5 / 32.4 |
| mul_elementwise | 10.8 / 10.2 | 11.3 / 10.6 | 5.20 / 10.1 |
| mul_elementwise_assign | 11.0 / 10.7 | 11.3 / 10.8 | 7.23 / 9.49 |
| sub_assign_scalar | 59.3 / 39.2 | 52.7 / 38.7 | 27.4 / 32.5 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.478 / 2.149 | 2.564 / 1.162 |
| m31 | scalar a * b | 4.181 / 1.612 | 2.092 / 0.881 |
| qm31 | scalar a * b | 16.13 / 6.768 | 7.046 / 3.037 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.8 |
| gld | dst += v | 97.5 / 78.4 | 103 / 141 |
| gld | dst -= v | 103 / 77.8 | 103 / 141 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.5 / 26.0 |
| m31 | dst += c * src | 32.9 / 36.0 | 49.8 / 54.7 |
| m31 | dst += v | 98.5 / 117 | 103 / 147 |
| m31 | dst -= v | 98.5 / 116 | 103 / 147 |
| m31 | dst = a * b | 34.8 / 39.5 | 59.9 / 66.5 |
| m31 | dst = c * src | 50.4 / 51.4 | 65.4 / 69.7 |
| qm31 | dst += c * src | 21.4 / 24.2 | 27.0 / 30.5 |
| qm31 | dst += v | 68.1 / 117 | 103 / 147 |
| qm31 | dst -= v | 67.7 / 116 | 103 / 147 |
| qm31 | dst = a * b | 23.9 / 27.4 | 24.1 / 25.3 |
| qm31 | dst = c * src | 29.2 / 32.8 | 30.6 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
