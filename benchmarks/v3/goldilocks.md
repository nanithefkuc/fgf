# Goldilocks: `gld` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v2](../v2/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| add_assign | 29.7 / 21.6 | 27.3 / 21.6 | 7.93 / 19.8 |
| add_assign_scalar | 44.5 / 37.8 | 44.4 / 39.1 | 26.0 / 32.3 |
| mul_elementwise | 10.8 / 10.2 | 11.3 / 10.8 | 5.20 / 10.1 |
| mul_elementwise_assign | 11.0 / 10.7 | 11.3 / 10.6 | 7.05 / 9.44 |
| sub_assign_scalar | 60.6 / 39.2 | 52.7 / 38.6 | 27.9 / 32.4 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.485 / 2.149 | 2.563 / 1.162 |
| m31 | scalar a * b | 4.201 / 1.611 | 2.098 / 0.881 |
| qm31 | scalar a * b | 4.524 / 2.417 | 6.959 / 3.053 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.6 |
| gld | dst += v | 97.4 / 78.1 | 103 / 141 |
| gld | dst -= v | 103 / 77.6 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.4 / 26.0 |
| m31 | dst += c * src | 38.1 / 48.2 | 49.8 / 54.7 |
| m31 | dst += v | 98.5 / 117 | 103 / 147 |
| m31 | dst -= v | 98.5 / 116 | 103 / 147 |
| m31 | dst = a * b | 51.4 / 58.6 | 59.9 / 66.5 |
| m31 | dst = c * src | 55.6 / 66.6 | 65.4 / 69.9 |
| qm31 | dst += c * src | 23.4 / 28.8 | 27.0 / 30.4 |
| qm31 | dst += v | 68.2 / 117 | 103 / 147 |
| qm31 | dst -= v | 67.8 / 117 | 103 / 147 |
| qm31 | dst = a * b | 28.2 / 36.3 | 24.0 / 25.2 |
| qm31 | dst = c * src | 33.2 / 42.1 | 30.6 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
