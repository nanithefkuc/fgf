# Goldilocks: `gld` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/goldilocks.md); the other snapshot at [v2](../v2/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| Revision | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| add_assign | 29.7 / 22.5 | 27.3 / 22.5 | 8.12 / 20.9 |
| add_assign_scalar | 45.0 / 39.8 | 44.4 / 40.8 | 25.9 / 33.7 |
| mul_elementwise | 10.8 / 10.5 | 11.3 / 11.3 | 5.16 / 10.5 |
| mul_elementwise_assign | 11.0 / 11.1 | 11.3 / 11.3 | 7.01 / 9.68 |
| sub_assign_scalar | 60.1 / 40.8 | 52.7 / 40.4 | 27.9 / 33.8 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.484 / 2.046 | 2.610 / 1.110 |
| m31 | scalar a * b | 4.192 / 1.547 | 2.043 / 0.803 |
| qm31 | scalar a * b | 9.691 / 4.959 | 6.953 / 2.911 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 17.0 | 22.5 / 20.6 |
| gld | dst += v | 97.5 / 81.6 | 103 / 149 |
| gld | dst -= v | 103 / 81.1 | 103 / 147 |
| gld | dst = a * b | 23.8 / 23.3 | 26.9 / 26.5 |
| gld | dst = c * src | 25.1 / 23.8 | 27.4 / 27.1 |
| m31 | dst += c * src | 32.9 / 37.6 | 49.8 / 57.0 |
| m31 | dst += v | 98.4 / 122 | 103 / 153 |
| m31 | dst -= v | 98.5 / 121 | 103 / 153 |
| m31 | dst = a * b | 34.9 / 41.2 | 59.9 / 69.3 |
| m31 | dst = c * src | 50.4 / 53.6 | 65.4 / 72.7 |
| qm31 | dst += c * src | 21.4 / 25.3 | 27.0 / 31.8 |
| qm31 | dst += v | 68.2 / 122 | 103 / 153 |
| qm31 | dst -= v | 67.8 / 121 | 103 / 153 |
| qm31 | dst = a * b | 23.9 / 28.5 | 24.0 / 26.2 |
| qm31 | dst = c * src | 29.2 / 34.2 | 30.6 / 36.1 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
