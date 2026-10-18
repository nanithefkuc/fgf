# Mersenne fields: `m31`, `qm31` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `m31`: `v4x` / `v3`; `qm31`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K): Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## Self-timings

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.4 / 33.4 | 28.1 / 34.2 | 7.91 / 20.4 |
| add_assign_scalar | 40.4 / 50.5 | 41.2 / 50.1 | 24.2 / 21.0 |
| mul_elementwise | 23.0 / 31.9 | 19.2 / 28.4 | 5.41 / 19.1 |
| mul_elementwise_assign | 25.0 / 31.7 | 21.3 / 32.6 | 7.87 / 21.6 |
| sub_assign_scalar | 32.6 / 39.6 | 33.9 / 41.7 | 22.5 / 24.1 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.1 / 32.0 | 18.3 / 30.9 | 7.62 / 23.9 |
| add_assign_scalar | 29.8 / 49.7 | 25.8 / 48.7 | 18.8 / 25.8 |
| mul_elementwise | 11.5 / 13.9 | 11.4 / 14.1 | 5.23 / 13.5 |
| mul_elementwise_assign | 7.24 / 14.1 | 7.10 / 14.3 | 6.56 / 13.3 |
| sub_assign_scalar | 23.6 / 41.0 | 21.1 / 41.3 | 16.6 / 31.8 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 28.2 / 32.3 |
| 1 KiB | 16 | 30.2 / 35.7 |
| 1 KiB | 2 | 26.9 / 35.1 |
| 1 KiB | 32 | 29.3 / 33.9 |
| 1 KiB | 4 | 29.1 / 33.2 |
| 1 KiB | 8 | 29.8 / 34.2 |
| 64 B | 1 | 5.73 / 11.6 |
| 64 B | 16 | 28.2 / 32.2 |
| 64 B | 2 | 10.4 / 19.2 |
| 64 B | 32 | 26.8 / 35.1 |
| 64 B | 4 | 17.9 / 24.4 |
| 64 B | 8 | 26.0 / 29.0 |
| 64 KiB | 1 | 27.9 / 34.1 |
| 64 KiB | 16 | 18.7 / 21.4 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 17.5 / 20.7 |
| 64 KiB | 4 | 26.1 / 35.3 |
| 64 KiB | 8 | 24.9 / 31.8 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 28.2 / 33.1 |
| add_assign per row | 1 KiB | 16 | 29.5 / 31.4 |
| add_assign per row | 1 KiB | 2 | 30.1 / 33.9 |
| add_assign per row | 1 KiB | 32 | 28.1 / 30.4 |
| add_assign per row | 1 KiB | 4 | 29.8 / 31.6 |
| add_assign per row | 1 KiB | 8 | 29.9 / 31.7 |
| add_assign per row | 64 B | 1 | 6.14 / 11.1 |
| add_assign per row | 64 B | 16 | 8.75 / 15.1 |
| add_assign per row | 64 B | 2 | 7.42 / 13.0 |
| add_assign per row | 64 B | 32 | 8.50 / 14.7 |
| add_assign per row | 64 B | 4 | 8.29 / 14.1 |
| add_assign per row | 64 B | 8 | 8.49 / 14.8 |
| add_assign per row | 64 KiB | 1 | 27.9 / 34.1 |
| add_assign per row | 64 KiB | 16 | 18.6 / 21.2 |
| add_assign per row | 64 KiB | 2 | 29.5 / 36.9 |
| add_assign per row | 64 KiB | 32 | 17.5 / 20.6 |
| add_assign per row | 64 KiB | 4 | 26.0 / 35.1 |
| add_assign per row | 64 KiB | 8 | 24.8 / 31.7 |
| add_assign | 1 KiB | 1 | 29.3 / 32.8 |
| add_assign | 1 KiB | 16 | 30.3 / 35.8 |
| add_assign | 1 KiB | 2 | 27.9 / 35.2 |
| add_assign | 1 KiB | 32 | 29.4 / 34.0 |
| add_assign | 1 KiB | 4 | 29.5 / 33.5 |
| add_assign | 1 KiB | 8 | 30.1 / 34.2 |
| add_assign | 64 B | 1 | 7.11 / 12.1 |
| add_assign | 64 B | 16 | 29.5 / 32.9 |
| add_assign | 64 B | 2 | 13.2 / 19.2 |
| add_assign | 64 B | 32 | 27.9 / 35.3 |
| add_assign | 64 B | 4 | 21.0 / 25.9 |
| add_assign | 64 B | 8 | 29.8 / 29.8 |
| add_assign | 64 KiB | 1 | 27.9 / 34.2 |
| add_assign | 64 KiB | 16 | 18.7 / 21.3 |
| add_assign | 64 KiB | 2 | 29.6 / 36.9 |
| add_assign | 64 KiB | 32 | 17.5 / 20.7 |
| add_assign | 64 KiB | 4 | 26.1 / 35.3 |
| add_assign | 64 KiB | 8 | 24.9 / 31.7 |

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
