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
| add_assign | 29.4 / 33.4 | 28.0 / 34.5 | 8.14 / 20.5 |
| add_assign_scalar | 40.4 / 50.5 | 41.2 / 50.1 | 24.7 / 20.9 |
| mul_elementwise | 23.0 / 31.9 | 20.2 / 31.1 | 5.47 / 17.9 |
| mul_elementwise_assign | 25.1 / 31.7 | 22.7 / 32.7 | 7.92 / 21.3 |
| sub_assign_scalar | 32.7 / 39.6 | 34.1 / 41.7 | 23.0 / 24.1 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.0 / 32.0 | 18.3 / 31.7 | 7.77 / 24.2 |
| add_assign_scalar | 29.8 / 49.7 | 25.8 / 48.7 | 19.1 / 25.8 |
| mul_elementwise | 11.5 / 13.9 | 11.7 / 14.2 | 5.31 / 13.6 |
| mul_elementwise_assign | 7.24 / 14.5 | 7.37 / 14.3 | 6.56 / 13.2 |
| sub_assign_scalar | 23.6 / 41.0 | 21.1 / 41.2 | 16.8 / 31.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 28.3 / 32.3 |
| 1 KiB | 16 | 30.3 / 35.7 |
| 1 KiB | 2 | 26.9 / 35.1 |
| 1 KiB | 32 | 29.4 / 34.0 |
| 1 KiB | 4 | 29.1 / 33.6 |
| 1 KiB | 8 | 29.8 / 34.8 |
| 64 B | 1 | 5.57 / 11.2 |
| 64 B | 16 | 28.3 / 32.2 |
| 64 B | 2 | 10.3 / 19.2 |
| 64 B | 32 | 26.9 / 35.1 |
| 64 B | 4 | 18.0 / 24.3 |
| 64 B | 8 | 25.3 / 29.0 |
| 64 KiB | 1 | 27.9 / 34.3 |
| 64 KiB | 16 | 18.5 / 21.7 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 17.9 / 20.7 |
| 64 KiB | 4 | 26.1 / 35.2 |
| 64 KiB | 8 | 25.6 / 30.9 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 28.1 / 33.0 |
| add_assign per row | 1 KiB | 16 | 29.1 / 31.3 |
| add_assign per row | 1 KiB | 2 | 30.0 / 33.9 |
| add_assign per row | 1 KiB | 32 | 28.4 / 30.4 |
| add_assign per row | 1 KiB | 4 | 29.9 / 31.5 |
| add_assign per row | 1 KiB | 8 | 30.0 / 31.6 |
| add_assign per row | 64 B | 1 | 6.14 / 11.1 |
| add_assign per row | 64 B | 16 | 8.98 / 15.1 |
| add_assign per row | 64 B | 2 | 7.42 / 13.0 |
| add_assign per row | 64 B | 32 | 8.43 / 14.8 |
| add_assign per row | 64 B | 4 | 8.19 / 14.1 |
| add_assign per row | 64 B | 8 | 8.80 / 14.8 |
| add_assign per row | 64 KiB | 1 | 27.9 / 34.3 |
| add_assign per row | 64 KiB | 16 | 18.5 / 21.6 |
| add_assign per row | 64 KiB | 2 | 29.5 / 36.8 |
| add_assign per row | 64 KiB | 32 | 18.0 / 20.6 |
| add_assign per row | 64 KiB | 4 | 26.0 / 35.1 |
| add_assign per row | 64 KiB | 8 | 25.5 / 30.4 |
| add_assign | 1 KiB | 1 | 30.3 / 32.8 |
| add_assign | 1 KiB | 16 | 30.4 / 35.8 |
| add_assign | 1 KiB | 2 | 27.8 / 35.3 |
| add_assign | 1 KiB | 32 | 29.5 / 34.0 |
| add_assign | 1 KiB | 4 | 29.6 / 33.5 |
| add_assign | 1 KiB | 8 | 30.2 / 35.1 |
| add_assign | 64 B | 1 | 7.41 / 12.1 |
| add_assign | 64 B | 16 | 30.2 / 33.0 |
| add_assign | 64 B | 2 | 13.2 / 19.1 |
| add_assign | 64 B | 32 | 27.9 / 35.4 |
| add_assign | 64 B | 4 | 21.0 / 25.9 |
| add_assign | 64 B | 8 | 27.8 / 29.9 |
| add_assign | 64 KiB | 1 | 27.9 / 34.3 |
| add_assign | 64 KiB | 16 | 18.5 / 21.7 |
| add_assign | 64 KiB | 2 | 29.6 / 36.9 |
| add_assign | 64 KiB | 32 | 18.0 / 20.7 |
| add_assign | 64 KiB | 4 | 26.1 / 35.3 |
| add_assign | 64 KiB | 8 | 25.6 / 30.9 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.742 / 1.712 | 2.570 / 1.161 |
| m31 | scalar a * b | 4.201 / 1.605 | 2.098 / 0.831 |
| qm31 | scalar a * b | 8.999 / 4.550 | 6.959 / 3.007 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.8 |
| gld | dst += v | 97.5 / 78.2 | 103 / 141 |
| gld | dst -= v | 103 / 78.3 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.4 / 26.0 |
| m31 | dst += c * src | 55.6 / 58.6 | 49.8 / 54.7 |
| m31 | dst += v | 87.6 / 103 | 103 / 147 |
| m31 | dst -= v | 70.3 / 82.9 | 103 / 147 |
| m31 | dst = a * b | 59.5 / 65.5 | 59.9 / 66.4 |
| m31 | dst = c * src | 73.1 / 73.7 | 65.4 / 69.7 |
| qm31 | dst += c * src | 22.9 / 27.2 | 27.0 / 30.5 |
| qm31 | dst += v | 62.5 / 103 | 103 / 146 |
| qm31 | dst -= v | 49.9 / 82.8 | 103 / 146 |
| qm31 | dst = a * b | 27.0 / 29.9 | 24.1 / 25.4 |
| qm31 | dst = c * src | 32.1 / 35.9 | 30.6 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
