# Byte fields: `gf8b`, `gf8d` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `gf8b`: `v4x` / `v3_gfni_crypto`; `gf8d`: `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64; Golden Cove (i7-12700K): Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64 |

## Self-timings

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Field | Operation | scalar mul (ns/op) |
| --- | --- | --- |
| gf8b | scalar mul | 0.701 / 0.357 |
| gf8d | scalar mul | 0.701 / 0.357 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 52.0 / 94.0 | 32.0 / 61.1 | 7.64 / 10.3 |
| add_assign_scalar | 42.8 / 77.1 | 41.5 / 64.8 | 28.5 / 24.6 |
| mul_add | 64.8 / 80.6 | 37.9 / 55.9 | 8.30 / 16.7 |
| mul_assign | 124 / 167 | 51.5 / 73.5 | 30.7 / 28.0 |
| mul_elementwise | 74.5 / 97.4 | 26.3 / 40.8 | 5.13 / 7.17 |
| mul_elementwise_assign | 79.8 / 114 | 31.2 / 47.7 | 6.71 / 9.58 |
| mul_into | 82.8 / 160 | 38.3 / 62.0 | 13.1 / 27.8 |
| sub_assign_scalar | 42.6 / 77.1 | 41.5 / 64.8 | 28.5 / 24.1 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.0 / 51.9 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.0 | - |
| gf8b | mul_into | 39.9 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.0 | - |
| gf8d | mul_elementwise | 28.7 / 47.6 | - |
| gf8d | mul_elementwise_assign | 40.7 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.2 / 40.0 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 14.37 / 4.096 | 14.39 / 3.857 |
| 32 B | 14.36 / 4.059 | 14.44 / 3.846 |
| 64 B | 15.69 / 4.283 | 15.68 / 4.060 |
| 128 B | 14.83 / 4.910 | 14.85 / 4.707 |
| 256 B | 16.34 / 5.978 | 16.14 / 5.783 |
| 512 B | 18.38 / 8.637 | 18.44 / 8.554 |
| 1 KiB | 22.21 / 14.29 | 22.12 / 14.29 |
| 2 KiB | 30.31 / 25.13 | 30.54 / 25.10 |
| 4 KiB | 62.07 / 47.44 | 61.59 / 47.42 |
| 16 KiB | 210.9 / 178.7 | 209.9 / 178.6 |
| 64 KiB | 1633 / 1097 | 1632 / 1098 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.3 / 69.1 | 42.9 / 75.6 | 43.7 / 66.6 | 44.5 / 62.1 |
| mul_add_gather_with | 44.3 / 69.8 | 42.9 / 75.4 | 43.8 / 65.8 | 44.4 / 62.2 |
| mul_add_matrix | 89.6 / 102 | 132 / 142 | 127 / 140 | 123 / 136 |
| mul_add_matrix_with | 89.3 / 102 | 132 / 143 | 123 / 138 | 119 / 135 |
| mul_add_scatter | 45.4 / 66.8 | 48.2 / 69.1 | 48.3 / 69.0 | 45.4 / 41.7 |
| mul_add_scatter_with | 45.4 / 67.0 | 48.2 / 69.3 | 48.3 / 69.1 | 45.4 / 41.8 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.4 / 80.3 | 105 / 84.7 |
| gf8d | mul_add_gather_with | 80.0 / 83.4 | 96.4 / 88.1 |
| gf8d | mul_into_gather | 90.9 / 101 | 105 / 106 |
| gf8d | mul_into_gather_with | 91.4 / 99.9 | 105 / 106 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.0 / 49.8 | 31.3 / 31.1 | 19.5 / 20.7 |
| gf8b | mul_add_matrix_at | 74.3 / 57.5 | 40.4 / 31.0 | 25.0 / 22.5 |
| gf8b | mul_add_matrix_with | 74.1 / 58.2 | 39.8 / 37.9 | 24.7 / 25.7 |
| gf8b | mul_into_matrix | 73.6 / 68.1 | 42.3 / 40.5 | 25.8 / 27.7 |
| gf8b | mul_into_matrix_with | 76.3 / 66.7 | 42.0 / 40.8 | 25.4 / 28.1 |
| gf8d | fill(0) + mul_add_matrix | 57.5 / 52.6 | 31.6 / 32.8 | 19.4 / 22.9 |
| gf8d | mul_into_matrix | 75.9 / 70.5 | 42.6 / 44.0 | 25.8 / 31.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 49.7 / 79.4 |
| 1 KiB | 16 | 84.0 / 129 |
| 1 KiB | 2 | 66.7 / 87.6 |
| 1 KiB | 32 | 34.9 / 63.9 |
| 1 KiB | 4 | 42.4 / 93.0 |
| 1 KiB | 8 | 80.6 / 124 |
| 64 B | 1 | 7.15 / 11.2 |
| 64 B | 16 | 49.9 / 79.1 |
| 64 B | 2 | 13.6 / 25.4 |
| 64 B | 32 | 66.7 / 87.5 |
| 64 B | 4 | 26.0 / 48.4 |
| 64 B | 8 | 26.8 / 69.5 |
| 64 KiB | 1 | 31.8 / 60.7 |
| 64 KiB | 16 | 18.9 / 25.4 |
| 64 KiB | 2 | 32.0 / 60.9 |
| 64 KiB | 32 | 18.5 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.1 |
| 64 KiB | 8 | 35.6 / 44.5 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 46.7 / 80.1 |
| add_assign per row | 1 KiB | 16 | 60.1 / 96.6 |
| add_assign per row | 1 KiB | 2 | 51.8 / 82.9 |
| add_assign per row | 1 KiB | 32 | 34.8 / 62.7 |
| add_assign per row | 1 KiB | 4 | 45.0 / 82.7 |
| add_assign per row | 1 KiB | 8 | 62.8 / 99.6 |
| add_assign per row | 64 B | 1 | 6.76 / 12.1 |
| add_assign per row | 64 B | 16 | 8.14 / 14.1 |
| add_assign per row | 64 B | 2 | 7.58 / 13.3 |
| add_assign per row | 64 B | 32 | 8.51 / 14.8 |
| add_assign per row | 64 B | 4 | 8.20 / 14.1 |
| add_assign per row | 64 B | 8 | 8.54 / 14.9 |
| add_assign per row | 64 KiB | 1 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.9 / 25.4 |
| add_assign per row | 64 KiB | 2 | 31.9 / 60.6 |
| add_assign per row | 64 KiB | 32 | 18.5 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 35.4 / 44.4 |
| add_assign | 1 KiB | 1 | 59.8 / 82.3 |
| add_assign | 1 KiB | 16 | 85.2 / 127 |
| add_assign | 1 KiB | 2 | 74.3 / 90.4 |
| add_assign | 1 KiB | 32 | 35.0 / 63.8 |
| add_assign | 1 KiB | 4 | 44.0 / 93.9 |
| add_assign | 1 KiB | 8 | 82.7 / 125 |
| add_assign | 64 B | 1 | 7.73 / 13.3 |
| add_assign | 64 B | 16 | 60.0 / 82.5 |
| add_assign | 64 B | 2 | 14.8 / 31.0 |
| add_assign | 64 B | 32 | 74.3 / 90.5 |
| add_assign | 64 B | 4 | 28.5 / 46.5 |
| add_assign | 64 B | 8 | 30.9 / 70.6 |
| add_assign | 64 KiB | 1 | 31.8 / 60.8 |
| add_assign | 64 KiB | 16 | 18.9 / 25.3 |
| add_assign | 64 KiB | 2 | 31.9 / 61.0 |
| add_assign | 64 KiB | 32 | 18.5 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 35.6 / 44.5 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / 12.1 | 27.2 / 46.5 | 30.4 / 71.4 | 55.8 / 81.6 | 60.7 / 97.7 | 56.1 / 82.5 | 57.6 / 98.0 | 57.3 / 82.2 | 62.3 / 99.2 | 59.0 / 83.1 | 62.8 / 103 |
| mul_add | 3.82 / 14.5 | 15.5 / 41.4 | 26.0 / 58.6 | 47.6 / 69.7 | 50.4 / 79.7 | 48.3 / 69.7 | 49.8 / 79.4 | 49.5 / 69.7 | 52.4 / 80.2 | 49.9 / 69.8 | 41.2 / 67.4 |
| mul_add_scatter (16 rows) | 11.4 / 19.0 | 31.1 / 62.1 | 43.7 / 91.8 | 87.5 / 108 | 58.4 / 88.5 | 65.4 / 107 | 56.4 / 87.7 | 89.2 / 108 | 58.2 / 90.7 | 66.6 / 108 | 46.4 / 60.3 |
| mul_add_scatter (4 rows) | 8.56 / 14.0 | 27.2 / 39.3 | 38.7 / 77.5 | 80.3 / 98.1 | 59.2 / 82.2 | 66.3 / 97.9 | 52.0 / 82.8 | 84.8 / 97.1 | 60.3 / 83.7 | 63.6 / 97.1 | 43.6 / 58.4 |
| mul_assign | 7.41 / 15.5 | 25.5 / 58.8 | 42.7 / 85.9 | 80.2 / 137 | 92.4 / 135 | 80.4 / 140 | 83.5 / 136 | 82.6 / 138 | 94.8 / 137 | 82.8 / 137 | 69.6 / 117 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 79.7 / 94.5 | 55.3 / 94.8 | 35.1 / 41.0 |
| 1 KiB | 16 | 60.4 / 89.4 | 49.0 / 81.2 | 31.5 / 40.8 |
| 16 KiB | 0 | 48.0 / 69.9 | 91.8 / 88.9 | 42.2 / 46.3 |
| 16 KiB | 16 | 47.3 / 69.0 | 86.7 / 72.4 | 39.5 / 43.6 |
| 2 KiB | 0 | 94.6 / 106 | 78.8 / 100 | 39.4 / 43.6 |
| 2 KiB | 16 | 83.2 / 101 | 67.2 / 86.5 | 35.5 / 42.9 |
| 256 B | 0 | 30.9 / 44.9 | 19.5 / 54.1 | 25.9 / 28.2 |
| 256 B | 16 | 27.7 / 36.8 | 18.4 / 45.1 | 23.1 / 28.0 |
| 3 KiB | 0 | 104 / 116 | 76.5 / 86.6 | 40.8 / 44.7 |
| 3 KiB | 16 | 93.1 / 112 | 61.2 / 73.1 | 36.0 / 43.2 |
| 4 KiB | 0 | 90.1 / 108 | 75.8 / 84.8 | 40.7 / 43.9 |
| 4 KiB | 16 | 82.5 / 105 | 65.7 / 70.0 | 33.5 / 40.3 |
| 512 B | 0 | 51.8 / 72.1 | 34.7 / 75.7 | 29.0 / 35.7 |
| 512 B | 16 | 37.0 / 67.9 | 31.4 / 67.1 | 26.7 / 35.4 |
| 8 KiB | 0 | 91.8 / 114 | 84.6 / 87.3 | 39.4 / 44.5 |
| 8 KiB | 16 | 88.3 / 111 | 77.8 / 71.4 | 35.0 / 42.5 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 96.2 / 95.0 |
| 16 | 96.5 / 94.9 |
| 32 | 96.5 / 94.9 |
| 48 | 96.5 / 94.9 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / 30.0 |
| 256 KiB | 16 | 29.8 / 22.3 |
| 64 KiB | 0 | 48.3 / 69.3 |
| 64 KiB | 16 | 48.3 / 68.9 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.4 / 70.4 |
| mul_add_scatter_with | 48.5 / 70.5 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 49.3 / 51.0 |
| gf8b | 16 KiB | 138 / 182 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 132 / 171 |
| gf8b | 64 KiB | 51.3 / 73.2 |
| gf8d | 1 MiB | 50.1 / 51.5 |
| gf8d | 16 KiB | 137 / 178 |
| gf8d | 256 KiB | 51.5 / 66.2 |
| gf8d | 4 KiB | 131 / 166 |
| gf8d | 64 KiB | 51.3 / 66.1 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.4 / 100 |
| 64 MiB | 6.06 / 14.4 |
| 8 MiB | 8.26 / 24.3 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 19.0 / 25.4 | 13.6 / 20.2 | 19.0 / 25.9 |
| 32 MiB | 9.40 / 27.3 | 5.98 / 14.0 | 6.24 / 16.1 |
| 8 MiB | 18.2 / 40.0 | 7.73 / 16.3 | 8.16 / 24.8 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 62.0 | 18.0 / 43.9 | 33.7 / 54.5 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.1 | 41.2 / 62.5 | 29.1 / 47.0 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 88.7 / 99.5 | 83.1 / 99.5 | 30.5 / 48.4 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 94.6 / 93.8 | 74.9 / 91.0 | 35.6 / 54.8 |
| systematic encode | 10 × 2 | 4 KiB | 73.8 / 73.0 | 65.5 / 71.2 | 54.1 / 72.0 |
| systematic encode | 10 × 4 | 4 KiB | 39.3 / 45.4 | 35.7 / 42.2 | 32.0 / 42.8 |
| systematic encode | 10 × 6 | 4 KiB | 26.6 / 31.3 | 24.4 / 30.0 | 22.5 / 30.1 |
| systematic encode | 10 × 2 | 16 KiB | 81.5 / 80.8 | 53.0 / 66.3 | 67.8 / 81.4 |
| systematic encode | 10 × 4 | 16 KiB | 42.8 / 48.8 | 36.7 / 37.9 | 38.7 / 48.3 |
| systematic encode | 10 × 6 | 16 KiB | 27.0 / 33.1 | 25.0 / 29.2 | 25.9 / 32.7 |
| systematic encode | 10 × 2 | 64 KiB | 83.5 / 75.8 | 54.0 / 65.5 | 73.4 / 83.1 |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / 48.4 | 35.9 / 32.3 | 40.6 / 45.0 |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / 32.5 | 23.7 / 23.0 | 26.6 / 29.4 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
