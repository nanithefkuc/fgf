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
| add_assign | 52.0 / 93.8 | 32.0 / 61.1 | 7.62 / 10.3 |
| add_assign_scalar | 59.9 / 110 | 51.9 / 73.5 | 29.2 / 27.6 |
| mul_add | 64.9 / 81.7 | 37.9 / 56.1 | 8.53 / 16.5 |
| mul_assign | 123 / 169 | 51.5 / 73.5 | 30.5 / 27.9 |
| mul_elementwise | 57.2 / 63.2 | 23.5 / 44.5 | 5.25 / 7.16 |
| mul_elementwise_assign | 61.7 / 66.8 | 28.0 / 60.7 | 7.21 / 9.66 |
| mul_into | 81.7 / 160 | 38.1 / 62.1 | 5.24 / 27.2 |
| sub_assign_scalar | 60.1 / 109 | 51.9 / 73.5 | 29.2 / 27.7 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.1 / 51.9 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.0 | - |
| gf8b | mul_into | 39.9 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.1 | - |
| gf8d | mul_elementwise | 28.7 / 47.5 | - |
| gf8d | mul_elementwise_assign | 40.7 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.2 / 40.1 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 14.40 / 4.096 | 14.40 / 3.848 |
| 32 B | 14.34 / 4.077 | 14.40 / 3.886 |
| 64 B | 15.91 / 4.285 | 16.00 / 4.063 |
| 128 B | 16.03 / 10.08 | 16.13 / 10.73 |
| 256 B | 16.54 / 5.984 | 16.55 / 5.766 |
| 512 B | 18.71 / 8.324 | 18.40 / 8.215 |
| 1 KiB | 21.92 / 14.21 | 21.83 / 14.27 |
| 2 KiB | 30.01 / 25.12 | 30.21 / 25.10 |
| 4 KiB | 61.08 / 47.43 | 60.81 / 47.40 |
| 16 KiB | 208.6 / 178.6 | 207.8 / 178.6 |
| 64 KiB | 1633 / 1084 | 1629 / 1097 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.4 / 65.1 | 42.8 / 71.9 | 43.9 / 64.1 | 44.2 / 62.6 |
| mul_add_gather_with | 44.3 / 65.1 | 42.8 / 72.1 | 44.0 / 63.6 | 44.2 / 62.5 |
| mul_add_matrix | 90.5 / 106 | 134 / 148 | 130 / 146 | 121 / 146 |
| mul_add_matrix_with | 90.4 / 106 | 134 / 148 | 126 / 146 | 119 / 146 |
| mul_add_scatter | 45.5 / 66.9 | 48.4 / 69.2 | 48.5 / 69.3 | 46.0 / 47.1 |
| mul_add_scatter_with | 45.6 / 67.0 | 48.4 / 69.3 | 48.5 / 69.4 | 46.2 / 47.2 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.0 / 80.1 | 105 / 84.6 |
| gf8d | mul_add_gather_with | 80.1 / 83.4 | 96.3 / 88.0 |
| gf8d | mul_into_gather | 91.2 / 101 | 105 / 107 |
| gf8d | mul_into_gather_with | 91.4 / 101 | 105 / 106 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 56.4 / 49.8 | 31.4 / 31.3 | 19.6 / 20.8 |
| gf8b | mul_add_matrix_at | 74.5 / 57.7 | 39.3 / 30.8 | 25.1 / 22.7 |
| gf8b | mul_add_matrix_with | 73.5 / 58.3 | 38.5 / 38.1 | 24.0 / 25.9 |
| gf8b | mul_into_matrix | 73.5 / 68.2 | 42.1 / 40.5 | 25.7 / 27.7 |
| gf8b | mul_into_matrix_with | 76.2 / 66.6 | 40.9 / 41.2 | 25.0 / 28.1 |
| gf8d | fill(0) + mul_add_matrix | 57.6 / 52.6 | 31.4 / 33.0 | 19.0 / 23.5 |
| gf8d | mul_into_matrix | 76.3 / 70.6 | 42.7 / 44.2 | 25.5 / 31.8 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 47.4 / 79.3 |
| 1 KiB | 16 | 83.9 / 128 |
| 1 KiB | 2 | 67.3 / 88.4 |
| 1 KiB | 32 | 34.5 / 63.5 |
| 1 KiB | 4 | 42.6 / 93.1 |
| 1 KiB | 8 | 80.0 / 126 |
| 64 B | 1 | 7.15 / 11.2 |
| 64 B | 16 | 47.5 / 79.3 |
| 64 B | 2 | 7.30 / 12.3 |
| 64 B | 32 | 66.9 / 88.4 |
| 64 B | 4 | 25.8 / 46.5 |
| 64 B | 8 | 25.7 / 69.8 |
| 64 KiB | 1 | 31.8 / 60.7 |
| 64 KiB | 16 | 18.9 / 25.5 |
| 64 KiB | 2 | 31.9 / 60.9 |
| 64 KiB | 32 | 18.3 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.1 |
| 64 KiB | 8 | 37.4 / 41.8 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 47.4 / 80.6 |
| add_assign per row | 1 KiB | 16 | 53.9 / 96.5 |
| add_assign per row | 1 KiB | 2 | 51.4 / 83.8 |
| add_assign per row | 1 KiB | 32 | 31.8 / 61.4 |
| add_assign per row | 1 KiB | 4 | 45.5 / 83.3 |
| add_assign per row | 1 KiB | 8 | 55.7 / 99.6 |
| add_assign per row | 64 B | 1 | 6.75 / 12.7 |
| add_assign per row | 64 B | 16 | 8.14 / 14.1 |
| add_assign per row | 64 B | 2 | 6.85 / 9.15 |
| add_assign per row | 64 B | 32 | 8.51 / 14.6 |
| add_assign per row | 64 B | 4 | 8.19 / 15.1 |
| add_assign per row | 64 B | 8 | 8.54 / 14.9 |
| add_assign per row | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.9 / 25.4 |
| add_assign per row | 64 KiB | 2 | 31.8 / 60.6 |
| add_assign per row | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 37.2 / 41.7 |
| add_assign | 1 KiB | 1 | 52.7 / 82.7 |
| add_assign | 1 KiB | 16 | 84.7 / 132 |
| add_assign | 1 KiB | 2 | 72.0 / 90.9 |
| add_assign | 1 KiB | 32 | 34.7 / 64.1 |
| add_assign | 1 KiB | 4 | 43.3 / 94.0 |
| add_assign | 1 KiB | 8 | 81.7 / 129 |
| add_assign | 64 B | 1 | 7.73 / 13.9 |
| add_assign | 64 B | 16 | 52.7 / 82.7 |
| add_assign | 64 B | 2 | 7.43 / 12.3 |
| add_assign | 64 B | 32 | 71.9 / 90.9 |
| add_assign | 64 B | 4 | 28.5 / 44.7 |
| add_assign | 64 B | 8 | 31.0 / 71.9 |
| add_assign | 64 KiB | 1 | 31.7 / 60.7 |
| add_assign | 64 KiB | 16 | 18.9 / 25.3 |
| add_assign | 64 KiB | 2 | 31.9 / 60.9 |
| add_assign | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 37.4 / 41.8 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / 13.3 | 27.8 / 43.0 | 29.8 / 71.9 | 55.9 / 82.3 | 60.6 / 98.0 | 59.9 / 83.2 | 68.0 / 98.8 | 57.6 / 83.1 | 61.7 / 98.8 | 60.9 / 84.0 | 62.9 / 100 |
| mul_add | 4.27 / 14.4 | 14.6 / 39.9 | 25.7 / 58.0 | 48.5 / 72.1 | 48.8 / 78.5 | 51.6 / 72.5 | 50.2 / 79.4 | 50.5 / 72.4 | 50.9 / 80.3 | 53.2 / 73.0 | 41.3 / 67.7 |
| mul_add_scatter (16 rows) | 11.9 / 19.7 | 30.9 / 45.0 | 41.6 / 82.3 | 68.4 / 102 | 57.9 / 87.4 | 62.2 / 102 | 54.6 / 88.3 | 70.7 / 103 | 60.2 / 89.1 | 62.3 / 102 | 45.1 / 61.5 |
| mul_add_scatter (4 rows) | 8.57 / 13.9 | 30.0 / 47.3 | 37.2 / 68.3 | 61.8 / 92.3 | 53.8 / 79.4 | 58.8 / 91.7 | 52.7 / 80.1 | 63.7 / 92.2 | 56.0 / 81.3 | 59.7 / 91.9 | 41.1 / 54.8 |
| mul_assign | 8.09 / 15.5 | 24.3 / 58.8 | 45.9 / 91.2 | 84.4 / 135 | 85.6 / 135 | 94.2 / 133 | 85.7 / 137 | 86.8 / 136 | 88.0 / 137 | 96.5 / 137 | 70.1 / 116 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 78.1 / 94.4 | 55.2 / 94.6 | 35.4 / 41.0 |
| 1 KiB | 16 | 58.6 / 89.4 | 49.0 / 80.6 | 31.6 / 40.8 |
| 16 KiB | 0 | 47.9 / 69.9 | 91.5 / 88.7 | 42.4 / 46.4 |
| 16 KiB | 16 | 47.5 / 69.1 | 85.7 / 71.7 | 39.5 / 43.7 |
| 2 KiB | 0 | 94.3 / 106 | 78.0 / 100 | 39.4 / 43.6 |
| 2 KiB | 16 | 82.3 / 101 | 65.3 / 85.2 | 35.1 / 42.9 |
| 256 B | 0 | 30.0 / 44.8 | 19.5 / 54.1 | 25.8 / 28.4 |
| 256 B | 16 | 26.8 / 36.8 | 18.4 / 45.0 | 23.4 / 27.8 |
| 3 KiB | 0 | 104 / 116 | 76.6 / 86.2 | 40.8 / 44.7 |
| 3 KiB | 16 | 92.3 / 112 | 58.7 / 70.5 | 36.2 / 43.5 |
| 4 KiB | 0 | 89.5 / 108 | 76.0 / 84.8 | 40.8 / 43.9 |
| 4 KiB | 16 | 82.0 / 105 | 65.2 / 69.1 | 33.8 / 40.5 |
| 512 B | 0 | 51.6 / 72.3 | 34.9 / 75.7 | 29.4 / 35.7 |
| 512 B | 16 | 36.7 / 67.9 | 31.4 / 66.5 | 27.0 / 35.5 |
| 8 KiB | 0 | 91.6 / 114 | 84.3 / 87.2 | 40.1 / 45.2 |
| 8 KiB | 16 | 88.1 / 111 | 77.5 / 70.7 | 35.5 / 42.3 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 96.3 / 95.0 |
| 16 | 96.3 / 94.9 |
| 32 | 96.5 / 94.9 |
| 48 | 96.4 / 94.9 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.5 / 29.6 |
| 256 KiB | 16 | 29.6 / 22.4 |
| 64 KiB | 0 | 48.5 / 69.3 |
| 64 KiB | 16 | 48.3 / 69.3 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.3 / 70.3 |
| mul_add_scatter_with | 48.5 / 70.8 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 47.9 / 52.6 |
| gf8b | 16 KiB | 138 / 182 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 131 / 170 |
| gf8b | 64 KiB | 51.3 / 73.2 |
| gf8d | 1 MiB | 48.4 / 52.6 |
| gf8d | 16 KiB | 138 / 178 |
| gf8d | 256 KiB | 51.5 / 66.3 |
| gf8d | 4 KiB | 130 / 166 |
| gf8d | 64 KiB | 51.3 / 65.6 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.4 / 100 |
| 64 MiB | 6.06 / 14.5 |
| 8 MiB | 8.35 / 24.1 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.8 / 25.3 | 13.7 / 20.1 | 18.8 / 24.0 |
| 32 MiB | 9.37 / 27.4 | 5.97 / 14.1 | 6.22 / 16.2 |
| 8 MiB | 18.1 / 40.0 | 7.67 / 16.5 | 8.15 / 24.5 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 62.0 | 18.0 / 43.8 | 33.6 / 54.5 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.3 | 41.2 / 62.5 | 29.3 / 47.2 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 90.3 / 100 | 82.9 / 99.5 | 30.6 / 48.3 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 95.7 / 94.0 | 74.8 / 91.0 | 35.6 / 55.1 |
| systematic encode | 10 × 2 | 4 KiB | 73.4 / 73.1 | 65.2 / 71.3 | 53.9 / 71.6 |
| systematic encode | 10 × 4 | 4 KiB | 39.5 / 45.5 | 35.7 / 42.4 | 31.9 / 42.8 |
| systematic encode | 10 × 6 | 4 KiB | 26.7 / 31.4 | 24.4 / 29.9 | 22.5 / 30.0 |
| systematic encode | 10 × 2 | 16 KiB | 81.3 / 80.9 | 52.7 / 66.6 | 67.6 / 81.4 |
| systematic encode | 10 × 4 | 16 KiB | 42.8 / 48.8 | 36.5 / 38.0 | 38.7 / 48.3 |
| systematic encode | 10 × 6 | 16 KiB | 27.1 / 33.1 | 24.9 / 29.2 | 25.8 / 32.6 |
| systematic encode | 10 × 2 | 64 KiB | 83.3 / 76.0 | 53.9 / 65.8 | 73.6 / 82.6 |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / 48.4 | 35.4 / 33.2 | 40.7 / 45.6 |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / 32.5 | 23.9 / 23.0 | 26.5 / 29.3 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
