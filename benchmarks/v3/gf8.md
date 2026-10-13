# Byte fields: `gf8b`, `gf8d` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v2](../v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| gf8d | scalar mul | 0.605 / 0.357 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 52.4 / 94.1 | 32.0 / 61.1 | 7.66 / 10.3 |
| add_assign_scalar | 41.5 / 77.1 | 41.5 / 64.8 | 28.3 / 24.6 |
| mul_add | 64.3 / 81.3 | 37.9 / 56.1 | 8.48 / 16.3 |
| mul_assign | 127 / 167 | 51.5 / 73.5 | 30.6 / 27.8 |
| mul_elementwise | 74.3 / 99.2 | 25.4 / 38.4 | 5.16 / 7.23 |
| mul_elementwise_assign | 74.1 / 113 | 29.7 / 47.8 | 6.56 / 9.64 |
| mul_into | 83.4 / 163 | 38.2 / 62.1 | 13.1 / 27.9 |
| sub_assign_scalar | 43.0 / 77.1 | 41.5 / 64.8 | 28.3 / 24.5 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.2 / 51.9 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.1 | - |
| gf8b | mul_into | 39.9 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.0 | - |
| gf8d | mul_elementwise | 28.8 / 48.4 | - |
| gf8d | mul_elementwise_assign | 40.8 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.2 / 40.0 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 10.74 / 4.272 | 10.31 / 3.859 |
| 32 B | 10.82 / 4.058 | 10.27 / 3.845 |
| 64 B | 12.32 / 4.287 | 11.76 / 4.071 |
| 128 B | 11.04 / 5.120 | 10.55 / 4.700 |
| 256 B | 14.01 / 6.187 | 13.17 / 5.764 |
| 512 B | 17.65 / 8.320 | 17.35 / 8.231 |
| 1 KiB | 20.66 / 14.03 | 21.08 / 13.92 |
| 2 KiB | 30.14 / 24.83 | 30.35 / 24.76 |
| 4 KiB | 61.86 / 47.00 | 61.11 / 46.98 |
| 16 KiB | 209.7 / 177.0 | 209.4 / 177.0 |
| 64 KiB | 1636 / 1092 | 1634 / 1092 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.5 / 69.2 | 43.9 / 76.4 | 45.0 / 67.1 | 45.0 / 62.0 |
| mul_add_gather_with | 44.4 / 69.3 | 43.8 / 76.5 | 44.9 / 66.8 | 45.0 / 62.4 |
| mul_add_matrix | 89.1 / 102 | 132 / 140 | 129 / 138 | 121 / 135 |
| mul_add_matrix_with | 89.5 / 102 | 132 / 139 | 128 / 139 | 120 / 135 |
| mul_add_scatter | 45.3 / 66.9 | 48.3 / 69.2 | 48.3 / 68.7 | 46.5 / 41.7 |
| mul_add_scatter_with | 45.3 / 67.0 | 48.3 / 69.3 | 48.3 / 68.7 | 46.5 / 41.7 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 90.9 / 83.0 | 104 / 85.5 |
| gf8d | mul_add_gather_with | 97.9 / 87.0 | 103 / 89.0 |
| gf8d | mul_into_gather | 91.0 / 102 | 105 / 107 |
| gf8d | mul_into_gather_with | 89.0 / 101 | 104 / 106 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.0 / 50.5 | 31.4 / 26.1 | 19.6 / 18.7 |
| gf8b | mul_add_matrix_at | 74.4 / 58.2 | 40.4 / 29.9 | 25.0 / 22.9 |
| gf8b | mul_add_matrix_with | 74.3 / 57.7 | 39.4 / 29.9 | 25.9 / 22.9 |
| gf8b | mul_into_matrix | 73.7 / 66.9 | 42.2 / 34.3 | 25.6 / 25.0 |
| gf8b | mul_into_matrix_with | 76.5 / 68.3 | 42.0 / 34.1 | 26.2 / 25.9 |
| gf8d | fill(0) + mul_add_matrix | 57.5 / 52.4 | 30.7 / 32.4 | 19.3 / 22.0 |
| gf8d | mul_into_matrix | 76.2 / 69.8 | 41.5 / 37.0 | 25.4 / 27.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 50.7 / 79.2 |
| 1 KiB | 16 | 84.3 / 128 |
| 1 KiB | 2 | 67.5 / 87.5 |
| 1 KiB | 32 | 34.8 / 63.8 |
| 1 KiB | 4 | 43.0 / 93.0 |
| 1 KiB | 8 | 81.0 / 127 |
| 64 B | 1 | 7.02 / 13.9 |
| 64 B | 16 | 50.8 / 79.3 |
| 64 B | 2 | 13.8 / 26.6 |
| 64 B | 32 | 67.3 / 87.6 |
| 64 B | 4 | 26.0 / 43.0 |
| 64 B | 8 | 25.5 / 69.6 |
| 64 KiB | 1 | 31.8 / 60.7 |
| 64 KiB | 16 | 18.9 / 25.2 |
| 64 KiB | 2 | 32.0 / 60.9 |
| 64 KiB | 32 | 18.5 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.1 |
| 64 KiB | 8 | 35.4 / 45.8 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 47.3 / 80.1 |
| add_assign per row | 1 KiB | 16 | 61.9 / 97.2 |
| add_assign per row | 1 KiB | 2 | 50.0 / 82.9 |
| add_assign per row | 1 KiB | 32 | 33.7 / 62.7 |
| add_assign per row | 1 KiB | 4 | 45.8 / 82.9 |
| add_assign per row | 1 KiB | 8 | 65.0 / 99.4 |
| add_assign per row | 64 B | 1 | 6.72 / 13.3 |
| add_assign per row | 64 B | 16 | 7.52 / 14.9 |
| add_assign per row | 64 B | 2 | 7.42 / 14.0 |
| add_assign per row | 64 B | 32 | 8.07 / 14.1 |
| add_assign per row | 64 B | 4 | 8.19 / 15.1 |
| add_assign per row | 64 B | 8 | 8.54 / 14.1 |
| add_assign per row | 64 KiB | 1 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.9 / 25.1 |
| add_assign per row | 64 KiB | 2 | 31.9 / 60.6 |
| add_assign per row | 64 KiB | 32 | 18.5 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 35.3 / 45.8 |
| add_assign | 1 KiB | 1 | 60.8 / 82.8 |
| add_assign | 1 KiB | 16 | 86.1 / 134 |
| add_assign | 1 KiB | 2 | 75.0 / 91.1 |
| add_assign | 1 KiB | 32 | 35.0 / 64.2 |
| add_assign | 1 KiB | 4 | 43.1 / 94.0 |
| add_assign | 1 KiB | 8 | 84.5 / 131 |
| add_assign | 64 B | 1 | 7.72 / 13.9 |
| add_assign | 64 B | 16 | 60.8 / 82.7 |
| add_assign | 64 B | 2 | 14.8 / 27.8 |
| add_assign | 64 B | 32 | 74.9 / 91.0 |
| add_assign | 64 B | 4 | 28.5 / 44.7 |
| add_assign | 64 B | 8 | 30.6 / 70.7 |
| add_assign | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign | 64 KiB | 16 | 18.9 / 25.2 |
| add_assign | 64 KiB | 2 | 31.9 / 61.0 |
| add_assign | 64 KiB | 32 | 18.5 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 35.4 / 45.7 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / 12.1 | 28.5 / 43.0 | 29.6 / 71.1 | 55.8 / 81.7 | 62.5 / 98.0 | 57.4 / 82.5 | 57.6 / 98.1 | 58.2 / 82.3 | 63.5 / 101 | 59.7 / 83.4 | 62.8 / 102 |
| mul_add | 4.90 / 13.9 | 19.3 / 39.9 | 28.9 / 58.7 | 52.8 / 70.7 | 54.8 / 78.4 | 54.5 / 71.0 | 63.0 / 78.2 | 54.3 / 70.7 | 56.4 / 80.3 | 54.8 / 71.9 | 49.6 / 67.9 |
| mul_add_scatter (16 rows) | 13.2 / 27.1 | 37.7 / 74.5 | 44.7 / 90.4 | 89.1 / 109 | 63.7 / 96.0 | 68.5 / 111 | 59.9 / 95.0 | 89.2 / 112 | 63.6 / 96.3 | 69.2 / 113 | 44.4 / 62.2 |
| mul_add_scatter (4 rows) | 9.01 / 17.9 | 31.0 / 43.7 | 37.3 / 72.9 | 79.8 / 96.6 | 63.0 / 86.6 | 68.0 / 98.4 | 55.4 / 87.8 | 83.8 / 98.7 | 64.6 / 89.3 | 69.1 / 99.8 | 41.4 / 59.6 |
| mul_assign | 8.09 / 19.9 | 25.5 / 74.4 | 43.2 / 74.5 | 87.8 / 137 | 95.6 / 136 | 86.7 / 140 | 90.1 / 137 | 89.1 / 137 | 95.3 / 136 | 86.9 / 135 | 73.5 / 121 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 81.2 / 98.6 | 110 / 111 | 35.4 / 41.4 |
| 1 KiB | 16 | 61.4 / 88.9 | 87.0 / 92.8 | 31.8 / 41.1 |
| 16 KiB | 0 | 47.8 / 69.5 | 93.5 / 90.1 | 42.5 / 46.3 |
| 16 KiB | 16 | 47.3 / 69.1 | 83.2 / 73.2 | 39.7 / 43.4 |
| 2 KiB | 0 | 92.9 / 110 | 123 / 116 | 39.5 / 43.8 |
| 2 KiB | 16 | 85.4 / 102 | 94.5 / 96.3 | 36.3 / 43.2 |
| 256 B | 0 | 33.0 / 47.3 | 70.0 / 82.3 | 26.1 / 29.0 |
| 256 B | 16 | 32.3 / 38.6 | 56.4 / 71.8 | 23.5 / 28.7 |
| 3 KiB | 0 | 97.4 / 117 | 98.4 / 92.7 | 40.9 / 44.8 |
| 3 KiB | 16 | 89.5 / 112 | 60.7 / 75.9 | 36.6 / 43.2 |
| 4 KiB | 0 | 90.9 / 110 | 90.8 / 88.7 | 40.9 / 43.6 |
| 4 KiB | 16 | 83.2 / 106 | 64.6 / 72.4 | 33.8 / 40.4 |
| 512 B | 0 | 54.7 / 79.2 | 95.9 / 101 | 29.4 / 36.2 |
| 512 B | 16 | 37.8 / 67.0 | 73.6 / 84.6 | 27.1 / 35.9 |
| 8 KiB | 0 | 92.0 / 115 | 92.1 / 89.4 | 41.7 / 45.6 |
| 8 KiB | 16 | 88.7 / 111 | 75.5 / 72.7 | 36.4 / 42.9 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 97.8 / 95.1 |
| 16 | 97.7 / 95.1 |
| 32 | 97.9 / 95.1 |
| 48 | 97.8 / 95.2 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / 30.1 |
| 256 KiB | 16 | 29.7 / 22.4 |
| 64 KiB | 0 | 48.4 / 69.4 |
| 64 KiB | 16 | 48.3 / 69.1 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.5 / 70.7 |
| mul_add_scatter_with | 48.6 / 70.9 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 49.5 / 53.0 |
| gf8b | 16 KiB | 137 / 179 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 128 / 166 |
| gf8b | 64 KiB | 51.3 / 73.3 |
| gf8d | 1 MiB | 47.5 / 50.0 |
| gf8d | 16 KiB | 137 / 180 |
| gf8d | 256 KiB | 51.5 / 66.4 |
| gf8d | 4 KiB | 128 / 166 |
| gf8d | 64 KiB | 51.3 / 65.6 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.3 / 101 |
| 64 MiB | 6.05 / 14.5 |
| 8 MiB | 8.39 / 24.1 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.8 / 25.7 | 13.7 / 20.3 | 18.8 / 26.3 |
| 32 MiB | 9.38 / 27.5 | 5.96 / 14.2 | 6.27 / 16.3 |
| 8 MiB | 18.2 / 40.0 | 7.68 / 16.2 | 8.30 / 24.8 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.0 / 62.0 | 17.8 / 43.7 | 33.7 / 54.8 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.1 | 41.2 / 62.5 | 29.1 / 46.9 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 88.7 / 98.7 | 83.1 / 99.6 | 30.6 / 48.2 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 92.5 / 93.8 | 75.0 / 91.0 | 35.7 / 54.6 |
| systematic encode | 10 × 2 | 4 KiB | 75.1 / 73.5 | 65.5 / 71.2 | 53.8 / 71.6 |
| systematic encode | 10 × 4 | 4 KiB | 39.7 / 45.7 | 35.8 / 43.0 | 32.0 / 42.7 |
| systematic encode | 10 × 6 | 4 KiB | 26.7 / 31.3 | 24.4 / 29.9 | 22.7 / 29.9 |
| systematic encode | 10 × 2 | 16 KiB | 81.8 / 80.5 | 52.9 / 66.6 | 67.5 / 81.5 |
| systematic encode | 10 × 4 | 16 KiB | 42.9 / 48.8 | 36.7 / 37.8 | 38.3 / 48.1 |
| systematic encode | 10 × 6 | 16 KiB | 27.1 / 33.1 | 25.1 / 29.2 | 25.7 / 32.6 |
| systematic encode | 10 × 2 | 64 KiB | 83.5 / 75.9 | 54.0 / 65.4 | 73.6 / 82.2 |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / 48.6 | 35.6 / 33.6 | 40.7 / 45.9 |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / 32.6 | 23.9 / 23.0 | 26.5 / 29.4 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
