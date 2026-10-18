# Byte fields: `gf8b`, `gf8d` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v2](../v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `388cbb4a43d5d20e8a663e89c3b44380c7c4f8b7` |
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
| gf8d | scalar mul | 0.605 / 0.356 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 52.1 / 93.6 | 32.0 / 61.1 | 7.60 / 10.4 |
| add_assign_scalar | 42.8 / 77.1 | 41.5 / 64.8 | 28.5 / 24.6 |
| mul_add | 64.4 / 81.3 | 37.9 / 56.1 | 8.36 / 15.9 |
| mul_assign | 127 / 167 | 51.5 / 73.5 | 30.6 / 27.8 |
| mul_elementwise | 73.6 / 98.7 | 26.3 / 38.5 | 5.23 / 7.21 |
| mul_elementwise_assign | 75.8 / 102 | 31.2 / 47.8 | 6.89 / 9.64 |
| mul_into | 84.6 / 163 | 38.3 / 62.1 | 13.1 / 27.8 |
| sub_assign_scalar | 42.7 / 77.1 | 41.5 / 64.8 | 28.5 / 24.6 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.2 / 52.0 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.1 | - |
| gf8b | mul_into | 40.0 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.1 | - |
| gf8d | mul_elementwise | 28.8 / 48.5 | - |
| gf8d | mul_elementwise_assign | 40.8 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.1 / 40.0 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 10.83 / 4.271 | 10.36 / 3.860 |
| 32 B | 10.77 / 4.270 | 10.20 / 3.844 |
| 64 B | 12.66 / 4.269 | 12.31 / 4.069 |
| 128 B | 13.60 / 5.123 | 13.07 / 4.705 |
| 256 B | 14.06 / 6.244 | 13.76 / 5.765 |
| 512 B | 16.00 / 8.242 | 15.92 / 8.197 |
| 1 KiB | 20.95 / 14.02 | 21.23 / 13.94 |
| 2 KiB | 30.96 / 24.84 | 31.48 / 24.77 |
| 4 KiB | 63.68 / 47.00 | 63.58 / 46.98 |
| 16 KiB | 211.2 / 177.0 | 211.7 / 177.0 |
| 64 KiB | 1634 / 1092 | 1633 / 1091 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.3 / 69.3 | 43.4 / 76.6 | 44.9 / 67.2 | 44.7 / 62.0 |
| mul_add_gather_with | 44.4 / 69.3 | 43.4 / 76.6 | 44.9 / 66.6 | 44.6 / 62.2 |
| mul_add_matrix | 89.2 / 102 | 133 / 139 | 129 / 139 | 121 / 136 |
| mul_add_matrix_with | 89.5 / 102 | 133 / 140 | 128 / 137 | 121 / 135 |
| mul_add_scatter | 45.2 / 66.9 | 48.2 / 69.2 | 48.2 / 69.2 | 46.0 / 44.3 |
| mul_add_scatter_with | 45.2 / 67.0 | 48.2 / 69.3 | 48.2 / 69.2 | 45.9 / 44.5 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.2 / 83.3 | 104 / 85.6 |
| gf8d | mul_add_gather_with | 97.9 / 87.0 | 100 / 89.2 |
| gf8d | mul_into_gather | 91.1 / 101 | 104 / 107 |
| gf8d | mul_into_gather_with | 89.2 / 101 | 104 / 106 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.4 / 50.7 | 31.1 / 26.2 | 19.4 / 18.8 |
| gf8b | mul_add_matrix_at | 74.4 / 58.3 | 39.2 / 30.0 | 24.3 / 23.0 |
| gf8b | mul_add_matrix_with | 74.4 / 58.0 | 39.7 / 30.0 | 25.4 / 22.8 |
| gf8b | mul_into_matrix | 76.3 / 67.2 | 42.7 / 34.3 | 26.1 / 25.1 |
| gf8b | mul_into_matrix_with | 76.4 / 68.0 | 42.2 / 34.2 | 26.2 / 25.9 |
| gf8d | fill(0) + mul_add_matrix | 57.2 / 52.5 | 31.4 / 32.6 | 19.6 / 22.2 |
| gf8d | mul_into_matrix | 76.2 / 70.1 | 42.6 / 37.1 | 26.2 / 27.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 52.2 / 79.4 |
| 1 KiB | 16 | 84.2 / 127 |
| 1 KiB | 2 | 67.0 / 87.3 |
| 1 KiB | 32 | 34.9 / 63.6 |
| 1 KiB | 4 | 43.0 / 92.6 |
| 1 KiB | 8 | 80.7 / 124 |
| 64 B | 1 | 6.79 / 14.7 |
| 64 B | 16 | 52.3 / 79.5 |
| 64 B | 2 | 13.1 / 29.1 |
| 64 B | 32 | 66.7 / 87.3 |
| 64 B | 4 | 26.0 / 44.7 |
| 64 B | 8 | 25.5 / 69.2 |
| 64 KiB | 1 | 31.7 / 60.6 |
| 64 KiB | 16 | 18.9 / 25.5 |
| 64 KiB | 2 | 31.9 / 60.9 |
| 64 KiB | 32 | 18.4 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.0 |
| 64 KiB | 8 | 36.6 / 41.6 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 56.3 / 80.0 |
| add_assign per row | 1 KiB | 16 | 63.7 / 97.1 |
| add_assign per row | 1 KiB | 2 | 52.1 / 82.7 |
| add_assign per row | 1 KiB | 32 | 34.0 / 62.5 |
| add_assign per row | 1 KiB | 4 | 45.7 / 82.6 |
| add_assign per row | 1 KiB | 8 | 63.2 / 99.4 |
| add_assign per row | 64 B | 1 | 6.78 / 13.9 |
| add_assign per row | 64 B | 16 | 8.74 / 15.7 |
| add_assign per row | 64 B | 2 | 8.26 / 15.5 |
| add_assign per row | 64 B | 32 | 9.38 / 17.5 |
| add_assign per row | 64 B | 4 | 9.02 / 16.9 |
| add_assign per row | 64 B | 8 | 9.44 / 17.7 |
| add_assign per row | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.9 / 25.4 |
| add_assign per row | 64 KiB | 2 | 31.9 / 60.7 |
| add_assign per row | 64 KiB | 32 | 18.4 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 36.5 / 41.6 |
| add_assign | 1 KiB | 1 | 68.8 / 82.6 |
| add_assign | 1 KiB | 16 | 85.9 / 134 |
| add_assign | 1 KiB | 2 | 76.0 / 90.5 |
| add_assign | 1 KiB | 32 | 35.1 / 63.8 |
| add_assign | 1 KiB | 4 | 43.1 / 93.6 |
| add_assign | 1 KiB | 8 | 84.3 / 131 |
| add_assign | 64 B | 1 | 8.08 / 15.5 |
| add_assign | 64 B | 16 | 68.8 / 82.6 |
| add_assign | 64 B | 2 | 14.8 / 31.0 |
| add_assign | 64 B | 32 | 75.9 / 90.5 |
| add_assign | 64 B | 4 | 29.6 / 48.6 |
| add_assign | 64 B | 8 | 30.9 / 70.6 |
| add_assign | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign | 64 KiB | 16 | 18.9 / 25.6 |
| add_assign | 64 KiB | 2 | 31.9 / 61.0 |
| add_assign | 64 KiB | 32 | 18.4 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 36.6 / 41.6 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 8.07 / 15.5 | 29.4 / 44.7 | 30.5 / 71.2 | 53.7 / 81.4 | 58.7 / 97.9 | 54.4 / 82.1 | 57.1 / 97.5 | 55.4 / 82.0 | 60.2 / 101 | 56.8 / 83.1 | 62.8 / 103 |
| mul_add | 5.05 / 13.9 | 19.4 / 39.9 | 33.8 / 58.9 | 49.3 / 70.8 | 52.3 / 78.4 | 51.0 / 71.1 | 57.9 / 79.4 | 51.4 / 70.7 | 53.4 / 79.1 | 52.5 / 71.8 | 49.5 / 68.2 |
| mul_add_scatter (16 rows) | 13.2 / 27.0 | 37.5 / 74.0 | 44.8 / 90.3 | 89.1 / 108 | 63.2 / 95.9 | 68.0 / 111 | 60.2 / 94.8 | 89.1 / 112 | 63.6 / 96.4 | 68.8 / 112 | 43.0 / 62.2 |
| mul_add_scatter (4 rows) | 9.05 / 17.8 | 30.0 / 46.1 | 37.1 / 72.7 | 81.3 / 96.9 | 63.5 / 86.8 | 68.3 / 98.2 | 55.8 / 87.8 | 82.8 / 98.8 | 63.4 / 89.2 | 67.6 / 99.9 | 40.9 / 59.7 |
| mul_assign | 8.09 / 19.9 | 27.4 / 74.5 | 44.6 / 74.5 | 86.7 / 137 | 93.7 / 136 | 86.4 / 139 | 87.6 / 136 | 87.8 / 137 | 95.0 / 136 | 87.1 / 133 | 74.7 / 121 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 81.1 / 98.3 | 110 / 111 | 35.5 / 41.4 |
| 1 KiB | 16 | 61.5 / 89.7 | 86.1 / 92.8 | 32.8 / 41.1 |
| 16 KiB | 0 | 47.8 / 69.4 | 93.1 / 89.8 | 40.1 / 46.3 |
| 16 KiB | 16 | 47.8 / 69.0 | 84.0 / 73.2 | 37.6 / 43.4 |
| 2 KiB | 0 | 94.4 / 110 | 122 / 116 | 39.6 / 43.8 |
| 2 KiB | 16 | 88.2 / 103 | 94.3 / 96.4 | 35.9 / 42.9 |
| 256 B | 0 | 33.1 / 47.7 | 70.8 / 82.3 | 26.2 / 29.1 |
| 256 B | 16 | 26.1 / 39.3 | 56.3 / 71.8 | 21.4 / 28.7 |
| 3 KiB | 0 | 97.4 / 117 | 97.3 / 92.6 | 40.8 / 44.8 |
| 3 KiB | 16 | 92.3 / 113 | 60.2 / 75.6 | 36.4 / 43.0 |
| 4 KiB | 0 | 91.1 / 110 | 90.9 / 88.7 | 40.8 / 43.5 |
| 4 KiB | 16 | 85.1 / 107 | 63.2 / 72.3 | 33.7 / 40.2 |
| 512 B | 0 | 54.8 / 79.4 | 96.3 / 101 | 29.5 / 36.2 |
| 512 B | 16 | 38.2 / 68.2 | 69.8 / 84.6 | 27.9 / 36.0 |
| 8 KiB | 0 | 93.4 / 115 | 90.2 / 89.4 | 41.7 / 45.6 |
| 8 KiB | 16 | 89.6 / 112 | 73.9 / 72.7 | 36.7 / 42.9 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 97.6 / 95.2 |
| 16 | 97.4 / 95.1 |
| 32 | 97.7 / 95.1 |
| 48 | 97.6 / 95.1 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / 29.8 |
| 256 KiB | 16 | 29.7 / 22.4 |
| 64 KiB | 0 | 48.3 / 69.3 |
| 64 KiB | 16 | 48.3 / 68.9 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.5 / 70.5 |
| mul_add_scatter_with | 48.5 / 70.8 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 47.4 / 52.9 |
| gf8b | 16 KiB | 137 / 179 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 126 / 166 |
| gf8b | 64 KiB | 51.3 / 73.3 |
| gf8d | 1 MiB | 49.1 / 49.8 |
| gf8d | 16 KiB | 137 / 180 |
| gf8d | 256 KiB | 51.5 / 68.0 |
| gf8d | 4 KiB | 125 / 166 |
| gf8d | 64 KiB | 51.4 / 65.6 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.3 / 99.9 |
| 64 MiB | 6.08 / 14.4 |
| 8 MiB | 8.43 / 24.4 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.9 / 25.8 | 13.6 / 20.5 | 18.9 / 26.4 |
| 32 MiB | 9.28 / 27.7 | 5.92 / 14.2 | 6.25 / 16.3 |
| 8 MiB | 18.2 / 40.0 | 7.59 / 16.3 | 8.08 / 24.7 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 62.0 | 18.0 / 43.7 | 33.6 / 54.5 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.1 | 41.2 / 62.5 | 29.1 / 47.0 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 89.6 / 99.0 | 83.0 / 99.5 | 30.5 / 48.9 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 93.9 / 93.8 | 74.9 / 91.0 | 35.6 / 54.7 |
| systematic encode | 10 × 2 | 4 KiB | 74.7 / 73.7 | 65.4 / 71.3 | 53.3 / 72.2 |
| systematic encode | 10 × 4 | 4 KiB | 39.7 / 45.8 | 35.9 / 42.7 | 31.8 / 42.8 |
| systematic encode | 10 × 6 | 4 KiB | 26.8 / 31.5 | 24.5 / 30.2 | 22.5 / 30.0 |
| systematic encode | 10 × 2 | 16 KiB | 81.9 / 80.5 | 52.9 / 66.4 | 67.3 / 81.4 |
| systematic encode | 10 × 4 | 16 KiB | 42.9 / 48.8 | 36.5 / 37.8 | 38.7 / 48.1 |
| systematic encode | 10 × 6 | 16 KiB | 27.2 / 33.1 | 25.0 / 29.2 | 25.7 / 32.7 |
| systematic encode | 10 × 2 | 64 KiB | 83.7 / 76.1 | 53.7 / 65.1 | 73.5 / 81.9 |
| systematic encode | 10 × 4 | 64 KiB | 43.4 / 48.4 | 35.2 / 33.4 | 40.6 / 45.9 |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / 32.7 | 23.8 / 23.0 | 26.6 / 29.4 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
