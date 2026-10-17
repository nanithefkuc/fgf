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
| add_assign | 47.4 / 94.0 | 32.0 / 61.1 | 7.55 / 10.3 |
| add_assign_scalar | 42.4 / 77.1 | 41.5 / 64.8 | 28.1 / 24.1 |
| mul_add | 63.4 / 80.6 | 37.9 / 56.6 | 8.39 / 16.2 |
| mul_assign | 122 / 167 | 51.5 / 73.5 | 30.4 / 28.0 |
| mul_elementwise | 70.3 / 96.5 | 25.4 / 40.4 | 5.07 / 7.08 |
| mul_elementwise_assign | 78.1 / 114 | 31.3 / 47.7 | 6.63 / 9.55 |
| mul_into | 79.2 / 160 | 38.2 / 62.0 | 13.1 / 27.8 |
| sub_assign_scalar | 42.3 / 77.1 | 41.5 / 64.9 | 28.1 / 24.2 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.1 / 52.0 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.0 | - |
| gf8b | mul_into | 39.9 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.1 | - |
| gf8d | mul_elementwise | 28.8 / 47.6 | - |
| gf8d | mul_elementwise_assign | 40.7 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.1 / 40.0 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 14.36 / 4.094 | 14.37 / 3.855 |
| 32 B | 14.37 / 4.059 | 14.41 / 3.844 |
| 64 B | 15.94 / 4.279 | 16.01 / 4.060 |
| 128 B | 16.32 / 4.912 | 16.30 / 4.698 |
| 256 B | 16.56 / 5.979 | 16.56 / 5.768 |
| 512 B | 18.10 / 8.626 | 17.87 / 8.612 |
| 1 KiB | 21.89 / 14.29 | 21.86 / 14.27 |
| 2 KiB | 29.95 / 25.11 | 30.35 / 25.06 |
| 4 KiB | 61.00 / 47.40 | 60.57 / 47.41 |
| 16 KiB | 209.8 / 178.6 | 208.1 / 178.6 |
| 64 KiB | 1634 / 1085 | 1632 / 1097 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.4 / 69.2 | 43.0 / 75.7 | 43.7 / 66.6 | 44.1 / 62.1 |
| mul_add_gather_with | 44.4 / 67.8 | 43.0 / 75.6 | 43.7 / 65.6 | 44.1 / 62.2 |
| mul_add_matrix | 89.5 / 102 | 132 / 142 | 127 / 138 | 123 / 136 |
| mul_add_matrix_with | 88.9 / 102 | 132 / 143 | 124 / 138 | 120 / 136 |
| mul_add_scatter | 45.5 / 66.8 | 48.2 / 69.1 | 48.2 / 69.0 | 44.8 / 43.8 |
| mul_add_scatter_with | 45.5 / 67.0 | 48.2 / 69.3 | 48.3 / 69.1 | 45.0 / 43.9 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 90.8 / 80.0 | 104 / 84.5 |
| gf8d | mul_add_gather_with | 79.9 / 83.0 | 95.1 / 87.9 |
| gf8d | mul_into_gather | 91.1 / 101 | 104 / 106 |
| gf8d | mul_into_gather_with | 90.9 / 100 | 104 / 105 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.2 / 49.9 | 31.2 / 31.2 | 19.5 / 20.7 |
| gf8b | mul_add_matrix_at | 74.0 / 57.4 | 40.6 / 30.9 | 25.0 / 22.6 |
| gf8b | mul_add_matrix_with | 73.9 / 58.2 | 40.6 / 37.5 | 24.5 / 25.7 |
| gf8b | mul_into_matrix | 76.1 / 68.2 | 42.0 / 40.5 | 25.6 / 27.6 |
| gf8b | mul_into_matrix_with | 76.1 / 66.7 | 42.6 / 40.7 | 25.4 / 28.1 |
| gf8d | fill(0) + mul_add_matrix | 57.2 / 52.7 | 31.5 / 32.9 | 19.6 / 23.0 |
| gf8d | mul_into_matrix | 76.1 / 70.5 | 42.6 / 44.2 | 26.1 / 31.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 55.3 / 79.1 |
| 1 KiB | 16 | 83.5 / 126 |
| 1 KiB | 2 | 63.8 / 87.3 |
| 1 KiB | 32 | 35.0 / 63.9 |
| 1 KiB | 4 | 42.3 / 93.0 |
| 1 KiB | 8 | 79.6 / 127 |
| 64 B | 1 | 7.18 / 13.3 |
| 64 B | 16 | 56.0 / 79.1 |
| 64 B | 2 | 14.0 / 24.3 |
| 64 B | 32 | 63.7 / 87.7 |
| 64 B | 4 | 25.4 / 48.4 |
| 64 B | 8 | 26.1 / 69.6 |
| 64 KiB | 1 | 31.8 / 60.7 |
| 64 KiB | 16 | 18.8 / 25.7 |
| 64 KiB | 2 | 31.9 / 60.9 |
| 64 KiB | 32 | 18.3 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.1 |
| 64 KiB | 8 | 37.3 / 44.7 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 57.8 / 80.0 |
| add_assign per row | 1 KiB | 16 | 64.2 / 96.5 |
| add_assign per row | 1 KiB | 2 | 51.8 / 82.8 |
| add_assign per row | 1 KiB | 32 | 34.5 / 62.8 |
| add_assign per row | 1 KiB | 4 | 45.8 / 82.7 |
| add_assign per row | 1 KiB | 8 | 63.7 / 99.5 |
| add_assign per row | 64 B | 1 | 6.76 / 12.7 |
| add_assign per row | 64 B | 16 | 8.16 / 14.1 |
| add_assign per row | 64 B | 2 | 7.58 / 13.3 |
| add_assign per row | 64 B | 32 | 8.52 / 14.1 |
| add_assign per row | 64 B | 4 | 8.19 / 14.3 |
| add_assign per row | 64 B | 8 | 8.54 / 14.9 |
| add_assign per row | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.7 / 25.6 |
| add_assign per row | 64 KiB | 2 | 31.9 / 60.6 |
| add_assign per row | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 37.1 / 44.6 |
| add_assign | 1 KiB | 1 | 67.8 / 82.3 |
| add_assign | 1 KiB | 16 | 84.2 / 127 |
| add_assign | 1 KiB | 2 | 68.0 / 90.1 |
| add_assign | 1 KiB | 32 | 35.0 / 64.3 |
| add_assign | 1 KiB | 4 | 44.4 / 93.9 |
| add_assign | 1 KiB | 8 | 80.9 / 131 |
| add_assign | 64 B | 1 | 7.73 / 13.3 |
| add_assign | 64 B | 16 | 67.8 / 82.3 |
| add_assign | 64 B | 2 | 14.8 / 29.4 |
| add_assign | 64 B | 32 | 67.9 / 90.1 |
| add_assign | 64 B | 4 | 28.5 / 48.5 |
| add_assign | 64 B | 8 | 30.9 / 70.6 |
| add_assign | 64 KiB | 1 | 31.8 / 60.8 |
| add_assign | 64 KiB | 16 | 18.8 / 25.7 |
| add_assign | 64 KiB | 2 | 31.9 / 61.0 |
| add_assign | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 37.3 / 44.7 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.42 / 13.3 | 27.4 / 48.5 | 30.8 / 71.3 | 53.7 / 81.7 | 53.8 / 98.0 | 53.0 / 82.4 | 57.6 / 98.0 | 52.0 / 82.2 | 56.8 / 101 | 54.6 / 83.4 | 62.4 / 103 |
| mul_add | 3.88 / 14.4 | 15.5 / 41.1 | 26.3 / 58.8 | 47.3 / 69.6 | 49.9 / 78.5 | 47.8 / 69.5 | 50.1 / 79.4 | 48.9 / 69.6 | 52.0 / 80.3 | 49.2 / 69.8 | 40.9 / 67.4 |
| mul_add_scatter (16 rows) | 11.4 / 19.0 | 31.0 / 62.3 | 43.5 / 92.4 | 87.5 / 108 | 58.7 / 88.4 | 65.1 / 108 | 56.2 / 87.6 | 88.8 / 108 | 58.3 / 90.6 | 66.8 / 108 | 45.6 / 60.3 |
| mul_add_scatter (4 rows) | 8.57 / 13.8 | 27.3 / 38.8 | 39.2 / 76.3 | 76.7 / 97.8 | 59.2 / 81.9 | 65.9 / 97.3 | 52.0 / 82.4 | 84.3 / 97.0 | 60.2 / 83.7 | 66.8 / 97.1 | 43.1 / 58.1 |
| mul_assign | 7.12 / 15.5 | 25.4 / 58.7 | 42.8 / 85.9 | 82.3 / 135 | 90.3 / 133 | 80.4 / 138 | 85.6 / 135 | 84.6 / 139 | 94.3 / 137 | 84.4 / 137 | 71.2 / 116 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 79.8 / 95.5 | 55.1 / 94.8 | 35.1 / 41.0 |
| 1 KiB | 16 | 60.6 / 90.4 | 49.0 / 81.2 | 31.3 / 40.7 |
| 16 KiB | 0 | 47.7 / 70.2 | 91.5 / 88.9 | 40.0 / 46.4 |
| 16 KiB | 16 | 48.2 / 69.0 | 87.3 / 72.4 | 37.5 / 43.5 |
| 2 KiB | 0 | 94.4 / 107 | 78.2 / 103 | 39.4 / 43.7 |
| 2 KiB | 16 | 83.6 / 102 | 67.0 / 88.2 | 35.5 / 43.0 |
| 256 B | 0 | 31.3 / 45.5 | 19.6 / 53.9 | 26.1 / 28.3 |
| 256 B | 16 | 22.6 / 37.4 | 18.4 / 45.3 | 21.7 / 28.0 |
| 3 KiB | 0 | 104 / 116 | 75.6 / 86.4 | 40.8 / 44.8 |
| 3 KiB | 16 | 92.2 / 112 | 58.5 / 72.9 | 36.0 / 43.4 |
| 4 KiB | 0 | 90.4 / 109 | 73.8 / 84.9 | 40.7 / 43.9 |
| 4 KiB | 16 | 81.6 / 105 | 65.0 / 70.0 | 33.6 / 40.3 |
| 512 B | 0 | 52.1 / 73.3 | 34.3 / 75.6 | 29.3 / 35.7 |
| 512 B | 16 | 38.2 / 68.9 | 31.2 / 67.1 | 25.8 / 35.5 |
| 8 KiB | 0 | 92.1 / 114 | 83.7 / 87.4 | 41.6 / 44.6 |
| 8 KiB | 16 | 79.0 / 113 | 74.0 / 71.4 | 36.4 / 42.4 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 96.3 / 94.9 |
| 16 | 96.5 / 94.8 |
| 32 | 96.7 / 94.8 |
| 48 | 96.8 / 93.6 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.4 / 30.0 |
| 256 KiB | 16 | 29.6 / 22.3 |
| 64 KiB | 0 | 48.3 / 69.4 |
| 64 KiB | 16 | 48.2 / 69.1 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.3 / 70.5 |
| mul_add_scatter_with | 48.4 / 70.7 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 50.3 / 53.2 |
| gf8b | 16 KiB | 139 / 182 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 131 / 171 |
| gf8b | 64 KiB | 51.3 / 73.3 |
| gf8d | 1 MiB | 49.4 / 47.8 |
| gf8d | 16 KiB | 138 / 179 |
| gf8d | 256 KiB | 51.5 / 66.4 |
| gf8d | 4 KiB | 129 / 167 |
| gf8d | 64 KiB | 51.3 / 66.1 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.4 / 99.3 |
| 64 MiB | 6.06 / 14.4 |
| 8 MiB | 8.28 / 24.5 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.9 / 25.8 | 13.8 / 20.3 | 18.9 / 26.4 |
| 32 MiB | 9.36 / 27.3 | 5.96 / 14.0 | 6.28 / 16.3 |
| 8 MiB | 18.1 / 40.0 | 7.60 / 16.2 | 8.37 / 24.2 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 62.0 | 17.9 / 44.0 | 33.6 / 54.3 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.3 | 41.2 / 62.5 | 29.2 / 47.3 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 90.0 / 99.8 | 83.1 / 99.4 | 30.5 / 48.6 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 97.2 / 93.8 | 75.0 / 91.0 | 35.6 / 54.9 |
| systematic encode | 10 × 2 | 4 KiB | 74.5 / 74.0 | 65.5 / 72.4 | 53.2 / 71.8 |
| systematic encode | 10 × 4 | 4 KiB | 39.5 / 45.4 | 35.9 / 42.4 | 32.0 / 42.8 |
| systematic encode | 10 × 6 | 4 KiB | 26.8 / 31.1 | 24.5 / 29.9 | 22.4 / 30.0 |
| systematic encode | 10 × 2 | 16 KiB | 81.4 / 80.4 | 52.8 / 66.2 | 67.8 / 81.3 |
| systematic encode | 10 × 4 | 16 KiB | 42.8 / 48.8 | 36.6 / 38.0 | 38.5 / 48.1 |
| systematic encode | 10 × 6 | 16 KiB | 27.1 / 33.1 | 24.9 / 29.2 | 25.7 / 32.7 |
| systematic encode | 10 × 2 | 64 KiB | 83.8 / 76.1 | 54.0 / 65.1 | 73.6 / 81.5 |
| systematic encode | 10 × 4 | 64 KiB | 43.4 / 48.3 | 35.5 / 33.3 | 40.6 / 45.0 |
| systematic encode | 10 × 6 | 64 KiB | 27.1 / 32.6 | 23.7 / 22.9 | 26.6 / 29.5 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
