# Byte fields: `gf8b`, `gf8d` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v2](../v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| add_assign | 51.5 / 93.7 | 32.0 / 61.1 | 7.68 / 10.3 |
| add_assign_scalar | 54.8 / 110 | 51.9 / 73.5 | 29.1 / 27.5 |
| mul_add | 64.7 / 81.7 | 37.9 / 56.1 | 8.55 / 16.7 |
| mul_assign | 125 / 172 | 51.5 / 73.5 | 30.7 / 27.9 |
| mul_elementwise | 55.6 / 64.8 | 24.3 / 37.6 | 5.15 / 7.13 |
| mul_elementwise_assign | 59.9 / 66.3 | 28.1 / 60.7 | 7.21 / 9.71 |
| mul_into | 82.7 / 159 | 38.1 / 62.2 | 5.24 / 27.1 |
| sub_assign_scalar | 59.9 / 109 | 51.9 / 73.5 | 29.1 / 27.7 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 62.0 | - |
| gf8b | mul_elementwise | 29.1 / 51.9 | - |
| gf8b | mul_elementwise_assign | 40.7 / 63.1 | - |
| gf8b | mul_into | 40.0 / 62.0 | - |
| gf8d | mul_add | 40.9 / 62.0 | - |
| gf8d | mul_elementwise | 28.6 / 48.4 | - |
| gf8d | mul_elementwise_assign | 40.8 / 54.4 | - |
| gf8d | mul_into | 40.0 / 62.0 | 18.2 / 40.1 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 10.72 / 4.271 | 10.24 / 3.859 |
| 32 B | 10.68 / 4.270 | 10.14 / 3.859 |
| 64 B | 12.69 / 4.272 | 12.25 / 4.076 |
| 128 B | 13.31 / 10.24 | 13.03 / 10.24 |
| 256 B | 14.06 / 6.243 | 13.48 / 5.790 |
| 512 B | 15.84 / 8.450 | 15.83 / 8.112 |
| 1 KiB | 20.69 / 14.15 | 21.14 / 14.11 |
| 2 KiB | 29.73 / 24.95 | 30.55 / 24.85 |
| 4 KiB | 61.06 / 47.07 | 61.16 / 47.05 |
| 16 KiB | 207.8 / 176.3 | 207.9 / 175.7 |
| 64 KiB | 1633 / 1093 | 1633 / 1092 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.7 / 64.8 | 43.7 / 71.8 | 45.3 / 63.9 | 44.9 / 63.0 |
| mul_add_gather_with | 44.6 / 64.4 | 43.5 / 71.9 | 45.1 / 63.8 | 44.6 / 62.6 |
| mul_add_matrix | 90.1 / 106 | 135 / 145 | 130 / 146 | 122 / 146 |
| mul_add_matrix_with | 90.6 / 106 | 135 / 147 | 130 / 147 | 123 / 145 |
| mul_add_scatter | 45.4 / 66.8 | 48.4 / 69.3 | 48.5 / 69.3 | 45.8 / 46.6 |
| mul_add_scatter_with | 45.5 / 67.0 | 48.4 / 69.4 | 48.5 / 69.4 | 45.8 / 46.2 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.2 / 83.2 | 104 / 85.5 |
| gf8d | mul_add_gather_with | 98.0 / 87.1 | 102 / 89.1 |
| gf8d | mul_into_gather | 90.9 / 101 | 104 / 107 |
| gf8d | mul_into_gather_with | 89.1 / 101 | 103 / 107 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.5 / 50.1 | 31.2 / 27.8 | 19.4 / 19.6 |
| gf8b | mul_add_matrix_at | 74.5 / 58.5 | 39.3 / 31.1 | 24.3 / 22.4 |
| gf8b | mul_add_matrix_with | 74.6 / 58.4 | 40.0 / 31.0 | 25.1 / 21.9 |
| gf8b | mul_into_matrix | 76.3 / 65.8 | 42.9 / 36.3 | 26.1 / 25.4 |
| gf8b | mul_into_matrix_with | 76.6 / 67.5 | 42.6 / 36.0 | 26.0 / 25.4 |
| gf8d | fill(0) + mul_add_matrix | 56.9 / 52.5 | 31.5 / 34.3 | 19.7 / 22.4 |
| gf8d | mul_into_matrix | 76.4 / 68.5 | 43.1 / 39.3 | 26.1 / 28.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 46.1 / 79.4 |
| 1 KiB | 16 | 83.8 / 128 |
| 1 KiB | 2 | 67.4 / 88.5 |
| 1 KiB | 32 | 34.5 / 63.8 |
| 1 KiB | 4 | 42.8 / 93.2 |
| 1 KiB | 8 | 80.5 / 124 |
| 64 B | 1 | 7.13 / 14.7 |
| 64 B | 16 | 46.2 / 79.3 |
| 64 B | 2 | 7.32 / 12.3 |
| 64 B | 32 | 67.8 / 88.5 |
| 64 B | 4 | 26.0 / 46.6 |
| 64 B | 8 | 26.3 / 69.8 |
| 64 KiB | 1 | 31.8 / 60.7 |
| 64 KiB | 16 | 18.9 / 25.8 |
| 64 KiB | 2 | 31.9 / 60.9 |
| 64 KiB | 32 | 18.3 / 24.7 |
| 64 KiB | 4 | 32.0 / 61.1 |
| 64 KiB | 8 | 37.1 / 45.5 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 44.3 / 80.3 |
| add_assign per row | 1 KiB | 16 | 53.0 / 97.3 |
| add_assign per row | 1 KiB | 2 | 51.6 / 83.8 |
| add_assign per row | 1 KiB | 32 | 31.9 / 61.2 |
| add_assign per row | 1 KiB | 4 | 45.2 / 83.2 |
| add_assign per row | 1 KiB | 8 | 55.5 / 99.4 |
| add_assign per row | 64 B | 1 | 6.74 / 13.3 |
| add_assign per row | 64 B | 16 | 8.32 / 15.7 |
| add_assign per row | 64 B | 2 | 6.85 / 11.4 |
| add_assign per row | 64 B | 32 | 8.89 / 16.6 |
| add_assign per row | 64 B | 4 | 8.69 / 15.1 |
| add_assign per row | 64 B | 8 | 9.03 / 16.7 |
| add_assign per row | 64 KiB | 1 | 31.7 / 60.8 |
| add_assign per row | 64 KiB | 16 | 18.9 / 25.7 |
| add_assign per row | 64 KiB | 2 | 31.9 / 60.7 |
| add_assign per row | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign per row | 64 KiB | 4 | 31.9 / 60.8 |
| add_assign per row | 64 KiB | 8 | 37.0 / 45.4 |
| add_assign | 1 KiB | 1 | 52.7 / 82.4 |
| add_assign | 1 KiB | 16 | 84.9 / 133 |
| add_assign | 1 KiB | 2 | 75.1 / 91.0 |
| add_assign | 1 KiB | 32 | 35.1 / 64.1 |
| add_assign | 1 KiB | 4 | 43.2 / 94.2 |
| add_assign | 1 KiB | 8 | 82.2 / 130 |
| add_assign | 64 B | 1 | 8.44 / 14.7 |
| add_assign | 64 B | 16 | 52.7 / 82.6 |
| add_assign | 64 B | 2 | 7.13 / 12.4 |
| add_assign | 64 B | 32 | 74.8 / 91.6 |
| add_assign | 64 B | 4 | 28.5 / 44.7 |
| add_assign | 64 B | 8 | 31.0 / 72.0 |
| add_assign | 64 KiB | 1 | 31.8 / 60.7 |
| add_assign | 64 KiB | 16 | 18.9 / 25.7 |
| add_assign | 64 KiB | 2 | 31.9 / 61.0 |
| add_assign | 64 KiB | 32 | 18.3 / 24.7 |
| add_assign | 64 KiB | 4 | 32.0 / 61.1 |
| add_assign | 64 KiB | 8 | 37.1 / 45.5 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 8.07 / 15.5 | 28.5 / 44.7 | 31.0 / 71.2 | 56.3 / 82.5 | 65.1 / 97.9 | 62.2 / 83.3 | 68.2 / 98.7 | 58.4 / 83.1 | 66.4 / 98.3 | 64.4 / 83.9 | 62.9 / 100 |
| mul_add | 5.91 / 13.8 | 18.6 / 39.9 | 35.0 / 57.3 | 54.3 / 73.0 | 55.1 / 78.4 | 56.7 / 72.7 | 64.1 / 79.4 | 55.4 / 71.7 | 55.0 / 80.3 | 57.0 / 72.8 | 48.9 / 68.1 |
| mul_add_scatter (16 rows) | 13.2 / 26.8 | 37.6 / 55.5 | 42.9 / 83.1 | 71.4 / 104 | 61.7 / 95.8 | 68.3 / 106 | 60.6 / 95.8 | 74.7 / 106 | 64.1 / 96.4 | 68.8 / 107 | 43.2 / 64.8 |
| mul_add_scatter (4 rows) | 9.03 / 17.7 | 31.1 / 55.2 | 36.6 / 66.7 | 64.3 / 92.3 | 57.5 / 86.5 | 63.9 / 94.2 | 54.7 / 87.5 | 66.1 / 95.1 | 59.1 / 88.5 | 64.2 / 96.5 | 39.7 / 57.4 |
| mul_assign | 9.37 / 21.5 | 26.4 / 69.9 | 49.1 / 112 | 86.7 / 139 | 85.6 / 138 | 94.1 / 139 | 90.3 / 137 | 86.8 / 139 | 92.7 / 137 | 96.5 / 140 | 69.6 / 122 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 80.4 / 98.5 | 110 / 111 | 35.5 / 41.4 |
| 1 KiB | 16 | 61.4 / 89.2 | 86.5 / 92.8 | 32.0 / 41.0 |
| 16 KiB | 0 | 47.8 / 69.5 | 95.2 / 89.9 | 42.6 / 44.8 |
| 16 KiB | 16 | 47.3 / 69.2 | 85.4 / 73.0 | 39.9 / 42.5 |
| 2 KiB | 0 | 92.7 / 110 | 122 / 116 | 39.5 / 43.8 |
| 2 KiB | 16 | 88.5 / 102 | 94.8 / 96.9 | 35.9 / 43.1 |
| 256 B | 0 | 32.9 / 47.3 | 70.2 / 82.3 | 26.1 / 29.0 |
| 256 B | 16 | 26.2 / 32.5 | 56.5 / 71.8 | 21.9 / 28.7 |
| 3 KiB | 0 | 97.5 / 117 | 98.6 / 91.5 | 40.8 / 44.5 |
| 3 KiB | 16 | 90.6 / 113 | 59.8 / 75.7 | 36.7 / 42.6 |
| 4 KiB | 0 | 91.1 / 110 | 89.5 / 88.6 | 40.9 / 43.7 |
| 4 KiB | 16 | 84.0 / 106 | 63.3 / 72.1 | 33.7 / 40.1 |
| 512 B | 0 | 54.3 / 79.4 | 96.4 / 101 | 29.6 / 36.2 |
| 512 B | 16 | 38.0 / 66.9 | 68.9 / 84.6 | 26.2 / 35.7 |
| 8 KiB | 0 | 93.3 / 115 | 91.2 / 89.3 | 41.7 / 45.7 |
| 8 KiB | 16 | 89.3 / 111 | 73.3 / 72.4 | 36.7 / 42.7 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 97.9 / 95.1 |
| 16 | 97.8 / 95.2 |
| 32 | 98.1 / 95.1 |
| 48 | 98.0 / 95.1 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.3 / 29.9 |
| 256 KiB | 16 | 29.6 / 22.6 |
| 64 KiB | 0 | 48.5 / 69.3 |
| 64 KiB | 16 | 48.3 / 69.4 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.6 / 70.6 |
| mul_add_scatter_with | 48.6 / 70.8 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 50.0 / 52.9 |
| gf8b | 16 KiB | 137 / 179 |
| gf8b | 256 KiB | 51.5 / 73.5 |
| gf8b | 4 KiB | 128 / 166 |
| gf8b | 64 KiB | 51.3 / 73.3 |
| gf8d | 1 MiB | 49.0 / 50.1 |
| gf8d | 16 KiB | 137 / 180 |
| gf8d | 256 KiB | 51.5 / 66.2 |
| gf8d | 4 KiB | 127 / 166 |
| gf8d | 64 KiB | 51.4 / 66.1 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 63.4 |
| 4 KiB | 43.3 / 99.4 |
| 64 MiB | 6.05 / 14.4 |
| 8 MiB | 8.39 / 24.1 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.9 / 25.4 | 13.8 / 19.9 | 18.9 / 24.3 |
| 32 MiB | 9.32 / 27.2 | 5.95 / 14.1 | 6.26 / 16.3 |
| 8 MiB | 18.2 / 40.0 | 7.60 / 16.2 | 8.40 / 24.2 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 62.0 | 17.7 / 43.7 | 33.7 / 54.5 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 62.1 | 41.2 / 62.5 | 29.1 / 46.9 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 89.4 / 98.6 | 83.1 / 99.6 | 30.6 / 48.2 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 95.5 / 93.7 | 74.9 / 91.1 | 35.7 / 54.8 |
| systematic encode | 10 × 2 | 4 KiB | 75.8 / 73.4 | 65.7 / 71.3 | 53.7 / 71.9 |
| systematic encode | 10 × 4 | 4 KiB | 39.6 / 45.6 | 35.8 / 43.1 | 31.9 / 42.7 |
| systematic encode | 10 × 6 | 4 KiB | 26.8 / 31.4 | 24.5 / 30.1 | 22.5 / 29.9 |
| systematic encode | 10 × 2 | 16 KiB | 81.8 / 81.3 | 53.1 / 66.4 | 67.2 / 81.6 |
| systematic encode | 10 × 4 | 16 KiB | 42.9 / 48.8 | 36.7 / 37.9 | 38.5 / 48.1 |
| systematic encode | 10 × 6 | 16 KiB | 27.1 / 33.2 | 25.0 / 29.0 | 25.7 / 32.6 |
| systematic encode | 10 × 2 | 64 KiB | 83.7 / 75.9 | 54.1 / 65.3 | 73.6 / 82.3 |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / 48.5 | 35.8 / 32.1 | 40.7 / 45.5 |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / 32.6 | 23.8 / 23.0 | 26.6 / 29.5 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
