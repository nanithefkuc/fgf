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
| gf8b | scalar mul | 0.701 / 0.343 |
| gf8d | scalar mul | 0.701 / 0.343 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 47.5 / 98.2 | 32.0 / 63.6 | 7.77 / 10.7 |
| add_assign_scalar | 42.1 / 80.3 | 41.6 / 67.6 | 29.0 / 25.3 |
| mul_add | 64.5 / 84.3 | 37.9 / 58.2 | 8.60 / 18.5 |
| mul_assign | 122 / 174 | 51.5 / 76.6 | 30.9 / 29.2 |
| mul_elementwise | 56.5 / 101 | 25.7 / 38.9 | 5.32 / 7.45 |
| mul_elementwise_assign | 62.1 / 119 | 28.0 / 49.7 | 7.15 / 9.99 |
| mul_into | 78.7 / 166 | 38.2 / 64.8 | 5.30 / 28.6 |
| sub_assign_scalar | 41.1 / 80.4 | 41.6 / 67.6 | 29.0 / 25.3 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 64.6 | - |
| gf8b | mul_elementwise | 29.0 / 54.2 | - |
| gf8b | mul_elementwise_assign | 40.7 / 65.7 | - |
| gf8b | mul_into | 39.9 / 64.6 | - |
| gf8d | mul_add | 40.9 / 64.7 | - |
| gf8d | mul_elementwise | 28.7 / 49.6 | - |
| gf8d | mul_elementwise_assign | 40.7 / 56.7 | - |
| gf8d | mul_into | 39.9 / 64.7 | 18.2 / 41.8 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 1 KiB | 21.94 / 13.72 | 21.89 / 13.72 |
| 128 B | 14.62 / 4.710 | 14.62 / 4.507 |
| 16 B | 14.41 / 3.927 | 14.39 / 3.704 |
| 16 KiB | 208.4 / 171.3 | 207.5 / 171.3 |
| 2 KiB | 29.93 / 24.14 | 30.27 / 24.10 |
| 256 B | 16.32 / 5.733 | 16.23 / 5.543 |
| 32 B | 14.38 / 3.891 | 14.42 / 3.687 |
| 4 KiB | 60.49 / 45.50 | 60.70 / 45.48 |
| 512 B | 18.30 / 8.380 | 18.18 / 8.326 |
| 64 B | 15.54 / 4.110 | 15.70 / 3.892 |
| 64 KiB | 1634 / 1053 | 1630 / 1052 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.6 / 72.5 | 43.1 / 78.7 | 44.2 / 68.9 | 43.8 / 64.7 |
| mul_add_gather_with | 44.6 / 71.4 | 43.1 / 78.2 | 44.2 / 69.1 | 43.8 / 64.3 |
| mul_add_matrix | 90.1 / 106 | 133 / 147 | 129 / 147 | 122 / 144 |
| mul_add_matrix_with | 90.0 / 107 | 133 / 147 | 125 / 146 | 119 / 143 |
| mul_add_scatter | 45.5 / 69.7 | 48.2 / 72.1 | 48.2 / 72.2 | 46.4 / 51.3 |
| mul_add_scatter_with | 45.5 / 69.8 | 48.2 / 72.3 | 48.2 / 72.2 | 46.4 / 51.2 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 90.8 / 83.7 | 103 / 88.3 |
| gf8d | mul_add_gather_with | 79.9 / 86.9 | 95.0 / 92.0 |
| gf8d | mul_into_gather | 90.7 / 106 | 103 / 111 |
| gf8d | mul_into_gather_with | 91.1 / 105 | 103 / 110 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.4 / 51.3 | 31.5 / 33.8 | 19.6 / 22.5 |
| gf8b | mul_add_matrix_at | 74.3 / 60.1 | 39.2 / 31.6 | 24.3 / 22.9 |
| gf8b | mul_add_matrix_with | 74.1 / 60.7 | 38.6 / 38.4 | 23.6 / 25.9 |
| gf8b | mul_into_matrix | 76.3 / 69.6 | 42.6 / 43.6 | 24.8 / 28.9 |
| gf8b | mul_into_matrix_with | 76.3 / 68.1 | 39.8 / 42.0 | 24.2 / 28.6 |
| gf8d | fill(0) + mul_add_matrix | 57.3 / 54.6 | 30.5 / 33.9 | 18.9 / 22.9 |
| gf8d | mul_into_matrix | 76.3 / 72.2 | 39.7 / 45.8 | 24.4 / 31.8 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 50.0 / 82.1 |
| 1 KiB | 16 | 83.0 / 132 |
| 1 KiB | 2 | 66.8 / 91.9 |
| 1 KiB | 32 | 34.8 / 66.6 |
| 1 KiB | 4 | 42.1 / 97.1 |
| 1 KiB | 8 | 78.5 / 130 |
| 64 B | 1 | 7.17 / 13.8 |
| 64 B | 16 | 50.0 / 82.0 |
| 64 B | 2 | 13.7 / 30.6 |
| 64 B | 32 | 66.6 / 92.0 |
| 64 B | 4 | 23.0 / 48.5 |
| 64 B | 8 | 26.8 / 71.9 |
| 64 KiB | 1 | 31.8 / 63.3 |
| 64 KiB | 16 | 18.9 / 26.9 |
| 64 KiB | 2 | 32.0 / 63.5 |
| 64 KiB | 32 | 18.3 / 25.8 |
| 64 KiB | 4 | 31.9 / 63.7 |
| 64 KiB | 8 | 37.2 / 44.9 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 48.3 / 84.1 |
| add_assign per row | 1 KiB | 16 | 53.6 / 101 |
| add_assign per row | 1 KiB | 2 | 51.9 / 87.1 |
| add_assign per row | 1 KiB | 32 | 34.0 / 64.9 |
| add_assign per row | 1 KiB | 4 | 43.9 / 86.6 |
| add_assign per row | 1 KiB | 8 | 55.7 / 104 |
| add_assign per row | 64 B | 1 | 6.77 / 13.8 |
| add_assign per row | 64 B | 16 | 8.18 / 14.7 |
| add_assign per row | 64 B | 2 | 7.58 / 14.2 |
| add_assign per row | 64 B | 32 | 8.52 / 14.7 |
| add_assign per row | 64 B | 4 | 8.19 / 14.9 |
| add_assign per row | 64 B | 8 | 8.54 / 16.4 |
| add_assign per row | 64 KiB | 1 | 31.9 / 63.4 |
| add_assign per row | 64 KiB | 16 | 18.9 / 26.9 |
| add_assign per row | 64 KiB | 2 | 31.9 / 63.2 |
| add_assign per row | 64 KiB | 32 | 18.2 / 25.7 |
| add_assign per row | 64 KiB | 4 | 31.8 / 63.3 |
| add_assign per row | 64 KiB | 8 | 37.0 / 44.6 |
| add_assign | 1 KiB | 1 | 52.7 / 86.4 |
| add_assign | 1 KiB | 16 | 84.5 / 140 |
| add_assign | 1 KiB | 2 | 66.4 / 94.8 |
| add_assign | 1 KiB | 32 | 35.0 / 66.9 |
| add_assign | 1 KiB | 4 | 42.2 / 98.1 |
| add_assign | 1 KiB | 8 | 81.1 / 134 |
| add_assign | 64 B | 1 | 7.73 / 13.2 |
| add_assign | 64 B | 16 | 52.7 / 86.4 |
| add_assign | 64 B | 2 | 14.8 / 30.6 |
| add_assign | 64 B | 32 | 66.3 / 95.0 |
| add_assign | 64 B | 4 | 22.7 / 46.6 |
| add_assign | 64 B | 8 | 30.9 / 73.6 |
| add_assign | 64 KiB | 1 | 31.9 / 63.4 |
| add_assign | 64 KiB | 16 | 18.9 / 26.9 |
| add_assign | 64 KiB | 2 | 32.0 / 63.6 |
| add_assign | 64 KiB | 32 | 18.3 / 25.8 |
| add_assign | 64 KiB | 4 | 31.9 / 63.7 |
| add_assign | 64 KiB | 8 | 37.2 / 44.9 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / 13.8 | 19.7 / 48.5 | 30.9 / 74.1 | 53.0 / 86.0 | 54.2 / 102 | 51.5 / 86.5 | 66.9 / 102 | 54.2 / 86.5 | 54.4 / 104 | 56.5 / 87.1 | 71.9 / 107 |
| mul_add | 3.88 / 15.0 | 14.9 / 43.1 | 26.4 / 61.1 | 47.7 / 73.4 | 48.2 / 83.0 | 48.4 / 73.0 | 53.1 / 82.7 | 48.9 / 73.4 | 49.8 / 83.7 | 50.4 / 73.2 | 42.2 / 70.2 |
| mul_add_scatter (16 rows) | 11.4 / 20.3 | 38.6 / 66.3 | 62.9 / 96.7 | 70.2 / 109 | 61.6 / 94.0 | 65.8 / 112 | 55.3 / 94.7 | 73.5 / 110 | 61.6 / 95.1 | 66.7 / 112 | 46.1 / 64.6 |
| mul_add_scatter (4 rows) | 8.51 / 14.7 | 27.1 / 41.0 | 51.2 / 81.7 | 65.1 / 101 | 54.3 / 85.3 | 62.8 / 102 | 55.0 / 87.0 | 67.1 / 101 | 55.8 / 87.1 | 63.4 / 101 | 43.2 / 58.7 |
| mul_assign | 7.12 / 16.1 | 32.4 / 61.3 | 44.5 / 89.6 | 82.3 / 141 | 81.3 / 141 | 86.7 / 143 | 90.3 / 142 | 84.6 / 144 | 83.7 / 141 | 86.9 / 143 | 73.0 / 122 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 79.3 / 98.4 | 54.6 / 98.8 | 35.2 / 42.8 |
| 1 KiB | 16 | 60.7 / 93.0 | 48.9 / 84.6 | 32.4 / 42.5 |
| 16 KiB | 0 | 47.8 / 72.5 | 86.4 / 92.3 | 42.5 / 48.2 |
| 16 KiB | 16 | 47.7 / 73.3 | 80.5 / 75.2 | 39.8 / 45.4 |
| 2 KiB | 0 | 94.1 / 110 | 77.3 / 107 | 39.4 / 45.5 |
| 2 KiB | 16 | 85.4 / 106 | 66.9 / 91.0 | 35.9 / 44.6 |
| 256 B | 0 | 31.1 / 46.8 | 19.4 / 55.9 | 26.1 / 29.6 |
| 256 B | 16 | 21.8 / 31.7 | 18.3 / 51.5 | 23.6 / 29.0 |
| 3 KiB | 0 | 102 / 121 | 76.2 / 90.0 | 40.8 / 46.7 |
| 3 KiB | 16 | 93.6 / 117 | 59.5 / 73.2 | 36.1 / 44.8 |
| 4 KiB | 0 | 90.0 / 113 | 76.3 / 88.4 | 40.7 / 45.8 |
| 4 KiB | 16 | 82.5 / 109 | 64.8 / 72.7 | 33.6 / 42.1 |
| 512 B | 0 | 51.8 / 75.3 | 34.0 / 79.0 | 29.5 / 37.2 |
| 512 B | 16 | 36.1 / 70.7 | 31.3 / 70.0 | 25.7 / 36.9 |
| 8 KiB | 0 | 92.0 / 118 | 83.4 / 90.9 | 41.7 / 48.3 |
| 8 KiB | 16 | 88.6 / 115 | 77.1 / 74.1 | 36.8 / 46.4 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 96.8 / 99.0 |
| 16 | 96.7 / 98.9 |
| 32 | 97.1 / 99.0 |
| 48 | 97.2 / 99.0 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.4 / 31.5 |
| 256 KiB | 16 | 29.3 / 23.8 |
| 64 KiB | 0 | 48.4 / 72.3 |
| 64 KiB | 16 | 48.2 / 72.0 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.4 / 73.3 |
| mul_add_scatter_with | 48.5 / 73.8 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 48.6 / 54.8 |
| gf8b | 16 KiB | 139 / 190 |
| gf8b | 256 KiB | 51.5 / 76.6 |
| gf8b | 4 KiB | 132 / 178 |
| gf8b | 64 KiB | 51.3 / 76.3 |
| gf8d | 1 MiB | 49.0 / 52.7 |
| gf8d | 16 KiB | 138 / 186 |
| gf8d | 256 KiB | 51.5 / 69.2 |
| gf8d | 4 KiB | 131 / 173 |
| gf8d | 64 KiB | 51.3 / 68.9 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 66.1 |
| 4 KiB | 43.4 / 103 |
| 64 MiB | 6.06 / 14.5 |
| 8 MiB | 8.47 / 25.4 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.8 / 27.0 | 13.7 / 21.2 | 18.8 / 27.4 |
| 32 MiB | 9.43 / 27.1 | 5.94 / 13.8 | 6.29 / 16.6 |
| 8 MiB | 18.2 / 41.8 | 7.75 / 16.6 | 8.35 / 25.5 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / 64.7 | 17.9 / 43.7 | 33.6 / 56.8 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / 64.7 | 41.2 / 65.1 | 29.3 / 49.1 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 89.7 / 104 | 83.1 / 104 | 30.6 / 50.3 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 92.3 / 97.6 | 74.9 / 94.8 | 35.6 / 57.0 |
| systematic encode | 10 × 2 | 4 KiB | 73.9 / 76.4 | 65.5 / 74.4 | 53.7 / 75.1 |
| systematic encode | 10 × 4 | 4 KiB | 39.3 / 47.3 | 35.6 / 44.3 | 32.0 / 44.6 |
| systematic encode | 10 × 6 | 4 KiB | 26.6 / 32.4 | 24.4 / 31.3 | 22.5 / 31.0 |
| systematic encode | 10 × 2 | 16 KiB | 81.6 / 83.7 | 52.9 / 69.2 | 67.5 / 84.9 |
| systematic encode | 10 × 4 | 16 KiB | 42.8 / 50.9 | 36.7 / 39.5 | 38.8 / 50.3 |
| systematic encode | 10 × 6 | 16 KiB | 27.0 / 34.5 | 25.1 / 30.6 | 25.8 / 33.9 |
| systematic encode | 10 × 2 | 64 KiB | 83.9 / 79.1 | 53.9 / 67.8 | 73.6 / 87.0 |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / 50.5 | 35.8 / 35.1 | 40.6 / 48.0 |
| systematic encode | 10 × 6 | 64 KiB | 26.9 / 33.8 | 23.6 / 24.1 | 26.6 / 30.7 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
