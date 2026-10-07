# Byte fields: `gf8b`, `gf8d` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v2](../v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| add_assign | 51.1 / 98.3 | 31.9 / 63.7 | 7.79 / 10.7 |
| add_assign_scalar | 43.1 / 80.4 | 41.6 / 67.6 | 28.7 / 25.5 |
| mul_add | 66.3 / 83.5 | 37.9 / 59.0 | 8.33 / 18.4 |
| mul_assign | 127 / 174 | 51.5 / 76.6 | 30.8 / 29.2 |
| mul_elementwise | 59.1 / 101 | 25.5 / 39.4 | 5.28 / 7.42 |
| mul_elementwise_assign | 60.6 / 106 | 28.1 / 50.3 | 7.02 / 10.0 |
| mul_into | 79.3 / 170 | 38.2 / 64.8 | 5.30 / 28.7 |
| sub_assign_scalar | 42.4 / 80.4 | 41.6 / 67.6 | 28.6 / 25.3 |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / 64.6 | - |
| gf8b | mul_elementwise | 29.1 / 54.2 | - |
| gf8b | mul_elementwise_assign | 40.7 / 65.8 | - |
| gf8b | mul_into | 39.9 / 64.6 | - |
| gf8d | mul_add | 40.9 / 64.7 | - |
| gf8d | mul_elementwise | 28.7 / 49.6 | - |
| gf8d | mul_elementwise_assign | 40.8 / 56.7 | - |
| gf8d | mul_into | 40.0 / 64.7 | 18.2 / 41.8 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 1 KiB | 20.87 / 13.68 | 21.35 / 13.59 |
| 128 B | 10.85 / 4.914 | 10.55 / 4.508 |
| 16 B | 10.74 / 4.095 | 10.29 / 3.696 |
| 16 KiB | 210.1 / 169.8 | 210.6 / 169.7 |
| 2 KiB | 31.88 / 24.05 | 32.09 / 23.96 |
| 256 B | 14.10 / 5.937 | 13.46 / 5.526 |
| 32 B | 10.86 / 4.096 | 10.14 / 3.698 |
| 4 KiB | 63.29 / 45.53 | 62.87 / 45.81 |
| 512 B | 16.29 / 7.989 | 16.52 / 8.023 |
| 64 B | 12.26 / 4.297 | 11.81 / 3.920 |
| 64 KiB | 1633 / 1052 | 1633 / 1052 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 45.1 / 72.9 | 44.5 / 80.0 | 45.5 / 69.2 | 44.5 / 64.9 |
| mul_add_gather_with | 45.2 / 73.3 | 44.6 / 80.2 | 45.4 / 69.2 | 44.5 / 64.9 |
| mul_add_matrix | 89.7 / 107 | 134 / 147 | 128 / 147 | 121 / 143 |
| mul_add_matrix_with | 90.1 / 107 | 133 / 147 | 128 / 146 | 121 / 143 |
| mul_add_scatter | 45.3 / 69.7 | 48.2 / 72.2 | 48.3 / 72.1 | 46.0 / 52.1 |
| mul_add_scatter_with | 45.3 / 69.8 | 48.3 / 72.3 | 48.3 / 72.1 | 46.0 / 51.8 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.1 / 87.1 | 104 / 89.2 |
| gf8d | mul_add_gather_with | 98.0 / 90.8 | 101 / 92.8 |
| gf8d | mul_into_gather | 91.0 / 106 | 104 / 112 |
| gf8d | mul_into_gather_with | 88.4 / 105 | 103 / 112 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.3 / 52.1 | 31.7 / 28.5 | 19.6 / 20.3 |
| gf8b | mul_add_matrix_at | 74.5 / 61.0 | 38.0 / 31.5 | 23.7 / 23.2 |
| gf8b | mul_add_matrix_with | 74.1 / 60.9 | 38.0 / 31.4 | 24.0 / 22.9 |
| gf8b | mul_into_matrix | 76.2 / 68.8 | 42.9 / 37.7 | 24.7 / 25.9 |
| gf8b | mul_into_matrix_with | 76.2 / 70.2 | 39.3 / 36.3 | 24.4 / 26.5 |
| gf8d | fill(0) + mul_add_matrix | 56.2 / 54.6 | 30.4 / 34.0 | 18.8 / 22.9 |
| gf8d | mul_into_matrix | 76.6 / 71.5 | 39.1 / 38.8 | 24.3 / 29.1 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 49.1 / 82.2 |
| 1 KiB | 16 | 83.4 / 134 |
| 1 KiB | 2 | 67.2 / 92.0 |
| 1 KiB | 32 | 34.8 / 66.6 |
| 1 KiB | 4 | 42.6 / 97.3 |
| 1 KiB | 8 | 79.3 / 132 |
| 64 B | 1 | 6.82 / 13.8 |
| 64 B | 16 | 49.0 / 82.3 |
| 64 B | 2 | 13.6 / 27.7 |
| 64 B | 32 | 67.2 / 92.1 |
| 64 B | 4 | 25.0 / 48.5 |
| 64 B | 8 | 26.8 / 72.7 |
| 64 KiB | 1 | 31.8 / 63.2 |
| 64 KiB | 16 | 18.8 / 26.6 |
| 64 KiB | 2 | 31.9 / 63.5 |
| 64 KiB | 32 | 18.3 / 25.8 |
| 64 KiB | 4 | 31.9 / 63.7 |
| 64 KiB | 8 | 35.3 / 45.4 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 48.3 / 84.6 |
| add_assign per row | 1 KiB | 16 | 51.4 / 101 |
| add_assign per row | 1 KiB | 2 | 51.4 / 87.3 |
| add_assign per row | 1 KiB | 32 | 33.1 / 65.1 |
| add_assign per row | 1 KiB | 4 | 37.3 / 86.6 |
| add_assign per row | 1 KiB | 8 | 52.4 / 104 |
| add_assign per row | 64 B | 1 | 6.59 / 13.2 |
| add_assign per row | 64 B | 16 | 7.02 / 14.0 |
| add_assign per row | 64 B | 2 | 6.85 / 13.5 |
| add_assign per row | 64 B | 32 | 7.13 / 13.4 |
| add_assign per row | 64 B | 4 | 7.28 / 14.4 |
| add_assign per row | 64 B | 8 | 7.51 / 14.8 |
| add_assign per row | 64 KiB | 1 | 31.9 / 63.4 |
| add_assign per row | 64 KiB | 16 | 18.8 / 26.5 |
| add_assign per row | 64 KiB | 2 | 31.8 / 63.2 |
| add_assign per row | 64 KiB | 32 | 18.3 / 25.8 |
| add_assign per row | 64 KiB | 4 | 31.7 / 63.4 |
| add_assign per row | 64 KiB | 8 | 35.6 / 45.4 |
| add_assign | 1 KiB | 1 | 52.7 / 86.5 |
| add_assign | 1 KiB | 16 | 84.1 / 140 |
| add_assign | 1 KiB | 2 | 74.1 / 95.3 |
| add_assign | 1 KiB | 32 | 34.9 / 66.9 |
| add_assign | 1 KiB | 4 | 43.2 / 98.2 |
| add_assign | 1 KiB | 8 | 80.6 / 136 |
| add_assign | 64 B | 1 | 7.11 / 13.8 |
| add_assign | 64 B | 16 | 52.7 / 86.6 |
| add_assign | 64 B | 2 | 14.3 / 27.7 |
| add_assign | 64 B | 32 | 74.0 / 95.3 |
| add_assign | 64 B | 4 | 28.5 / 46.6 |
| add_assign | 64 B | 8 | 30.8 / 73.7 |
| add_assign | 64 KiB | 1 | 31.8 / 63.3 |
| add_assign | 64 KiB | 16 | 18.8 / 26.6 |
| add_assign | 64 KiB | 2 | 32.0 / 63.6 |
| add_assign | 64 KiB | 32 | 18.3 / 25.8 |
| add_assign | 64 KiB | 4 | 31.9 / 63.7 |
| add_assign | 64 KiB | 8 | 35.3 / 45.2 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.12 / 14.5 | 27.4 / 46.6 | 30.8 / 74.4 | 53.6 / 85.9 | 54.6 / 102 | 54.7 / 86.5 | 65.4 / 102 | 55.7 / 86.7 | 56.6 / 105 | 56.7 / 87.2 | 70.6 / 107 |
| mul_add | 5.03 / 14.6 | 21.3 / 41.6 | 34.3 / 61.6 | 50.5 / 73.6 | 52.5 / 81.8 | 52.9 / 73.2 | 58.2 / 82.7 | 51.9 / 73.4 | 53.7 / 83.7 | 54.4 / 73.4 | 49.9 / 70.3 |
| mul_add_scatter (16 rows) | 13.2 / 28.0 | 42.8 / 77.5 | 67.1 / 93.9 | 72.8 / 113 | 65.4 / 101 | 70.6 / 115 | 60.7 / 100 | 75.0 / 117 | 65.9 / 101 | 70.1 / 117 | 43.8 / 67.7 |
| mul_add_scatter (4 rows) | 9.05 / 18.6 | 30.9 / 48.1 | 51.7 / 76.2 | 65.3 / 101 | 58.4 / 90.3 | 66.8 / 102 | 57.7 / 91.9 | 67.3 / 103 | 58.1 / 93.3 | 66.6 / 104 | 42.2 / 59.5 |
| mul_assign | 7.42 / 22.4 | 35.6 / 77.6 | 47.5 / 89.6 | 84.7 / 140 | 86.5 / 143 | 89.2 / 142 | 95.4 / 145 | 89.1 / 142 | 88.2 / 144 | 93.6 / 140 | 73.1 / 127 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 80.9 / 103 | 110 / 115 | 35.3 / 43.3 |
| 1 KiB | 16 | 61.6 / 92.9 | 87.8 / 96.6 | 32.6 / 42.7 |
| 16 KiB | 0 | 47.9 / 72.6 | 88.8 / 93.4 | 42.3 / 48.2 |
| 16 KiB | 16 | 48.2 / 73.6 | 80.2 / 75.9 | 39.6 / 45.5 |
| 2 KiB | 0 | 92.6 / 115 | 122 / 118 | 39.4 / 45.5 |
| 2 KiB | 16 | 87.5 / 107 | 88.4 / 101 | 35.7 / 44.7 |
| 256 B | 0 | 33.0 / 49.5 | 70.7 / 85.8 | 26.2 / 30.3 |
| 256 B | 16 | 25.8 / 32.9 | 56.6 / 74.6 | 23.6 / 30.1 |
| 3 KiB | 0 | 97.1 / 122 | 103 / 97.1 | 40.9 / 46.7 |
| 3 KiB | 16 | 91.3 / 117 | 60.0 / 78.1 | 36.0 / 44.9 |
| 4 KiB | 0 | 91.2 / 115 | 88.6 / 92.2 | 40.8 / 45.5 |
| 4 KiB | 16 | 84.2 / 112 | 63.7 / 74.9 | 33.8 / 41.8 |
| 512 B | 0 | 54.5 / 82.8 | 96.7 / 105 | 29.1 / 37.9 |
| 512 B | 16 | 38.0 / 70.3 | 73.1 / 88.0 | 25.9 / 37.6 |
| 8 KiB | 0 | 93.2 / 119 | 91.8 / 93.0 | 41.5 / 48.3 |
| 8 KiB | 16 | 80.5 / 116 | 75.6 / 75.6 | 36.6 / 46.3 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 97.9 / 99.2 |
| 16 | 97.9 / 99.3 |
| 32 | 98.2 / 99.2 |
| 48 | 97.8 / 99.4 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / 31.1 |
| 256 KiB | 16 | 29.5 / 24.1 |
| 64 KiB | 0 | 48.5 / 72.2 |
| 64 KiB | 16 | 48.3 / 72.0 |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.5 / 73.5 |
| mul_add_scatter_with | 48.6 / 73.9 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 50.4 / 58.1 |
| gf8b | 16 KiB | 133 / 187 |
| gf8b | 256 KiB | 51.5 / 76.6 |
| gf8b | 4 KiB | 126 / 173 |
| gf8b | 64 KiB | 51.4 / 76.3 |
| gf8d | 1 MiB | 48.7 / 51.0 |
| gf8d | 16 KiB | 133 / 183 |
| gf8d | 256 KiB | 51.5 / 69.1 |
| gf8d | 4 KiB | 126 / 172 |
| gf8d | 64 KiB | 51.4 / 68.4 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / 66.1 |
| 4 KiB | 43.0 / 103 |
| 64 MiB | 6.06 / 14.6 |
| 8 MiB | 8.48 / 25.2 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.9 / 26.7 | 13.7 / 21.1 | 18.9 / 27.2 |
| 32 MiB | 9.42 / 27.8 | 6.00 / 13.9 | 6.30 / 16.6 |
| 8 MiB | 16.3 / 41.8 | 7.57 / 16.7 | 8.36 / 25.3 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.0 / 64.6 | 17.6 / 45.6 | 33.7 / 57.0 |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.8 / 64.7 | 41.2 / 65.1 | 29.1 / 49.0 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 89.5 / 104 | 83.1 / 104 | 30.6 / 50.6 |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 94.0 / 98.1 | 75.0 / 94.9 | 35.6 / 57.1 |
| systematic encode | 10 × 2 | 4 KiB | 74.9 / 76.5 | 65.5 / 74.3 | 53.8 / 74.4 |
| systematic encode | 10 × 4 | 4 KiB | 39.7 / 47.2 | 35.9 / 44.4 | 31.8 / 44.4 |
| systematic encode | 10 × 6 | 4 KiB | 26.9 / 32.4 | 24.4 / 31.2 | 22.5 / 31.3 |
| systematic encode | 10 × 2 | 16 KiB | 81.8 / 85.0 | 52.9 / 69.4 | 67.3 / 84.8 |
| systematic encode | 10 × 4 | 16 KiB | 42.9 / 50.9 | 36.7 / 39.6 | 38.7 / 50.0 |
| systematic encode | 10 × 6 | 16 KiB | 27.3 / 34.6 | 25.0 / 30.3 | 25.8 / 34.0 |
| systematic encode | 10 × 2 | 64 KiB | 84.1 / 79.3 | 54.0 / 69.0 | 73.4 / 89.6 |
| systematic encode | 10 × 4 | 64 KiB | 43.5 / 50.4 | 35.5 / 34.4 | 40.6 / 47.2 |
| systematic encode | 10 × 6 | 64 KiB | 27.2 / 34.1 | 24.0 / 24.3 | 26.6 / 30.7 |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
