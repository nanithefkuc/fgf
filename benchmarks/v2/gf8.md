# Byte fields: `gf8b`, `gf8d` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `-` |
| Resolved field backends | `gf8b`: `v4x` / -; `gf8d`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64 |

## Self-timings

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Field | Operation | scalar mul (ns/op) |
| --- | --- | --- |
| gf8b | scalar mul | 0.701 / - |
| gf8d | scalar mul | 0.701 / - |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 52.0 / - | 32.0 / - | 7.64 / - |
| add_assign_scalar | 42.8 / - | 41.5 / - | 28.5 / - |
| mul_add | 64.8 / - | 37.9 / - | 8.30 / - |
| mul_assign | 124 / - | 51.5 / - | 30.7 / - |
| mul_elementwise | 74.5 / - | 26.3 / - | 5.13 / - |
| mul_elementwise_assign | 79.8 / - | 31.2 / - | 6.71 / - |
| mul_into | 82.8 / - | 38.3 / - | 13.1 / - |
| sub_assign_scalar | 42.6 / - | 41.5 / - | 28.5 / - |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / - | - |
| gf8b | mul_elementwise | 29.0 / - | - |
| gf8b | mul_elementwise_assign | 40.7 / - | - |
| gf8b | mul_into | 39.9 / - | - |
| gf8d | mul_add | 40.9 / - | - |
| gf8d | mul_elementwise | 28.7 / - | - |
| gf8d | mul_elementwise_assign | 40.7 / - | - |
| gf8d | mul_into | 40.0 / - | 18.2 / - |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 14.37 / - | 14.39 / - |
| 32 B | 14.36 / - | 14.44 / - |
| 64 B | 15.69 / - | 15.68 / - |
| 128 B | 14.83 / - | 14.85 / - |
| 256 B | 16.34 / - | 16.14 / - |
| 512 B | 18.38 / - | 18.44 / - |
| 1 KiB | 22.21 / - | 22.12 / - |
| 2 KiB | 30.31 / - | 30.54 / - |
| 4 KiB | 62.07 / - | 61.59 / - |
| 16 KiB | 210.9 / - | 209.9 / - |
| 64 KiB | 1633 / - | 1632 / - |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.3 / - | 42.9 / - | 43.7 / - | 44.5 / - |
| mul_add_gather_with | 44.3 / - | 42.9 / - | 43.8 / - | 44.4 / - |
| mul_add_matrix | 89.6 / - | 132 / - | 127 / - | 123 / - |
| mul_add_matrix_with | 89.3 / - | 132 / - | 123 / - | 119 / - |
| mul_add_scatter | 45.4 / - | 48.2 / - | 48.3 / - | 45.4 / - |
| mul_add_scatter_with | 45.4 / - | 48.2 / - | 48.3 / - | 45.4 / - |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 91.4 / - | 105 / - |
| gf8d | mul_add_gather_with | 80.0 / - | 96.4 / - |
| gf8d | mul_into_gather | 90.9 / - | 105 / - |
| gf8d | mul_into_gather_with | 91.4 / - | 105 / - |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.0 / - | 31.3 / - | 19.5 / - |
| gf8b | mul_add_matrix_at | 74.3 / - | 40.4 / - | 25.0 / - |
| gf8b | mul_add_matrix_with | 74.1 / - | 39.8 / - | 24.7 / - |
| gf8b | mul_into_matrix | 73.6 / - | 42.3 / - | 25.8 / - |
| gf8b | mul_into_matrix_with | 76.3 / - | 42.0 / - | 25.4 / - |
| gf8d | fill(0) + mul_add_matrix | 57.5 / - | 31.6 / - | 19.4 / - |
| gf8d | mul_into_matrix | 75.9 / - | 42.6 / - | 25.8 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 49.7 / - |
| 1 KiB | 16 | 84.0 / - |
| 1 KiB | 2 | 66.7 / - |
| 1 KiB | 32 | 34.9 / - |
| 1 KiB | 4 | 42.4 / - |
| 1 KiB | 8 | 80.6 / - |
| 64 B | 1 | 7.15 / - |
| 64 B | 16 | 49.9 / - |
| 64 B | 2 | 13.6 / - |
| 64 B | 32 | 66.7 / - |
| 64 B | 4 | 26.0 / - |
| 64 B | 8 | 26.8 / - |
| 64 KiB | 1 | 31.8 / - |
| 64 KiB | 16 | 18.9 / - |
| 64 KiB | 2 | 32.0 / - |
| 64 KiB | 32 | 18.5 / - |
| 64 KiB | 4 | 32.0 / - |
| 64 KiB | 8 | 35.6 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 46.7 / - |
| add_assign per row | 1 KiB | 16 | 60.1 / - |
| add_assign per row | 1 KiB | 2 | 51.8 / - |
| add_assign per row | 1 KiB | 32 | 34.8 / - |
| add_assign per row | 1 KiB | 4 | 45.0 / - |
| add_assign per row | 1 KiB | 8 | 62.8 / - |
| add_assign per row | 64 B | 1 | 6.76 / - |
| add_assign per row | 64 B | 16 | 8.14 / - |
| add_assign per row | 64 B | 2 | 7.58 / - |
| add_assign per row | 64 B | 32 | 8.51 / - |
| add_assign per row | 64 B | 4 | 8.20 / - |
| add_assign per row | 64 B | 8 | 8.54 / - |
| add_assign per row | 64 KiB | 1 | 31.9 / - |
| add_assign per row | 64 KiB | 16 | 18.9 / - |
| add_assign per row | 64 KiB | 2 | 31.9 / - |
| add_assign per row | 64 KiB | 32 | 18.5 / - |
| add_assign per row | 64 KiB | 4 | 31.9 / - |
| add_assign per row | 64 KiB | 8 | 35.4 / - |
| add_assign | 1 KiB | 1 | 59.8 / - |
| add_assign | 1 KiB | 16 | 85.2 / - |
| add_assign | 1 KiB | 2 | 74.3 / - |
| add_assign | 1 KiB | 32 | 35.0 / - |
| add_assign | 1 KiB | 4 | 44.0 / - |
| add_assign | 1 KiB | 8 | 82.7 / - |
| add_assign | 64 B | 1 | 7.73 / - |
| add_assign | 64 B | 16 | 60.0 / - |
| add_assign | 64 B | 2 | 14.8 / - |
| add_assign | 64 B | 32 | 74.3 / - |
| add_assign | 64 B | 4 | 28.5 / - |
| add_assign | 64 B | 8 | 30.9 / - |
| add_assign | 64 KiB | 1 | 31.8 / - |
| add_assign | 64 KiB | 16 | 18.9 / - |
| add_assign | 64 KiB | 2 | 31.9 / - |
| add_assign | 64 KiB | 32 | 18.5 / - |
| add_assign | 64 KiB | 4 | 32.0 / - |
| add_assign | 64 KiB | 8 | 35.6 / - |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / - | 27.2 / - | 30.4 / - | 55.8 / - | 60.7 / - | 56.1 / - | 57.6 / - | 57.3 / - | 62.3 / - | 59.0 / - | 62.8 / - |
| mul_add | 3.82 / - | 15.5 / - | 26.0 / - | 47.6 / - | 50.4 / - | 48.3 / - | 49.8 / - | 49.5 / - | 52.4 / - | 49.9 / - | 41.2 / - |
| mul_add_scatter (16 rows) | 11.4 / - | 31.1 / - | 43.7 / - | 87.5 / - | 58.4 / - | 65.4 / - | 56.4 / - | 89.2 / - | 58.2 / - | 66.6 / - | 46.4 / - |
| mul_add_scatter (4 rows) | 8.56 / - | 27.2 / - | 38.7 / - | 80.3 / - | 59.2 / - | 66.3 / - | 52.0 / - | 84.8 / - | 60.3 / - | 63.6 / - | 43.6 / - |
| mul_assign | 7.41 / - | 25.5 / - | 42.7 / - | 80.2 / - | 92.4 / - | 80.4 / - | 83.5 / - | 82.6 / - | 94.8 / - | 82.8 / - | 69.6 / - |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 79.7 / - | 55.3 / - | 35.1 / - |
| 1 KiB | 16 | 60.4 / - | 49.0 / - | 31.5 / - |
| 16 KiB | 0 | 48.0 / - | 91.8 / - | 42.2 / - |
| 16 KiB | 16 | 47.3 / - | 86.7 / - | 39.5 / - |
| 2 KiB | 0 | 94.6 / - | 78.8 / - | 39.4 / - |
| 2 KiB | 16 | 83.2 / - | 67.2 / - | 35.5 / - |
| 256 B | 0 | 30.9 / - | 19.5 / - | 25.9 / - |
| 256 B | 16 | 27.7 / - | 18.4 / - | 23.1 / - |
| 3 KiB | 0 | 104 / - | 76.5 / - | 40.8 / - |
| 3 KiB | 16 | 93.1 / - | 61.2 / - | 36.0 / - |
| 4 KiB | 0 | 90.1 / - | 75.8 / - | 40.7 / - |
| 4 KiB | 16 | 82.5 / - | 65.7 / - | 33.5 / - |
| 512 B | 0 | 51.8 / - | 34.7 / - | 29.0 / - |
| 512 B | 16 | 37.0 / - | 31.4 / - | 26.7 / - |
| 8 KiB | 0 | 91.8 / - | 84.6 / - | 39.4 / - |
| 8 KiB | 16 | 88.3 / - | 77.8 / - | 35.0 / - |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 96.2 / - |
| 16 | 96.5 / - |
| 32 | 96.5 / - |
| 48 | 96.5 / - |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / - |
| 256 KiB | 16 | 29.8 / - |
| 64 KiB | 0 | 48.3 / - |
| 64 KiB | 16 | 48.3 / - |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.4 / - |
| mul_add_scatter_with | 48.5 / - |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 49.3 / - |
| gf8b | 16 KiB | 138 / - |
| gf8b | 256 KiB | 51.5 / - |
| gf8b | 4 KiB | 132 / - |
| gf8b | 64 KiB | 51.3 / - |
| gf8d | 1 MiB | 50.1 / - |
| gf8d | 16 KiB | 137 / - |
| gf8d | 256 KiB | 51.5 / - |
| gf8d | 4 KiB | 131 / - |
| gf8d | 64 KiB | 51.3 / - |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / - |
| 4 KiB | 43.4 / - |
| 64 MiB | 6.06 / - |
| 8 MiB | 8.26 / - |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 19.0 / - | 13.6 / - | 19.0 / - |
| 32 MiB | 9.40 / - | 5.98 / - | 6.24 / - |
| 8 MiB | 18.2 / - | 7.73 / - | 8.16 / - |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.1 / - | 18.0 / - | 33.7 / - |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / - | 41.2 / - | 29.1 / - |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 88.7 / - | 83.1 / - | 30.5 / - |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 94.6 / - | 74.9 / - | 35.6 / - |
| systematic encode | 10 × 2 | 4 KiB | 73.8 / - | 65.5 / - | 54.1 / - |
| systematic encode | 10 × 4 | 4 KiB | 39.3 / - | 35.7 / - | 32.0 / - |
| systematic encode | 10 × 6 | 4 KiB | 26.6 / - | 24.4 / - | 22.5 / - |
| systematic encode | 10 × 2 | 16 KiB | 81.5 / - | 53.0 / - | 67.8 / - |
| systematic encode | 10 × 4 | 16 KiB | 42.8 / - | 36.7 / - | 38.7 / - |
| systematic encode | 10 × 6 | 16 KiB | 27.0 / - | 25.0 / - | 25.9 / - |
| systematic encode | 10 × 2 | 64 KiB | 83.5 / - | 54.0 / - | 73.4 / - |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / - | 35.9 / - | 40.6 / - |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / - | 23.7 / - | 26.6 / - |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
