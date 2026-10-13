# Byte fields: `gf8b`, `gf8d` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/gf8.md); the other snapshot at [v2](../v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| gf8d | scalar mul | 0.605 / - |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 52.4 / - | 32.0 / - | 7.66 / - |
| add_assign_scalar | 41.5 / - | 41.5 / - | 28.3 / - |
| mul_add | 64.3 / - | 37.9 / - | 8.48 / - |
| mul_assign | 127 / - | 51.5 / - | 30.6 / - |
| mul_elementwise | 74.3 / - | 25.4 / - | 5.16 / - |
| mul_elementwise_assign | 74.1 / - | 29.7 / - | 6.56 / - |
| mul_into | 83.4 / - | 38.2 / - | 13.1 / - |
| sub_assign_scalar | 43.0 / - | 41.5 / - | 28.3 / - |

### Aligned single-row operations

64-byte-aligned buffers.

| Field | Operation | 64 KiB (GiB/s) | 4 MiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_add | 40.9 / - | - |
| gf8b | mul_elementwise | 29.2 / - | - |
| gf8b | mul_elementwise_assign | 40.7 / - | - |
| gf8b | mul_into | 39.9 / - | - |
| gf8d | mul_add | 40.9 / - | - |
| gf8d | mul_elementwise | 28.8 / - | - |
| gf8d | mul_elementwise_assign | 40.8 / - | - |
| gf8d | mul_into | 40.0 / - | 18.2 / - |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | mul_add (ns/op) | mul_add_with (ns/op) |
| --- | --- | --- |
| 16 B | 10.74 / - | 10.31 / - |
| 32 B | 10.82 / - | 10.27 / - |
| 64 B | 12.32 / - | 11.76 / - |
| 128 B | 11.04 / - | 10.55 / - |
| 256 B | 14.01 / - | 13.17 / - |
| 512 B | 17.65 / - | 17.35 / - |
| 1 KiB | 20.66 / - | 21.08 / - |
| 2 KiB | 30.14 / - | 30.35 / - |
| 4 KiB | 61.86 / - | 61.11 / - |
| 16 KiB | 209.7 / - | 209.4 / - |
| 64 KiB | 1636 / - | 1634 / - |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | 2 rows (GiB/s) | 4 rows (GiB/s) | 8 rows (GiB/s) | 16 rows (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather | 44.5 / - | 43.9 / - | 45.0 / - | 45.0 / - |
| mul_add_gather_with | 44.4 / - | 43.8 / - | 44.9 / - | 45.0 / - |
| mul_add_matrix | 89.1 / - | 132 / - | 129 / - | 121 / - |
| mul_add_matrix_with | 89.5 / - | 132 / - | 128 / - | 120 / - |
| mul_add_scatter | 45.3 / - | 48.3 / - | 48.3 / - | 46.5 / - |
| mul_add_scatter_with | 45.3 / - | 48.3 / - | 48.3 / - | 46.5 / - |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Field | Operation | 4 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- |
| gf8b | mul_into_gather_with | 90.9 / - | 104 / - |
| gf8d | mul_add_gather_with | 97.9 / - | 103 / - |
| gf8d | mul_into_gather | 91.0 / - | 105 / - |
| gf8d | mul_into_gather_with | 89.0 / - | 104 / - |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| gf8b | fill(0) + mul_add_matrix | 57.0 / - | 31.4 / - | 19.6 / - |
| gf8b | mul_add_matrix_at | 74.4 / - | 40.4 / - | 25.0 / - |
| gf8b | mul_add_matrix_with | 74.3 / - | 39.4 / - | 25.9 / - |
| gf8b | mul_into_matrix | 73.7 / - | 42.2 / - | 25.6 / - |
| gf8b | mul_into_matrix_with | 76.5 / - | 42.0 / - | 26.2 / - |
| gf8d | fill(0) + mul_add_matrix | 57.5 / - | 30.7 / - | 19.3 / - |
| gf8d | mul_into_matrix | 76.2 / - | 41.5 / - | 25.4 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 50.7 / - |
| 1 KiB | 16 | 84.3 / - |
| 1 KiB | 2 | 67.5 / - |
| 1 KiB | 32 | 34.8 / - |
| 1 KiB | 4 | 43.0 / - |
| 1 KiB | 8 | 81.0 / - |
| 64 B | 1 | 7.02 / - |
| 64 B | 16 | 50.8 / - |
| 64 B | 2 | 13.8 / - |
| 64 B | 32 | 67.3 / - |
| 64 B | 4 | 26.0 / - |
| 64 B | 8 | 25.5 / - |
| 64 KiB | 1 | 31.8 / - |
| 64 KiB | 16 | 18.9 / - |
| 64 KiB | 2 | 32.0 / - |
| 64 KiB | 32 | 18.5 / - |
| 64 KiB | 4 | 32.0 / - |
| 64 KiB | 8 | 35.4 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 47.3 / - |
| add_assign per row | 1 KiB | 16 | 61.9 / - |
| add_assign per row | 1 KiB | 2 | 50.0 / - |
| add_assign per row | 1 KiB | 32 | 33.7 / - |
| add_assign per row | 1 KiB | 4 | 45.8 / - |
| add_assign per row | 1 KiB | 8 | 65.0 / - |
| add_assign per row | 64 B | 1 | 6.72 / - |
| add_assign per row | 64 B | 16 | 7.52 / - |
| add_assign per row | 64 B | 2 | 7.42 / - |
| add_assign per row | 64 B | 32 | 8.07 / - |
| add_assign per row | 64 B | 4 | 8.19 / - |
| add_assign per row | 64 B | 8 | 8.54 / - |
| add_assign per row | 64 KiB | 1 | 31.9 / - |
| add_assign per row | 64 KiB | 16 | 18.9 / - |
| add_assign per row | 64 KiB | 2 | 31.9 / - |
| add_assign per row | 64 KiB | 32 | 18.5 / - |
| add_assign per row | 64 KiB | 4 | 31.9 / - |
| add_assign per row | 64 KiB | 8 | 35.3 / - |
| add_assign | 1 KiB | 1 | 60.8 / - |
| add_assign | 1 KiB | 16 | 86.1 / - |
| add_assign | 1 KiB | 2 | 75.0 / - |
| add_assign | 1 KiB | 32 | 35.0 / - |
| add_assign | 1 KiB | 4 | 43.1 / - |
| add_assign | 1 KiB | 8 | 84.5 / - |
| add_assign | 64 B | 1 | 7.72 / - |
| add_assign | 64 B | 16 | 60.8 / - |
| add_assign | 64 B | 2 | 14.8 / - |
| add_assign | 64 B | 32 | 74.9 / - |
| add_assign | 64 B | 4 | 28.5 / - |
| add_assign | 64 B | 8 | 30.6 / - |
| add_assign | 64 KiB | 1 | 31.7 / - |
| add_assign | 64 KiB | 16 | 18.9 / - |
| add_assign | 64 KiB | 2 | 31.9 / - |
| add_assign | 64 KiB | 32 | 18.5 / - |
| add_assign | 64 KiB | 4 | 32.0 / - |
| add_assign | 64 KiB | 8 | 35.4 / - |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | 64 B (GiB/s) | 256 B (GiB/s) | 512 B (GiB/s) | 1152 B (GiB/s) | 1168 B (GiB/s) | 1184 B (GiB/s) | 1200 B (GiB/s) | 1216 B (GiB/s) | 1232 B (GiB/s) | 1248 B (GiB/s) | 1400 B (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_assign | 7.41 / - | 28.5 / - | 29.6 / - | 55.8 / - | 62.5 / - | 57.4 / - | 57.6 / - | 58.2 / - | 63.5 / - | 59.7 / - | 62.8 / - |
| mul_add | 4.90 / - | 19.3 / - | 28.9 / - | 52.8 / - | 54.8 / - | 54.5 / - | 63.0 / - | 54.3 / - | 56.4 / - | 54.8 / - | 49.6 / - |
| mul_add_scatter (16 rows) | 13.2 / - | 37.7 / - | 44.7 / - | 89.1 / - | 63.7 / - | 68.5 / - | 59.9 / - | 89.2 / - | 63.6 / - | 69.2 / - | 44.4 / - |
| mul_add_scatter (4 rows) | 9.01 / - | 31.0 / - | 37.3 / - | 79.8 / - | 63.0 / - | 68.0 / - | 55.4 / - | 83.8 / - | 64.6 / - | 69.1 / - | 41.4 / - |
| mul_assign | 8.09 / - | 25.5 / - | 43.2 / - | 87.8 / - | 95.6 / - | 86.7 / - | 90.1 / - | 89.1 / - | 95.3 / - | 86.9 / - | 73.5 / - |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | mul_add_scatter (GiB/s) | mul_add_gather (GiB/s) | mul_add_matrix (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 KiB | 0 | 81.2 / - | 110 / - | 35.4 / - |
| 1 KiB | 16 | 61.4 / - | 87.0 / - | 31.8 / - |
| 16 KiB | 0 | 47.8 / - | 93.5 / - | 42.5 / - |
| 16 KiB | 16 | 47.3 / - | 83.2 / - | 39.7 / - |
| 2 KiB | 0 | 92.9 / - | 123 / - | 39.5 / - |
| 2 KiB | 16 | 85.4 / - | 94.5 / - | 36.3 / - |
| 256 B | 0 | 33.0 / - | 70.0 / - | 26.1 / - |
| 256 B | 16 | 32.3 / - | 56.4 / - | 23.5 / - |
| 3 KiB | 0 | 97.4 / - | 98.4 / - | 40.9 / - |
| 3 KiB | 16 | 89.5 / - | 60.7 / - | 36.6 / - |
| 4 KiB | 0 | 90.9 / - | 90.8 / - | 40.9 / - |
| 4 KiB | 16 | 83.2 / - | 64.6 / - | 33.8 / - |
| 512 B | 0 | 54.7 / - | 95.9 / - | 29.4 / - |
| 512 B | 16 | 37.8 / - | 73.6 / - | 27.1 / - |
| 8 KiB | 0 | 92.0 / - | 92.1 / - | 41.7 / - |
| 8 KiB | 16 | 88.7 / - | 75.5 / - | 36.4 / - |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | mul_add (GiB/s) |
| --- | --- |
| 0 | 97.8 / - |
| 16 | 97.7 / - |
| 32 | 97.9 / - |
| 48 | 97.8 / - |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | mul_add_scatter (GiB/s) |
| --- | --- | --- |
| 256 KiB | 0 | 29.6 / - |
| 256 KiB | 16 | 29.7 / - |
| 64 KiB | 0 | 48.4 / - |
| 64 KiB | 16 | 48.3 / - |

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | Throughput (GiB/s) |
| --- | --- |
| mul_add_scatter | 48.5 / - |
| mul_add_scatter_with | 48.6 / - |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Field | Buffer size | mul_assign (GiB/s) |
| --- | --- | --- |
| gf8b | 1 MiB | 49.5 / - |
| gf8b | 16 KiB | 137 / - |
| gf8b | 256 KiB | 51.5 / - |
| gf8b | 4 KiB | 128 / - |
| gf8b | 64 KiB | 51.3 / - |
| gf8d | 1 MiB | 47.5 / - |
| gf8d | 16 KiB | 137 / - |
| gf8d | 256 KiB | 51.5 / - |
| gf8d | 4 KiB | 128 / - |
| gf8d | 64 KiB | 51.3 / - |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | add_assign (GiB/s) |
| --- | --- |
| 256 KiB | 41.4 / - |
| 4 KiB | 43.3 / - |
| 64 MiB | 6.05 / - |
| 8 MiB | 8.39 / - |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | mul_into (GiB/s) | mul_into + read (GiB/s) | mul_add (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.8 / - | 13.7 / - | 18.8 / - |
| 32 MiB | 9.38 / - | 5.96 / - | 6.27 / - |
| 8 MiB | 18.2 / - | 7.68 / - | 8.30 / - |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients. ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding matrix, or `EncodeIdx` for accumulation. Outputs are validated before timing.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| dst = c * src | 1 × 1 | 64 KiB | 40.0 / - | 17.8 / - | 33.7 / - |
| dst ^= c * src | 1 × 1 | 64 KiB | 40.9 / - | 41.2 / - | 29.1 / - |
| dst = sum(c[i] * src[i]) | 16 × 1 | 4 KiB | 88.7 / - | 83.1 / - | 30.6 / - |
| dst = sum(c[i] * src[i]) | 16 × 1 | 16 KiB | 92.5 / - | 75.0 / - | 35.7 / - |
| systematic encode | 10 × 2 | 4 KiB | 75.1 / - | 65.5 / - | 53.8 / - |
| systematic encode | 10 × 4 | 4 KiB | 39.7 / - | 35.8 / - | 32.0 / - |
| systematic encode | 10 × 6 | 4 KiB | 26.7 / - | 24.4 / - | 22.7 / - |
| systematic encode | 10 × 2 | 16 KiB | 81.8 / - | 52.9 / - | 67.5 / - |
| systematic encode | 10 × 4 | 16 KiB | 42.9 / - | 36.7 / - | 38.3 / - |
| systematic encode | 10 × 6 | 16 KiB | 27.1 / - | 25.1 / - | 25.7 / - |
| systematic encode | 10 × 2 | 64 KiB | 83.5 / - | 54.0 / - | 73.6 / - |
| systematic encode | 10 × 4 | 64 KiB | 43.3 / - | 35.6 / - | 40.7 / - |
| systematic encode | 10 × 6 | 64 KiB | 27.0 / - | 23.9 / - | 26.5 / - |

## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
