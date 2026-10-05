# GF(2^8): polynomial byte fields

Public API timings for flat byte fields and the matched `0x11D` competitor
panel. Every result cell reports Tiger Lake / Golden Cove; shared hosts,
toolchain, sampling, and number format: [BENCHMARKS.md](../BENCHMARKS.md).
The v2-versus-v3 comparison lives at [v2 comparison](v2/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gf-comp` |
| Resolved backend | `v4x` / `v3_gfni_crypto` |
| Competitors | Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 through a Go 1.27.1 C archive |
| Competitor build | `external/bench-trio` with `fgf/simd512` |
| Competitor protocol | Arms interleave in one process across all six orders; outputs are validated before timing |
| Throughput | Single-row cases count buffer bytes. Accumulating scatter, gather, and matrix cases count row bytes × sources × destinations. Overwrite-matrix and competitor cases count source bytes. |

## Self-timings

### Scalar multiplication

| Field | Case | Latency (ns/op) |
| --- | --- | --- |
| `Gf8<Poly<AES>>` | Dependent multiply chain | 0.70 / 0.34 |
| `Gf8<Poly<REED_SOLOMON>>` | Dependent multiply chain | 0.61 / 0.34 |

### Packed operations

`Gf8<Poly<AES>>`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| `add_assign` | 52.0 / 96.4 | 32.0 / 63.7 | 7.61 / 10.7 |
| `add_assign_scalar` | 63.5 / 113 | 51.9 / 76.6 | 29.2 / 28.8 |
| `sub_assign_scalar` | 63.5 / 113 | 51.9 / 76.6 | 29.1 / 29.0 |
| `mul_add` | 65.4 / 84.4 | 37.8 / 58.5 | 8.39 / 18.5 |
| `mul_into` | 80.0 / 164 | 38.3 / 63.4 | 5.28 / 28.7 |
| `mul_assign` | 125 / 177 | 51.5 / 76.6 | 30.5 / 29.2 |
| `mul_elementwise` | 54.2 / 68.1 | 23.7 / 42.5 | 5.36 / 7.39 |
| `mul_elementwise_assign` | 58.8 / 69.6 | 28.1 / 59.1 | 7.21 / 10.1 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Operation | Throughput (GiB/s) |
| --- | --- | --- |
| `Gf8<Poly<AES>>` | `mul_add` | 40.9 / 64.5 |
| `Gf8<Poly<AES>>` | `mul_into` | 39.9 / 64.6 |
| `Gf8<Poly<AES>>` | `mul_elementwise` | 29.3 / 54.2 |
| `Gf8<Poly<AES>>` | `mul_elementwise_assign` | 40.7 / 65.8 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_add` | 40.9 / 64.6 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_into` | 39.9 / 64.6 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_elementwise` | 28.8 / 49.6 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_elementwise_assign` | 40.8 / 56.4 |

### One-shot and prepared multiplication

`Gf8<Poly<AES>>`, ordinary `Vec<u8>` buffers; latency per call. Preparation is
excluded.

| Row bytes | `mul_add` (ns/op) | `mul_add_with` (ns/op) |
| --- | --- | --- |
| 16 | 10.9 / 4.34 | 10.7 / 4.16 |
| 32 | 12.0 / 4.34 | 11.9 / 4.16 |
| 64 | 12.0 / 4.53 | 11.9 / 4.34 |
| 128 | 11.1 / 6.03 | 10.9 / 5.78 |
| 256 | 13.0 / 7.19 | 13.0 / 6.12 |
| 512 | 15.2 / 8.53 | 15.2 / 8.50 |
| 1024 | 20.6 / 13.3 | 20.1 / 13.2 |
| 2048 | 27.4 / 23.8 | 27.5 / 23.6 |
| 4096 | 55.5 / 45.4 | 55.7 / 45.1 |
| 16384 | 202 / 170 | 202 / 170 |
| 65536 | 1640 / 1050 | 1640 / 1050 |

### Scatter, gather, and matrix

`Gf8<Poly<AES>>`, 64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one
source; gather has one destination; matrix has eight sources. Column counts
are destinations for scatter and matrix, sources for gather.

| Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- |
| `mul_add_scatter` | 45.5 / 69.8 | 48.4 / 72.3 | 48.5 / 72.2 | 45.0 / 49.2 |
| `mul_add_scatter_with` | 45.5 / 69.8 | 48.4 / 72.3 | 48.5 / 72.2 | 44.9 / 49.1 |
| `mul_add_gather` | 44.5 / 67.5 | 43.9 / 75.0 | 45.0 / 66.5 | 44.9 / 65.6 |
| `mul_add_gather_with` | 44.4 / 66.2 | 44.0 / 75.8 | 45.2 / 67.1 | 44.9 / 65.6 |
| `mul_add_matrix` | 90.4 / 110 | 135 / 153 | 130 / 153 | 124 / 152 |
| `mul_add_matrix_with` | 90.8 / 110 | 135 / 155 | 130 / 154 | 126 / 151 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8<Poly<AES>>` | `mul_into_gather_with` | 90.0 / 88.6 | 101 / 89.1 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_into_gather` | 88.3 / 105 | 101 / 94.8 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_into_gather_with` | 86.2 / 102 | 100 / 94.9 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_add_gather_with` | 101 / 91.6 | 96.7 / 93.1 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous row
offsets. The `fill(0)` cases include clearing the destination.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8<Poly<AES>>` | `fill(0)` + `mul_add_matrix` | 56.3 / 50.8 | 31.1 / 27.0 | 19.3 / 19.6 |
| `Gf8<Poly<AES>>` | `mul_into_matrix` | 73.4 / 67.2 | 41.9 / 35.1 | 25.8 / 25.8 |
| `Gf8<Poly<AES>>` | `mul_add_matrix_with` | 73.1 / 57.2 | 40.2 / 31.2 | 25.4 / 22.8 |
| `Gf8<Poly<AES>>` | `mul_into_matrix_with` | 75.7 / 68.2 | 42.5 / 35.3 | 26.4 / 26.3 |
| `Gf8<Poly<AES>>` | `mul_add_matrix_at` | 72.9 / 57.1 | 40.3 / 31.3 | 24.6 / 22.8 |
| `Gf8<Poly<REED_SOLOMON>>` | `fill(0)` + `mul_add_matrix` | 55.4 / 52.4 | 31.3 / 33.7 | 19.5 / 23.4 |
| `Gf8<Poly<REED_SOLOMON>>` | `mul_into_matrix` | 74.8 / 69.1 | 42.6 / 37.6 | 25.9 / 28.6 |

### Row-wise addition

`Gf8<Poly<AES>>` `add_assign_rows`, ordinary `Vec<u8>` buffers.

| Row bytes | Rows | `Gf8<Poly<AES>>` (GiB/s) |
| --- | --- | --- |
| 64 | 1 | 6.55 / 14.1 |
| 64 | 2 | 12.7 / 27.3 |
| 64 | 4 | 23.6 / 43.6 |
| 64 | 8 | 24.9 / 63.6 |
| 64 | 16 | 43.0 / 69.0 |
| 64 | 32 | 65.6 / 90.0 |
| 1024 | 1 | 43.2 / 80.1 |
| 1024 | 2 | 66.3 / 90.3 |
| 1024 | 4 | 42.5 / 95.6 |
| 1024 | 8 | 79.7 / 128 |
| 1024 | 16 | 83.6 / 131 |
| 1024 | 32 | 35.0 / 66.2 |
| 65536 | 1 | 31.8 / 63.2 |
| 65536 | 2 | 32.0 / 63.5 |
| 65536 | 4 | 32.0 / 63.6 |
| 65536 | 8 | 36.2 / 49.4 |
| 65536 | 16 | 18.9 / 26.4 |
| 65536 | 32 | 18.4 / 25.8 |

### Network-size payloads

`Gf8<Poly<AES>>`, ordinary `Vec<u8>` buffers; scatter rows use the payload
length as the row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 7.12 / 13.1 | 5.51 / 13.2 | 7.98 / 19.1 | 8.73 / 18.0 | 13.1 / 27.5 |
| 256 | 26.9 / 46.0 | 17.7 / 38.7 | 25.2 / 67.5 | 30.1 / 55.7 | 37.2 / 58.1 |
| 512 | 29.3 / 69.0 | 30.8 / 58.7 | 43.1 / 85.7 | 35.5 / 62.5 | 43.2 / 83.3 |
| 1152 | 54.1 / 83.3 | 52.5 / 72.7 | 80.8 / 137 | 63.0 / 95.1 | 71.0 / 108 |
| 1168 | 62.8 / 98.1 | 53.1 / 78.1 | 84.7 / 136 | 56.4 / 89.4 | 62.0 / 99.4 |
| 1184 | 64.2 / 84.8 | 55.4 / 73.7 | 88.4 / 138 | 63.3 / 97.2 | 65.7 / 110 |
| 1200 | 65.0 / 98.3 | 61.6 / 80.2 | 83.0 / 137 | 54.2 / 90.4 | 57.6 / 99.2 |
| 1216 | 56.0 / 84.3 | 53.9 / 73.2 | 82.4 / 138 | 65.5 / 98.4 | 73.6 / 111 |
| 1232 | 65.1 / 99.2 | 54.7 / 81.2 | 87.4 / 136 | 56.4 / 91.8 | 63.3 / 100 |
| 1248 | 65.3 / 85.9 | 56.1 / 73.9 | 94.9 / 139 | 64.3 / 99.6 | 67.0 / 112 |
| 1400 | 60.5 / 103 | 47.6 / 69.9 | 70.6 / 123 | 41.3 / 62.0 | 43.0 / 65.4 |

### Fixed-offset multi-row operations

`Gf8<Poly<REED_SOLOMON>>`; every source and destination base has the stated
offset modulo 64. Scatter has four destinations, gather has sixteen sources,
and matrix has ten sources and four destinations. Warm batches use 512 calls.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 33.1 / 49.6 | 75.5 / 86.6 | 26.3 / 30.1 |
| 256 | 16 | 31.8 / 40.5 | 59.0 / 74.8 | 22.0 / 29.8 |
| 512 | 0 | 54.8 / 83.7 | 100 / 107 | 29.5 / 37.8 |
| 512 | 16 | 38.0 / 70.1 | 76.9 / 87.2 | 26.1 / 37.4 |
| 1024 | 0 | 81.1 / 115 | 111 / 117 | 35.4 / 43.2 |
| 1024 | 16 | 61.7 / 103 | 87.7 / 96.0 | 32.8 / 42.9 |
| 2048 | 0 | 95.2 / 136 | 124 / 122 | 39.5 / 45.6 |
| 2048 | 16 | 88.3 / 129 | 94.0 / 99.6 | 36.1 / 45.0 |
| 3072 | 0 | 97.9 / 122 | 102 / 96.6 | 40.9 / 46.7 |
| 3072 | 16 | 94.1 / 118 | 60.3 / 76.7 | 36.4 / 45.0 |
| 4096 | 0 | 91.2 / 115 | 92.2 / 92.9 | 40.8 / 45.6 |
| 4096 | 16 | 84.8 / 111 | 64.6 / 74.9 | 33.8 / 41.8 |
| 8192 | 0 | 88.1 / 119 | 93.5 / 93.5 | 41.8 / 47.4 |
| 8192 | 16 | 80.6 / 116 | 76.8 / 74.7 | 36.8 / 44.5 |
| 16384 | 0 | 48.9 / 74.1 | 92.4 / 93.9 | 42.4 / 48.2 |
| 16384 | 16 | 48.4 / 73.9 | 82.3 / 75.8 | 39.7 / 45.3 |

### Fixed-offset single-row multiplication

`Gf8<Poly<REED_SOLOMON>>` `mul_add`, 16 KiB. Source and destination start at
the stated offset from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 97.8 / 96.9 |
| 16 | 97.3 / 97.4 |
| 32 | 97.5 / 98.9 |
| 48 | 97.4 / 98.8 |

### Scatter destination alignment

`Gf8<Poly<AES>>` `mul_add_scatter`, eight destinations, ordinary `Vec<u8>`
source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Throughput (GiB/s) |
| --- | --- | --- |
| 64 KiB | 0 | 48.5 / 72.2 |
| 64 KiB | 16 | 48.3 / 72.1 |
| 256 KiB | 0 | 29.4 / 31.1 |
| 256 KiB | 16 | 29.4 / 23.6 |

`Gf8<Poly<REED_SOLOMON>>`, one source and four 16 KiB destination rows,
64-byte-aligned buffers.

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.5 / 73.5 |
| `mul_add_scatter_with` | 48.6 / 73.8 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | `Gf8<Poly<AES>>` (GiB/s) | `Gf8<Poly<REED_SOLOMON>>` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 122 / 170 | 124 / 170 |
| 16 KiB | 132 / 187 | 133 / 187 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 69.3 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 69.1 |
| 1 MiB | 47.1 / 54.2 | 46.6 / 50.4 |

### Large destinations

`Gf8<Poly<AES>>`, ordinary `Vec<u8>` buffers. Read-back XOR-folds all
written 64-bit words.

| Buffer size | `mul_into` (GiB/s) | `mul_into` + read (GiB/s) | `mul_add` (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 19.0 / 26.5 | 13.7 / 21.0 | 19.0 / 26.9 |
| 8 MiB | 18.2 / 41.8 | 7.77 / 16.7 | 8.14 / 25.8 |
| 32 MiB | 9.23 / 28.3 | 5.92 / 14.1 | 6.17 / 16.6 |

`Gf8<Poly<REED_SOLOMON>>`, 64-byte-aligned buffers.

| Operation | Buffer size | Throughput (GiB/s) |
| --- | --- | --- |
| `mul_into` | 4 MiB | 18.2 / 41.8 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients.
ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding
matrix, or `EncodeIdx` for accumulation.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `dst = c * src` | 1 × 1 | 64 KiB | 40.1 / 64.7 | 18.1 / 45.4 | 33.7 / 57.0 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.9 / 64.7 | 41.2 / 65.2 | 29.2 / 48.9 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 89.5 / 103 | 82.8 / 97.8 | 30.6 / 50.9 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 93.5 / 97.6 | 75.1 / 89.9 | 35.6 / 57.4 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 75.2 / 75.4 | 64.9 / 69.7 | 53.8 / 74.7 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 79.4 / 83.3 | 53.3 / 65.1 | 67.6 / 85.1 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 83.8 / 79.5 | 54.2 / 63.9 | 73.6 / 85.0 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 39.6 / 47.6 | 35.5 / 45.0 | 32.1 / 44.2 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 42.9 / 51.0 | 36.7 / 39.4 | 38.4 / 49.9 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 43.4 / 50.5 | 36.0 / 35.1 | 40.7 / 47.8 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 26.4 / 32.7 | 24.2 / 31.4 | 22.7 / 31.2 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 26.8 / 34.5 | 25.0 / 29.4 | 25.7 / 34.0 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 27.0 / 34.1 | 23.8 / 24.0 | 26.7 / 30.9 |

- Preparation and Go bridge pointer staging are excluded.
