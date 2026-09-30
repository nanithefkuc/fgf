# Benchmarks

Current public API performance and matched competitor comparisons. Every result
cell reports **Tiger Lake / Golden Cove**. Latency is in ns/op; throughput is in
GiB/s. The hosts are separate machines, not interchangeable measurement controls.

### Measurement provenance

| Setting | Value |
| --- | --- |
| CPU | Intel Core i5-1135G7 / Intel Core i7-12700K |
| OS | Arch Linux, Linux 7.2.7-arch1-1 / CachyOS, Linux 7.2.6-1-cachyos |
| Toolchain | Rust 1.98.1; Go 1.27.1 |
| Source | `fgf` 2.0.0, identical measurement sources on both hosts |
| Dependencies | `simdispatch` 0.2.0, `archmage` 0.9.29; ISA-L 2.32.0, klauspost/reedsolomon v1.14.2, Plonky3 0.7.0 |
| Execution | Isolated CPU 3 / 8; `SCHED_IDLE`; `performance` / `powersave` governor; `RAYON_NUM_THREADS=1`, `GOMAXPROCS=1` |
| Build | Self suites: `--all-features`; competitor harnesses: `fgf/simd512`; prime harness: `-C target-cpu=native` |
| Resolved backend | Process and byte fields: `v4x` / `v3_gfni_crypto`; Mersenne31 and Goldilocks: `v4x` / `v3`; QuadMersenne31: `v3` / `v3` |
| Plonky3 packing | Mersenne31: 16 / 8 elements; Goldilocks: 8 / 4; quadratic extension: 16 / 8 |
| Sampling | One complete campaign across both hosts. Every cell is the median of five per-run medians, rounded to three significant figures. Warm batches use 32 calls, or 512 in the fixed-offset multi-row sweep; scalar batches contain 1,048,576 operations. |
| Competitor protocol | Arms interleave within each process: all six orders for ISA-L/klauspost, alternating order for Plonky3. Outputs are validated before timing; duplicate-operation controls check measurement bias. |
| Setup | Allocation, input generation, coefficient preparation, and layout conversion occur outside timed regions. Buffers are warm; alignment is stated per table. |
| Throughput accounting | Single-row and GF(2) cases count buffer bytes. Accumulating scatter/gather/matrix cases count row bytes × sources × destinations. Overwrite-matrix cases and byte-field competitor cases count source bytes. Prime competitor cases count twice the buffer bytes. Range XOR counts the entire packed buffer, not only the updated range. |

Reproduce on each host's isolated CPU:

```sh
export FEC_GOLDEN_CORE=<cpu> RAYON_NUM_THREADS=1 GOMAXPROCS=1
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gf-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gdl-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-m31-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gf2-comp
```

## Public API self-timings

### Scalar multiplication

| Field | Case | Latency (ns/op) |
| --- | --- | --- |
| `Gf8B` | Dependent multiply chain | 0.7 / 0.34 |
| `Gf8D` | Dependent multiply chain | 0.7 / 0.34 |
| `Gf16` | Dependent multiply chain | 2.41 / 1.34 |
| `Mersenne31` | Indexed products with accumulation | 4.18 / 1.55 |
| `Goldilocks` | Indexed products with accumulation | 3.74 / 1.64 |
| `QuadMersenne31` | Indexed products with accumulation | 8.97 / 4.41 |

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `add_assign` | 42.9 / 96.7 | 32 / 63.7 | 7.61 / 10.8 |
| `Gf8B` | `add_assign_scalar` | 42.5 / 80.2 | 41.6 / 67.5 | 31.8 / 25.2 |
| `Gf8B` | `sub_assign_scalar` | 42.9 / 80.2 | 41.6 / 67.5 | 31.8 / 25.6 |
| `Gf8B` | `mul_add` | 64.8 / 84 | 37.9 / 58.5 | 8.52 / 17.2 |
| `Gf8B` | `mul_into` | 82 / 163 | 38.2 / 64.7 | 13.1 / 29 |
| `Gf8B` | `mul_assign` | 122 / 171 | 51.5 / 76.6 | 30.8 / 29.2 |
| `Gf8B` | `mul_elementwise` | 73.4 / 98.6 | 26.3 / 40.8 | 5.16 / 7.54 |
| `Gf8B` | `mul_elementwise_assign` | 74.7 / 117 | 31.2 / 49.8 | 6.71 / 10 |
| `Gf16` | `add_assign` | 42.9 / 96.4 | 32 / 63.7 | 7.63 / 10.8 |
| `Gf16` | `add_assign_scalar` | 41.6 / 78.1 | 41.5 / 67.5 | 31.9 / 25.5 |
| `Gf16` | `sub_assign_scalar` | 41.1 / 78.2 | 41.5 / 67.5 | 31.8 / 25.2 |
| `Gf16` | `mul_add` | 36.6 / 52.7 | 32 / 48.5 | 6.95 / 8.18 |
| `Gf16` | `mul_add_with` | 37.2 / 53.3 | 32 / 49 | 6.96 / 8.17 |
| `Gf16` | `mul_into` | 49.9 / 81.1 | 36.6 / 61.5 | 6.81 / 9.5 |
| `Gf16` | `mul_assign` | 55 / 82.7 | 48.5 / 76.9 | 28.8 / 31.9 |
| `Gf16` | `mul_elementwise` | 33.4 / 34.8 | 23 / 29.3 | 4.64 / 5.16 |
| `Gf16` | `mul_elementwise_assign` | 34.3 / 35.2 | 27.6 / 30.7 | 5.56 / 7.12 |
| `Mersenne31` | `add_assign` | 29.3 / 34.8 | 23.1 / 35.7 | 8.26 / 21.4 |
| `Mersenne31` | `add_assign_scalar` | 40.1 / 52.2 | 41.2 / 53 | 24.7 / 21.7 |
| `Mersenne31` | `sub_assign_scalar` | 32.4 / 42.8 | 33.9 / 43.5 | 23 / 25.5 |
| `Mersenne31` | `mul_elementwise` | 17 / 21.9 | 16.3 / 22.6 | 5.74 / 19 |
| `Mersenne31` | `mul_elementwise_assign` | 17.9 / 22.9 | 16.9 / 23.6 | 7.96 / 22.6 |
| `Goldilocks` | `add_assign` | 30 / 18.5 | 27.3 / 18.5 | 8.18 / 17.6 |
| `Goldilocks` | `add_assign_scalar` | 44.2 / 24.3 | 44.4 / 23.9 | 25.9 / 23.8 |
| `Goldilocks` | `sub_assign_scalar` | 59.8 / 25.2 | 52.7 / 25.4 | 27.9 / 25.4 |
| `Goldilocks` | `mul_elementwise` | 10.8 / 10.5 | 11.3 / 11.1 | 5.59 / 10.5 |
| `Goldilocks` | `mul_elementwise_assign` | 11 / 11.1 | 11.3 / 11.1 | 7.32 / 9.91 |
| `QuadMersenne31` | `add_assign` | 20.3 / 38.7 | 18.5 / 33 | 7.77 / 25.2 |
| `QuadMersenne31` | `add_assign_scalar` | 29.6 / 52.8 | 27.3 / 51.4 | 19.1 / 26.9 |
| `QuadMersenne31` | `sub_assign_scalar` | 23.5 / 41 | 22.6 / 43.4 | 16.9 / 33.1 |
| `QuadMersenne31` | `mul_elementwise` | 3.13 / 5.78 | 3.12 / 5.81 | 3.17 / 5.73 |
| `QuadMersenne31` | `mul_elementwise_assign` | 3.29 / 5.86 | 3.29 / 5.9 | 3.24 / 5.52 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Operation | Throughput (GiB/s) |
| --- | --- | --- |
| `Gf8B` | `mul_add` | 40.7 / 64.8 |
| `Gf8B` | `mul_into` | 39.8 / 64.7 |
| `Gf8D` | `mul_add` | 40.7 / 64.7 |
| `Gf8D` | `mul_into` | 39.8 / 64.7 |
| `Gf16` | `mul_add` | 40.1 / 55.7 |
| `Gf16` | `mul_into` | 36 / 63.6 |
| `Gf8B` | `mul_elementwise` | 29.1 / 54.1 |
| `Gf8B` | `mul_elementwise_assign` | 40.7 / 65.7 |
| `Gf8D` | `mul_elementwise` | 28.8 / 49.5 |
| `Gf8D` | `mul_elementwise_assign` | 40.6 / 56.7 |

### Wider binary fields

256 KiB `mul_add`, ordinary `Vec<u8>` buffers.

| Field | Throughput (GiB/s) |
| --- | --- |
| `Gf32` | 17.1 / 31.4 |
| `Gf64` | 9.09 / 17 |
| `FanPaar16` | 9.97 / 17.2 |
| `FanPaar32` | 3.71 / 6.34 |
| `FanPaar64` | 1.36 / 2.21 |

### One-shot and prepared multiplication

Ordinary `Vec<u8>` buffers; latency per call. Preparation is excluded.

| Row bytes | Gf8B `mul_add` (ns/op) | Gf8B `mul_add_with` (ns/op) | Gf16 `mul_add` (ns/op) | Gf16 `mul_add_with` (ns/op) |
| --- | --- | --- | --- | --- |
| 16 | 14.5 / 4.22 | 14.1 / 4.19 | 14.2 / 6.59 | 12.5 / 5.97 |
| 32 | 15.5 / 4.41 | 15.1 / 4.16 | 14 / 6.59 | 12.6 / 5.94 |
| 64 | 15.6 / 4.47 | 15.4 / 5.12 | 12.4 / 8.16 | 11.2 / 7.41 |
| 128 | 15.8 / 5.81 | 15.6 / 5.78 | 13.1 / 9.16 | 11.8 / 8.41 |
| 256 | 16.3 / 6.94 | 16 / 6.97 | 14.6 / 10.6 | 13.4 / 8.5 |
| 512 | 17.2 / 8.47 | 17.3 / 8.31 | 16.2 / 12.3 | 14.9 / 12 |
| 1024 | 22 / 13.5 | 21.8 / 13.4 | 27.5 / 20 | 25.9 / 19.4 |
| 2048 | 28.8 / 23.8 | 28.8 / 23.8 | 41.7 / 35.2 | 40.7 / 34.7 |
| 4096 | 60.4 / 45.3 | 60.1 / 45.2 | 108 / 70.8 | 106 / 69.9 |
| 16384 | 209 / 169 | 209 / 170 | 363 / 267 | 362 / 267 |
| 65536 | 1.63e+03 / 1.05e+03 | 1.63e+03 / 1.05e+03 | 1.93e+03 / 1.25e+03 | 1.93e+03 / 1.25e+03 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one source; gather
has one destination; matrix has eight sources. Column counts refer to
destinations for scatter/matrix and sources for gather.

| Field | Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `Gf8B` | `mul_add_scatter` | 45.5 / 69.9 | 48.2 / 72.2 | 48.3 / 72.2 | 46.3 / 48 |
| `Gf8B` | `mul_add_scatter_with` | 45.5 / 69.9 | 48.2 / 72.2 | 48.3 / 72.2 | 46.3 / 48.1 |
| `Gf8B` | `mul_add_gather` | 44.5 / 71 | 43 / 79.7 | 43.7 / 69.7 | 44.8 / 64.5 |
| `Gf8B` | `mul_add_gather_with` | 44.4 / 71.4 | 43 / 79.9 | 43.6 / 69.7 | 44.9 / 64.6 |
| `Gf8B` | `mul_add_matrix` | 79.2 / 103 | 110 / 114 | 109 / 114 | 106 / 114 |
| `Gf8B` | `mul_add_matrix_with` | 75.4 / 103 | 102 / 114 | 101 / 114 | 98.3 / 113 |
| `Gf16` | `mul_add_scatter` | 30.6 / 56.2 | 28.4 / 64.9 | 28.5 / 62.6 | 27.5 / 39.3 |
| `Gf16` | `mul_add_scatter_with` | 30.5 / 56.2 | 28.4 / 64 | 28.5 / 61.7 | 27.4 / 39.3 |
| `Gf16` | `mul_add_gather` | 33.8 / 66.7 | 39.5 / 77.7 | 38.6 / 78.2 | 40.9 / 72.7 |
| `Gf16` | `mul_add_gather_with` | 33.8 / 65.9 | 39.5 / 77.7 | 38.6 / 78.1 | 40.9 / 73.4 |
| `Gf16` | `mul_add_matrix` | 46.3 / 53.2 | 56.5 / 56.5 | 56.3 / 56 | 56.2 / 56.5 |
| `Gf16` | `mul_add_matrix_with` | 45.7 / 53.5 | 55.2 / 55.8 | 55.1 / 55.1 | 55.1 / 55.4 |

### Short multi-row operations

`Gf16`, ordinary `Vec<u8>` buffers; coefficient sets include zero and one.

| Operation | Sources × destinations | 64 B (GiB/s) | 256 B (GiB/s) | 1 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `mul_add_scatter` | 1 × 16 | 4.08 / 8.71 | 13.1 / 27.4 | 27.4 / 57.2 | 38.1 / 65 |
| `mul_add_gather` | 8 × 1 | 6.89 / 14.3 | 19.2 / 36.8 | 31.1 / 58.4 | 38.6 / 72.2 |
| `mul_add_matrix` | 8 × 16 | 4 / 29.5 | 13.8 / 48.5 | 34.6 / 57 | 55.5 / 58.4 |
| `mul_into` | 1 × 1 | 2.76 / 7.42 | 9.77 / 25.1 | 30.3 / 51.3 | 55.8 / 82 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8B` | `mul_into_gather_with` | 81.8 / 84.3 | 91.1 / 88.1 |
| `Gf8D` | `mul_into_gather` | 81.3 / 106 | 92.4 / 111 |
| `Gf8D` | `mul_into_gather_with` | 86.5 / 105 | 98 / 111 |
| `Gf8D` | `mul_add_gather_with` | 79.1 / 86.8 | 95 / 91.8 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous
row offsets. The `fill(0)` cases include clearing the destination.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `fill(0)` + `mul_add_matrix` | 47.8 / 50.1 | 26.2 / 27.4 | 16.7 / 17.6 |
| `Gf8B` | `mul_into_matrix` | 57.3 / 60.5 | 32.2 / 34.8 | 20.9 / 22.3 |
| `Gf8B` | `mul_add_matrix_with` | 54.6 / 57.2 | 30.4 / 32.5 | 19.4 / 20.6 |
| `Gf8B` | `mul_into_matrix_with` | 57.4 / 60.8 | 32.3 / 35.5 | 20.4 / 22.2 |
| `Gf8B` | `mul_add_matrix_at` | 58 / 56.9 | 32 / 32.5 | 20.7 / 20.6 |
| `Gf8D` | `fill(0)` + `mul_add_matrix` | 46.8 / 52.1 | 26.3 / 29.7 | 16.6 / 18.8 |
| `Gf8D` | `mul_into_matrix` | 56.3 / 63.5 | 32.1 / 39.3 | 20.5 / 24.1 |

### Bit-packed GF(2)

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where
`N` is the number of packed bits.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.6 / 88.3 | 41.4 / 66.1 | 8.76 / 25.7 | 6.09 / 14.8 |
| `bits::and_into` | 22.2 / 68 | 19.3 / 47.1 | 5.83 / 17.7 | 4.62 / 10.9 |
| `bits::weight` | 109 / 14.5 | 67.9 / 15 | 38.6 / 15 | 18 / 14.4 |
| `bits::dot_product` | 18 / 39.7 | 17.8 / 43.5 | 10.6 / 32.2 | 8.75 / 18.4 |
| `bits::xor_range` | 53.6 / 90 | 65.9 / 74.2 | 21.8 / 39.8 | 9.75 / 24.4 |
| `bits::xor_range_with` | 57.8 / 100 | 66.1 / 74.5 | 21.9 / 39.8 | 9.83 / 24.8 |

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.7 / 5.39 |
| `bits::xor_range_with` | 16-byte rows; bits `[61, 69)` | 4.45 / 2.07 |

### Row-wise addition

`add_assign_rows` on ordinary `Vec<u8>` buffers.

| Row bytes | Rows | Gf8B (GiB/s) | Gf16 (GiB/s) | Mersenne31 (GiB/s) |
| --- | --- | --- | --- | --- |
| 64 | 1 | 6.6 / 13.5 | 5.02 / 12.1 | 5.23 / 9.44 |
| 64 | 2 | 12.8 / 27.6 | 9.91 / 22.8 | 9.58 / 16 |
| 64 | 4 | 24.1 / 46.2 | 20.1 / 40.8 | 14.5 / 24.2 |
| 64 | 8 | 37.8 / 58.5 | 30.7 / 62 | 23.1 / 30.2 |
| 64 | 16 | 53.5 / 69 | 40.4 / 76.5 | 27 / 33.2 |
| 64 | 32 | 65.1 / 89.5 | 58.2 / 86.1 | 26.4 / 36.3 |
| 1024 | 1 | 53.6 / 80.5 | 40.4 / 76.1 | 26.9 / 33.3 |
| 1024 | 2 | 65.2 / 89.6 | 58.2 / 86.2 | 26.3 / 36.3 |
| 1024 | 4 | 42.4 / 95.8 | 40.8 / 93.5 | 28.7 / 34.5 |
| 1024 | 8 | 79.9 / 131 | 77.5 / 125 | 29.7 / 36.2 |
| 1024 | 16 | 83.7 / 131 | 82.4 / 131 | 30.2 / 37.1 |
| 1024 | 32 | 34.9 / 66.5 | 34.7 / 66.2 | 29.5 / 35.2 |
| 65536 | 1 | 31.8 / 63.2 | 31.7 / 63.2 | 27.9 / 34.2 |
| 65536 | 2 | 31.9 / 63.5 | 31.9 / 63.5 | 29.7 / 38.4 |
| 65536 | 4 | 32 / 63.7 | 30.7 / 63.7 | 26.1 / 36.7 |
| 65536 | 8 | 36.7 / 46.6 | 30.7 / 44.1 | 24.4 / 32.7 |
| 65536 | 16 | 18.8 / 26.4 | 18.8 / 25.8 | 18.6 / 22.4 |
| 65536 | 32 | 18.5 / 25.8 | 18.2 / 24.8 | 18.4 / 21.5 |

### Network-size payloads

`Gf8B`, ordinary `Vec<u8>` buffers; scatter rows use the payload length as
the row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 7.09 / 13.2 | 3.65 / 12.5 | 6.51 / 17.3 | 8.49 / 14.4 | 11.2 / 19.6 |
| 256 | 26.9 / 45.7 | 14.6 / 39.7 | 22.7 / 54.5 | 26.9 / 34.7 | 30.9 / 63.9 |
| 512 | 43.2 / 69.7 | 27.4 / 57.6 | 51.7 / 100 | 35.8 / 63.7 | 42.4 / 83.6 |
| 1152 | 53.6 / 82.9 | 46 / 72 | 75.1 / 135 | 78.6 / 102 | 85.7 / 112 |
| 1168 | 59 / 97.8 | 48.1 / 79.3 | 82.9 / 135 | 58.7 / 85.2 | 57.9 / 92 |
| 1184 | 55 / 84 | 46.5 / 72.5 | 76 / 135 | 64.7 / 101 | 65 / 112 |
| 1200 | 55.3 / 97.2 | 48.3 / 81.3 | 79.3 / 135 | 51.7 / 85.8 | 56 / 91.2 |
| 1216 | 55.7 / 83.5 | 47.6 / 71.9 | 77.6 / 135 | 82.6 / 100 | 88.4 / 112 |
| 1232 | 59.5 / 100 | 50.6 / 81 | 84.6 / 136 | 60.2 / 86.8 | 57.8 / 94.2 |
| 1248 | 57 / 84.5 | 48.3 / 72.9 | 78.5 / 136 | 66.1 / 101 | 66.8 / 112 |
| 1400 | 60.5 / 103 | 40.3 / 68.5 | 63.7 / 117 | 43.6 / 60.6 | 45.4 / 62.8 |

### Fixed-offset multi-row operations

`Gf8D`; every source and destination base has the stated offset modulo
64. Scatter has four destinations, gather has sixteen sources, and matrix
has ten sources and four destinations.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 31 / 46.3 | 19.9 / 53.1 | 25.8 / 32.2 |
| 256 | 16 | 27.4 / 38 | 18.5 / 45.5 | 23.3 / 29.8 |
| 512 | 0 | 52.4 / 74.7 | 34.8 / 80.3 | 30.3 / 39.1 |
| 512 | 16 | 36.2 / 70.4 | 31.6 / 71 | 27.9 / 35.1 |
| 1024 | 0 | 79.4 / 98.2 | 55.6 / 100 | 33.7 / 43.7 |
| 1024 | 16 | 59.1 / 92.9 | 48.9 / 85.3 | 30.7 / 38.6 |
| 2048 | 0 | 95 / 110 | 79.1 / 108 | 35.2 / 45.9 |
| 2048 | 16 | 85.1 / 106 | 66.9 / 92.2 | 31.8 / 40.2 |
| 3072 | 0 | 104 / 117 | 78.8 / 91.8 | 35.6 / 42.2 |
| 3072 | 16 | 92.4 / 116 | 60.5 / 74.3 | 31.7 / 39.3 |
| 4096 | 0 | 89.8 / 113 | 76.5 / 88.8 | 35.5 / 39.4 |
| 4096 | 16 | 81.1 / 109 | 65.1 / 72.6 | 30.3 / 36.8 |
| 8192 | 0 | 92.1 / 119 | 85.6 / 91.3 | 34.7 / 39.4 |
| 8192 | 16 | 88.3 / 115 | 78 / 74.3 | 30.6 / 37 |
| 16384 | 0 | 48.9 / 73.8 | 89.7 / 92.8 | 34.9 / 40.1 |
| 16384 | 16 | 48.5 / 73.5 | 85.2 / 75.5 | 32.8 / 37.8 |

### Fixed-offset single-row multiplication

`Gf8D` `mul_add`, 16 KiB. Both source and destination start at the stated
offset from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 93.6 / 88.8 |
| 16 | 93.3 / 102 |
| 32 | 93.5 / 103 |
| 48 | 93.7 / 102 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at
the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Gf8B `mul_add_scatter` (GiB/s) | Gf16 `mul_add_scatter` (GiB/s) |
| --- | --- | --- | --- |
| 64 KiB | 0 | 48.2 / 72 | 26.9 / 58.9 |
| 64 KiB | 16 | 48.2 / 72.1 | 29.1 / 61.6 |
| 256 KiB | 0 | 29.6 / 24.1 | 25.6 / 24.6 |
| 256 KiB | 16 | 29.6 / 23.8 | 26.8 / 24.9 |

`Gf8D`, one source and four 16 KiB destination rows, 64-byte-aligned buffers:

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.4 / 73.8 |
| `mul_add_scatter_with` | 48.4 / 73.7 |

### In-place scaling by buffer size

64-byte-aligned buffers.

| Buffer size | Gf8B `mul_assign` (GiB/s) | Gf8D `mul_assign` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 127 / 175 | 126 / 172 |
| 16 KiB | 137 / 187 | 138 / 188 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 69 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 69.2 |
| 1 MiB | 49 / 55.5 | 50.8 / 51.7 |

### Large destinations

Ordinary `Vec<u8>` buffers. Read-back XOR-folds all written 64-bit words.

| Buffer size | Gf8B `mul_into` (GiB/s) | Gf8B `mul_into` + read (GiB/s) | Gf8B `mul_add` (GiB/s) | Gf16 `mul_into` (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 MiB | 18.8 / 26.5 | 13.8 / 20.8 | 18.8 / 27 | 18.4 / 24.1 |
| 8 MiB | 18.2 / 41.8 | 7.67 / 16.7 | 8.27 / 25.1 | 8.18 / 41.7 |
| 32 MiB | 9.31 / 28.2 | 5.94 / 14.3 | 6.26 / 16.5 | 6.28 / 27.6 |

| Field | Operation | Buffer size | Alignment | Throughput (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8D` | `mul_into` | 4 MiB | 64-byte-aligned | 18.2 / 41.8 |

## vs Competitor timings

### GF(256), polynomial `0x11D`

Page-aligned buffers and matched coding coefficients. ISA-L uses its
field kernels; klauspost uses `Encode` with a supplied coding matrix, or
`EncodeIdx` for accumulation. Preparation and Go bridge pointer staging
are excluded.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `dst = c * src` | 1 × 1 | 64 KiB | 40 / 64.6 | 17.9 / 45.6 | 33.7 / 57.2 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.9 / 64.9 | 41.2 / 65.2 | 29.1 / 49.2 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 82.6 / 104 | 82.8 / 98 | 30.6 / 50.9 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 89.8 / 106 | 75 / 90.2 | 35.7 / 57.4 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 64.6 / 73.5 | 65.1 / 69.4 | 53.4 / 74.9 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 63.5 / 72 | 53.1 / 65.5 | 67.5 / 84.8 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 58.4 / 72.6 | 54 / 63.9 | 73.5 / 89.3 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 34.4 / 40.1 | 35.6 / 34.9 | 31.9 / 40.5 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 35.1 / 41.1 | 36.7 / 29.9 | 38.5 / 44.8 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 33.1 / 39.6 | 35.1 / 31.7 | 40.6 / 45.4 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 22.1 / 26.4 | 24.3 / 24.5 | 22.6 / 27.1 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 22.6 / 26.2 | 25 / 22.1 | 25.8 / 30 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 20.6 / 25.7 | 23.7 / 23.6 | 26.6 / 30.4 |

### Prime fields

Page-aligned 64 KiB regions on each library's native layout. Throughput
counts one operand pair (twice the region bytes); conversions are excluded.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.18 / 1.55 | 2.09 / 0.8 |
| `Goldilocks` | `acc += a * b` | 3.74 / 1.64 | 2.57 / 1.11 |
| `QuadMersenne31` | `acc += a * b` | 8.97 / 4.41 | 6.9 / 2.91 |

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 48 / 58.6 | 65.4 / 72.7 |
| `Mersenne31` | `dst += c * src` | 33.5 / 42.5 | 49.8 / 57 |
| `Mersenne31` | `dst = a * b` | 37.1 / 47.1 | 59.8 / 69.4 |
| `Mersenne31` | `dst += v` | 87.6 / 108 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 70.3 / 86.4 | 103 / 153 |
| `Goldilocks` | `dst = c * src` | 25.1 / 24.1 | 27.4 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17 | 22.5 / 20.7 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.3 | 26.9 / 26.4 |
| `Goldilocks` | `dst += v` | 97.5 / 49.5 | 103 / 148 |
| `Goldilocks` | `dst -= v` | 103 / 52.7 | 103 / 148 |
| `QuadMersenne31` | `dst = c * src` | 14.8 / 25.9 | 30.6 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 12 / 20.9 | 27 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 7.16 / 12.3 | 24.1 / 26.4 |
| `QuadMersenne31` | `dst += v` | 62.6 / 108 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 49.9 / 86.4 | 103 / 153 |

No matched competitor measurements are available for bit-packed GF(2)
or the wider binary fields.
