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
| Resolved backend | Process and byte fields: `v4x` / `v3_gfni_crypto`; Mersenne31 and Goldilocks: `v4x` / `v3`; QuadMersenne31: `v4x` / `v3` |
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
| `Gf8D` | Dependent multiply chain | 0.61 / 0.34 |
| `Gf16` | Dependent multiply chain | 2.44 / 1.35 |
| `Mersenne31` | Indexed products with accumulation | 4.22 / 1.55 |
| `Goldilocks` | Indexed products with accumulation | 3.74 / 1.64 |
| `QuadMersenne31` | Indexed products with accumulation | 8.97 / 4.4 |

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `add_assign` | 42.8 / 96.7 | 32 / 63.7 | 7.65 / 10.8 |
| `Gf8B` | `add_assign_scalar` | 42.8 / 80.2 | 41.5 / 67.6 | 26.2 / 25.6 |
| `Gf8B` | `sub_assign_scalar` | 43 / 80.3 | 41.5 / 67.5 | 27.3 / 25.4 |
| `Gf8B` | `mul_add` | 64 / 84.1 | 37.9 / 58.6 | 8.3 / 16.9 |
| `Gf8B` | `mul_into` | 81.6 / 163 | 38.2 / 64.7 | 12.9 / 29.1 |
| `Gf8B` | `mul_assign` | 124 / 171 | 51.5 / 76.6 | 29.4 / 29 |
| `Gf8B` | `mul_elementwise` | 73.4 / 108 | 26.3 / 42.4 | 5.08 / 7.48 |
| `Gf8B` | `mul_elementwise_assign` | 79.8 / 106 | 31.2 / 50.6 | 6.61 / 10 |
| `Gf16` | `add_assign` | 43 / 96.2 | 32 / 63.7 | 7.56 / 10.8 |
| `Gf16` | `add_assign_scalar` | 41.5 / 78.3 | 41.5 / 67.5 | 26.4 / 25.2 |
| `Gf16` | `sub_assign_scalar` | 41.4 / 78.3 | 41.5 / 67.5 | 27.1 / 25.1 |
| `Gf16` | `mul_add` | 36.9 / 52.7 | 32 / 48.4 | 7.03 / 8.17 |
| `Gf16` | `mul_add_with` | 38 / 53.6 | 32 / 48.4 | 7.06 / 8.17 |
| `Gf16` | `mul_into` | 49.6 / 81.1 | 36.6 / 61.7 | 6.71 / 9.51 |
| `Gf16` | `mul_assign` | 54.2 / 82.4 | 48.5 / 76.9 | 26.4 / 31.9 |
| `Gf16` | `mul_elementwise` | 33.5 / 34.8 | 23 / 29.7 | 4.56 / 5.13 |
| `Gf16` | `mul_elementwise_assign` | 34.4 / 35.1 | 27.7 / 30.7 | 5.34 / 7.11 |
| `Mersenne31` | `add_assign` | 29.5 / 34.8 | 23.2 / 35.6 | 8.21 / 21.5 |
| `Mersenne31` | `add_assign_scalar` | 40.1 / 52.3 | 41.2 / 53 | 23.5 / 21.8 |
| `Mersenne31` | `sub_assign_scalar` | 32.5 / 41.2 | 34.1 / 43.5 | 22.1 / 25.3 |
| `Mersenne31` | `mul_elementwise` | 22.8 / 26.6 | 20.6 / 30.4 | 5.82 / 19.9 |
| `Mersenne31` | `mul_elementwise_assign` | 25 / 32.7 | 19.1 / 34.1 | 8.28 / 22.7 |
| `Goldilocks` | `add_assign` | 29.8 / 22.4 | 27.2 / 22.5 | 8.25 / 21.9 |
| `Goldilocks` | `add_assign_scalar` | 44.8 / 40.9 | 44.4 / 40.9 | 25.3 / 33.8 |
| `Goldilocks` | `sub_assign_scalar` | 59.1 / 39.2 | 52.7 / 40.2 | 27 / 33.9 |
| `Goldilocks` | `mul_elementwise` | 10.8 / 10.6 | 11.3 / 11.1 | 5.59 / 10.5 |
| `Goldilocks` | `mul_elementwise_assign` | 11 / 11.1 | 10.2 / 11.1 | 7.18 / 9.9 |
| `QuadMersenne31` | `add_assign` | 20.3 / 38.6 | 17.7 / 33 | 7.81 / 25.2 |
| `QuadMersenne31` | `add_assign_scalar` | 29.4 / 52.8 | 26.9 / 51.1 | 18.2 / 26.8 |
| `QuadMersenne31` | `sub_assign_scalar` | 23.7 / 42.4 | 22.9 / 43.4 | 15.5 / 33.1 |
| `QuadMersenne31` | `mul_elementwise` | 11.8 / 14.8 | 11.7 / 14.8 | 5.22 / 13.9 |
| `QuadMersenne31` | `mul_elementwise_assign` | 7.23 / 15 | 7.38 / 15 | 6.48 / 14.2 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Operation | Throughput (GiB/s) |
| --- | --- | --- |
| `Gf8B` | `mul_add` | 40.7 / 64.8 |
| `Gf8B` | `mul_into` | 39.8 / 64.8 |
| `Gf8D` | `mul_add` | 40.7 / 64.8 |
| `Gf8D` | `mul_into` | 39.8 / 64.7 |
| `Gf16` | `mul_add` | 40.1 / 56.6 |
| `Gf16` | `mul_into` | 36 / 63.6 |
| `Gf8B` | `mul_elementwise` | 29 / 54.2 |
| `Gf8B` | `mul_elementwise_assign` | 40.7 / 65.6 |
| `Gf8D` | `mul_elementwise` | 28.8 / 50.4 |
| `Gf8D` | `mul_elementwise_assign` | 40.6 / 56.7 |

### Wider binary fields

256 KiB `mul_add`, ordinary `Vec<u8>` buffers.

| Field | Throughput (GiB/s) |
| --- | --- |
| `Gf32` | 17.1 / 30.7 |
| `Gf64` | 9.09 / 17 |
| `FanPaar16` | 9.97 / 17.2 |
| `FanPaar32` | 3.71 / 6.34 |
| `FanPaar64` | 1.37 / 2.21 |

### One-shot and prepared multiplication

Ordinary `Vec<u8>` buffers; latency per call. Preparation is excluded.

| Row bytes | Gf8B `mul_add` (ns/op) | Gf8B `mul_add_with` (ns/op) | Gf16 `mul_add` (ns/op) | Gf16 `mul_add_with` (ns/op) |
| --- | --- | --- | --- | --- |
| 16 | 14.5 / 4.12 | 14.2 / 4.12 | 13.7 / 6.62 | 12.5 / 6 |
| 32 | 15.4 / 4.31 | 15.1 / 4.19 | 13.8 / 6.59 | 12.6 / 5.97 |
| 64 | 15.8 / 4.47 | 15.5 / 4.5 | 12.8 / 8.16 | 11.2 / 7.44 |
| 128 | 16 / 5.81 | 15.7 / 5.78 | 14 / 9.16 | 12.2 / 8.38 |
| 256 | 16.4 / 6.97 | 16 / 6 | 16 / 9.22 | 14.7 / 8.5 |
| 512 | 17.3 / 8.22 | 17.3 / 8.03 | 18.7 / 12.2 | 17.2 / 12.1 |
| 1024 | 22 / 13.5 | 21.8 / 13.3 | 26.8 / 20 | 26.2 / 19.4 |
| 2048 | 30.9 / 23.8 | 30.9 / 23.8 | 43.5 / 35.3 | 43.2 / 34.8 |
| 4096 | 64.5 / 45.3 | 63.8 / 45.2 | 109 / 70.9 | 108 / 69.9 |
| 16384 | 213 / 170 | 212 / 170 | 364 / 267 | 363 / 267 |
| 65536 | 1.63e+03 / 1.05e+03 | 1.63e+03 / 1.05e+03 | 1.93e+03 / 1.26e+03 | 1.93e+03 / 1.26e+03 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one source; gather
has one destination; matrix has eight sources. Column counts refer to
destinations for scatter/matrix and sources for gather.

| Field | Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `Gf8B` | `mul_add_scatter` | 45.5 / 69.9 | 48.2 / 72.2 | 48.2 / 72.3 | 45.9 / 44.5 |
| `Gf8B` | `mul_add_scatter_with` | 45.5 / 69.9 | 48.2 / 72.2 | 48.2 / 72.3 | 46.2 / 44.4 |
| `Gf8B` | `mul_add_gather` | 44.3 / 71.1 | 42.9 / 79.6 | 43.7 / 69.6 | 43.4 / 64.9 |
| `Gf8B` | `mul_add_gather_with` | 44.2 / 71.4 | 42.9 / 79.9 | 43.7 / 69.6 | 43.4 / 65 |
| `Gf8B` | `mul_add_matrix` | 89.5 / 105 | 132 / 114 | 126 / 113 | 123 / 112 |
| `Gf8B` | `mul_add_matrix_with` | 89.3 / 105 | 132 / 115 | 123 / 114 | 117 / 113 |
| `Gf16` | `mul_add_scatter` | 30.6 / 56.2 | 28.5 / 65 | 27.3 / 65.3 | 26.9 / 37.8 |
| `Gf16` | `mul_add_scatter_with` | 30.5 / 56.2 | 28.4 / 65.7 | 27.3 / 64.5 | 26.9 / 37.5 |
| `Gf16` | `mul_add_gather` | 33.9 / 66.5 | 39.6 / 78.1 | 38.6 / 77.2 | 39.7 / 73.1 |
| `Gf16` | `mul_add_gather_with` | 33.8 / 65.8 | 39.6 / 78 | 38.5 / 77 | 39.8 / 73.1 |
| `Gf16` | `mul_add_matrix` | 46.4 / 51.9 | 56.1 / 56.3 | 55.5 / 56.4 | 55 / 56.8 |
| `Gf16` | `mul_add_matrix_with` | 45.8 / 52 | 55.3 / 54.5 | 54.9 / 54.5 | 54.1 / 54.8 |

### Short multi-row operations

`Gf16`, ordinary `Vec<u8>` buffers; coefficient sets include zero and one.

| Operation | Sources × destinations | 64 B (GiB/s) | 256 B (GiB/s) | 1 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `mul_add_scatter` | 1 × 16 | 4.13 / 8.57 | 13.1 / 27.5 | 27.9 / 57.3 | 38.1 / 65.1 |
| `mul_add_gather` | 8 × 1 | 6.8 / 13.2 | 18.8 / 36.7 | 31.4 / 53.9 | 38.7 / 73.2 |
| `mul_add_matrix` | 8 × 16 | 4.07 / 29.5 | 13.8 / 48.5 | 34.8 / 56.9 | 55.1 / 58.6 |
| `mul_into` | 1 × 1 | 2.75 / 7.42 | 9.63 / 25.1 | 29.8 / 51.3 | 55.7 / 81.9 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8B` | `mul_into_gather_with` | 91.8 / 83.9 | 102 / 88.2 |
| `Gf8D` | `mul_into_gather` | 90.5 / 106 | 102 / 100 |
| `Gf8D` | `mul_into_gather_with` | 90.8 / 106 | 102 / 100 |
| `Gf8D` | `mul_add_gather_with` | 79.2 / 87.1 | 95 / 91.8 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous
row offsets. The `fill(0)` cases include clearing the destination.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `fill(0)` + `mul_add_matrix` | 56.2 / 50.5 | 31.1 / 26.7 | 19.2 / 17.2 |
| `Gf8B` | `mul_into_matrix` | 72.7 / 62.2 | 42 / 33.1 | 25.7 / 21.6 |
| `Gf8B` | `mul_add_matrix_with` | 72.8 / 56.7 | 40 / 30.5 | 24.1 / 19.8 |
| `Gf8B` | `mul_into_matrix_with` | 75.4 / 62.8 | 42.3 / 33.3 | 25.1 / 21.7 |
| `Gf8B` | `mul_add_matrix_at` | 72.8 / 56.7 | 40.3 / 30.7 | 24.3 / 19.9 |
| `Gf8D` | `fill(0)` + `mul_add_matrix` | 56.6 / 52 | 31.1 / 28.7 | 19.1 / 18.2 |
| `Gf8D` | `mul_into_matrix` | 75.2 / 66.1 | 42.4 / 36.9 | 25.4 / 23.7 |

### Bit-packed GF(2)

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where
`N` is the number of packed bits.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.4 / 88 | 41.4 / 66.1 | 8.46 / 25.7 | 6.05 / 14.7 |
| `bits::and_into` | 22.4 / 67.5 | 19.3 / 45.4 | 5.87 / 18.3 | 4.61 / 10.8 |
| `bits::weight` | 113 / 14.5 | 67.9 / 15 | 38.2 / 15 | 18 / 14.4 |
| `bits::dot_product` | 20.3 / 46.1 | 18.3 / 46.7 | 10.8 / 36.1 | 8.73 / 18.9 |
| `bits::xor_range` | 53.3 / 90.2 | 65.9 / 74.5 | 20.3 / 39.8 | 9.62 / 23.9 |
| `bits::xor_range_with` | 57.9 / 100 | 66 / 74.5 | 20.4 / 39.9 | 9.75 / 24.2 |

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.9 / 5.39 |
| `bits::xor_range_with` | 16-byte rows; bits `[61, 69)` | 4.41 / 2.08 |

### Row-wise addition

`add_assign_rows` on ordinary `Vec<u8>` buffers.

| Row bytes | Rows | Gf8B (GiB/s) | Gf16 (GiB/s) | Mersenne31 (GiB/s) |
| --- | --- | --- | --- | --- |
| 64 | 1 | 6.58 / 14.4 | 5.02 / 11.8 | 5.31 / 9.4 |
| 64 | 2 | 12.7 / 27.2 | 9.96 / 21.7 | 9.86 / 16 |
| 64 | 4 | 24.2 / 43.9 | 20.1 / 40.4 | 17 / 22.1 |
| 64 | 8 | 35.8 / 57.6 | 30.3 / 62.8 | 24 / 30.3 |
| 64 | 16 | 53.4 / 80.3 | 40.7 / 77.3 | 26.3 / 33.4 |
| 64 | 32 | 66.3 / 89.2 | 58.4 / 86 | 26.8 / 36.3 |
| 1024 | 1 | 53.1 / 80.3 | 40.6 / 77.5 | 26.3 / 33.3 |
| 1024 | 2 | 66.5 / 89.2 | 58.2 / 85.8 | 26.8 / 36.4 |
| 1024 | 4 | 42.5 / 95.4 | 40.9 / 93.6 | 28.7 / 34.5 |
| 1024 | 8 | 80.5 / 128 | 76.2 / 126 | 29.7 / 35.5 |
| 1024 | 16 | 84 / 133 | 81.6 / 130 | 30.3 / 35.9 |
| 1024 | 32 | 34.9 / 66.3 | 34.6 / 66 | 29.6 / 34.3 |
| 65536 | 1 | 31.7 / 63.2 | 31.7 / 63.2 | 27.9 / 35.7 |
| 65536 | 2 | 31.9 / 63.5 | 31.9 / 63.5 | 29.7 / 38.4 |
| 65536 | 4 | 30.7 / 63.6 | 32 / 63.6 | 26.1 / 35.5 |
| 65536 | 8 | 38.5 / 47.2 | 30.1 / 42.2 | 25.3 / 33 |
| 65536 | 16 | 18.3 / 26.4 | 18 / 25.6 | 17.8 / 22.6 |
| 65536 | 32 | 17.9 / 25.8 | 17.7 / 25 | 17.6 / 21.6 |

### Network-size payloads

`Gf8B`, ordinary `Vec<u8>` buffers; scatter rows use the payload length as
the row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 7.06 / 13.5 | 3.69 / 13.6 | 6.76 / 11.8 | 8.47 / 14.5 | 11.2 / 19.6 |
| 256 | 26.1 / 40.6 | 14.8 / 34.2 | 23.5 / 47.7 | 26.9 / 34.5 | 30.6 / 63.9 |
| 512 | 38.3 / 68.7 | 26.9 / 57.8 | 50.2 / 90.8 | 35.4 / 63.7 | 42.6 / 83.3 |
| 1152 | 51.4 / 82.7 | 45.7 / 72 | 71.2 / 135 | 77 / 102 | 86.7 / 113 |
| 1168 | 56.5 / 98 | 47.2 / 80.2 | 75.3 / 133 | 58.6 / 84.8 | 57.4 / 92.1 |
| 1184 | 52.1 / 83 | 45.7 / 72.3 | 69.3 / 134 | 65.5 / 101 | 64.8 / 112 |
| 1200 | 54.6 / 98.2 | 48.2 / 81.3 | 68.5 / 131 | 51.5 / 85 | 56 / 91.2 |
| 1216 | 53.2 / 83.5 | 46.9 / 72 | 72.9 / 135 | 82.6 / 100 | 88.4 / 112 |
| 1232 | 58.8 / 101 | 49.1 / 81 | 72.1 / 128 | 60 / 86 | 57.7 / 94.1 |
| 1248 | 54.3 / 84.7 | 47.9 / 73.1 | 69.8 / 130 | 64.3 / 101 | 65.8 / 112 |
| 1400 | 60.3 / 102 | 40 / 68.4 | 62.5 / 116 | 43.4 / 60.7 | 45.9 / 62.7 |

### Fixed-offset multi-row operations

`Gf8D`; every source and destination base has the stated offset modulo
64. Scatter has four destinations, gather has sixteen sources, and matrix
has ten sources and four destinations.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 30.8 / 47.1 | 19.9 / 58.3 | 26.1 / 31.7 |
| 256 | 16 | 27.4 / 38.6 | 18.5 / 45.9 | 23.6 / 29.1 |
| 512 | 0 | 52.1 / 75.5 | 34.8 / 80.5 | 29.4 / 38.7 |
| 512 | 16 | 35.8 / 71.2 | 31.6 / 70.5 | 26.9 / 34.6 |
| 1024 | 0 | 78.9 / 98.7 | 55.5 / 100 | 35.4 / 43.5 |
| 1024 | 16 | 59.6 / 93.7 | 48.8 / 85.3 | 32.3 / 38.6 |
| 2048 | 0 | 95.2 / 111 | 79.1 / 105 | 39.5 / 45.6 |
| 2048 | 16 | 84.8 / 106 | 67.2 / 92.5 | 35.7 / 40.1 |
| 3072 | 0 | 103 / 118 | 78.6 / 92.4 | 40.6 / 41.9 |
| 3072 | 16 | 92.8 / 117 | 60.1 / 74.3 | 36.2 / 38.9 |
| 4096 | 0 | 90.1 / 113 | 76.8 / 88.8 | 40.7 / 39.3 |
| 4096 | 16 | 80.2 / 110 | 64.4 / 72.3 | 33.7 / 36.8 |
| 8192 | 0 | 92.1 / 118 | 85.9 / 91.2 | 41.6 / 40.1 |
| 8192 | 16 | 88.3 / 115 | 77.9 / 73.9 | 36.8 / 37.5 |
| 16384 | 0 | 48.8 / 73.7 | 88.5 / 93 | 42.2 / 41.6 |
| 16384 | 16 | 48.3 / 73.3 | 84.4 / 75.5 | 39.7 / 38.7 |

### Fixed-offset single-row multiplication

`Gf8D` `mul_add`, 16 KiB. Both source and destination start at the stated
offset from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 93.2 / 102 |
| 16 | 93.2 / 103 |
| 32 | 93.1 / 103 |
| 48 | 93.4 / 103 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at
the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Gf8B `mul_add_scatter` (GiB/s) | Gf16 `mul_add_scatter` (GiB/s) |
| --- | --- | --- | --- |
| 64 KiB | 0 | 48.2 / 72 | 27.6 / 61.4 |
| 64 KiB | 16 | 48.3 / 72.1 | 30 / 64.2 |
| 256 KiB | 0 | 28.6 / 23.7 | 24 / 24.7 |
| 256 KiB | 16 | 28 / 23.6 | 25.3 / 24.9 |

`Gf8D`, one source and four 16 KiB destination rows, 64-byte-aligned buffers:

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.5 / 73.7 |
| `mul_add_scatter_with` | 48.5 / 73.7 |

### In-place scaling by buffer size

64-byte-aligned buffers.

| Buffer size | Gf8B `mul_assign` (GiB/s) | Gf8D `mul_assign` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 126 / 174 | 125 / 171 |
| 16 KiB | 134 / 189 | 135 / 181 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 69 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 70.9 |
| 1 MiB | 49.6 / 55.4 | 47.5 / 51.5 |

### Large destinations

Ordinary `Vec<u8>` buffers. Read-back XOR-folds all written 64-bit words.

| Buffer size | Gf8B `mul_into` (GiB/s) | Gf8B `mul_into` + read (GiB/s) | Gf8B `mul_add` (GiB/s) | Gf16 `mul_into` (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 MiB | 18.2 / 26.6 | 13.2 / 21.1 | 18.4 / 27.1 | 17.8 / 24.2 |
| 8 MiB | 17.9 / 41.8 | 7.46 / 16.5 | 7.92 / 25.9 | 7.87 / 41.7 |
| 32 MiB | 9.41 / 27.3 | 5.94 / 14.1 | 6.27 / 16.3 | 6.13 / 27 |

| Field | Operation | Buffer size | Alignment | Throughput (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8D` | `mul_into` | 4 MiB | 64-byte-aligned | 18.1 / 41.8 |

## vs Competitor timings

### GF(256), polynomial `0x11D`

Page-aligned buffers and matched coding coefficients. ISA-L uses its
field kernels; klauspost uses `Encode` with a supplied coding matrix, or
`EncodeIdx` for accumulation. Preparation and Go bridge pointer staging
are excluded.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `dst = c * src` | 1 × 1 | 64 KiB | 40.1 / 64.7 | 17.8 / 45.5 | 33.7 / 56.6 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.8 / 65 | 41.2 / 65.1 | 29.1 / 49.2 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 90.9 / 104 | 82.9 / 97.9 | 30.6 / 51 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 95.1 / 97.9 | 74.9 / 89.9 | 35.5 / 57.1 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 74 / 73.6 | 65.2 / 69.6 | 53.3 / 75 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 78.8 / 80.7 | 53.3 / 65.3 | 67.3 / 85 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 83.7 / 75.8 | 54 / 64.1 | 73.3 / 88.5 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 39.3 / 40.2 | 35.6 / 35 | 31.8 / 40.4 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 42.6 / 45.8 | 36.7 / 29.8 | 38.2 / 45.1 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 42.1 / 43 | 33.9 / 31.5 | 39.7 / 45.8 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 26.2 / 26.5 | 24.3 / 24.5 | 22.5 / 27.9 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 26.4 / 29.3 | 25 / 22 | 25.7 / 30.2 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 25.8 / 27.4 | 23 / 23.6 | 26.1 / 30.4 |

### Prime fields

Page-aligned 64 KiB regions on each library's native layout. Throughput
counts one operand pair (twice the region bytes); conversions are excluded.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.22 / 1.55 | 2.1 / 0.85 |
| `Goldilocks` | `acc += a * b` | 3.74 / 1.64 | 2.57 / 1.11 |
| `QuadMersenne31` | `acc += a * b` | 8.97 / 4.4 | 6.99 / 2.91 |

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 73.3 / 76.8 | 65.5 / 72.7 |
| `Mersenne31` | `dst += c * src` | 55.6 / 61.1 | 49.8 / 57 |
| `Mersenne31` | `dst = a * b` | 59.7 / 68.2 | 59.8 / 69.4 |
| `Mersenne31` | `dst += v` | 87.6 / 108 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 70.3 / 86.4 | 103 / 153 |
| `Goldilocks` | `dst = c * src` | 25.1 / 23.8 | 27.5 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17 | 22.5 / 20.6 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.3 | 26.9 / 26.5 |
| `Goldilocks` | `dst += v` | 97.5 / 81.6 | 103 / 148 |
| `Goldilocks` | `dst -= v` | 103 / 81.1 | 103 / 147 |
| `QuadMersenne31` | `dst = c * src` | 31.1 / 37.7 | 30.6 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 22.9 / 28.3 | 27 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 27 / 31.1 | 24.1 / 26.2 |
| `QuadMersenne31` | `dst += v` | 62.5 / 108 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 49.9 / 86.4 | 103 / 153 |

No matched competitor measurements are available for bit-packed GF(2)
or the wider binary fields.
