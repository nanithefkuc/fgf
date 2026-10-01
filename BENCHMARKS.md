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
| `Gf8D` | Dependent multiply chain | 0.7 / 0.34 |
| `Gf16` | Dependent multiply chain | 2.41 / 1.34 |
| `Mersenne31` | Indexed products with accumulation | 4.22 / 1.54 |
| `Goldilocks` | Indexed products with accumulation | 3.74 / 1.63 |
| `QuadMersenne31` | Indexed products with accumulation | 8.98 / 4.36 |

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `add_assign` | 42.8 / 92.5 | 32 / 63.7 | 7.58 / 10.8 |
| `Gf8B` | `add_assign_scalar` | 42.6 / 80.1 | 41.5 / 67.6 | 28.6 / 25.7 |
| `Gf8B` | `sub_assign_scalar` | 42.7 / 80.1 | 41.5 / 67.6 | 28.6 / 25.4 |
| `Gf8B` | `mul_add` | 62.2 / 83 | 37.9 / 59 | 8.32 / 17.2 |
| `Gf8B` | `mul_into` | 77.8 / 163 | 38.3 / 64.7 | 13.1 / 29.1 |
| `Gf8B` | `mul_assign` | 122 / 170 | 51.5 / 76.6 | 30.9 / 29.2 |
| `Gf8B` | `mul_elementwise` | 72.5 / 108 | 26.3 / 38.6 | 5.14 / 7.46 |
| `Gf8B` | `mul_elementwise_assign` | 71.9 / 105 | 31.2 / 50.6 | 6.73 / 9.99 |
| `Gf16` | `add_assign` | 42.8 / 93.1 | 32 / 63.7 | 7.54 / 10.6 |
| `Gf16` | `add_assign_scalar` | 41.2 / 77.8 | 41.5 / 67.5 | 28.6 / 25.4 |
| `Gf16` | `sub_assign_scalar` | 40.8 / 77.9 | 41.5 / 67.5 | 28.6 / 25.4 |
| `Gf16` | `mul_add` | 35.9 / 52.7 | 32 / 48.4 | 7 / 8.19 |
| `Gf16` | `mul_add_with` | 36.5 / 53.2 | 32 / 48.4 | 7 / 8.19 |
| `Gf16` | `mul_into` | 49.8 / 81.1 | 36.5 / 61.8 | 6.87 / 9.62 |
| `Gf16` | `mul_assign` | 53.9 / 82.2 | 48.4 / 76.9 | 29 / 31.9 |
| `Gf16` | `mul_elementwise` | 33.3 / 34.8 | 23 / 27.2 | 4.53 / 5.11 |
| `Gf16` | `mul_elementwise_assign` | 34 / 34.2 | 27.7 / 30.8 | 5.61 / 7.08 |
| `Mersenne31` | `add_assign` | 29.5 / 34.7 | 23.1 / 35.5 | 8.28 / 21.4 |
| `Mersenne31` | `add_assign_scalar` | 40 / 49.9 | 41.2 / 53 | 24.4 / 21.8 |
| `Mersenne31` | `sub_assign_scalar` | 32.4 / 41.2 | 33.9 / 43.5 | 22.8 / 25.3 |
| `Mersenne31` | `mul_elementwise` | 22.8 / 27.8 | 21.2 / 32.2 | 5.78 / 18.2 |
| `Mersenne31` | `mul_elementwise_assign` | 24.8 / 32.7 | 21.2 / 34.2 | 8.1 / 22.2 |
| `Goldilocks` | `add_assign` | 30.1 / 21.9 | 27.3 / 22.5 | 8.12 / 20.9 |
| `Goldilocks` | `add_assign_scalar` | 44.9 / 41.1 | 44.4 / 40.8 | 25.6 / 33.7 |
| `Goldilocks` | `sub_assign_scalar` | 60.3 / 39.2 | 52.7 / 40.4 | 27.6 / 33.8 |
| `Goldilocks` | `mul_elementwise` | 10.8 / 10.6 | 11.3 / 11.2 | 5.7 / 10.6 |
| `Goldilocks` | `mul_elementwise_assign` | 11 / 11.1 | 11.3 / 11.1 | 7.48 / 9.84 |
| `QuadMersenne31` | `add_assign` | 20.2 / 37.4 | 18.3 / 33.4 | 7.98 / 25.2 |
| `QuadMersenne31` | `add_assign_scalar` | 29.6 / 52.8 | 27.2 / 51.4 | 18.9 / 27 |
| `QuadMersenne31` | `sub_assign_scalar` | 23.7 / 40.9 | 22.9 / 43.5 | 16.7 / 33.1 |
| `QuadMersenne31` | `mul_elementwise` | 11.8 / 14.8 | 11.7 / 14.9 | 5.26 / 13.8 |
| `QuadMersenne31` | `mul_elementwise_assign` | 7.23 / 15 | 7.37 / 15.1 | 6.46 / 13.6 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Operation | Throughput (GiB/s) |
| --- | --- | --- |
| `Gf8B` | `mul_add` | 40.7 / 64.8 |
| `Gf8B` | `mul_into` | 39.8 / 64.7 |
| `Gf8D` | `mul_add` | 40.7 / 64.8 |
| `Gf8D` | `mul_into` | 39.8 / 64.7 |
| `Gf16` | `mul_add` | 40.1 / 56.6 |
| `Gf16` | `mul_into` | 36.2 / 63.6 |
| `Gf8B` | `mul_elementwise` | 29.1 / 54.1 |
| `Gf8B` | `mul_elementwise_assign` | 40.6 / 65.7 |
| `Gf8D` | `mul_elementwise` | 28.8 / 49.6 |
| `Gf8D` | `mul_elementwise_assign` | 40.6 / 56.7 |

### Wider binary fields

256 KiB `mul_add`, ordinary `Vec<u8>` buffers.

| Field | Throughput (GiB/s) |
| --- | --- |
| `Gf32` | 17.1 / 30.7 |
| `Gf64` | 9.09 / 16.9 |
| `FanPaar16` | 9.97 / 17.2 |
| `FanPaar32` | 3.71 / 6.34 |
| `FanPaar64` | 1.36 / 1.91 |

### One-shot and prepared multiplication

Ordinary `Vec<u8>` buffers; latency per call. Preparation is excluded.

| Row bytes | Gf8B `mul_add` (ns/op) | Gf8B `mul_add_with` (ns/op) | Gf16 `mul_add` (ns/op) | Gf16 `mul_add_with` (ns/op) |
| --- | --- | --- | --- | --- |
| 16 | 14.4 / 4.12 | 14.1 / 4.12 | 13.7 / 6.62 | 12.5 / 5.97 |
| 32 | 15.5 / 4.16 | 15.1 / 4.12 | 13.8 / 6.81 | 12.6 / 5.97 |
| 64 | 15.7 / 4.34 | 15.4 / 4.34 | 12.3 / 7.03 | 11.1 / 7.44 |
| 128 | 15.8 / 5.78 | 15.7 / 4.97 | 13.1 / 7.91 | 11.8 / 7.28 |
| 256 | 16.3 / 6.94 | 15.9 / 6.97 | 14.4 / 9.16 | 13.4 / 8.5 |
| 512 | 17.7 / 8.38 | 17.3 / 8.12 | 15.9 / 12.4 | 14.7 / 12.1 |
| 1024 | 22.8 / 13.7 | 22.4 / 13.6 | 31.2 / 20 | 31.1 / 19.4 |
| 2048 | 29.2 / 24.1 | 29.2 / 24 | 41.5 / 35.3 | 40 / 34.8 |
| 4096 | 59.2 / 45.7 | 58.9 / 45.6 | 106 / 70.9 | 104 / 69.9 |
| 16384 | 208 / 170 | 207 / 170 | 363 / 267 | 361 / 266 |
| 65536 | 1.63e+03 / 1.04e+03 | 1.63e+03 / 1.04e+03 | 1.93e+03 / 1.26e+03 | 1.93e+03 / 1.26e+03 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one source; gather
has one destination; matrix has eight sources. Column counts refer to
destinations for scatter/matrix and sources for gather.

| Field | Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `Gf8B` | `mul_add_scatter` | 45.5 / 69.9 | 48.2 / 72.2 | 48.2 / 71.9 | 45.9 / 43.7 |
| `Gf8B` | `mul_add_scatter_with` | 45.5 / 69.9 | 48.2 / 72.2 | 48.2 / 71.9 | 45.9 / 43.8 |
| `Gf8B` | `mul_add_gather` | 44.4 / 71.1 | 43 / 78.6 | 43.6 / 69.4 | 44 / 64.8 |
| `Gf8B` | `mul_add_gather_with` | 44.3 / 71.8 | 43 / 78.8 | 43.6 / 69.5 | 44 / 65 |
| `Gf8B` | `mul_add_matrix` | 79.2 / 103 | 110 / 115 | 109 / 113 | 105 / 113 |
| `Gf8B` | `mul_add_matrix_with` | 75.7 / 103 | 102 / 115 | 101 / 113 | 97.8 / 114 |
| `Gf16` | `mul_add_scatter` | 30.6 / 56.1 | 28.5 / 65.6 | 27.6 / 60.6 | 27.4 / 36.2 |
| `Gf16` | `mul_add_scatter_with` | 30.5 / 56.1 | 28.5 / 65.5 | 27.5 / 60.5 | 27.4 / 36.4 |
| `Gf16` | `mul_add_gather` | 33.9 / 67.1 | 39.6 / 77.5 | 38.6 / 77.5 | 40.5 / 73.4 |
| `Gf16` | `mul_add_gather_with` | 33.8 / 67.4 | 39.6 / 77.5 | 38.5 / 77.4 | 40.4 / 73.4 |
| `Gf16` | `mul_add_matrix` | 46.3 / 52.2 | 56.2 / 55.7 | 56.2 / 56.4 | 55.9 / 56.5 |
| `Gf16` | `mul_add_matrix_with` | 45.8 / 52 | 55.4 / 54.9 | 55.4 / 55.6 | 55.2 / 55.9 |

### Short multi-row operations

`Gf16`, ordinary `Vec<u8>` buffers; coefficient sets include zero and one.

| Operation | Sources × destinations | 64 B (GiB/s) | 256 B (GiB/s) | 1 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `mul_add_scatter` | 1 × 16 | 4.13 / 8.24 | 13.2 / 27.2 | 27.8 / 56.9 | 38.1 / 63.7 |
| `mul_add_gather` | 8 × 1 | 6.82 / 14.3 | 19 / 37.1 | 31 / 55.4 | 38.6 / 74.2 |
| `mul_add_matrix` | 8 × 16 | 4.05 / 29.6 | 13.8 / 48.5 | 34.7 / 57 | 55.1 / 58.5 |
| `mul_into` | 1 × 1 | 2.77 / 7.42 | 10 / 25.1 | 29.9 / 51.3 | 55.8 / 82 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8B` | `mul_into_gather_with` | 81.1 / 83.9 | 90.9 / 88.3 |
| `Gf8D` | `mul_into_gather` | 81.3 / 106 | 92 / 113 |
| `Gf8D` | `mul_into_gather_with` | 86.5 / 105 | 98 / 113 |
| `Gf8D` | `mul_add_gather_with` | 79.7 / 86.8 | 94.8 / 92 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous
row offsets. The `fill(0)` cases include clearing the destination.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `fill(0)` + `mul_add_matrix` | 47.8 / 50.5 | 26.2 / 27.5 | 16.6 / 17.6 |
| `Gf8B` | `mul_into_matrix` | 58 / 61 | 32.7 / 34.9 | 20.7 / 22.5 |
| `Gf8B` | `mul_add_matrix_with` | 54.5 / 57.5 | 30.7 / 32.2 | 19.4 / 20.7 |
| `Gf8B` | `mul_into_matrix_with` | 57 / 61.4 | 32.6 / 35.2 | 20.4 / 22.4 |
| `Gf8B` | `mul_add_matrix_at` | 58.1 / 57.3 | 32.3 / 32.3 | 20.5 / 20.7 |
| `Gf8D` | `fill(0)` + `mul_add_matrix` | 46.8 / 52.1 | 26.3 / 30 | 16.7 / 18.8 |
| `Gf8D` | `mul_into_matrix` | 56.3 / 64.5 | 32.4 / 38.7 | 20.7 / 24.3 |

### Bit-packed GF(2)

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where
`N` is the number of packed bits.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.8 / 88 | 41.4 / 66.1 | 8.56 / 25.8 | 6.11 / 14.8 |
| `bits::and_into` | 22.4 / 65.4 | 19.3 / 46 | 5.81 / 19.8 | 4.64 / 11 |
| `bits::weight` | 116 / 14.7 | 67.9 / 15 | 38.8 / 15 | 18 / 14.4 |
| `bits::dot_product` | 20.4 / 35.3 | 18.3 / 36.9 | 10.9 / 36 | 8.79 / 19.4 |
| `bits::xor_range` | 53.6 / 90 | 65.9 / 74.5 | 21 / 39.9 | 9.77 / 24.1 |
| `bits::xor_range_with` | 57.9 / 100 | 66 / 74.5 | 21.1 / 39.9 | 9.84 / 24.5 |

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.8 / 5.39 |
| `bits::xor_range_with` | 16-byte rows; bits `[61, 69)` | 4.14 / 2.08 |

### Row-wise addition

`add_assign_rows` on ordinary `Vec<u8>` buffers.

| Row bytes | Rows | Gf8B (GiB/s) | Gf16 (GiB/s) | Mersenne31 (GiB/s) |
| --- | --- | --- | --- | --- |
| 64 | 1 | 6.58 / 12.5 | 4.63 / 10.9 | 5.17 / 9.44 |
| 64 | 2 | 12.6 / 25.9 | 9.28 / 22.6 | 9.37 / 15.9 |
| 64 | 4 | 23.7 / 44.1 | 18.7 / 38.1 | 15.8 / 20.9 |
| 64 | 8 | 38.2 / 58.2 | 30.1 / 62.8 | 23.3 / 30.3 |
| 64 | 16 | 54.1 / 69 | 39.4 / 77.3 | 27 / 33.2 |
| 64 | 32 | 63.2 / 89.4 | 52.7 / 86.1 | 26.9 / 36.2 |
| 1024 | 1 | 53.4 / 80.3 | 39.5 / 77.5 | 26.7 / 33.4 |
| 1024 | 2 | 63 / 89.2 | 52.7 / 86 | 26.9 / 36.2 |
| 1024 | 4 | 42.1 / 95.1 | 39.7 / 91.2 | 29.1 / 34.5 |
| 1024 | 8 | 79.5 / 121 | 75.1 / 119 | 30.1 / 36.3 |
| 1024 | 16 | 83.5 / 132 | 81 / 130 | 30.6 / 37.2 |
| 1024 | 32 | 34.9 / 66.2 | 34.8 / 66.9 | 29.8 / 35.4 |
| 65536 | 1 | 31.7 / 63.3 | 31.7 / 63.3 | 28 / 34.3 |
| 65536 | 2 | 31.9 / 63.5 | 31.9 / 63.6 | 29.9 / 38.4 |
| 65536 | 4 | 32 / 63.7 | 32 / 63.7 | 26 / 36.9 |
| 65536 | 8 | 38.7 / 46.5 | 30.2 / 45.1 | 25.1 / 33.1 |
| 65536 | 16 | 18.8 / 26.8 | 19 / 26 | 18.6 / 22.5 |
| 65536 | 32 | 18.3 / 25.8 | 18.2 / 24.6 | 18.2 / 21.6 |

### Network-size payloads

`Gf8B`, ordinary `Vec<u8>` buffers; scatter rows use the payload length as
the row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 7.34 / 12 | 3.63 / 13.5 | 6.32 / 15.5 | 8.31 / 14.4 | 11.2 / 19.6 |
| 256 | 24.1 / 39.3 | 14.6 / 34.2 | 20.7 / 52.3 | 26.4 / 35.5 | 30.7 / 64 |
| 512 | 40.9 / 65.8 | 26.6 / 57.4 | 50.5 / 92.5 | 37.2 / 63.8 | 42.9 / 83.6 |
| 1152 | 53.6 / 82.7 | 45.9 / 70.8 | 74.2 / 136 | 79 / 102 | 86.6 / 113 |
| 1168 | 54.3 / 98 | 48.3 / 80 | 81 / 135 | 58.4 / 85.3 | 57.1 / 92.1 |
| 1184 | 53.1 / 83.6 | 46.5 / 70.9 | 75.1 / 136 | 65.3 / 101 | 64.7 / 112 |
| 1200 | 55.4 / 97.5 | 48.3 / 80.2 | 77.2 / 133 | 51.5 / 85.3 | 56 / 91.3 |
| 1216 | 54.8 / 82.9 | 47.5 / 70.9 | 76.3 / 135 | 83.3 / 100 | 88.3 / 112 |
| 1232 | 56.6 / 101 | 50.6 / 81 | 82.9 / 135 | 59.7 / 86.6 | 57.6 / 94.4 |
| 1248 | 56.9 / 84.5 | 48.1 / 71.1 | 77 / 136 | 64.6 / 101 | 66.2 / 112 |
| 1400 | 60.6 / 104 | 40.2 / 68.6 | 64.6 / 117 | 43.8 / 60.7 | 46 / 62.9 |

### Fixed-offset multi-row operations

`Gf8D`; every source and destination base has the stated offset modulo
64. Scatter has four destinations, gather has sixteen sources, and matrix
has ten sources and four destinations.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 30.5 / 46.3 | 19.8 / 53.1 | 25.8 / 32.2 |
| 256 | 16 | 27.5 / 38 | 18.5 / 45.5 | 23.4 / 29.9 |
| 512 | 0 | 52 / 74.1 | 34.8 / 80.5 | 30.3 / 39 |
| 512 | 16 | 38.1 / 70.3 | 31.6 / 70.9 | 28.1 / 35.1 |
| 1024 | 0 | 79.1 / 98 | 55.5 / 100 | 33.9 / 43.6 |
| 1024 | 16 | 60.1 / 92.8 | 48.8 / 85.2 | 30.7 / 38.6 |
| 2048 | 0 | 95.2 / 110 | 79.2 / 108 | 35.4 / 46 |
| 2048 | 16 | 85.6 / 106 | 67 / 92.2 | 31.8 / 40.1 |
| 3072 | 0 | 103 / 117 | 78.3 / 91.8 | 35.8 / 41.8 |
| 3072 | 16 | 92.5 / 116 | 60.3 / 74.7 | 31.8 / 39 |
| 4096 | 0 | 90.1 / 113 | 76.5 / 88.8 | 35.6 / 39.6 |
| 4096 | 16 | 82.2 / 109 | 65.3 / 72.6 | 30.3 / 36.9 |
| 8192 | 0 | 92.2 / 119 | 85.6 / 91.5 | 34.8 / 39.4 |
| 8192 | 16 | 88.4 / 115 | 78 / 74.4 | 30.9 / 37 |
| 16384 | 0 | 48.8 / 73.7 | 88.2 / 92.8 | 34.6 / 40.1 |
| 16384 | 16 | 48.4 / 73.4 | 84 / 75.5 | 32.4 / 37.6 |

### Fixed-offset single-row multiplication

`Gf8D` `mul_add`, 16 KiB. Both source and destination start at the stated
offset from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 93 / 103 |
| 16 | 93.2 / 102 |
| 32 | 93.2 / 103 |
| 48 | 93.5 / 103 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at
the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Gf8B `mul_add_scatter` (GiB/s) | Gf16 `mul_add_scatter` (GiB/s) |
| --- | --- | --- | --- |
| 64 KiB | 0 | 48.2 / 71.8 | 27.3 / 60.6 |
| 64 KiB | 16 | 48.3 / 71.9 | 29.6 / 62.8 |
| 256 KiB | 0 | 29.4 / 23.6 | 25.5 / 24.4 |
| 256 KiB | 16 | 29.4 / 23.6 | 26.6 / 24.8 |

`Gf8D`, one source and four 16 KiB destination rows, 64-byte-aligned buffers:

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.4 / 73.8 |
| `mul_add_scatter_with` | 48.5 / 73.8 |

### In-place scaling by buffer size

64-byte-aligned buffers.

| Buffer size | Gf8B `mul_assign` (GiB/s) | Gf8D `mul_assign` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 129 / 175 | 129 / 172 |
| 16 KiB | 137 / 187 | 137 / 185 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 69 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 70.9 |
| 1 MiB | 50.4 / 54.5 | 49.6 / 54.4 |

### Large destinations

Ordinary `Vec<u8>` buffers. Read-back XOR-folds all written 64-bit words.

| Buffer size | Gf8B `mul_into` (GiB/s) | Gf8B `mul_into` + read (GiB/s) | Gf8B `mul_add` (GiB/s) | Gf16 `mul_into` (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 MiB | 18.9 / 26.7 | 13.7 / 21.1 | 18.9 / 27.3 | 18.5 / 24.4 |
| 8 MiB | 18.2 / 41.8 | 7.74 / 16.6 | 8.16 / 25.1 | 8.03 / 41.7 |
| 32 MiB | 9.42 / 28.1 | 6 / 14.2 | 6.26 / 16.3 | 6.28 / 27.7 |

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
| `dst = c * src` | 1 × 1 | 64 KiB | 40.1 / 64.6 | 17.8 / 45.7 | 33.7 / 57.1 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.9 / 65 | 41.2 / 65.2 | 29.1 / 49.3 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 82.7 / 104 | 82.8 / 98 | 30.7 / 51.4 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 89.5 / 105 | 74.9 / 90.3 | 35.7 / 57.5 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 64.5 / 73.9 | 65 / 69.6 | 53.6 / 74.7 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 64.5 / 72 | 53.1 / 65.4 | 67.4 / 85.4 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 58.8 / 71.8 | 54.2 / 63.5 | 73.6 / 86.2 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 34.4 / 40.1 | 35.6 / 35 | 31.9 / 40.4 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 35.7 / 41.2 | 36.4 / 29.9 | 38.5 / 44.7 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 33.3 / 39.9 | 35.5 / 31.5 | 40.7 / 45.8 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 22.1 / 26.5 | 24.4 / 24.5 | 22.7 / 27.8 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 22.7 / 26.2 | 24.9 / 22 | 25.8 / 30.1 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 20.7 / 25.7 | 23.9 / 23.6 | 26.6 / 30.5 |

### Prime fields

Page-aligned 64 KiB regions on each library's native layout. Throughput
counts one operand pair (twice the region bytes); conversions are excluded.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.22 / 1.54 | 2.1 / 0.84 |
| `Goldilocks` | `acc += a * b` | 3.74 / 1.63 | 2.57 / 1.11 |
| `QuadMersenne31` | `acc += a * b` | 8.98 / 4.36 | 6.98 / 2.92 |

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 73.3 / 76.8 | 65.5 / 72.8 |
| `Mersenne31` | `dst += c * src` | 55.6 / 61.1 | 49.8 / 57 |
| `Mersenne31` | `dst = a * b` | 59.5 / 68.2 | 59.8 / 69.4 |
| `Mersenne31` | `dst += v` | 87.6 / 108 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 70.3 / 86.4 | 103 / 153 |
| `Goldilocks` | `dst = c * src` | 25.1 / 23.8 | 27.5 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17 | 22.5 / 20.6 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.3 | 26.9 / 26.5 |
| `Goldilocks` | `dst += v` | 97.5 / 81.6 | 103 / 148 |
| `Goldilocks` | `dst -= v` | 103 / 81.1 | 103 / 148 |
| `QuadMersenne31` | `dst = c * src` | 31.1 / 37.7 | 30.5 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 22.9 / 28 | 27 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 27 / 31.2 | 24.1 / 26.2 |
| `QuadMersenne31` | `dst += v` | 62.5 / 108 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 49.9 / 86.4 | 103 / 153 |

No matched competitor measurements are available for bit-packed GF(2)
or the wider binary fields.
