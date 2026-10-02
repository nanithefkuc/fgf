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
| `Gf16` | Dependent multiply chain | 2.42 / 1.31 |
| `Mersenne31` | Indexed products with accumulation | 4.21 / 1.55 |
| `Goldilocks` | Indexed products with accumulation | 3.74 / 1.64 |
| `QuadMersenne31` | Indexed products with accumulation | 8.98 / 4.41 |

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `add_assign` | 42.9 / 96.6 | 32 / 63.7 | 7.63 / 10.8 |
| `Gf8B` | `add_assign_scalar` | 42.9 / 80.2 | 41.5 / 67.6 | 28.6 / 25.7 |
| `Gf8B` | `sub_assign_scalar` | 42.8 / 80.1 | 41.5 / 67.6 | 28.5 / 25.6 |
| `Gf8B` | `mul_add` | 62 / 84.1 | 37.9 / 58.5 | 8.54 / 17.5 |
| `Gf8B` | `mul_into` | 81.5 / 164 | 38.2 / 64.7 | 13.1 / 29.1 |
| `Gf8B` | `mul_assign` | 122 / 171 | 51.5 / 76.6 | 30.6 / 29.2 |
| `Gf8B` | `mul_elementwise` | 73.3 / 99.8 | 25.3 / 39.3 | 5.19 / 7.48 |
| `Gf8B` | `mul_elementwise_assign` | 76.6 / 118 | 31.2 / 49.8 | 6.72 / 10 |
| `Gf16` | `add_assign` | 42.8 / 96.4 | 32 / 63.7 | 7.7 / 10.8 |
| `Gf16` | `add_assign_scalar` | 41.5 / 78.2 | 41.5 / 67.6 | 28.7 / 25.4 |
| `Gf16` | `sub_assign_scalar` | 41 / 78 | 41.5 / 67.6 | 28.5 / 25.4 |
| `Gf16` | `mul_add` | 36.2 / 52.9 | 32 / 48.5 | 6.97 / 8.19 |
| `Gf16` | `mul_add_with` | 37 / 53.5 | 32 / 49 | 7 / 8.19 |
| `Gf16` | `mul_into` | 49.9 / 81 | 36.5 / 61.2 | 6.86 / 9.6 |
| `Gf16` | `mul_assign` | 53.7 / 82.5 | 48.5 / 76.9 | 28.9 / 31.9 |
| `Gf16` | `mul_elementwise` | 33.4 / 34.8 | 22.3 / 28.8 | 4.62 / 5.13 |
| `Gf16` | `mul_elementwise_assign` | 34.2 / 35.1 | 27.7 / 30.7 | 5.56 / 7.14 |
| `Mersenne31` | `add_assign` | 29.3 / 34.8 | 23.2 / 35.6 | 8.18 / 21.1 |
| `Mersenne31` | `add_assign_scalar` | 40 / 52.4 | 41.2 / 52.3 | 24.5 / 21.9 |
| `Mersenne31` | `sub_assign_scalar` | 32.5 / 42.8 | 34.1 / 43.5 | 22.7 / 25.6 |
| `Mersenne31` | `mul_elementwise` | 22.6 / 27.8 | 21.3 / 29.5 | 5.77 / 20 |
| `Mersenne31` | `mul_elementwise_assign` | 24.9 / 32.8 | 21.2 / 34.2 | 8.09 / 22.7 |
| `Goldilocks` | `add_assign` | 29.9 / 22.4 | 27.3 / 22.2 | 8.17 / 21.8 |
| `Goldilocks` | `add_assign_scalar` | 44.8 / 41.1 | 44.4 / 40.8 | 25.5 / 33.7 |
| `Goldilocks` | `sub_assign_scalar` | 59.7 / 39.1 | 52.7 / 40.2 | 27.5 / 33.8 |
| `Goldilocks` | `mul_elementwise` | 10.8 / 10.6 | 11.3 / 11.1 | 5.48 / 10.5 |
| `Goldilocks` | `mul_elementwise_assign` | 11 / 11.1 | 11.3 / 11.1 | 7.4 / 9.92 |
| `QuadMersenne31` | `add_assign` | 20.3 / 38.6 | 18.3 / 32.6 | 7.86 / 24.3 |
| `QuadMersenne31` | `add_assign_scalar` | 29.3 / 53.1 | 26.9 / 51.2 | 18.8 / 26.9 |
| `QuadMersenne31` | `sub_assign_scalar` | 23.7 / 42.6 | 22.9 / 43.4 | 16.7 / 33.1 |
| `QuadMersenne31` | `mul_elementwise` | 11.8 / 14.7 | 11.7 / 14.2 | 5.22 / 14 |
| `QuadMersenne31` | `mul_elementwise_assign` | 7.23 / 14.8 | 7.37 / 15.2 | 6.6 / 13.9 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Operation | Throughput (GiB/s) |
| --- | --- | --- |
| `Gf8B` | `mul_add` | 40.7 / 64.8 |
| `Gf8B` | `mul_into` | 39.8 / 64.7 |
| `Gf8D` | `mul_add` | 40.7 / 65 |
| `Gf8D` | `mul_into` | 39.8 / 64.7 |
| `Gf16` | `mul_add` | 40.1 / 56.6 |
| `Gf16` | `mul_into` | 36.1 / 63.7 |
| `Gf8B` | `mul_elementwise` | 29.1 / 54.1 |
| `Gf8B` | `mul_elementwise_assign` | 40.7 / 65.7 |
| `Gf8D` | `mul_elementwise` | 28.8 / 49.6 |
| `Gf8D` | `mul_elementwise_assign` | 40.6 / 56.6 |

### Wider binary fields

256 KiB `mul_add`, ordinary `Vec<u8>` buffers.

| Field | Throughput (GiB/s) |
| --- | --- |
| `Gf32` | 17.1 / 31.8 |
| `Gf64` | 9.09 / 17 |
| `FanPaar16` | 9.97 / 17.2 |
| `FanPaar32` | 3.71 / 6.34 |
| `FanPaar64` | 1.37 / 1.91 |

### One-shot and prepared multiplication

Ordinary `Vec<u8>` buffers; latency per call. Preparation is excluded.

| Row bytes | Gf8B `mul_add` (ns/op) | Gf8B `mul_add_with` (ns/op) | Gf16 `mul_add` (ns/op) | Gf16 `mul_add_with` (ns/op) |
| --- | --- | --- | --- | --- |
| 16 | 14.4 / 4.19 | 14.2 / 4.53 | 13.7 / 6.59 | 12.4 / 5.75 |
| 32 | 15.4 / 4.12 | 15.1 / 4.19 | 13.8 / 6.59 | 12.6 / 5.97 |
| 64 | 15.7 / 4.38 | 15.4 / 4.53 | 12.5 / 8.16 | 11.1 / 7.19 |
| 128 | 15.8 / 5.75 | 15.6 / 5.78 | 13.1 / 9.12 | 11.7 / 8.16 |
| 256 | 16.3 / 6.91 | 15.9 / 6.97 | 14.5 / 10.6 | 13.4 / 8.31 |
| 512 | 17.1 / 8.25 | 16.8 / 8.06 | 16.4 / 12.2 | 15 / 11.9 |
| 1024 | 22.8 / 13.4 | 22.2 / 13.4 | 28 / 20 | 28.5 / 19.5 |
| 2048 | 30 / 23.8 | 29.8 / 23.8 | 42 / 35.2 | 41 / 34.8 |
| 4096 | 64.2 / 45.3 | 64.2 / 45.2 | 108 / 70.8 | 108 / 69.8 |
| 16384 | 213 / 170 | 212 / 170 | 364 / 262 | 363 / 266 |
| 65536 | 1.63e+03 / 1.05e+03 | 1.63e+03 / 1.05e+03 | 1.93e+03 / 1.25e+03 | 1.93e+03 / 1.26e+03 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one source; gather
has one destination; matrix has eight sources. Column counts refer to
destinations for scatter/matrix and sources for gather.

| Field | Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `Gf8B` | `mul_add_scatter` | 45.5 / 69.8 | 48.2 / 72.2 | 48.2 / 72.2 | 46.2 / 44.2 |
| `Gf8B` | `mul_add_scatter_with` | 45.5 / 69.9 | 48.2 / 72.2 | 48.2 / 72.2 | 46.2 / 44.2 |
| `Gf8B` | `mul_add_gather` | 44.4 / 71.5 | 42.8 / 79.6 | 43.6 / 69.7 | 43.9 / 64.8 |
| `Gf8B` | `mul_add_gather_with` | 44.3 / 71.9 | 42.8 / 79.8 | 43.6 / 69.7 | 43.9 / 64.9 |
| `Gf8B` | `mul_add_matrix` | 89.6 / 106 | 133 / 145 | 129 / 146 | 122 / 141 |
| `Gf8B` | `mul_add_matrix_with` | 89.3 / 106 | 132 / 146 | 123 / 145 | 118 / 141 |
| `Gf16` | `mul_add_scatter` | 30.6 / 56.2 | 28.5 / 65.7 | 27.6 / 62.7 | 26.9 / 37.6 |
| `Gf16` | `mul_add_scatter_with` | 30.5 / 56.2 | 28.5 / 65.6 | 27.6 / 62.7 | 26.9 / 37.6 |
| `Gf16` | `mul_add_gather` | 33.9 / 65.6 | 39.6 / 78.6 | 38.6 / 78.1 | 40.3 / 73.1 |
| `Gf16` | `mul_add_gather_with` | 33.8 / 65.8 | 39.5 / 78.5 | 38.6 / 78 | 40.4 / 73.1 |
| `Gf16` | `mul_add_matrix` | 45.7 / 52.1 | 55.8 / 56.8 | 55.6 / 56.2 | 55.5 / 56.7 |
| `Gf16` | `mul_add_matrix_with` | 45.5 / 50.9 | 55 / 54.8 | 55 / 54.1 | 54.7 / 55.5 |

### Short multi-row operations

`Gf16`, ordinary `Vec<u8>` buffers; coefficient sets include zero and one.

| Operation | Sources × destinations | 64 B (GiB/s) | 256 B (GiB/s) | 1 KiB (GiB/s) | 16 KiB (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `mul_add_scatter` | 1 × 16 | 4.13 / 8.14 | 13.2 / 27.4 | 27.9 / 57.3 | 38.1 / 63.5 |
| `mul_add_gather` | 8 × 1 | 6.86 / 14.1 | 19.1 / 36.7 | 31.7 / 54.8 | 38.6 / 73.2 |
| `mul_add_matrix` | 8 × 16 | 4.06 / 29.6 | 13.9 / 48.5 | 34.8 / 56.9 | 54.9 / 58.4 |
| `mul_into` | 1 × 1 | 2.8 / 7.22 | 9.99 / 25.1 | 29.8 / 51.4 | 55.8 / 81.9 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8B` | `mul_into_gather_with` | 91.1 / 83.9 | 102 / 88.2 |
| `Gf8D` | `mul_into_gather` | 91.3 / 105 | 102 / 100 |
| `Gf8D` | `mul_into_gather_with` | 90.8 / 105 | 102 / 99.7 |
| `Gf8D` | `mul_add_gather_with` | 79.2 / 86.9 | 94 / 91.9 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous
row offsets. The `fill(0)` cases include clearing the destination.

| Field | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | `fill(0)` + `mul_add_matrix` | 56.2 / 50.8 | 30.9 / 32.3 | 19.3 / 21.8 |
| `Gf8B` | `mul_into_matrix` | 72.6 / 68.7 | 42 / 41.5 | 25.5 / 28.8 |
| `Gf8B` | `mul_add_matrix_with` | 72.8 / 57.4 | 39.6 / 38 | 24.1 / 26.1 |
| `Gf8B` | `mul_into_matrix_with` | 75.5 / 66.9 | 41.9 / 41.8 | 25 / 28.8 |
| `Gf8B` | `mul_add_matrix_at` | 72.8 / 57.3 | 40 / 31.1 | 24.5 / 22.9 |
| `Gf8D` | `fill(0)` + `mul_add_matrix` | 56.4 / 52.5 | 31.1 / 33.3 | 19.4 / 23.3 |
| `Gf8D` | `mul_into_matrix` | 74.5 / 70.5 | 42.2 / 45 | 25.7 / 32.2 |

### Bit-packed GF(2)

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where
`N` is the number of packed bits.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.8 / 88.4 | 41.4 / 66.1 | 8.58 / 25.8 | 6.12 / 14.8 |
| `bits::and_into` | 22.1 / 68.2 | 19.3 / 46.8 | 5.86 / 20.2 | 4.62 / 10.8 |
| `bits::weight` | 118 / 14.7 | 67.9 / 15 | 38.6 / 15 | 18.1 / 14.5 |
| `bits::dot_product` | 20.3 / 39.7 | 18.3 / 43.5 | 11.2 / 39.1 | 8.72 / 18.3 |
| `bits::xor_range` | 53.5 / 90 | 65.9 / 74.5 | 21.1 / 39.9 | 9.77 / 24.2 |
| `bits::xor_range_with` | 57.9 / 100 | 66 / 74.6 | 21.2 / 39.8 | 9.87 / 24.6 |

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.9 / 5.35 |
| `bits::xor_range_with` | 16-byte rows; bits `[61, 69)` | 4.14 / 2.08 |

### Row-wise addition

`add_assign_rows` on ordinary `Vec<u8>` buffers.

| Row bytes | Rows | Gf8B (GiB/s) | Gf16 (GiB/s) | Mersenne31 (GiB/s) |
| --- | --- | --- | --- | --- |
| 64 | 1 | 6.6 / 13.8 | 5.05 / 11.2 | 5.2 / 9.44 |
| 64 | 2 | 12.8 / 25.8 | 9.88 / 22.7 | 9.54 / 16 |
| 64 | 4 | 23.7 / 39.1 | 19 / 40.8 | 15.1 / 24.1 |
| 64 | 8 | 37.5 / 57.8 | 30.1 / 62.8 | 20.7 / 30.4 |
| 64 | 16 | 53.5 / 69 | 43 / 77.3 | 27.3 / 33.2 |
| 64 | 32 | 63.9 / 89.9 | 54.6 / 86.7 | 26.1 / 36.4 |
| 1024 | 1 | 53.5 / 80.3 | 43 / 77.5 | 27.2 / 33.2 |
| 1024 | 2 | 64 / 89.9 | 54.6 / 86.7 | 26.1 / 36.4 |
| 1024 | 4 | 42.1 / 96 | 39.9 / 94.1 | 28.6 / 34.5 |
| 1024 | 8 | 79.4 / 131 | 75.7 / 127 | 29.7 / 36.2 |
| 1024 | 16 | 83.9 / 133 | 81.3 / 131 | 30.2 / 37.1 |
| 1024 | 32 | 34.8 / 66.6 | 34.6 / 66.4 | 29.5 / 35.5 |
| 65536 | 1 | 31.7 / 63.2 | 31.7 / 63.2 | 27.9 / 35.7 |
| 65536 | 2 | 29.4 / 63.5 | 31.9 / 63.5 | 29.7 / 38.4 |
| 65536 | 4 | 32 / 63.7 | 32 / 63.7 | 26.1 / 36.8 |
| 65536 | 8 | 38.2 / 44.6 | 30.2 / 46.9 | 25.4 / 31.1 |
| 65536 | 16 | 19 / 26.9 | 18.8 / 26 | 18.9 / 22.1 |
| 65536 | 32 | 18.4 / 25.8 | 18.2 / 24.8 | 18.3 / 21.6 |

### Network-size payloads

`Gf8B`, ordinary `Vec<u8>` buffers; scatter rows use the payload length as
the row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 6.76 / 13.7 | 3.69 / 13.4 | 6.76 / 17.2 | 8.46 / 14.5 | 11.3 / 19.6 |
| 256 | 25.3 / 41.2 | 14.6 / 34.4 | 22.7 / 52.3 | 26.9 / 35.3 | 30.3 / 56.8 |
| 512 | 41.9 / 69.7 | 27.2 / 58 | 49.7 / 94.2 | 36.1 / 64.1 | 43.4 / 83.5 |
| 1152 | 52.7 / 83.1 | 45.6 / 72 | 76.1 / 134 | 76.8 / 102 | 87.4 / 112 |
| 1168 | 53.9 / 98 | 47.6 / 79.1 | 79.3 / 134 | 58.9 / 85.4 | 57.5 / 92 |
| 1184 | 52 / 84 | 46.2 / 72.5 | 76 / 135 | 65.2 / 101 | 65 / 112 |
| 1200 | 55.4 / 98.2 | 48.2 / 80.2 | 81.1 / 135 | 51.4 / 85.8 | 55.7 / 91.5 |
| 1216 | 53.3 / 83.9 | 47.1 / 72.3 | 79.8 / 135 | 83.9 / 100 | 87.8 / 112 |
| 1232 | 55.5 / 99.8 | 49.1 / 81 | 87.4 / 136 | 60 / 86.4 | 58.5 / 94.5 |
| 1248 | 55.3 / 85.1 | 47.7 / 73.1 | 80.2 / 134 | 65.9 / 101 | 66.3 / 112 |
| 1400 | 60.6 / 103 | 39.7 / 68.5 | 63.6 / 117 | 42.1 / 60.8 | 45.1 / 62.7 |

### Fixed-offset multi-row operations

`Gf8D`; every source and destination base has the stated offset modulo
64. Scatter has four destinations, gather has sixteen sources, and matrix
has ten sources and four destinations.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 30.8 / 47.3 | 19.9 / 52.7 | 26.1 / 29.2 |
| 256 | 16 | 27.3 / 38.7 | 18.7 / 45.5 | 23.6 / 28.7 |
| 512 | 0 | 52.1 / 77.7 | 35 / 80.5 | 28.9 / 36.8 |
| 512 | 16 | 36.7 / 71.3 | 31.8 / 70.8 | 26.4 / 36.6 |
| 1024 | 0 | 79.2 / 112 | 55.9 / 100 | 35 / 42.6 |
| 1024 | 16 | 58.4 / 102 | 49 / 85.2 | 31.9 / 42.2 |
| 2048 | 0 | 95.5 / 134 | 78 / 108 | 39.2 / 45 |
| 2048 | 16 | 83.7 / 126 | 67.7 / 92.2 | 35.5 / 44.2 |
| 3072 | 0 | 102 / 118 | 79.2 / 91 | 40.7 / 46.2 |
| 3072 | 16 | 92.2 / 116 | 59.9 / 74.8 | 36.2 / 44.8 |
| 4096 | 0 | 90.2 / 113 | 76.5 / 88.7 | 40.8 / 45.2 |
| 4096 | 16 | 82.2 / 110 | 64 / 72.6 | 33.7 / 42.3 |
| 8192 | 0 | 92.1 / 119 | 85.8 / 91.4 | 41.7 / 47.6 |
| 8192 | 16 | 88.4 / 116 | 78 / 74.4 | 36.9 / 44.5 |
| 16384 | 0 | 48.8 / 72.6 | 89 / 92.8 | 42.2 / 48.2 |
| 16384 | 16 | 48.3 / 71 | 84.4 / 75.5 | 39.6 / 45.4 |

### Fixed-offset single-row multiplication

`Gf8D` `mul_add`, 16 KiB. Both source and destination start at the stated
offset from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 92.7 / 102 |
| 16 | 92.8 / 103 |
| 32 | 92.9 / 103 |
| 48 | 92.5 / 103 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at
the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Gf8B `mul_add_scatter` (GiB/s) | Gf16 `mul_add_scatter` (GiB/s) |
| --- | --- | --- | --- |
| 64 KiB | 0 | 48.2 / 72.1 | 27.2 / 58.7 |
| 64 KiB | 16 | 48.3 / 72.2 | 29.5 / 61.6 |
| 256 KiB | 0 | 29.4 / 24.1 | 25.8 / 24.8 |
| 256 KiB | 16 | 29.4 / 23.7 | 26.7 / 25 |

`Gf8D`, one source and four 16 KiB destination rows, 64-byte-aligned buffers:

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.4 / 73.5 |
| `mul_add_scatter_with` | 48.5 / 73.7 |

### In-place scaling by buffer size

64-byte-aligned buffers.

| Buffer size | Gf8B `mul_assign` (GiB/s) | Gf8D `mul_assign` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 128 / 174 | 128 / 172 |
| 16 KiB | 137 / 186 | 138 / 185 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 68.8 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 70.9 |
| 1 MiB | 48 / 52.2 | 48.8 / 47.8 |

### Large destinations

Ordinary `Vec<u8>` buffers. Read-back XOR-folds all written 64-bit words.

| Buffer size | Gf8B `mul_into` (GiB/s) | Gf8B `mul_into` + read (GiB/s) | Gf8B `mul_add` (GiB/s) | Gf16 `mul_into` (GiB/s) |
| --- | --- | --- | --- | --- |
| 1 MiB | 18.9 / 27 | 13.8 / 21.3 | 18.9 / 27.5 | 18.6 / 24.2 |
| 8 MiB | 18.2 / 41.8 | 7.63 / 16.6 | 8.22 / 25.4 | 8.15 / 41.7 |
| 32 MiB | 9.4 / 28 | 5.88 / 14.3 | 6.26 / 16.4 | 6.22 / 27.8 |

| Field | Operation | Buffer size | Alignment | Throughput (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8D` | `mul_into` | 4 MiB | 64-byte-aligned | 18.2 / 41.7 |

## vs Competitor timings

### GF(256), polynomial `0x11D`

Page-aligned buffers and matched coding coefficients. ISA-L uses its
field kernels; klauspost uses `Encode` with a supplied coding matrix, or
`EncodeIdx` for accumulation. Preparation and Go bridge pointer staging
are excluded.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `dst = c * src` | 1 × 1 | 64 KiB | 40.1 / 64.6 | 18.2 / 45.6 | 33.7 / 56.8 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.9 / 64.9 | 41.2 / 65.1 | 29.1 / 49.1 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 88.6 / 104 | 82.8 / 98 | 30.6 / 50.2 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 93.3 / 97.8 | 75 / 90.2 | 35.7 / 56.8 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 74.8 / 74.6 | 65 / 69.6 | 53.7 / 74.5 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 79.3 / 83.3 | 53.1 / 65.6 | 67.7 / 84.8 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 83.9 / 79.7 | 54.2 / 63.7 | 73.5 / 83.2 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 39.4 / 47.5 | 35.5 / 44.5 | 31.8 / 44 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 42.8 / 50.8 | 36.6 / 39.5 | 38.5 / 50.2 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 43.5 / 50.6 | 35.3 / 34.4 | 40.5 / 48.2 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 26.4 / 32.7 | 24.4 / 31.1 | 22.6 / 31.3 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 26.7 / 34.4 | 25 / 29.6 | 25.8 / 33.9 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 27 / 34 | 23.5 / 23.9 | 26.7 / 30.8 |

### Prime fields

Page-aligned 64 KiB regions on each library's native layout. Throughput
counts one operand pair (twice the region bytes); conversions are excluded.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.21 / 1.55 | 2.1 / 0.8 |
| `Goldilocks` | `acc += a * b` | 3.74 / 1.64 | 2.57 / 1.11 |
| `QuadMersenne31` | `acc += a * b` | 8.98 / 4.41 | 6.98 / 2.9 |

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 73.3 / 76.8 | 65.4 / 72.7 |
| `Mersenne31` | `dst += c * src` | 55.6 / 61.1 | 49.8 / 57 |
| `Mersenne31` | `dst = a * b` | 59.5 / 68.2 | 59.9 / 69.3 |
| `Mersenne31` | `dst += v` | 87.5 / 108 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 70.3 / 86.4 | 103 / 153 |
| `Goldilocks` | `dst = c * src` | 25.1 / 23.8 | 27.5 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17 | 22.5 / 20.6 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.3 | 26.9 / 26.4 |
| `Goldilocks` | `dst += v` | 97.5 / 81.6 | 103 / 148 |
| `Goldilocks` | `dst -= v` | 103 / 81 | 103 / 147 |
| `QuadMersenne31` | `dst = c * src` | 31.1 / 37.5 | 30.6 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 22.9 / 28.4 | 27 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 27 / 31.1 | 24.1 / 26.4 |
| `QuadMersenne31` | `dst += v` | 62.5 / 108 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 49.9 / 86.4 | 103 / 153 |

No matched competitor measurements are available for bit-packed GF(2)
or the wider binary fields.
