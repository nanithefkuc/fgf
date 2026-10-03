# GF(2^8): polynomial byte fields

Public API timings for flat byte fields and the matched `0x11D` competitor
panel. Snapshot rows measure `fgf` 2.1.0; paired rows compare its baseline
with the `Gf8<POLY>` cutover. Every result cell reports Tiger Lake / Golden
Cove; shared snapshot method: [BENCHMARKS.md](../BENCHMARKS.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gf-comp` |
| Resolved backend | `v4x` / `v3_gfni_crypto` |
| Competitors | Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 through a Go 1.27.1 C archive |
| Competitor build | `external/bench-trio` with `fgf/simd512` |
| Competitor protocol | Arms interleave in one process across all six orders; outputs are validated before timing |
| Throughput | Single-row cases count buffer bytes. Accumulating scatter, gather, and matrix cases count row bytes × sources × destinations. Overwrite-matrix and competitor cases count source bytes. |
| Paired baseline | Revision `46afc76`, portable x86-64, all features |
| Paired generic | `Gf8<POLY>` API, portable x86-64, all features |
| Paired compiler | rustc 1.98.1, LLVM 22.1.8, thin LTO, one codegen unit |
| Paired protocol | Five paired runs of each binary, shuffled order, isolated cores 3 / 8, no backend override |
| Paired aggregation | Per-case median; XOR and GF(2^16) operations are unchanged controls |
| Paired run A artifacts | `kernels`: `9202948b735f`; `compare`: `1bfdec0864cc` |
| Paired run B artifacts | `kernels`: `d7e44ca60c1e`; `compare`: `b7f7cdf2e324` |
| Paired run B scope | Lint-adjusted build; baseline and candidate remeasured together |
| Paired run C artifacts | `kernels`: `bfc8de17861a`; `compare`: `8561d4c45cca` |
| Paired run C scope | Validated build; baseline and candidate remeasured together |

## Self-timings

### Scalar multiplication

| Field | Case | Latency (ns/op) |
| --- | --- | --- |
| `Gf8B` | Dependent multiply chain | 0.61 / 0.34 |
| `Gf8D` | Dependent multiply chain | 0.70 / 0.34 |

### Packed operations

`Gf8B`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| `add_assign` | 42.9 / 96.7 | 32.0 / 63.7 | 7.88 / 10.8 |
| `add_assign_scalar` | 42.5 / 80.2 | 41.5 / 67.5 | 29.1 / 25.2 |
| `sub_assign_scalar` | 43.0 / 80.3 | 41.5 / 67.6 | 29.1 / 25.5 |
| `mul_add` | 63.8 / 82.6 | 37.9 / 58.4 | 8.54 / 17.3 |
| `mul_into` | 79.4 / 164 | 38.2 / 64.7 | 13.0 / 29.0 |
| `mul_assign` | 122 / 171 | 51.5 / 76.6 | 30.9 / 29.2 |
| `mul_elementwise` | 73.3 / 99.3 | 26.3 / 42.7 | 5.17 / 7.51 |
| `mul_elementwise_assign` | 74.1 / 118 | 31.2 / 49.8 | 6.76 / 10.0 |

### Aligned single-row operations

64 KiB buffers, 64-byte-aligned.

| Field | Run | Build | Operation | Throughput (GiB/s) |
| --- | --- | --- | --- | --- |
| `Gf8B` | 2.1.0 | snapshot | `mul_add` | 40.7 / 64.8 |
| `Gf8B` | 2.1.0 | snapshot | `mul_into` | 39.8 / 64.7 |
| `Gf8D` | 2.1.0 | snapshot | `mul_add` | 40.7 / 64.8 |
| `Gf8D` | 2.1.0 | snapshot | `mul_into` | 39.8 / 64.7 |
| `Gf8B` | 2.1.0 | snapshot | `mul_elementwise` | 29.1 / 54.1 |
| `Gf8B` | 2.1.0 | snapshot | `mul_elementwise_assign` | 40.7 / 65.6 |
| `Gf8D` | 2.1.0 | snapshot | `mul_elementwise` | 28.7 / 49.6 |
| `Gf8D` | 2.1.0 | snapshot | `mul_elementwise_assign` | 40.6 / 56.6 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_add` | 40.6 / 64.6 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_into` | 39.6 / 64.6 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_elementwise` | 29.5 / 54.2 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_elementwise` | 29.4 / 54.0 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_elementwise_assign` | 40.7 / 65.6 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_elementwise_assign` | 40.7 / 65.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `mul_into` | 39.7 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `mul_elementwise` | 29.0 / 49.4 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `mul_elementwise` | 29.0 / 50.2 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `mul_elementwise_assign` | 40.6 / 56.6 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `mul_elementwise_assign` | 40.8 / 56.5 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_add` | 40.6 / 64.6 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_into` | 39.6 / 64.6 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_elementwise` | 29.4 / 54.1 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_elementwise` | 29.4 / 54.2 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_elementwise_assign` | 40.7 / 65.6 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_elementwise_assign` | 40.7 / 65.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `mul_into` | 39.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `mul_elementwise` | 29.0 / 49.5 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `mul_elementwise` | 29.1 / 49.5 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `mul_elementwise_assign` | 40.6 / 56.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `mul_elementwise_assign` | 40.8 / 56.8 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_add` | 40.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_into` | 39.6 / 64.6 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_elementwise` | 29.4 / 54.1 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_elementwise` | 29.6 / 54.2 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_elementwise_assign` | 40.6 / 65.6 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_elementwise_assign` | 40.7 / 65.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `mul_add` | 40.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `mul_into` | 39.6 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `mul_into` | 39.7 / 64.7 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `mul_elementwise` | 29.1 / 49.4 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `mul_elementwise` | 29.2 / 49.5 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `mul_elementwise_assign` | 40.6 / 56.6 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `mul_elementwise_assign` | 40.7 / 56.6 |

### One-shot and prepared multiplication

`Gf8B`, ordinary `Vec<u8>` buffers; latency per call. Preparation is excluded.

| Row bytes | `mul_add` (ns/op) | `mul_add_with` (ns/op) |
| --- | --- | --- |
| 16 | 14.5 / 4.09 | 14.2 / 4.09 |
| 32 | 15.4 / 4.16 | 15.1 / 4.16 |
| 64 | 15.8 / 4.38 | 15.4 / 4.44 |
| 128 | 15.8 / 5.72 | 15.8 / 5.75 |
| 256 | 16.8 / 6.75 | 16.5 / 6.94 |
| 512 | 17.1 / 8.31 | 17.0 / 8.12 |
| 1024 | 21.9 / 13.7 | 21.6 / 13.7 |
| 2048 | 28.8 / 24.1 | 28.8 / 24.0 |
| 4096 | 58.9 / 45.7 | 59.0 / 45.8 |
| 16384 | 208 / 170 | 207 / 172 |
| 65536 | 1630 / 1040 | 1630 / 1040 |

### Scatter, gather, and matrix

`Gf8B`, 64 KiB rows, ordinary `Vec<u8>` buffers. Scatter has one source;
gather has one destination; matrix has eight sources. Column counts are
destinations for scatter and matrix, sources for gather.

| Operation | 2 (GiB/s) | 4 (GiB/s) | 8 (GiB/s) | 16 (GiB/s) |
| --- | --- | --- | --- | --- |
| `mul_add_scatter` | 45.5 / 69.8 | 48.2 / 72.2 | 48.3 / 72.0 | 45.8 / 48.3 |
| `mul_add_scatter_with` | 45.5 / 69.9 | 48.2 / 72.3 | 48.3 / 72.0 | 45.8 / 48.4 |
| `mul_add_gather` | 44.4 / 71.2 | 43.1 / 79.1 | 43.7 / 69.4 | 44.0 / 64.8 |
| `mul_add_gather_with` | 44.3 / 72.0 | 43.0 / 79.6 | 43.7 / 69.5 | 44.0 / 64.9 |
| `mul_add_matrix` | 89.6 / 107 | 132 / 147 | 128 / 143 | 122 / 141 |
| `mul_add_matrix_with` | 89.3 / 107 | 132 / 148 | 123 / 144 | 119 / 141 |

### Aligned gather

16 sources, 64-byte-aligned buffers.

| Field | Operation | 4 KiB rows (GiB/s) | 16 KiB rows (GiB/s) |
| --- | --- | --- | --- |
| `Gf8B` | `mul_into_gather_with` | 90.9 / 84.0 | 102 / 88.2 |
| `Gf8D` | `mul_into_gather` | 90.8 / 106 | 102 / 100 |
| `Gf8D` | `mul_into_gather_with` | 91.1 / 105 | 102 / 100 |
| `Gf8D` | `mul_add_gather_with` | 79.2 / 87.0 | 94.3 / 92.0 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. `_at` uses contiguous row
offsets. The `fill(0)` cases include clearing the destination.

| Field | Run | Build | Operation | 2 destinations (GiB/s) | 4 destinations (GiB/s) | 6 destinations (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| `Gf8B` | 2.1.0 | snapshot | `fill(0)` + `mul_add_matrix` | 56.4 / 49.8 | 31.1 / 32.4 | 19.4 / 21.9 |
| `Gf8B` | 2.1.0 | snapshot | `mul_into_matrix` | 72.9 / 68.3 | 42.3 / 41.6 | 25.7 / 28.8 |
| `Gf8B` | 2.1.0 | snapshot | `mul_add_matrix_with` | 72.8 / 57.4 | 40.0 / 37.8 | 24.2 / 26.1 |
| `Gf8B` | 2.1.0 | snapshot | `mul_into_matrix_with` | 75.5 / 67.0 | 42.4 / 41.5 | 25.2 / 28.8 |
| `Gf8B` | 2.1.0 | snapshot | `mul_add_matrix_at` | 72.9 / 56.6 | 40.0 / 31.2 | 24.7 / 22.8 |
| `Gf8D` | 2.1.0 | snapshot | `fill(0)` + `mul_add_matrix` | 56.8 / 52.5 | 31.2 / 33.7 | 19.4 / 23.4 |
| `Gf8D` | 2.1.0 | snapshot | `mul_into_matrix` | 75.0 / 70.3 | 42.4 / 45.0 | 25.9 / 32.2 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `fill(0)` + `mul_add_matrix` | 56.6 / 49.9 | 31.0 / 32.3 | 19.3 / 21.4 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `fill(0)` + `mul_add_matrix` | 56.5 / 51.0 | 31.0 / 27.0 | 19.4 / 19.2 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_into_matrix` | 73.0 / 68.4 | 42.0 / 41.6 | 25.5 / 28.6 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_into_matrix` | 72.6 / 68.9 | 41.9 / 35.5 | 25.5 / 25.8 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_add_matrix_with` | 73.0 / 56.8 | 40.2 / 37.9 | 24.2 / 25.6 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_add_matrix_with` | 73.0 / 56.9 | 40.2 / 30.9 | 25.2 / 22.4 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_into_matrix_with` | 75.6 / 66.5 | 42.5 / 41.8 | 25.1 / 28.8 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_into_matrix_with` | 75.8 / 66.7 | 42.7 / 35.2 | 26.4 / 25.9 |
| `Gf8<AES>` | 2026-10-03 A | paired baseline | `mul_add_matrix_at` | 72.6 / 56.4 | 40.0 / 30.9 | 24.7 / 22.5 |
| `Gf8<AES>` | 2026-10-03 A | paired generic | `mul_add_matrix_at` | 73.0 / 57.3 | 39.6 / 31.0 | 24.7 / 22.4 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `fill(0)` + `mul_add_matrix` | 56.9 / 52.5 | 31.1 / 33.8 | 19.5 / 22.9 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `fill(0)` + `mul_add_matrix` | 56.6 / 52.8 | 31.1 / 33.8 | 19.6 / 22.9 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired baseline | `mul_into_matrix` | 75.5 / 70.3 | 42.3 / 45.6 | 26.0 / 31.9 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 A | paired generic | `mul_into_matrix` | 75.0 / 72.2 | 42.3 / 38.2 | 26.0 / 28.7 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `fill(0)` + `mul_add_matrix` | 56.6 / 50.0 | 30.9 / 32.3 | 19.2 / 21.4 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `fill(0)` + `mul_add_matrix` | 56.4 / 51.1 | 31.0 / 27.1 | 19.4 / 19.1 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_into_matrix` | 73.0 / 68.4 | 41.9 / 41.6 | 25.6 / 28.4 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_into_matrix` | 72.8 / 68.8 | 42.1 / 35.4 | 25.7 / 25.8 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_add_matrix_with` | 73.0 / 56.7 | 39.9 / 37.7 | 24.2 / 25.6 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_add_matrix_with` | 73.2 / 57.0 | 40.0 / 31.1 | 25.4 / 22.3 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_into_matrix_with` | 75.7 / 66.7 | 42.4 / 41.8 | 25.2 / 28.8 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_into_matrix_with` | 76.0 / 66.6 | 42.5 / 35.2 | 26.2 / 25.9 |
| `Gf8<AES>` | 2026-10-03 B | paired baseline | `mul_add_matrix_at` | 72.7 / 56.3 | 39.9 / 31.0 | 24.7 / 22.4 |
| `Gf8<AES>` | 2026-10-03 B | paired generic | `mul_add_matrix_at` | 73.0 / 57.4 | 40.2 / 31.2 | 24.7 / 22.4 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `fill(0)` + `mul_add_matrix` | 56.9 / 52.6 | 31.0 / 33.7 | 19.4 / 22.9 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `fill(0)` + `mul_add_matrix` | 56.8 / 52.6 | 31.3 / 33.8 | 19.6 / 22.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired baseline | `mul_into_matrix` | 75.3 / 70.2 | 42.2 / 45.0 | 26.0 / 31.6 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 B | paired generic | `mul_into_matrix` | 75.0 / 72.4 | 42.6 / 38.1 | 25.8 / 28.5 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `fill(0)` + `mul_add_matrix` | 56.5 / 49.8 | 30.9 / 32.3 | 19.3 / 21.4 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `fill(0)` + `mul_add_matrix` | 56.5 / 50.9 | 31.1 / 27.0 | 19.4 / 19.2 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_into_matrix` | 72.9 / 68.3 | 41.9 / 41.5 | 25.6 / 28.7 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_into_matrix` | 72.7 / 69.0 | 42.2 / 35.4 | 25.7 / 25.8 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_add_matrix_with` | 73.0 / 56.9 | 39.8 / 37.7 | 24.1 / 25.6 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_add_matrix_with` | 72.9 / 57.0 | 40.0 / 31.0 | 25.3 / 22.4 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_into_matrix_with` | 75.7 / 66.6 | 42.3 / 41.7 | 25.1 / 28.9 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_into_matrix_with` | 75.6 / 66.8 | 42.5 / 35.3 | 26.3 / 25.9 |
| `Gf8<AES>` | 2026-10-03 C | paired baseline | `mul_add_matrix_at` | 72.8 / 56.4 | 40.1 / 31.2 | 24.5 / 22.4 |
| `Gf8<AES>` | 2026-10-03 C | paired generic | `mul_add_matrix_at` | 73.2 / 57.2 | 40.1 / 31.1 | 24.6 / 22.5 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `fill(0)` + `mul_add_matrix` | 56.8 / 52.5 | 31.1 / 33.7 | 19.5 / 22.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `fill(0)` + `mul_add_matrix` | 56.8 / 52.8 | 31.3 / 34.0 | 19.6 / 22.9 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired baseline | `mul_into_matrix` | 75.5 / 70.0 | 42.4 / 45.0 | 25.9 / 31.8 |
| `Gf8<REED_SOLOMON>` | 2026-10-03 C | paired generic | `mul_into_matrix` | 75.3 / 72.3 | 42.5 / 38.2 | 26.0 / 28.7 |

### Row-wise addition

`Gf8B` `add_assign_rows`, ordinary `Vec<u8>` buffers.

| Row bytes | Rows | `Gf8B` (GiB/s) |
| --- | --- | --- |
| 64 | 1 | 6.67 / 10.5 |
| 64 | 2 | 12.9 / 27.4 |
| 64 | 4 | 24.1 / 46.0 |
| 64 | 8 | 38.6 / 58.5 |
| 64 | 16 | 48.7 / 68.6 |
| 64 | 32 | 65.5 / 89.8 |
| 1024 | 1 | 48.4 / 79.9 |
| 1024 | 2 | 65.5 / 89.9 |
| 1024 | 4 | 42.5 / 95.9 |
| 1024 | 8 | 80.4 / 129 |
| 1024 | 16 | 83.8 / 131 |
| 1024 | 32 | 34.9 / 66.5 |
| 65536 | 1 | 31.8 / 63.2 |
| 65536 | 2 | 31.9 / 63.5 |
| 65536 | 4 | 32.0 / 63.7 |
| 65536 | 8 | 36.9 / 47.3 |
| 65536 | 16 | 18.9 / 26.7 |
| 65536 | 32 | 18.5 / 25.8 |

### Network-size payloads

`Gf8B`, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the
row pitch.

| Payload bytes | `add_assign` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | `mul_add_scatter`, 4 rows (GiB/s) | `mul_add_scatter`, 16 rows (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| 64 | 6.86 / 12.1 | 3.67 / 13.6 | 6.53 / 15.0 | 8.48 / 14.3 | 11.4 / 19.6 |
| 256 | 26.1 / 40.4 | 14.6 / 34.4 | 22.1 / 52.6 | 26.9 / 34.8 | 31.0 / 63.8 |
| 512 | 44.1 / 67.2 | 27.4 / 57.8 | 51.2 / 88.7 | 38.1 / 63.3 | 43.8 / 83.5 |
| 1152 | 53.8 / 82.9 | 46.1 / 70.8 | 76.0 / 135 | 77.0 / 101 | 85.3 / 112 |
| 1168 | 59.6 / 98.1 | 48.4 / 79.3 | 85.1 / 134 | 58.9 / 84.8 | 57.7 / 91.9 |
| 1184 | 54.6 / 84.0 | 46.7 / 71.0 | 74.4 / 136 | 64.6 / 101 | 65.1 / 112 |
| 1200 | 55.4 / 98.3 | 48.8 / 80.2 | 79.0 / 134 | 51.7 / 85.2 | 56.2 / 91.3 |
| 1216 | 55.8 / 83.7 | 47.7 / 71.1 | 78.4 / 135 | 82.3 / 100 | 88.3 / 112 |
| 1232 | 60.6 / 101 | 50.9 / 81.1 | 87.4 / 135 | 60.1 / 86.4 | 58.0 / 94.3 |
| 1248 | 55.1 / 84.9 | 48.5 / 71.1 | 78.5 / 135 | 65.0 / 100 | 66.9 / 112 |
| 1400 | 60.6 / 103 | 40.2 / 68.6 | 63.2 / 117 | 42.8 / 60.6 | 46.0 / 62.8 |

### Fixed-offset multi-row operations

`Gf8D`; every source and destination base has the stated offset modulo 64.
Scatter has four destinations, gather has sixteen sources, and matrix has ten
sources and four destinations. Warm batches use 512 calls.

| Row bytes | Base offset mod 64 | `mul_add_scatter` (GiB/s) | `mul_add_gather` (GiB/s) | `mul_add_matrix` (GiB/s) |
| --- | --- | --- | --- | --- |
| 256 | 0 | 30.5 / 47.2 | 19.7 / 57.0 | 26.0 / 29.4 |
| 256 | 16 | 27.5 / 38.6 | 18.5 / 45.7 | 23.7 / 28.7 |
| 512 | 0 | 52.0 / 77.8 | 34.6 / 80.3 | 29.3 / 37.0 |
| 512 | 16 | 36.9 / 71.3 | 31.5 / 70.9 | 26.9 / 36.8 |
| 1024 | 0 | 79.3 / 112 | 55.5 / 100 | 35.4 / 42.6 |
| 1024 | 16 | 60.2 / 102 | 48.8 / 85.1 | 32.2 / 42.4 |
| 2048 | 0 | 95.3 / 134 | 78.9 / 108 | 39.4 / 45.5 |
| 2048 | 16 | 85.5 / 126 | 67.4 / 92.2 | 35.5 / 44.8 |
| 3072 | 0 | 103 / 118 | 80.0 / 91.4 | 40.8 / 46.6 |
| 3072 | 16 | 93.4 / 117 | 60.8 / 74.5 | 36.1 / 45.0 |
| 4096 | 0 | 89.7 / 113 | 76.6 / 89.0 | 40.7 / 45.8 |
| 4096 | 16 | 82.2 / 110 | 65.5 / 72.8 | 33.6 / 42.2 |
| 8192 | 0 | 92.1 / 119 | 86.3 / 91.4 | 41.8 / 47.6 |
| 8192 | 16 | 88.6 / 116 | 78.1 / 74.4 | 36.9 / 44.7 |
| 16384 | 0 | 48.8 / 74.0 | 89.5 / 92.8 | 42.3 / 48.2 |
| 16384 | 16 | 48.3 / 73.6 | 85.0 / 75.6 | 39.6 / 45.4 |

### Fixed-offset single-row multiplication

`Gf8D` `mul_add`, 16 KiB. Source and destination start at the stated offset
from a 64-byte boundary.

| Base offset mod 64 | Throughput (GiB/s) |
| --- | --- |
| 0 | 92.8 / 88.8 |
| 16 | 92.8 / 103 |
| 32 | 93.2 / 103 |
| 48 | 93.0 / 103 |

### Scatter destination alignment

`Gf8B` `mul_add_scatter`, eight destinations, ordinary `Vec<u8>` source; the
destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | Throughput (GiB/s) |
| --- | --- | --- |
| 64 KiB | 0 | 48.3 / 72.1 |
| 64 KiB | 16 | 48.3 / 72.2 |
| 256 KiB | 0 | 29.4 / 23.5 |
| 256 KiB | 16 | 29.4 / 23.5 |

`Gf8D`, one source and four 16 KiB destination rows, 64-byte-aligned buffers.

| Operation | Throughput (GiB/s) |
| --- | --- |
| `mul_add_scatter` | 48.4 / 73.4 |
| `mul_add_scatter_with` | 48.4 / 73.7 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | `Gf8B` (GiB/s) | `Gf8D` (GiB/s) |
| --- | --- | --- |
| 4 KiB | 129 / 174 | 128 / 171 |
| 16 KiB | 137 / 184 | 137 / 181 |
| 64 KiB | 51.3 / 76.3 | 51.3 / 68.9 |
| 256 KiB | 51.5 / 76.6 | 51.5 / 69.1 |
| 1 MiB | 49.3 / 54.3 | 49.1 / 53.5 |

### Large destinations

`Gf8B`, ordinary `Vec<u8>` buffers. Read-back XOR-folds all written 64-bit
words.

| Buffer size | `mul_into` (GiB/s) | `mul_into` + read (GiB/s) | `mul_add` (GiB/s) |
| --- | --- | --- | --- |
| 1 MiB | 18.9 / 26.5 | 13.7 / 21.0 | 18.9 / 27.0 |
| 8 MiB | 18.2 / 41.8 | 7.71 / 16.6 | 8.08 / 25.8 |
| 32 MiB | 9.44 / 28.0 | 5.92 / 14.2 | 6.21 / 16.6 |

`Gf8D`, 64-byte-aligned buffers.

| Operation | Buffer size | Throughput (GiB/s) |
| --- | --- | --- |
| `mul_into` | 4 MiB | 18.2 / 41.7 |

## Competitors

Polynomial `0x11D`, page-aligned buffers, and matched coding coefficients.
ISA-L uses its field kernels; klauspost uses `Encode` with a supplied coding
matrix, or `EncodeIdx` for accumulation.

| Operation | Sources × destinations | Row size | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | klauspost/reedsolomon (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| `dst = c * src` | 1 × 1 | 64 KiB | 40.1 / 64.6 | 18.0 / 45.6 | 33.8 / 56.8 |
| `dst ^= c * src` | 1 × 1 | 64 KiB | 40.9 / 64.9 | 41.2 / 65.2 | 29.2 / 49.1 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 4 KiB | 89.5 / 104 | 82.9 / 97.6 | 30.6 / 50.2 |
| `dst = Σ c[i] * src[i]` | 16 × 1 | 16 KiB | 95.2 / 97.7 | 74.9 / 90.0 | 35.7 / 57.1 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 4 KiB | 75.0 / 75.0 | 65.0 / 69.5 | 54.0 / 74.6 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 16 KiB | 80.7 / 82.7 | 53.1 / 65.0 | 67.6 / 84.7 |
| `dst = Σ c[i] * src[i]` | 10 × 2 | 64 KiB | 83.9 / 79.7 | 54.3 / 63.5 | 73.5 / 85.3 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 4 KiB | 39.3 / 47.5 | 35.6 / 45.0 | 32.2 / 44.4 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 16 KiB | 42.8 / 50.8 | 36.7 / 39.4 | 38.3 / 50.4 |
| `dst = Σ c[i] * src[i]` | 10 × 4 | 64 KiB | 43.4 / 50.4 | 35.7 / 34.1 | 40.7 / 47.8 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 4 KiB | 26.5 / 32.8 | 24.4 / 31.3 | 22.5 / 30.3 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 16 KiB | 26.9 / 34.6 | 25.1 / 29.5 | 25.9 / 34.0 |
| `dst = Σ c[i] * src[i]` | 10 × 6 | 64 KiB | 27.0 / 34.1 | 23.3 / 23.9 | 26.6 / 30.6 |

- Preparation and Go bridge pointer staging are excluded.
