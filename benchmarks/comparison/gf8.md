# Byte fields: `gf8b`, `gf8d`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf8.md), [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `-`; v3: `v4x` / `-` |
| Resolved field backends | v2 `gf8b`: `v4x` / -; v2 `gf8d`: `v4x` / -; v3 `gf8b`: `v4x` / -; v3 `gf8d`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |
| Competitors | Willow Cove (i5-1135G7) v2: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64; Willow Cove (i5-1135G7) v3: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64 |

## gf8b

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 1.0000 / - |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 1.0088 / - |
| add_assign | 256 KiB | 0.9998 / - |
| add_assign | 8 MiB | 1.0029 / - |
| add_assign_scalar | 4 KiB | 0.9706 / - |
| add_assign_scalar | 256 KiB | 1.0000 / - |
| add_assign_scalar | 8 MiB | 0.9916 / - |
| mul_add | 4 KiB | 0.9910 / - |
| mul_add | 256 KiB | 0.9997 / - |
| mul_add | 8 MiB | 1.0216 / - |
| mul_assign | 4 KiB | 1.0263 / - |
| mul_assign | 256 KiB | 1.0004 / - |
| mul_assign | 8 MiB | 0.9962 / - |
| mul_elementwise | 4 KiB | 0.9985 / - |
| mul_elementwise | 256 KiB | 0.9658 / - |
| mul_elementwise | 8 MiB | 1.0058 / - |
| mul_elementwise_assign | 4 KiB | 0.9292 / - |
| mul_elementwise_assign | 256 KiB | 0.9520 / - |
| mul_elementwise_assign | 8 MiB | 0.9778 / - |
| mul_into | 4 KiB | 1.0081 / - |
| mul_into | 256 KiB | 0.9995 / - |
| mul_into | 8 MiB | 1.0034 / - |
| sub_assign_scalar | 4 KiB | 1.0114 / - |
| sub_assign_scalar | 256 KiB | 1.0002 / - |
| sub_assign_scalar | 8 MiB | 0.9938 / - |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 0.9998 / - |
| mul_elementwise | 64 KiB | 1.0051 / - |
| mul_elementwise_assign | 64 KiB | 1.0002 / - |
| mul_into | 64 KiB | 1.0000 / - |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | Case | v3 speed factor |
| --- | --- | --- |
| 16 B | mul_add | 1.3377 / - |
| 16 B | mul_add_with | 1.3959 / - |
| 32 B | mul_add | 1.3276 / - |
| 32 B | mul_add_with | 1.4059 / - |
| 64 B | mul_add | 1.2734 / - |
| 64 B | mul_add_with | 1.3340 / - |
| 128 B | mul_add | 1.3426 / - |
| 128 B | mul_add_with | 1.4075 / - |
| 256 B | mul_add | 1.1663 / - |
| 256 B | mul_add_with | 1.2254 / - |
| 512 B | mul_add | 1.0408 / - |
| 512 B | mul_add_with | 1.0629 / - |
| 1 KiB | mul_add | 1.0751 / - |
| 1 KiB | mul_add_with | 1.0496 / - |
| 2 KiB | mul_add | 1.0057 / - |
| 2 KiB | mul_add_with | 1.0062 / - |
| 4 KiB | mul_add | 1.0034 / - |
| 4 KiB | mul_add_with | 1.0077 / - |
| 16 KiB | mul_add | 1.0060 / - |
| 16 KiB | mul_add_with | 1.0028 / - |
| 64 KiB | mul_add | 0.9986 / - |
| 64 KiB | mul_add_with | 0.9986 / - |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather | 2 rows | 1.0037 / - |
| mul_add_gather | 4 rows | 1.0243 / - |
| mul_add_gather | 8 rows | 1.0295 / - |
| mul_add_gather | 16 rows | 1.0134 / - |
| mul_add_gather_with | 2 rows | 1.0034 / - |
| mul_add_gather_with | 4 rows | 1.0217 / - |
| mul_add_gather_with | 8 rows | 1.0267 / - |
| mul_add_gather_with | 16 rows | 1.0144 / - |
| mul_add_matrix | 2 rows | 0.9942 / - |
| mul_add_matrix | 4 rows | 1.0047 / - |
| mul_add_matrix | 8 rows | 1.0129 / - |
| mul_add_matrix | 16 rows | 0.9886 / - |
| mul_add_matrix_with | 2 rows | 1.0024 / - |
| mul_add_matrix_with | 4 rows | 1.0012 / - |
| mul_add_matrix_with | 8 rows | 1.0382 / - |
| mul_add_matrix_with | 16 rows | 1.0114 / - |
| mul_add_scatter | 2 rows | 0.9965 / - |
| mul_add_scatter | 4 rows | 1.0018 / - |
| mul_add_scatter | 8 rows | 1.0014 / - |
| mul_add_scatter | 16 rows | 1.0258 / - |
| mul_add_scatter_with | 2 rows | 0.9963 / - |
| mul_add_scatter_with | 4 rows | 1.0011 / - |
| mul_add_scatter_with | 8 rows | 1.0008 / - |
| mul_add_scatter_with | 16 rows | 1.0253 / - |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_into_gather_with | 4 KiB | 0.9940 / - |
| mul_into_gather_with | 16 KiB | 0.9966 / - |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 1.0003 / - |
| fill(0) + mul_add_matrix | 4 destinations | 1.0033 / - |
| fill(0) + mul_add_matrix | 6 destinations | 1.0052 / - |
| mul_add_matrix_at | 2 destinations | 1.0017 / - |
| mul_add_matrix_at | 4 destinations | 1.0007 / - |
| mul_add_matrix_at | 6 destinations | 0.9974 / - |
| mul_add_matrix_with | 2 destinations | 1.0035 / - |
| mul_add_matrix_with | 4 destinations | 0.9898 / - |
| mul_add_matrix_with | 6 destinations | 1.0489 / - |
| mul_into_matrix | 2 destinations | 1.0024 / - |
| mul_into_matrix | 4 destinations | 0.9968 / - |
| mul_into_matrix | 6 destinations | 0.9942 / - |
| mul_into_matrix_with | 2 destinations | 1.0031 / - |
| mul_into_matrix_with | 4 destinations | 1.0003 / - |
| mul_into_matrix_with | 6 destinations | 1.0293 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 1.0190 / - |
| 1 KiB | 16 | 1.0034 / - |
| 1 KiB | 2 | 1.0127 / - |
| 1 KiB | 32 | 0.9964 / - |
| 1 KiB | 4 | 1.0157 / - |
| 1 KiB | 8 | 1.0058 / - |
| 64 B | 1 | 0.9809 / - |
| 64 B | 16 | 1.0182 / - |
| 64 B | 2 | 1.0149 / - |
| 64 B | 32 | 1.0081 / - |
| 64 B | 4 | 1.0017 / - |
| 64 B | 8 | 0.9516 / - |
| 64 KiB | 1 | 0.9982 / - |
| 64 KiB | 16 | 1.0002 / - |
| 64 KiB | 2 | 0.9999 / - |
| 64 KiB | 32 | 0.9991 / - |
| 64 KiB | 4 | 0.9999 / - |
| 64 KiB | 8 | 0.9964 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 1.0120 / - |
| add_assign per row | 1 KiB | 16 | 1.0290 / - |
| add_assign per row | 1 KiB | 2 | 0.9645 / - |
| add_assign per row | 1 KiB | 32 | 0.9688 / - |
| add_assign per row | 1 KiB | 4 | 1.0181 / - |
| add_assign per row | 1 KiB | 8 | 1.0343 / - |
| add_assign per row | 64 B | 1 | 0.9939 / - |
| add_assign per row | 64 B | 16 | 0.9240 / - |
| add_assign per row | 64 B | 2 | 0.9793 / - |
| add_assign per row | 64 B | 32 | 0.9474 / - |
| add_assign per row | 64 B | 4 | 0.9998 / - |
| add_assign per row | 64 B | 8 | 0.9999 / - |
| add_assign per row | 64 KiB | 1 | 0.9992 / - |
| add_assign per row | 64 KiB | 16 | 1.0004 / - |
| add_assign per row | 64 KiB | 2 | 0.9996 / - |
| add_assign per row | 64 KiB | 32 | 0.9989 / - |
| add_assign per row | 64 KiB | 4 | 1.0001 / - |
| add_assign per row | 64 KiB | 8 | 0.9981 / - |
| add_assign | 1 KiB | 1 | 1.0176 / - |
| add_assign | 1 KiB | 16 | 1.0112 / - |
| add_assign | 1 KiB | 2 | 1.0091 / - |
| add_assign | 1 KiB | 32 | 1.0010 / - |
| add_assign | 1 KiB | 4 | 0.9795 / - |
| add_assign | 1 KiB | 8 | 1.0220 / - |
| add_assign | 64 B | 1 | 0.9989 / - |
| add_assign | 64 B | 16 | 1.0134 / - |
| add_assign | 64 B | 2 | 0.9977 / - |
| add_assign | 64 B | 32 | 1.0088 / - |
| add_assign | 64 B | 4 | 0.9997 / - |
| add_assign | 64 B | 8 | 0.9880 / - |
| add_assign | 64 KiB | 1 | 0.9993 / - |
| add_assign | 64 KiB | 16 | 1.0002 / - |
| add_assign | 64 KiB | 2 | 1.0005 / - |
| add_assign | 64 KiB | 32 | 1.0027 / - |
| add_assign | 64 KiB | 4 | 1.0005 / - |
| add_assign | 64 KiB | 8 | 0.9961 / - |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 64 B | 0.9998 / - |
| add_assign | 256 B | 1.0470 / - |
| add_assign | 512 B | 0.9712 / - |
| add_assign | 1152 B | 1.0016 / - |
| add_assign | 1168 B | 1.0305 / - |
| add_assign | 1184 B | 1.0229 / - |
| add_assign | 1200 B | 0.9994 / - |
| add_assign | 1216 B | 1.0161 / - |
| add_assign | 1232 B | 1.0191 / - |
| add_assign | 1248 B | 1.0108 / - |
| add_assign | 1400 B | 1.0006 / - |
| mul_add | 64 B | 1.2807 / - |
| mul_add | 256 B | 1.2447 / - |
| mul_add | 512 B | 1.1095 / - |
| mul_add | 1152 B | 1.1107 / - |
| mul_add | 1168 B | 1.0858 / - |
| mul_add | 1184 B | 1.1300 / - |
| mul_add | 1200 B | 1.2633 / - |
| mul_add | 1216 B | 1.0962 / - |
| mul_add | 1232 B | 1.0764 / - |
| mul_add | 1248 B | 1.0972 / - |
| mul_add | 1400 B | 1.2046 / - |
| mul_add_scatter (16 rows) | 64 B | 1.1576 / - |
| mul_add_scatter (16 rows) | 256 B | 1.2092 / - |
| mul_add_scatter (16 rows) | 512 B | 1.0234 / - |
| mul_add_scatter (16 rows) | 1152 B | 1.0184 / - |
| mul_add_scatter (16 rows) | 1168 B | 1.0907 / - |
| mul_add_scatter (16 rows) | 1184 B | 1.0476 / - |
| mul_add_scatter (16 rows) | 1200 B | 1.0629 / - |
| mul_add_scatter (16 rows) | 1216 B | 0.9999 / - |
| mul_add_scatter (16 rows) | 1232 B | 1.0924 / - |
| mul_add_scatter (16 rows) | 1248 B | 1.0403 / - |
| mul_add_scatter (16 rows) | 1400 B | 0.9567 / - |
| mul_add_scatter (4 rows) | 64 B | 1.0525 / - |
| mul_add_scatter (4 rows) | 256 B | 1.1380 / - |
| mul_add_scatter (4 rows) | 512 B | 0.9631 / - |
| mul_add_scatter (4 rows) | 1152 B | 0.9934 / - |
| mul_add_scatter (4 rows) | 1168 B | 1.0631 / - |
| mul_add_scatter (4 rows) | 1184 B | 1.0244 / - |
| mul_add_scatter (4 rows) | 1200 B | 1.0656 / - |
| mul_add_scatter (4 rows) | 1216 B | 0.9881 / - |
| mul_add_scatter (4 rows) | 1232 B | 1.0709 / - |
| mul_add_scatter (4 rows) | 1248 B | 1.0860 / - |
| mul_add_scatter (4 rows) | 1400 B | 0.9503 / - |
| mul_assign | 64 B | 1.0912 / - |
| mul_assign | 256 B | 1.0003 / - |
| mul_assign | 512 B | 1.0130 / - |
| mul_assign | 1152 B | 1.0949 / - |
| mul_assign | 1168 B | 1.0352 / - |
| mul_assign | 1184 B | 1.0783 / - |
| mul_assign | 1200 B | 1.0787 / - |
| mul_assign | 1216 B | 1.0786 / - |
| mul_assign | 1232 B | 1.0055 / - |
| mul_assign | 1248 B | 1.0499 / - |
| mul_assign | 1400 B | 1.0564 / - |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | v3 speed factor |
| --- | --- | --- |
| 256 KiB | 0 | 0.9988 / - |
| 256 KiB | 16 | 0.9975 / - |
| 64 KiB | 0 | 1.0002 / - |
| 64 KiB | 16 | 1.0005 / - |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 1.0043 / - |
| 16 KiB | 0.9931 / - |
| 256 KiB | 1.0003 / - |
| 4 KiB | 0.9685 / - |
| 64 KiB | 1.0012 / - |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | v3 speed factor |
| --- | --- |
| 256 KiB | 1.0000 / - |
| 4 KiB | 0.9992 / - |
| 64 MiB | 0.9991 / - |
| 8 MiB | 1.0151 / - |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | Case | v3 speed factor |
| --- | --- | --- |
| 1 MiB | mul_into | 0.9929 / - |
| 1 MiB | mul_into + read | 1.0036 / - |
| 1 MiB | mul_add | 0.9923 / - |
| 32 MiB | mul_into | 0.9975 / - |
| 32 MiB | mul_into + read | 0.9975 / - |
| 32 MiB | mul_add | 1.0042 / - |
| 8 MiB | mul_into | 1.0006 / - |
| 8 MiB | mul_into + read | 0.9926 / - |
| 8 MiB | mul_add | 1.0170 / - |

## gf8d

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 1.1579 / - |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0001 / - |
| mul_elementwise | 64 KiB | 1.0011 / - |
| mul_elementwise_assign | 64 KiB | 1.0032 / - |
| mul_into | 64 KiB | 1.0000 / - |
| mul_into | 4 MiB | 0.9994 / - |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather_with | 4 KiB | 1.2227 / - |
| mul_add_gather_with | 16 KiB | 1.0662 / - |
| mul_into_gather | 4 KiB | 1.0010 / - |
| mul_into_gather | 16 KiB | 0.9989 / - |
| mul_into_gather_with | 4 KiB | 0.9740 / - |
| mul_into_gather_with | 16 KiB | 0.9937 / - |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9995 / - |
| fill(0) + mul_add_matrix | 4 destinations | 0.9715 / - |
| fill(0) + mul_add_matrix | 6 destinations | 0.9974 / - |
| mul_into_matrix | 2 destinations | 1.0028 / - |
| mul_into_matrix | 4 destinations | 0.9737 / - |
| mul_into_matrix | 6 destinations | 0.9826 / - |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | Case | v3 speed factor |
| --- | --- | --- | --- |
| 1 KiB | 0 | mul_add_scatter | 1.0183 / - |
| 1 KiB | 0 | mul_add_gather | 1.9896 / - |
| 1 KiB | 0 | mul_add_matrix | 1.0106 / - |
| 1 KiB | 16 | mul_add_scatter | 1.0168 / - |
| 1 KiB | 16 | mul_add_gather | 1.7762 / - |
| 1 KiB | 16 | mul_add_matrix | 1.0093 / - |
| 16 KiB | 0 | mul_add_scatter | 0.9962 / - |
| 16 KiB | 0 | mul_add_gather | 1.0185 / - |
| 16 KiB | 0 | mul_add_matrix | 1.0085 / - |
| 16 KiB | 16 | mul_add_scatter | 1.0005 / - |
| 16 KiB | 16 | mul_add_gather | 0.9598 / - |
| 16 KiB | 16 | mul_add_matrix | 1.0061 / - |
| 2 KiB | 0 | mul_add_scatter | 0.9822 / - |
| 2 KiB | 0 | mul_add_gather | 1.5574 / - |
| 2 KiB | 0 | mul_add_matrix | 1.0017 / - |
| 2 KiB | 16 | mul_add_scatter | 1.0274 / - |
| 2 KiB | 16 | mul_add_gather | 1.4070 / - |
| 2 KiB | 16 | mul_add_matrix | 1.0239 / - |
| 256 B | 0 | mul_add_scatter | 1.0696 / - |
| 256 B | 0 | mul_add_gather | 3.5832 / - |
| 256 B | 0 | mul_add_matrix | 1.0065 / - |
| 256 B | 16 | mul_add_scatter | 1.1652 / - |
| 256 B | 16 | mul_add_gather | 3.0648 / - |
| 256 B | 16 | mul_add_matrix | 1.0151 / - |
| 3 KiB | 0 | mul_add_scatter | 0.9403 / - |
| 3 KiB | 0 | mul_add_gather | 1.2877 / - |
| 3 KiB | 0 | mul_add_matrix | 1.0020 / - |
| 3 KiB | 16 | mul_add_scatter | 0.9609 / - |
| 3 KiB | 16 | mul_add_gather | 0.9920 / - |
| 3 KiB | 16 | mul_add_matrix | 1.0148 / - |
| 4 KiB | 0 | mul_add_scatter | 1.0085 / - |
| 4 KiB | 0 | mul_add_gather | 1.1979 / - |
| 4 KiB | 0 | mul_add_matrix | 1.0042 / - |
| 4 KiB | 16 | mul_add_scatter | 1.0085 / - |
| 4 KiB | 16 | mul_add_gather | 0.9832 / - |
| 4 KiB | 16 | mul_add_matrix | 1.0118 / - |
| 512 B | 0 | mul_add_scatter | 1.0545 / - |
| 512 B | 0 | mul_add_gather | 2.7601 / - |
| 512 B | 0 | mul_add_matrix | 1.0118 / - |
| 512 B | 16 | mul_add_scatter | 1.0215 / - |
| 512 B | 16 | mul_add_gather | 2.3422 / - |
| 512 B | 16 | mul_add_matrix | 1.0118 / - |
| 8 KiB | 0 | mul_add_scatter | 1.0026 / - |
| 8 KiB | 0 | mul_add_gather | 1.0885 / - |
| 8 KiB | 0 | mul_add_matrix | 1.0575 / - |
| 8 KiB | 16 | mul_add_scatter | 1.0042 / - |
| 8 KiB | 16 | mul_add_gather | 0.9701 / - |
| 8 KiB | 16 | mul_add_matrix | 1.0394 / - |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | v3 speed factor |
| --- | --- |
| 0 | 1.0169 / - |
| 16 | 1.0122 / - |
| 32 | 1.0145 / - |
| 48 | 1.0133 / - |

### Scatter destination alignment

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | v3 speed factor |
| --- | --- |
| mul_add_scatter | 1.0029 / - |
| mul_add_scatter_with | 1.0026 / - |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 0.9479 / - |
| 16 KiB | 0.9987 / - |
| 256 KiB | 1.0001 / - |
| 4 KiB | 0.9777 / - |
| 64 KiB | 1.0006 / - |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst = c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9998 / - |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=262144; preparation=one-shot; row_bytes=16384; sources=16 | 0.9782 / - |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=65536; preparation=one-shot; row_bytes=4096; sources=16 | 1.0000 / - |
| dst ^= c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9990 / - |
| systematic encode | alignment=page; destinations=2; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0042 / - |
| systematic encode | alignment=page; destinations=2; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0167 / - |
| systematic encode | alignment=page; destinations=2; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0003 / - |
| systematic encode | alignment=page; destinations=4; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0026 / - |
| systematic encode | alignment=page; destinations=4; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0105 / - |
| systematic encode | alignment=page; destinations=4; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 0.9999 / - |
| systematic encode | alignment=page; destinations=6; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0013 / - |
| systematic encode | alignment=page; destinations=6; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0049 / - |
| systematic encode | alignment=page; destinations=6; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0023 / - |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
