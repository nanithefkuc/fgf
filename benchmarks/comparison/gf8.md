# Byte fields: `gf8b`, `gf8d`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf8.md), [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `v3_gfni_crypto`; v3: `v4x` / `v3_gfni_crypto` |
| Resolved field backends | v2 `gf8b`: `v4x` / `v3_gfni_crypto`; v2 `gf8d`: `v4x` / `v3_gfni_crypto`; v3 `gf8b`: `v4x` / `v3_gfni_crypto`; v3 `gf8d`: `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |
| Competitors | Willow Cove (i5-1135G7) v2: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64; Willow Cove (i5-1135G7) v3: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64; Golden Cove (i7-12700K) v2: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64; Golden Cove (i7-12700K) v3: Intel ISA-L 2.32.0; klauspost/reedsolomon v1.14.2 via Go go1.27.1-X:nodwarf5 linux/amd64 |

## gf8b

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 1.0000 / 1.0002 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 1.0088 / 1.0003 |
| add_assign | 256 KiB | 0.9998 / 1.0005 |
| add_assign | 8 MiB | 1.0029 / 0.9988 |
| add_assign_scalar | 4 KiB | 0.9706 / 1.0003 |
| add_assign_scalar | 256 KiB | 1.0000 / 0.9995 |
| add_assign_scalar | 8 MiB | 0.9916 / 1.0023 |
| mul_add | 4 KiB | 0.9910 / 1.0088 |
| mul_add | 256 KiB | 0.9997 / 1.0050 |
| mul_add | 8 MiB | 1.0216 / 0.9787 |
| mul_assign | 4 KiB | 1.0263 / 1.0004 |
| mul_assign | 256 KiB | 1.0004 / 1.0003 |
| mul_assign | 8 MiB | 0.9962 / 0.9934 |
| mul_elementwise | 4 KiB | 0.9985 / 1.0180 |
| mul_elementwise | 256 KiB | 0.9658 / 0.9418 |
| mul_elementwise | 8 MiB | 1.0058 / 1.0079 |
| mul_elementwise_assign | 4 KiB | 0.9292 / 0.9861 |
| mul_elementwise_assign | 256 KiB | 0.9520 / 1.0018 |
| mul_elementwise_assign | 8 MiB | 0.9778 / 1.0063 |
| mul_into | 4 KiB | 1.0081 / 1.0162 |
| mul_into | 256 KiB | 0.9995 / 1.0007 |
| mul_into | 8 MiB | 1.0034 / 1.0038 |
| sub_assign_scalar | 4 KiB | 1.0114 / 1.0002 |
| sub_assign_scalar | 256 KiB | 1.0002 / 0.9998 |
| sub_assign_scalar | 8 MiB | 0.9938 / 1.0136 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 0.9998 / 0.9996 |
| mul_elementwise | 64 KiB | 1.0051 / 0.9983 |
| mul_elementwise_assign | 64 KiB | 1.0002 / 1.0015 |
| mul_into | 64 KiB | 1.0000 / 1.0001 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | Case | v3 speed factor |
| --- | --- | --- |
| 16 B | mul_add | 1.3377 / 0.9589 |
| 16 B | mul_add_with | 1.3959 / 0.9995 |
| 32 B | mul_add | 1.3276 / 1.0002 |
| 32 B | mul_add_with | 1.4059 / 1.0004 |
| 64 B | mul_add | 1.2734 / 0.9989 |
| 64 B | mul_add_with | 1.3340 / 0.9972 |
| 128 B | mul_add | 1.3426 / 0.9590 |
| 128 B | mul_add_with | 1.4075 / 1.0016 |
| 256 B | mul_add | 1.1663 / 0.9662 |
| 256 B | mul_add_with | 1.2254 / 1.0034 |
| 512 B | mul_add | 1.0408 / 1.0382 |
| 512 B | mul_add_with | 1.0629 / 1.0392 |
| 1 KiB | mul_add | 1.0751 / 1.0188 |
| 1 KiB | mul_add_with | 1.0496 / 1.0260 |
| 2 KiB | mul_add | 1.0057 / 1.0120 |
| 2 KiB | mul_add_with | 1.0062 / 1.0140 |
| 4 KiB | mul_add | 1.0034 / 1.0093 |
| 4 KiB | mul_add_with | 1.0077 / 1.0094 |
| 16 KiB | mul_add | 1.0060 / 1.0093 |
| 16 KiB | mul_add_with | 1.0028 / 1.0094 |
| 64 KiB | mul_add | 0.9986 / 1.0050 |
| 64 KiB | mul_add_with | 0.9986 / 1.0055 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather | 2 rows | 1.0037 / 1.0024 |
| mul_add_gather | 4 rows | 1.0243 / 1.0116 |
| mul_add_gather | 8 rows | 1.0295 / 1.0073 |
| mul_add_gather | 16 rows | 1.0134 / 0.9984 |
| mul_add_gather_with | 2 rows | 1.0034 / 0.9941 |
| mul_add_gather_with | 4 rows | 1.0217 / 1.0145 |
| mul_add_gather_with | 8 rows | 1.0267 / 1.0154 |
| mul_add_gather_with | 16 rows | 1.0144 / 1.0030 |
| mul_add_matrix | 2 rows | 0.9942 / 0.9995 |
| mul_add_matrix | 4 rows | 1.0047 / 0.9862 |
| mul_add_matrix | 8 rows | 1.0129 / 0.9911 |
| mul_add_matrix | 16 rows | 0.9886 / 0.9958 |
| mul_add_matrix_with | 2 rows | 1.0024 / 0.9975 |
| mul_add_matrix_with | 4 rows | 1.0012 / 0.9766 |
| mul_add_matrix_with | 8 rows | 1.0382 / 1.0109 |
| mul_add_matrix_with | 16 rows | 1.0114 / 0.9938 |
| mul_add_scatter | 2 rows | 0.9965 / 1.0014 |
| mul_add_scatter | 4 rows | 1.0018 / 1.0015 |
| mul_add_scatter | 8 rows | 1.0014 / 0.9952 |
| mul_add_scatter | 16 rows | 1.0258 / 0.9985 |
| mul_add_scatter_with | 2 rows | 0.9963 / 1.0005 |
| mul_add_scatter_with | 4 rows | 1.0011 / 0.9998 |
| mul_add_scatter_with | 8 rows | 1.0008 / 0.9948 |
| mul_add_scatter_with | 16 rows | 1.0253 / 0.9988 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_into_gather_with | 4 KiB | 0.9940 / 1.0343 |
| mul_into_gather_with | 16 KiB | 0.9966 / 1.0098 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 1.0003 / 1.0145 |
| fill(0) + mul_add_matrix | 4 destinations | 1.0033 / 0.8404 |
| fill(0) + mul_add_matrix | 6 destinations | 1.0052 / 0.9001 |
| mul_add_matrix_at | 2 destinations | 1.0017 / 1.0115 |
| mul_add_matrix_at | 4 destinations | 1.0007 / 0.9659 |
| mul_add_matrix_at | 6 destinations | 0.9974 / 1.0145 |
| mul_add_matrix_with | 2 destinations | 1.0035 / 0.9917 |
| mul_add_matrix_with | 4 destinations | 0.9898 / 0.7901 |
| mul_add_matrix_with | 6 destinations | 1.0489 / 0.8907 |
| mul_into_matrix | 2 destinations | 1.0024 / 0.9817 |
| mul_into_matrix | 4 destinations | 0.9968 / 0.8465 |
| mul_into_matrix | 6 destinations | 0.9942 / 0.9022 |
| mul_into_matrix_with | 2 destinations | 1.0031 / 1.0226 |
| mul_into_matrix_with | 4 destinations | 1.0003 / 0.8366 |
| mul_into_matrix_with | 6 destinations | 1.0293 / 0.9236 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 1.0190 / 0.9980 |
| 1 KiB | 16 | 1.0034 / 0.9979 |
| 1 KiB | 2 | 1.0127 / 0.9989 |
| 1 KiB | 32 | 0.9964 / 0.9974 |
| 1 KiB | 4 | 1.0157 / 1.0000 |
| 1 KiB | 8 | 1.0058 / 1.0180 |
| 64 B | 1 | 0.9809 / 1.2459 |
| 64 B | 16 | 1.0182 / 1.0022 |
| 64 B | 2 | 1.0149 / 1.0471 |
| 64 B | 32 | 1.0081 / 1.0012 |
| 64 B | 4 | 1.0017 / 0.8872 |
| 64 B | 8 | 0.9516 / 1.0014 |
| 64 KiB | 1 | 0.9982 / 0.9997 |
| 64 KiB | 16 | 1.0002 / 0.9897 |
| 64 KiB | 2 | 0.9999 / 0.9999 |
| 64 KiB | 32 | 0.9991 / 1.0011 |
| 64 KiB | 4 | 0.9999 / 1.0001 |
| 64 KiB | 8 | 0.9964 / 1.0286 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 1.0120 / 1.0001 |
| add_assign per row | 1 KiB | 16 | 1.0290 / 1.0064 |
| add_assign per row | 1 KiB | 2 | 0.9645 / 0.9999 |
| add_assign per row | 1 KiB | 32 | 0.9688 / 1.0008 |
| add_assign per row | 1 KiB | 4 | 1.0181 / 1.0032 |
| add_assign per row | 1 KiB | 8 | 1.0343 / 0.9986 |
| add_assign per row | 64 B | 1 | 0.9939 / 1.0939 |
| add_assign per row | 64 B | 16 | 0.9240 / 1.0542 |
| add_assign per row | 64 B | 2 | 0.9793 / 1.0492 |
| add_assign per row | 64 B | 32 | 0.9474 / 0.9536 |
| add_assign per row | 64 B | 4 | 0.9998 / 1.0675 |
| add_assign per row | 64 B | 8 | 0.9999 / 0.9494 |
| add_assign per row | 64 KiB | 1 | 0.9992 / 0.9998 |
| add_assign per row | 64 KiB | 16 | 1.0004 / 0.9877 |
| add_assign per row | 64 KiB | 2 | 0.9996 / 1.0000 |
| add_assign per row | 64 KiB | 32 | 0.9989 / 1.0008 |
| add_assign per row | 64 KiB | 4 | 1.0001 / 0.9999 |
| add_assign per row | 64 KiB | 8 | 0.9981 / 1.0306 |
| add_assign | 1 KiB | 1 | 1.0176 / 1.0055 |
| add_assign | 1 KiB | 16 | 1.0112 / 1.0598 |
| add_assign | 1 KiB | 2 | 1.0091 / 1.0075 |
| add_assign | 1 KiB | 32 | 1.0010 / 1.0070 |
| add_assign | 1 KiB | 4 | 0.9795 / 1.0017 |
| add_assign | 1 KiB | 8 | 1.0220 / 1.0473 |
| add_assign | 64 B | 1 | 0.9989 / 1.0488 |
| add_assign | 64 B | 16 | 1.0134 / 1.0022 |
| add_assign | 64 B | 2 | 0.9977 / 0.8985 |
| add_assign | 64 B | 32 | 1.0088 / 1.0057 |
| add_assign | 64 B | 4 | 0.9997 / 0.9618 |
| add_assign | 64 B | 8 | 0.9880 / 1.0005 |
| add_assign | 64 KiB | 1 | 0.9993 / 0.9999 |
| add_assign | 64 KiB | 16 | 1.0002 / 0.9958 |
| add_assign | 64 KiB | 2 | 1.0005 / 1.0001 |
| add_assign | 64 KiB | 32 | 1.0027 / 1.0012 |
| add_assign | 64 KiB | 4 | 1.0005 / 1.0001 |
| add_assign | 64 KiB | 8 | 0.9961 / 1.0261 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 64 B | 0.9998 / 0.9975 |
| add_assign | 256 B | 1.0470 / 0.9231 |
| add_assign | 512 B | 0.9712 / 0.9964 |
| add_assign | 1152 B | 1.0016 / 1.0001 |
| add_assign | 1168 B | 1.0305 / 1.0029 |
| add_assign | 1184 B | 1.0229 / 0.9997 |
| add_assign | 1200 B | 0.9994 / 1.0010 |
| add_assign | 1216 B | 1.0161 / 1.0020 |
| add_assign | 1232 B | 1.0191 / 1.0152 |
| add_assign | 1248 B | 1.0108 / 1.0041 |
| add_assign | 1400 B | 1.0006 / 0.9925 |
| mul_add | 64 B | 1.2807 / 0.9579 |
| mul_add | 256 B | 1.2447 / 0.9635 |
| mul_add | 512 B | 1.1095 / 1.0006 |
| mul_add | 1152 B | 1.1107 / 1.0149 |
| mul_add | 1168 B | 1.0858 / 0.9847 |
| mul_add | 1184 B | 1.1300 / 1.0189 |
| mul_add | 1200 B | 1.2633 / 0.9851 |
| mul_add | 1216 B | 1.0962 / 1.0153 |
| mul_add | 1232 B | 1.0764 / 1.0002 |
| mul_add | 1248 B | 1.0972 / 1.0288 |
| mul_add | 1400 B | 1.2046 / 1.0079 |
| mul_add_scatter (16 rows) | 64 B | 1.1576 / 1.4293 |
| mul_add_scatter (16 rows) | 256 B | 1.2092 / 1.1999 |
| mul_add_scatter (16 rows) | 512 B | 1.0234 / 0.9840 |
| mul_add_scatter (16 rows) | 1152 B | 1.0184 / 1.0089 |
| mul_add_scatter (16 rows) | 1168 B | 1.0907 / 1.0841 |
| mul_add_scatter (16 rows) | 1184 B | 1.0476 / 1.0310 |
| mul_add_scatter (16 rows) | 1200 B | 1.0629 / 1.0840 |
| mul_add_scatter (16 rows) | 1216 B | 0.9999 / 1.0376 |
| mul_add_scatter (16 rows) | 1232 B | 1.0924 / 1.0615 |
| mul_add_scatter (16 rows) | 1248 B | 1.0403 / 1.0449 |
| mul_add_scatter (16 rows) | 1400 B | 0.9567 / 1.0319 |
| mul_add_scatter (4 rows) | 64 B | 1.0525 / 1.2780 |
| mul_add_scatter (4 rows) | 256 B | 1.1380 / 1.1111 |
| mul_add_scatter (4 rows) | 512 B | 0.9631 / 0.9409 |
| mul_add_scatter (4 rows) | 1152 B | 0.9934 / 0.9850 |
| mul_add_scatter (4 rows) | 1168 B | 1.0631 / 1.0540 |
| mul_add_scatter (4 rows) | 1184 B | 1.0244 / 1.0046 |
| mul_add_scatter (4 rows) | 1200 B | 1.0656 / 1.0605 |
| mul_add_scatter (4 rows) | 1216 B | 0.9881 / 1.0169 |
| mul_add_scatter (4 rows) | 1232 B | 1.0709 / 1.0673 |
| mul_add_scatter (4 rows) | 1248 B | 1.0860 / 1.0272 |
| mul_add_scatter (4 rows) | 1400 B | 0.9503 / 1.0199 |
| mul_assign | 64 B | 1.0912 / 1.2881 |
| mul_assign | 256 B | 1.0003 / 1.2652 |
| mul_assign | 512 B | 1.0130 / 0.8670 |
| mul_assign | 1152 B | 1.0949 / 0.9954 |
| mul_assign | 1168 B | 1.0352 / 1.0090 |
| mul_assign | 1184 B | 1.0783 / 1.0005 |
| mul_assign | 1200 B | 1.0787 / 1.0062 |
| mul_assign | 1216 B | 1.0786 / 0.9925 |
| mul_assign | 1232 B | 1.0055 / 0.9934 |
| mul_assign | 1248 B | 1.0499 / 0.9812 |
| mul_assign | 1400 B | 1.0564 / 1.0359 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | v3 speed factor |
| --- | --- | --- |
| 256 KiB | 0 | 0.9988 / 1.0050 |
| 256 KiB | 16 | 0.9975 / 1.0044 |
| 64 KiB | 0 | 1.0002 / 1.0003 |
| 64 KiB | 16 | 1.0005 / 1.0020 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 1.0043 / 1.0398 |
| 16 KiB | 0.9931 / 0.9830 |
| 256 KiB | 1.0003 / 1.0002 |
| 4 KiB | 0.9685 / 0.9736 |
| 64 KiB | 1.0012 / 1.0011 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | v3 speed factor |
| --- | --- |
| 256 KiB | 1.0000 / 1.0000 |
| 4 KiB | 0.9992 / 1.0021 |
| 64 MiB | 0.9991 / 1.0099 |
| 8 MiB | 1.0151 / 0.9913 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | Case | v3 speed factor |
| --- | --- | --- |
| 1 MiB | mul_into | 0.9929 / 1.0103 |
| 1 MiB | mul_into + read | 1.0036 / 1.0065 |
| 1 MiB | mul_add | 0.9923 / 1.0146 |
| 32 MiB | mul_into | 0.9975 / 1.0081 |
| 32 MiB | mul_into + read | 0.9975 / 1.0201 |
| 32 MiB | mul_add | 1.0042 / 1.0111 |
| 8 MiB | mul_into | 1.0006 / 0.9996 |
| 8 MiB | mul_into + read | 0.9926 / 0.9923 |
| 8 MiB | mul_add | 1.0170 / 0.9968 |

## gf8d

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 1.1579 / 1.0015 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0001 / 1.0003 |
| mul_elementwise | 64 KiB | 1.0011 / 1.0173 |
| mul_elementwise_assign | 64 KiB | 1.0032 / 1.0003 |
| mul_into | 64 KiB | 1.0000 / 1.0001 |
| mul_into | 4 MiB | 0.9994 / 0.9997 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather_with | 4 KiB | 1.2227 / 1.0423 |
| mul_add_gather_with | 16 KiB | 1.0662 / 1.0096 |
| mul_into_gather | 4 KiB | 1.0010 / 1.0091 |
| mul_into_gather | 16 KiB | 0.9989 / 1.0041 |
| mul_into_gather_with | 4 KiB | 0.9740 / 1.0133 |
| mul_into_gather_with | 16 KiB | 0.9937 / 1.0041 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9995 / 0.9951 |
| fill(0) + mul_add_matrix | 4 destinations | 0.9715 / 0.9880 |
| fill(0) + mul_add_matrix | 6 destinations | 0.9974 / 0.9611 |
| mul_into_matrix | 2 destinations | 1.0028 / 0.9906 |
| mul_into_matrix | 4 destinations | 0.9737 / 0.8416 |
| mul_into_matrix | 6 destinations | 0.9826 / 0.8720 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | Case | v3 speed factor |
| --- | --- | --- | --- |
| 1 KiB | 0 | mul_add_scatter | 1.0183 / 1.0437 |
| 1 KiB | 0 | mul_add_gather | 1.9896 / 1.1733 |
| 1 KiB | 0 | mul_add_matrix | 1.0106 / 1.0099 |
| 1 KiB | 16 | mul_add_scatter | 1.0168 / 0.9943 |
| 1 KiB | 16 | mul_add_gather | 1.7762 / 1.1428 |
| 1 KiB | 16 | mul_add_matrix | 1.0093 / 1.0063 |
| 16 KiB | 0 | mul_add_scatter | 0.9962 / 0.9939 |
| 16 KiB | 0 | mul_add_gather | 1.0185 / 1.0130 |
| 16 KiB | 0 | mul_add_matrix | 1.0085 / 1.0011 |
| 16 KiB | 16 | mul_add_scatter | 1.0005 / 1.0015 |
| 16 KiB | 16 | mul_add_gather | 0.9598 / 1.0110 |
| 16 KiB | 16 | mul_add_matrix | 1.0061 / 0.9968 |
| 2 KiB | 0 | mul_add_scatter | 0.9822 / 1.0378 |
| 2 KiB | 0 | mul_add_gather | 1.5574 / 1.1557 |
| 2 KiB | 0 | mul_add_matrix | 1.0017 / 1.0045 |
| 2 KiB | 16 | mul_add_scatter | 1.0274 / 1.0097 |
| 2 KiB | 16 | mul_add_gather | 1.4070 / 1.1141 |
| 2 KiB | 16 | mul_add_matrix | 1.0239 / 1.0060 |
| 256 B | 0 | mul_add_scatter | 1.0696 / 1.0540 |
| 256 B | 0 | mul_add_gather | 3.5832 / 1.5205 |
| 256 B | 0 | mul_add_matrix | 1.0065 / 1.0274 |
| 256 B | 16 | mul_add_scatter | 1.1652 / 1.0487 |
| 256 B | 16 | mul_add_gather | 3.0648 / 1.5911 |
| 256 B | 16 | mul_add_matrix | 1.0151 / 1.0267 |
| 3 KiB | 0 | mul_add_scatter | 0.9403 / 1.0084 |
| 3 KiB | 0 | mul_add_gather | 1.2877 / 1.0709 |
| 3 KiB | 0 | mul_add_matrix | 1.0020 / 1.0010 |
| 3 KiB | 16 | mul_add_scatter | 0.9609 / 1.0064 |
| 3 KiB | 16 | mul_add_gather | 0.9920 / 1.0379 |
| 3 KiB | 16 | mul_add_matrix | 1.0148 / 0.9995 |
| 4 KiB | 0 | mul_add_scatter | 1.0085 / 1.0187 |
| 4 KiB | 0 | mul_add_gather | 1.1979 / 1.0459 |
| 4 KiB | 0 | mul_add_matrix | 1.0042 / 0.9917 |
| 4 KiB | 16 | mul_add_scatter | 1.0085 / 1.0085 |
| 4 KiB | 16 | mul_add_gather | 0.9832 / 1.0345 |
| 4 KiB | 16 | mul_add_matrix | 1.0118 / 1.0045 |
| 512 B | 0 | mul_add_scatter | 1.0545 / 1.0988 |
| 512 B | 0 | mul_add_gather | 2.7601 / 1.3363 |
| 512 B | 0 | mul_add_matrix | 1.0118 / 1.0153 |
| 512 B | 16 | mul_add_scatter | 1.0215 / 0.9868 |
| 512 B | 16 | mul_add_gather | 2.3422 / 1.2601 |
| 512 B | 16 | mul_add_matrix | 1.0118 / 1.0158 |
| 8 KiB | 0 | mul_add_scatter | 1.0026 / 1.0112 |
| 8 KiB | 0 | mul_add_gather | 1.0885 / 1.0242 |
| 8 KiB | 0 | mul_add_matrix | 1.0575 / 1.0242 |
| 8 KiB | 16 | mul_add_scatter | 1.0042 / 1.0051 |
| 8 KiB | 16 | mul_add_gather | 0.9701 / 1.0181 |
| 8 KiB | 16 | mul_add_matrix | 1.0394 / 1.0086 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | v3 speed factor |
| --- | --- |
| 0 | 1.0169 / 1.0012 |
| 16 | 1.0122 / 1.0017 |
| 32 | 1.0145 / 1.0028 |
| 48 | 1.0133 / 1.0035 |

### Scatter destination alignment

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | v3 speed factor |
| --- | --- |
| mul_add_scatter | 1.0029 / 1.0039 |
| mul_add_scatter_with | 1.0026 / 1.0051 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 0.9479 / 0.9707 |
| 16 KiB | 0.9987 / 1.0107 |
| 256 KiB | 1.0001 / 1.0030 |
| 4 KiB | 0.9777 / 0.9980 |
| 64 KiB | 1.0006 / 0.9920 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst = c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9998 / 0.9996 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=262144; preparation=one-shot; row_bytes=16384; sources=16 | 0.9782 / 0.9998 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=65536; preparation=one-shot; row_bytes=4096; sources=16 | 1.0000 / 0.9916 |
| dst ^= c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9990 / 0.9995 |
| systematic encode | alignment=page; destinations=2; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0042 / 0.9967 |
| systematic encode | alignment=page; destinations=2; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0167 / 1.0063 |
| systematic encode | alignment=page; destinations=2; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0003 / 1.0015 |
| systematic encode | alignment=page; destinations=4; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0026 / 1.0000 |
| systematic encode | alignment=page; destinations=4; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0105 / 1.0068 |
| systematic encode | alignment=page; destinations=4; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 0.9999 / 1.0034 |
| systematic encode | alignment=page; destinations=6; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0013 / 1.0013 |
| systematic encode | alignment=page; destinations=6; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0049 / 1.0001 |
| systematic encode | alignment=page; destinations=6; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0023 / 1.0020 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
