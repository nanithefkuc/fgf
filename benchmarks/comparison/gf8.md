# Byte fields: `gf8b`, `gf8d`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf8.md), [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `388cbb4a43d5d20e8a663e89c3b44380c7c4f8b7` |
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
| scalar mul | 1.0000 / 0.9998 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 1.1000 / 0.9961 |
| add_assign | 256 KiB | 0.9995 / 0.9999 |
| add_assign | 8 MiB | 1.0068 / 1.0029 |
| add_assign_scalar | 4 KiB | 1.0085 / 1.0004 |
| add_assign_scalar | 256 KiB | 0.9999 / 1.0001 |
| add_assign_scalar | 8 MiB | 1.0127 / 1.0210 |
| mul_add | 4 KiB | 1.0164 / 1.0096 |
| mul_add | 256 KiB | 0.9992 / 0.9916 |
| mul_add | 8 MiB | 0.9966 / 0.9841 |
| mul_assign | 4 KiB | 1.0400 / 1.0002 |
| mul_assign | 256 KiB | 1.0003 / 1.0003 |
| mul_assign | 8 MiB | 1.0053 / 0.9942 |
| mul_elementwise | 4 KiB | 1.0470 / 1.0221 |
| mul_elementwise | 256 KiB | 1.0377 / 0.9528 |
| mul_elementwise | 8 MiB | 1.0307 / 1.0195 |
| mul_elementwise_assign | 4 KiB | 0.9698 / 0.8906 |
| mul_elementwise_assign | 256 KiB | 0.9982 / 1.0021 |
| mul_elementwise_assign | 8 MiB | 1.0393 / 1.0092 |
| mul_into | 4 KiB | 1.0683 / 1.0167 |
| mul_into | 256 KiB | 1.0011 / 1.0013 |
| mul_into | 8 MiB | 0.9998 / 0.9986 |
| sub_assign_scalar | 4 KiB | 1.0096 / 1.0001 |
| sub_assign_scalar | 256 KiB | 1.0001 / 0.9991 |
| sub_assign_scalar | 8 MiB | 1.0126 / 1.0170 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0002 / 0.9993 |
| mul_elementwise | 64 KiB | 1.0025 / 1.0005 |
| mul_elementwise_assign | 64 KiB | 1.0014 / 1.0019 |
| mul_into | 64 KiB | 1.0017 / 0.9993 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | Case | v3 speed factor |
| --- | --- | --- |
| 16 B | mul_add | 1.3268 / 0.9586 |
| 16 B | mul_add_with | 1.3871 / 0.9988 |
| 32 B | mul_add | 1.3349 / 0.9506 |
| 32 B | mul_add_with | 1.4133 / 1.0000 |
| 64 B | mul_add | 1.2586 / 1.0022 |
| 64 B | mul_add_with | 1.3005 / 0.9980 |
| 128 B | mul_add | 1.2005 / 0.9587 |
| 128 B | mul_add_with | 1.2467 / 0.9984 |
| 256 B | mul_add | 1.1781 / 0.9576 |
| 256 B | mul_add_with | 1.2040 / 1.0005 |
| 512 B | mul_add | 1.1310 / 1.0466 |
| 512 B | mul_add_with | 1.1228 / 1.0506 |
| 1 KiB | mul_add | 1.0448 / 1.0191 |
| 1 KiB | mul_add_with | 1.0294 / 1.0235 |
| 2 KiB | mul_add | 0.9673 / 1.0105 |
| 2 KiB | mul_add_with | 0.9640 / 1.0118 |
| 4 KiB | mul_add | 0.9580 / 1.0085 |
| 4 KiB | mul_add_with | 0.9527 / 1.0093 |
| 16 KiB | mul_add | 0.9930 / 1.0088 |
| 16 KiB | mul_add_with | 0.9826 / 1.0090 |
| 64 KiB | mul_add | 1.0002 / 0.9936 |
| 64 KiB | mul_add_with | 0.9997 / 1.0053 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather | 2 rows | 0.9990 / 1.0015 |
| mul_add_gather | 4 rows | 1.0091 / 1.0116 |
| mul_add_gather | 8 rows | 1.0283 / 1.0083 |
| mul_add_gather | 16 rows | 1.0128 / 0.9974 |
| mul_add_gather_with | 2 rows | 1.0002 / 1.0224 |
| mul_add_gather_with | 4 rows | 1.0082 / 1.0144 |
| mul_add_gather_with | 8 rows | 1.0276 / 1.0143 |
| mul_add_gather_with | 16 rows | 1.0116 / 1.0009 |
| mul_add_matrix | 2 rows | 0.9969 / 0.9988 |
| mul_add_matrix | 4 rows | 1.0061 / 0.9773 |
| mul_add_matrix | 8 rows | 1.0089 / 1.0103 |
| mul_add_matrix | 16 rows | 0.9827 / 0.9965 |
| mul_add_matrix_with | 2 rows | 1.0077 / 0.9994 |
| mul_add_matrix_with | 4 rows | 1.0045 / 0.9829 |
| mul_add_matrix_with | 8 rows | 1.0343 / 0.9975 |
| mul_add_matrix_with | 16 rows | 1.0086 / 0.9983 |
| mul_add_scatter | 2 rows | 0.9941 / 1.0017 |
| mul_add_scatter | 4 rows | 1.0007 / 1.0012 |
| mul_add_scatter | 8 rows | 1.0000 / 1.0026 |
| mul_add_scatter | 16 rows | 1.0283 / 1.0115 |
| mul_add_scatter_with | 2 rows | 0.9946 / 1.0000 |
| mul_add_scatter_with | 4 rows | 0.9997 / 1.0004 |
| mul_add_scatter_with | 8 rows | 0.9993 / 1.0024 |
| mul_add_scatter_with | 16 rows | 1.0202 / 1.0126 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_into_gather_with | 4 KiB | 1.0049 / 1.0409 |
| mul_into_gather_with | 16 KiB | 1.0012 / 1.0127 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 1.0034 / 1.0176 |
| fill(0) + mul_add_matrix | 4 destinations | 0.9957 / 0.8377 |
| fill(0) + mul_add_matrix | 6 destinations | 0.9980 / 0.9048 |
| mul_add_matrix_at | 2 destinations | 1.0055 / 1.0154 |
| mul_add_matrix_at | 4 destinations | 0.9672 / 0.9715 |
| mul_add_matrix_at | 6 destinations | 0.9752 / 1.0155 |
| mul_add_matrix_with | 2 destinations | 1.0068 / 0.9965 |
| mul_add_matrix_with | 4 destinations | 0.9786 / 0.7998 |
| mul_add_matrix_with | 6 destinations | 1.0365 / 0.8889 |
| mul_into_matrix | 2 destinations | 1.0025 / 0.9851 |
| mul_into_matrix | 4 destinations | 1.0164 / 0.8485 |
| mul_into_matrix | 6 destinations | 1.0189 / 0.9096 |
| mul_into_matrix_with | 2 destinations | 1.0047 / 1.0198 |
| mul_into_matrix_with | 4 destinations | 0.9924 / 0.8400 |
| mul_into_matrix_with | 6 destinations | 1.0289 / 0.9225 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9441 / 1.0038 |
| 1 KiB | 16 | 1.0085 / 1.0077 |
| 1 KiB | 2 | 1.0492 / 1.0003 |
| 1 KiB | 32 | 0.9985 / 0.9950 |
| 1 KiB | 4 | 1.0144 / 0.9950 |
| 1 KiB | 8 | 1.0147 / 0.9828 |
| 64 B | 1 | 0.9459 / 1.1037 |
| 64 B | 16 | 0.9342 / 1.0039 |
| 64 B | 2 | 0.9373 / 1.1989 |
| 64 B | 32 | 1.0480 / 0.9962 |
| 64 B | 4 | 1.0255 / 0.9226 |
| 64 B | 8 | 0.9780 / 0.9953 |
| 64 KiB | 1 | 0.9979 / 0.9994 |
| 64 KiB | 16 | 1.0056 / 0.9917 |
| 64 KiB | 2 | 0.9993 / 1.0001 |
| 64 KiB | 32 | 1.0047 / 1.0003 |
| 64 KiB | 4 | 1.0002 / 0.9996 |
| 64 KiB | 8 | 0.9820 / 0.9311 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 0.9737 / 1.0004 |
| add_assign per row | 1 KiB | 16 | 0.9917 / 1.0055 |
| add_assign per row | 1 KiB | 2 | 1.0043 / 0.9993 |
| add_assign per row | 1 KiB | 32 | 0.9870 / 0.9958 |
| add_assign per row | 1 KiB | 4 | 0.9975 / 0.9994 |
| add_assign per row | 1 KiB | 8 | 0.9927 / 0.9987 |
| add_assign per row | 64 B | 1 | 1.0020 / 1.0979 |
| add_assign per row | 64 B | 16 | 1.0714 / 1.1123 |
| add_assign per row | 64 B | 2 | 1.0894 / 1.1617 |
| add_assign per row | 64 B | 32 | 1.1014 / 1.2457 |
| add_assign per row | 64 B | 4 | 1.1008 / 1.1799 |
| add_assign per row | 64 B | 8 | 1.1055 / 1.1901 |
| add_assign per row | 64 KiB | 1 | 1.0002 / 1.0000 |
| add_assign per row | 64 KiB | 16 | 1.0059 / 0.9926 |
| add_assign per row | 64 KiB | 2 | 1.0005 / 1.0002 |
| add_assign per row | 64 KiB | 32 | 1.0047 / 1.0011 |
| add_assign per row | 64 KiB | 4 | 0.9996 / 1.0002 |
| add_assign per row | 64 KiB | 8 | 0.9832 / 0.9328 |
| add_assign | 1 KiB | 1 | 1.0148 / 1.0042 |
| add_assign | 1 KiB | 16 | 1.0202 / 1.0578 |
| add_assign | 1 KiB | 2 | 1.1184 / 1.0048 |
| add_assign | 1 KiB | 32 | 1.0011 / 0.9931 |
| add_assign | 1 KiB | 4 | 0.9710 / 0.9969 |
| add_assign | 1 KiB | 8 | 1.0413 / 1.0003 |
| add_assign | 64 B | 1 | 1.0463 / 1.1659 |
| add_assign | 64 B | 16 | 1.0140 / 1.0041 |
| add_assign | 64 B | 2 | 0.9998 / 1.0557 |
| add_assign | 64 B | 32 | 1.1184 / 1.0052 |
| add_assign | 64 B | 4 | 1.0395 / 1.0017 |
| add_assign | 64 B | 8 | 0.9996 / 1.0009 |
| add_assign | 64 KiB | 1 | 0.9984 / 1.0002 |
| add_assign | 64 KiB | 16 | 1.0056 / 0.9961 |
| add_assign | 64 KiB | 2 | 1.0000 / 1.0004 |
| add_assign | 64 KiB | 32 | 1.0048 / 1.0010 |
| add_assign | 64 KiB | 4 | 1.0003 / 0.9992 |
| add_assign | 64 KiB | 8 | 0.9831 / 0.9299 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 64 B | 1.0887 / 1.1679 |
| add_assign | 256 B | 1.0738 / 0.9205 |
| add_assign | 512 B | 0.9926 / 0.9983 |
| add_assign | 1152 B | 1.0009 / 0.9961 |
| add_assign | 1168 B | 1.0912 / 0.9992 |
| add_assign | 1184 B | 1.0269 / 0.9959 |
| add_assign | 1200 B | 0.9918 / 0.9951 |
| add_assign | 1216 B | 1.0645 / 0.9980 |
| add_assign | 1232 B | 1.0599 / 0.9990 |
| add_assign | 1248 B | 1.0397 / 0.9966 |
| add_assign | 1400 B | 1.0058 / 0.9995 |
| mul_add | 64 B | 1.3023 / 0.9613 |
| mul_add | 256 B | 1.2512 / 0.9696 |
| mul_add | 512 B | 1.2870 / 1.0028 |
| mul_add | 1152 B | 1.0431 / 1.0170 |
| mul_add | 1168 B | 1.0473 / 0.9999 |
| mul_add | 1184 B | 1.0669 / 1.0227 |
| mul_add | 1200 B | 1.1575 / 1.0000 |
| mul_add | 1216 B | 1.0528 / 1.0163 |
| mul_add | 1232 B | 1.0268 / 0.9855 |
| mul_add | 1248 B | 1.0681 / 1.0298 |
| mul_add | 1400 B | 1.2103 / 1.0118 |
| mul_add_scatter (16 rows) | 64 B | 1.1594 / 1.4204 |
| mul_add_scatter (16 rows) | 256 B | 1.2094 / 1.1883 |
| mul_add_scatter (16 rows) | 512 B | 1.0297 / 0.9766 |
| mul_add_scatter (16 rows) | 1152 B | 1.0185 / 1.0028 |
| mul_add_scatter (16 rows) | 1168 B | 1.0770 / 1.0838 |
| mul_add_scatter (16 rows) | 1184 B | 1.0444 / 1.0290 |
| mul_add_scatter (16 rows) | 1200 B | 1.0699 / 1.0817 |
| mul_add_scatter (16 rows) | 1216 B | 1.0027 / 1.0359 |
| mul_add_scatter (16 rows) | 1232 B | 1.0898 / 1.0633 |
| mul_add_scatter (16 rows) | 1248 B | 1.0293 / 1.0423 |
| mul_add_scatter (16 rows) | 1400 B | 0.9421 / 1.0320 |
| mul_add_scatter (4 rows) | 64 B | 1.0559 / 1.2867 |
| mul_add_scatter (4 rows) | 256 B | 1.1015 / 1.1897 |
| mul_add_scatter (4 rows) | 512 B | 0.9447 / 0.9530 |
| mul_add_scatter (4 rows) | 1152 B | 1.0595 / 0.9907 |
| mul_add_scatter (4 rows) | 1168 B | 1.0719 / 1.0593 |
| mul_add_scatter (4 rows) | 1184 B | 1.0362 / 1.0095 |
| mul_add_scatter (4 rows) | 1200 B | 1.0720 / 1.0657 |
| mul_add_scatter (4 rows) | 1216 B | 0.9819 / 1.0185 |
| mul_add_scatter (4 rows) | 1232 B | 1.0530 / 1.0663 |
| mul_add_scatter (4 rows) | 1248 B | 1.0119 / 1.0283 |
| mul_add_scatter (4 rows) | 1400 B | 0.9500 / 1.0271 |
| mul_assign | 64 B | 1.1363 / 1.2863 |
| mul_assign | 256 B | 1.0772 / 1.2680 |
| mul_assign | 512 B | 1.0399 / 0.8672 |
| mul_assign | 1152 B | 1.0539 / 1.0181 |
| mul_assign | 1168 B | 1.0373 / 1.0222 |
| mul_assign | 1184 B | 1.0736 / 1.0043 |
| mul_assign | 1200 B | 1.0235 / 1.0083 |
| mul_assign | 1216 B | 1.0380 / 0.9864 |
| mul_assign | 1232 B | 1.0073 / 0.9950 |
| mul_assign | 1248 B | 1.0317 / 0.9675 |
| mul_assign | 1400 B | 1.0496 / 1.0429 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | v3 speed factor |
| --- | --- | --- |
| 256 KiB | 0 | 1.0048 / 0.9926 |
| 256 KiB | 16 | 1.0051 / 1.0046 |
| 64 KiB | 0 | 1.0005 / 0.9999 |
| 64 KiB | 16 | 1.0006 / 0.9971 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 0.9420 / 0.9941 |
| 16 KiB | 0.9912 / 0.9856 |
| 256 KiB | 1.0004 / 1.0001 |
| 4 KiB | 0.9598 / 0.9736 |
| 64 KiB | 1.0018 / 1.0003 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | v3 speed factor |
| --- | --- |
| 256 KiB | 1.0000 / 1.0001 |
| 4 KiB | 0.9973 / 1.0057 |
| 64 MiB | 1.0020 / 0.9981 |
| 8 MiB | 1.0178 / 0.9972 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | Case | v3 speed factor |
| --- | --- | --- |
| 1 MiB | mul_into | 1.0001 / 0.9992 |
| 1 MiB | mul_into + read | 0.9863 / 1.0092 |
| 1 MiB | mul_add | 1.0000 / 0.9995 |
| 32 MiB | mul_into | 0.9916 / 1.0128 |
| 32 MiB | mul_into + read | 0.9933 / 1.0128 |
| 32 MiB | mul_add | 0.9957 / 0.9986 |
| 8 MiB | mul_into | 1.0022 / 1.0003 |
| 8 MiB | mul_into + read | 0.9984 / 1.0054 |
| 8 MiB | mul_add | 0.9663 / 1.0243 |

## gf8d

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 1.1573 / 1.0023 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 0.9999 / 1.0000 |
| mul_elementwise | 64 KiB | 0.9976 / 1.0179 |
| mul_elementwise_assign | 64 KiB | 1.0040 / 1.0003 |
| mul_into | 64 KiB | 1.0010 / 1.0000 |
| mul_into | 4 MiB | 1.0003 / 1.0003 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather_with | 4 KiB | 1.2259 / 1.0485 |
| mul_add_gather_with | 16 KiB | 1.0566 / 1.0144 |
| mul_into_gather | 4 KiB | 1.0002 / 1.0072 |
| mul_into_gather | 16 KiB | 1.0012 / 1.0082 |
| mul_into_gather_with | 4 KiB | 0.9808 / 1.0047 |
| mul_into_gather_with | 16 KiB | 0.9972 / 1.0103 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9994 / 0.9971 |
| fill(0) + mul_add_matrix | 4 destinations | 0.9958 / 0.9917 |
| fill(0) + mul_add_matrix | 6 destinations | 1.0020 / 0.9648 |
| mul_into_matrix | 2 destinations | 1.0014 / 0.9937 |
| mul_into_matrix | 4 destinations | 0.9995 / 0.8388 |
| mul_into_matrix | 6 destinations | 1.0026 / 0.8727 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | Case | v3 speed factor |
| --- | --- | --- | --- |
| 1 KiB | 0 | mul_add_scatter | 1.0158 / 1.0299 |
| 1 KiB | 0 | mul_add_gather | 1.9974 / 1.1761 |
| 1 KiB | 0 | mul_add_matrix | 1.0123 / 1.0083 |
| 1 KiB | 16 | mul_add_scatter | 1.0150 / 0.9927 |
| 1 KiB | 16 | mul_add_gather | 1.7580 / 1.1423 |
| 1 KiB | 16 | mul_add_matrix | 1.0462 / 1.0096 |
| 16 KiB | 0 | mul_add_scatter | 1.0004 / 0.9886 |
| 16 KiB | 0 | mul_add_gather | 1.0182 / 1.0098 |
| 16 KiB | 0 | mul_add_matrix | 1.0039 / 0.9996 |
| 16 KiB | 16 | mul_add_scatter | 0.9917 / 1.0005 |
| 16 KiB | 16 | mul_add_gather | 0.9624 / 1.0113 |
| 16 KiB | 16 | mul_add_matrix | 1.0047 / 0.9986 |
| 2 KiB | 0 | mul_add_scatter | 0.9993 / 1.0335 |
| 2 KiB | 0 | mul_add_gather | 1.5653 / 1.1238 |
| 2 KiB | 0 | mul_add_matrix | 1.0037 / 1.0022 |
| 2 KiB | 16 | mul_add_scatter | 1.0555 / 1.0079 |
| 2 KiB | 16 | mul_add_gather | 1.4068 / 1.0927 |
| 2 KiB | 16 | mul_add_matrix | 1.0112 / 0.9976 |
| 256 B | 0 | mul_add_scatter | 1.0560 / 1.0479 |
| 256 B | 0 | mul_add_gather | 3.6181 / 1.5254 |
| 256 B | 0 | mul_add_matrix | 1.0057 / 1.0288 |
| 256 B | 16 | mul_add_scatter | 1.1555 / 1.0514 |
| 256 B | 16 | mul_add_gather | 3.0601 / 1.5865 |
| 256 B | 16 | mul_add_matrix | 0.9891 / 1.0237 |
| 3 KiB | 0 | mul_add_scatter | 0.9403 / 1.0093 |
| 3 KiB | 0 | mul_add_gather | 1.2878 / 1.0727 |
| 3 KiB | 0 | mul_add_matrix | 1.0009 / 1.0006 |
| 3 KiB | 16 | mul_add_scatter | 1.0007 / 1.0085 |
| 3 KiB | 16 | mul_add_gather | 1.0288 / 1.0374 |
| 3 KiB | 16 | mul_add_matrix | 1.0112 / 0.9891 |
| 4 KiB | 0 | mul_add_scatter | 1.0077 / 1.0156 |
| 4 KiB | 0 | mul_add_gather | 1.2316 / 1.0448 |
| 4 KiB | 0 | mul_add_matrix | 1.0015 / 0.9909 |
| 4 KiB | 16 | mul_add_scatter | 1.0433 / 1.0193 |
| 4 KiB | 16 | mul_add_gather | 0.9721 / 1.0338 |
| 4 KiB | 16 | mul_add_matrix | 1.0033 / 0.9990 |
| 512 B | 0 | mul_add_scatter | 1.0525 / 1.0824 |
| 512 B | 0 | mul_add_gather | 2.8042 / 1.3383 |
| 512 B | 0 | mul_add_matrix | 1.0067 / 1.0131 |
| 512 B | 16 | mul_add_scatter | 0.9988 / 0.9899 |
| 512 B | 16 | mul_add_gather | 2.2400 / 1.2605 |
| 512 B | 16 | mul_add_matrix | 1.0805 / 1.0142 |
| 8 KiB | 0 | mul_add_scatter | 1.0139 / 1.0116 |
| 8 KiB | 0 | mul_add_gather | 1.0768 / 1.0235 |
| 8 KiB | 0 | mul_add_matrix | 1.0027 / 1.0205 |
| 8 KiB | 16 | mul_add_scatter | 1.1336 / 0.9868 |
| 8 KiB | 16 | mul_add_gather | 0.9996 / 1.0177 |
| 8 KiB | 16 | mul_add_matrix | 1.0068 / 1.0124 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | v3 speed factor |
| --- | --- |
| 0 | 1.0127 / 1.0031 |
| 16 | 1.0101 / 1.0034 |
| 32 | 1.0099 / 1.0030 |
| 48 | 1.0085 / 1.0164 |

### Scatter destination alignment

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | v3 speed factor |
| --- | --- |
| mul_add_scatter | 1.0028 / 0.9997 |
| mul_add_scatter_with | 1.0024 / 1.0006 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 0.9926 / 1.0409 |
| 16 KiB | 0.9959 / 1.0087 |
| 256 KiB | 1.0005 / 1.0243 |
| 4 KiB | 0.9684 / 0.9938 |
| 64 KiB | 1.0015 / 0.9926 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst = c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 1.0006 / 1.0000 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=262144; preparation=one-shot; row_bytes=16384; sources=16 | 0.9663 / 1.0000 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=65536; preparation=one-shot; row_bytes=4096; sources=16 | 0.9948 / 0.9922 |
| dst ^= c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 1.0003 / 0.9966 |
| systematic encode | alignment=page; destinations=2; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0069 / 1.0012 |
| systematic encode | alignment=page; destinations=2; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0025 / 0.9956 |
| systematic encode | alignment=page; destinations=2; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 0.9997 / 1.0008 |
| systematic encode | alignment=page; destinations=4; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0019 / 1.0003 |
| systematic encode | alignment=page; destinations=4; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0060 / 1.0093 |
| systematic encode | alignment=page; destinations=4; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 0.9994 / 1.0032 |
| systematic encode | alignment=page; destinations=6; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0042 / 0.9995 |
| systematic encode | alignment=page; destinations=6; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0002 / 1.0135 |
| systematic encode | alignment=page; destinations=6; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 0.9967 / 1.0028 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
