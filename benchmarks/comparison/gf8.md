# Byte fields: `gf8b`, `gf8d`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf8.md), [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| scalar mul | 0.9999 / 1.0000 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9914 / 0.9993 |
| add_assign | 256 KiB | 0.9999 / 1.0000 |
| add_assign | 8 MiB | 1.0077 / 0.9992 |
| add_assign_scalar | 4 KiB | 0.9143 / 1.0039 |
| add_assign_scalar | 256 KiB | 1.0000 / 1.0003 |
| add_assign_scalar | 8 MiB | 0.9956 / 0.9950 |
| mul_add | 4 KiB | 0.9971 / 0.9989 |
| mul_add | 256 KiB | 0.9999 / 1.0000 |
| mul_add | 8 MiB | 1.0017 / 1.0142 |
| mul_assign | 4 KiB | 1.0126 / 1.0130 |
| mul_assign | 256 KiB | 1.0004 / 1.0005 |
| mul_assign | 8 MiB | 1.0052 / 0.9997 |
| mul_elementwise | 4 KiB | 0.9728 / 1.0251 |
| mul_elementwise | 256 KiB | 1.0342 / 0.8457 |
| mul_elementwise | 8 MiB | 0.9809 / 0.9963 |
| mul_elementwise_assign | 4 KiB | 0.9712 / 0.9924 |
| mul_elementwise_assign | 256 KiB | 1.0016 / 1.0000 |
| mul_elementwise_assign | 8 MiB | 1.0002 / 1.0048 |
| mul_into | 4 KiB | 1.0121 / 0.9981 |
| mul_into | 256 KiB | 1.0010 / 1.0007 |
| mul_into | 8 MiB | 0.9999 / 0.9991 |
| sub_assign_scalar | 4 KiB | 0.9978 / 1.0042 |
| sub_assign_scalar | 256 KiB | 1.0000 / 1.0003 |
| sub_assign_scalar | 8 MiB | 0.9966 / 1.0010 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0002 / 0.9999 |
| mul_elementwise | 64 KiB | 0.9991 / 0.9995 |
| mul_elementwise_assign | 64 KiB | 1.0002 / 1.0016 |
| mul_into | 64 KiB | 1.0002 / 0.9999 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | Case | v3 speed factor |
| --- | --- | --- |
| 16 B | mul_add | 1.3440 / 0.9590 |
| 16 B | mul_add_with | 1.4057 / 0.9973 |
| 32 B | mul_add | 1.3434 / 0.9547 |
| 32 B | mul_add_with | 1.4196 / 1.0071 |
| 64 B | mul_add | 1.2541 / 1.0031 |
| 64 B | mul_add_with | 1.3069 / 0.9966 |
| 128 B | mul_add | 1.2047 / 0.9844 |
| 128 B | mul_add_with | 1.2379 / 1.0481 |
| 256 B | mul_add | 1.1764 / 0.9585 |
| 256 B | mul_add_with | 1.2274 / 0.9959 |
| 512 B | mul_add | 1.1813 / 0.9851 |
| 512 B | mul_add_with | 1.1618 / 1.0127 |
| 1 KiB | mul_add | 1.0596 / 1.0044 |
| 1 KiB | mul_add_with | 1.0326 / 1.0111 |
| 2 KiB | mul_add | 1.0092 / 1.0069 |
| 2 KiB | mul_add_with | 0.9889 / 1.0101 |
| 4 KiB | mul_add | 1.0003 / 1.0077 |
| 4 KiB | mul_add_with | 0.9942 / 1.0075 |
| 16 KiB | mul_add | 1.0037 / 1.0134 |
| 16 KiB | mul_add_with | 0.9995 / 1.0167 |
| 64 KiB | mul_add | 1.0001 / 0.9918 |
| 64 KiB | mul_add_with | 0.9978 / 1.0048 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather | 2 rows | 1.0069 / 0.9954 |
| mul_add_gather | 4 rows | 1.0206 / 0.9995 |
| mul_add_gather | 8 rows | 1.0319 / 0.9957 |
| mul_add_gather | 16 rows | 1.0155 / 1.0070 |
| mul_add_gather_with | 2 rows | 1.0075 / 0.9894 |
| mul_add_gather_with | 4 rows | 1.0165 / 0.9971 |
| mul_add_gather_with | 8 rows | 1.0262 / 1.0025 |
| mul_add_gather_with | 16 rows | 1.0109 / 1.0021 |
| mul_add_matrix | 2 rows | 0.9952 / 1.0026 |
| mul_add_matrix | 4 rows | 1.0074 / 0.9847 |
| mul_add_matrix | 8 rows | 0.9963 / 0.9981 |
| mul_add_matrix | 16 rows | 1.0070 / 0.9962 |
| mul_add_matrix_with | 2 rows | 1.0023 / 0.9994 |
| mul_add_matrix_with | 4 rows | 1.0041 / 0.9915 |
| mul_add_matrix_with | 8 rows | 1.0277 / 1.0107 |
| mul_add_matrix_with | 16 rows | 1.0282 / 0.9979 |
| mul_add_scatter | 2 rows | 0.9969 / 0.9989 |
| mul_add_scatter | 4 rows | 1.0007 / 1.0008 |
| mul_add_scatter | 8 rows | 0.9998 / 0.9999 |
| mul_add_scatter | 16 rows | 0.9945 / 0.9895 |
| mul_add_scatter_with | 2 rows | 0.9978 / 0.9999 |
| mul_add_scatter_with | 4 rows | 1.0005 / 1.0003 |
| mul_add_scatter_with | 8 rows | 0.9997 / 1.0001 |
| mul_add_scatter_with | 16 rows | 0.9916 / 0.9792 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_into_gather_with | 4 KiB | 1.0024 / 1.0383 |
| mul_into_gather_with | 16 KiB | 0.9909 / 1.0110 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 1.0188 / 1.0073 |
| fill(0) + mul_add_matrix | 4 destinations | 0.9933 / 0.8877 |
| fill(0) + mul_add_matrix | 6 destinations | 0.9935 / 0.9441 |
| mul_add_matrix_at | 2 destinations | 1.0001 / 1.0144 |
| mul_add_matrix_at | 4 destinations | 1.0007 / 1.0109 |
| mul_add_matrix_at | 6 destinations | 0.9704 / 0.9835 |
| mul_add_matrix_with | 2 destinations | 1.0155 / 1.0018 |
| mul_add_matrix_with | 4 destinations | 1.0386 / 0.8124 |
| mul_add_matrix_with | 6 destinations | 1.0454 / 0.8445 |
| mul_into_matrix | 2 destinations | 1.0381 / 0.9652 |
| mul_into_matrix | 4 destinations | 1.0200 / 0.8959 |
| mul_into_matrix | 6 destinations | 1.0161 / 0.9167 |
| mul_into_matrix_with | 2 destinations | 1.0054 / 1.0136 |
| mul_into_matrix_with | 4 destinations | 1.0394 / 0.8733 |
| mul_into_matrix_with | 6 destinations | 1.0430 / 0.9034 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9727 / 1.0017 |
| 1 KiB | 16 | 0.9996 / 0.9991 |
| 1 KiB | 2 | 1.0016 / 1.0002 |
| 1 KiB | 32 | 1.0001 / 1.0038 |
| 1 KiB | 4 | 1.0038 / 1.0008 |
| 1 KiB | 8 | 1.0061 / 0.9872 |
| 64 B | 1 | 0.9971 / 1.3132 |
| 64 B | 16 | 0.9709 / 1.0004 |
| 64 B | 2 | 1.0036 / 0.9977 |
| 64 B | 32 | 1.0140 / 1.0008 |
| 64 B | 4 | 1.0069 / 1.0007 |
| 64 B | 8 | 1.0252 / 1.0002 |
| 64 KiB | 1 | 0.9999 / 1.0005 |
| 64 KiB | 16 | 1.0000 / 1.0117 |
| 64 KiB | 2 | 0.9998 / 1.0000 |
| 64 KiB | 32 | 1.0002 / 0.9982 |
| 64 KiB | 4 | 1.0000 / 1.0001 |
| 64 KiB | 8 | 0.9911 / 1.0886 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 0.9348 / 0.9972 |
| add_assign per row | 1 KiB | 16 | 0.9846 / 1.0085 |
| add_assign per row | 1 KiB | 2 | 1.0040 / 1.0004 |
| add_assign per row | 1 KiB | 32 | 1.0044 / 0.9972 |
| add_assign per row | 1 KiB | 4 | 0.9948 / 0.9983 |
| add_assign per row | 1 KiB | 8 | 0.9975 / 0.9973 |
| add_assign per row | 64 B | 1 | 0.9977 / 1.0471 |
| add_assign per row | 64 B | 16 | 1.0216 / 1.1152 |
| add_assign per row | 64 B | 2 | 0.9997 / 1.2452 |
| add_assign per row | 64 B | 32 | 1.0451 / 1.1313 |
| add_assign per row | 64 B | 4 | 1.0608 / 1.0002 |
| add_assign per row | 64 B | 8 | 1.0570 / 1.1194 |
| add_assign per row | 64 KiB | 1 | 1.0006 / 1.0001 |
| add_assign per row | 64 KiB | 16 | 1.0003 / 1.0124 |
| add_assign per row | 64 KiB | 2 | 1.0004 / 1.0007 |
| add_assign per row | 64 KiB | 32 | 1.0006 / 0.9983 |
| add_assign per row | 64 KiB | 4 | 1.0006 / 1.0006 |
| add_assign per row | 64 KiB | 8 | 0.9944 / 1.0892 |
| add_assign | 1 KiB | 1 | 0.9992 / 0.9968 |
| add_assign | 1 KiB | 16 | 1.0023 / 1.0063 |
| add_assign | 1 KiB | 2 | 1.0431 / 1.0008 |
| add_assign | 1 KiB | 32 | 1.0097 / 1.0002 |
| add_assign | 1 KiB | 4 | 0.9977 / 1.0013 |
| add_assign | 1 KiB | 8 | 1.0059 / 1.0046 |
| add_assign | 64 B | 1 | 1.0920 / 1.0525 |
| add_assign | 64 B | 16 | 0.9989 / 0.9992 |
| add_assign | 64 B | 2 | 0.9602 / 1.0024 |
| add_assign | 64 B | 32 | 1.0393 / 1.0079 |
| add_assign | 64 B | 4 | 1.0002 / 1.0002 |
| add_assign | 64 B | 8 | 0.9998 / 1.0012 |
| add_assign | 64 KiB | 1 | 1.0013 / 1.0005 |
| add_assign | 64 KiB | 16 | 1.0005 / 1.0158 |
| add_assign | 64 KiB | 2 | 1.0001 / 1.0002 |
| add_assign | 64 KiB | 32 | 1.0002 / 0.9982 |
| add_assign | 64 KiB | 4 | 1.0003 / 1.0003 |
| add_assign | 64 KiB | 8 | 0.9912 / 1.0888 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 64 B | 1.0892 / 1.1656 |
| add_assign | 256 B | 1.0242 / 1.0392 |
| add_assign | 512 B | 1.0378 / 0.9905 |
| add_assign | 1152 B | 1.0062 / 1.0014 |
| add_assign | 1168 B | 1.0748 / 0.9993 |
| add_assign | 1184 B | 1.0386 / 1.0013 |
| add_assign | 1200 B | 1.0025 / 0.9983 |
| add_assign | 1216 B | 1.0139 / 0.9994 |
| add_assign | 1232 B | 1.0767 / 0.9945 |
| add_assign | 1248 B | 1.0568 / 0.9993 |
| add_assign | 1400 B | 0.9999 / 0.9996 |
| mul_add | 64 B | 1.3857 / 0.9585 |
| mul_add | 256 B | 1.2789 / 0.9993 |
| mul_add | 512 B | 1.3624 / 0.9868 |
| mul_add | 1152 B | 1.1179 / 1.0137 |
| mul_add | 1168 B | 1.1286 / 0.9999 |
| mul_add | 1184 B | 1.0999 / 1.0025 |
| mul_add | 1200 B | 1.2754 / 1.0002 |
| mul_add | 1216 B | 1.0985 / 0.9906 |
| mul_add | 1232 B | 1.0790 / 1.0000 |
| mul_add | 1248 B | 1.0713 / 0.9967 |
| mul_add | 1400 B | 1.1852 / 1.0052 |
| mul_add_scatter (16 rows) | 64 B | 1.1129 / 1.3591 |
| mul_add_scatter (16 rows) | 256 B | 1.2155 / 1.2347 |
| mul_add_scatter (16 rows) | 512 B | 1.0328 / 1.0101 |
| mul_add_scatter (16 rows) | 1152 B | 1.0447 / 1.0173 |
| mul_add_scatter (16 rows) | 1168 B | 1.0647 / 1.0966 |
| mul_add_scatter (16 rows) | 1184 B | 1.0983 / 1.0341 |
| mul_add_scatter (16 rows) | 1200 B | 1.1106 / 1.0854 |
| mul_add_scatter (16 rows) | 1216 B | 1.0569 / 1.0361 |
| mul_add_scatter (16 rows) | 1232 B | 1.0656 / 1.0820 |
| mul_add_scatter (16 rows) | 1248 B | 1.1033 / 1.0487 |
| mul_add_scatter (16 rows) | 1400 B | 0.9577 / 1.0540 |
| mul_add_scatter (4 rows) | 64 B | 1.0543 / 1.2777 |
| mul_add_scatter (4 rows) | 256 B | 1.0336 / 1.1688 |
| mul_add_scatter (4 rows) | 512 B | 0.9844 / 0.9774 |
| mul_add_scatter (4 rows) | 1152 B | 1.0406 / 1.0000 |
| mul_add_scatter (4 rows) | 1168 B | 1.0677 / 1.0891 |
| mul_add_scatter (4 rows) | 1184 B | 1.0876 / 1.0269 |
| mul_add_scatter (4 rows) | 1200 B | 1.0390 / 1.0921 |
| mul_add_scatter (4 rows) | 1216 B | 1.0379 / 1.0319 |
| mul_add_scatter (4 rows) | 1232 B | 1.0564 / 1.0887 |
| mul_add_scatter (4 rows) | 1248 B | 1.0744 / 1.0495 |
| mul_add_scatter (4 rows) | 1400 B | 0.9676 / 1.0466 |
| mul_assign | 64 B | 1.1578 / 1.3856 |
| mul_assign | 256 B | 1.0878 / 1.1875 |
| mul_assign | 512 B | 1.0703 / 1.2249 |
| mul_assign | 1152 B | 1.0271 / 1.0248 |
| mul_assign | 1168 B | 0.9996 / 1.0165 |
| mul_assign | 1184 B | 0.9994 / 1.0498 |
| mul_assign | 1200 B | 1.0539 / 1.0000 |
| mul_assign | 1216 B | 1.0000 / 1.0243 |
| mul_assign | 1232 B | 1.0537 / 1.0029 |
| mul_assign | 1248 B | 0.9998 / 1.0224 |
| mul_assign | 1400 B | 0.9923 / 1.0481 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | v3 speed factor |
| --- | --- | --- |
| 256 KiB | 0 | 0.9937 / 1.0118 |
| 256 KiB | 16 | 0.9994 / 1.0126 |
| 64 KiB | 0 | 0.9997 / 1.0001 |
| 64 KiB | 16 | 1.0008 / 1.0004 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 1.0436 / 1.0048 |
| 16 KiB | 0.9930 / 0.9848 |
| 256 KiB | 1.0004 / 1.0002 |
| 4 KiB | 0.9711 / 0.9738 |
| 64 KiB | 1.0018 / 1.0009 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | v3 speed factor |
| --- | --- |
| 256 KiB | 1.0000 / 1.0000 |
| 4 KiB | 0.9987 / 0.9908 |
| 64 MiB | 0.9985 / 0.9937 |
| 8 MiB | 1.0046 / 1.0011 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | Case | v3 speed factor |
| --- | --- | --- |
| 1 MiB | mul_into | 1.0046 / 1.0041 |
| 1 MiB | mul_into + read | 1.0094 / 0.9894 |
| 1 MiB | mul_add | 1.0030 / 1.0104 |
| 32 MiB | mul_into | 0.9950 / 0.9948 |
| 32 MiB | mul_into + read | 0.9979 / 1.0019 |
| 32 MiB | mul_add | 1.0060 / 1.0037 |
| 8 MiB | mul_into | 1.0009 / 0.9997 |
| 8 MiB | mul_into + read | 0.9899 / 0.9861 |
| 8 MiB | mul_add | 1.0296 / 0.9892 |

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
| mul_add | 64 KiB | 1.0000 / 0.9996 |
| mul_elementwise | 64 KiB | 0.9963 / 1.0181 |
| mul_elementwise_assign | 64 KiB | 1.0036 / 1.0001 |
| mul_into | 64 KiB | 0.9998 / 1.0000 |
| mul_into | 4 MiB | 0.9998 / 0.9993 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather_with | 4 KiB | 1.2243 / 1.0448 |
| mul_add_gather_with | 16 KiB | 1.0583 / 1.0118 |
| mul_into_gather | 4 KiB | 0.9961 / 1.0041 |
| mul_into_gather | 16 KiB | 0.9894 / 1.0052 |
| mul_into_gather_with | 4 KiB | 0.9753 / 0.9993 |
| mul_into_gather_with | 16 KiB | 0.9857 / 1.0037 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9879 / 0.9987 |
| fill(0) + mul_add_matrix | 4 destinations | 1.0016 / 1.0417 |
| fill(0) + mul_add_matrix | 6 destinations | 1.0368 / 0.9563 |
| mul_into_matrix | 2 destinations | 1.0013 / 0.9703 |
| mul_into_matrix | 4 destinations | 1.0090 / 0.8900 |
| mul_into_matrix | 6 destinations | 1.0237 / 0.9002 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | Case | v3 speed factor |
| --- | --- | --- | --- |
| 1 KiB | 0 | mul_add_scatter | 1.0298 / 1.0430 |
| 1 KiB | 0 | mul_add_gather | 1.9865 / 1.1776 |
| 1 KiB | 0 | mul_add_matrix | 1.0038 / 1.0079 |
| 1 KiB | 16 | mul_add_scatter | 1.0479 / 0.9974 |
| 1 KiB | 16 | mul_add_gather | 1.7665 / 1.1511 |
| 1 KiB | 16 | mul_add_matrix | 1.0125 / 1.0060 |
| 16 KiB | 0 | mul_add_scatter | 0.9985 / 0.9941 |
| 16 KiB | 0 | mul_add_gather | 1.0406 / 1.0127 |
| 16 KiB | 0 | mul_add_matrix | 1.0050 / 0.9652 |
| 16 KiB | 16 | mul_add_scatter | 0.9964 / 1.0023 |
| 16 KiB | 16 | mul_add_gather | 0.9961 / 1.0183 |
| 16 KiB | 16 | mul_add_matrix | 1.0106 / 0.9737 |
| 2 KiB | 0 | mul_add_scatter | 0.9826 / 1.0396 |
| 2 KiB | 0 | mul_add_gather | 1.5661 / 1.1557 |
| 2 KiB | 0 | mul_add_matrix | 1.0013 / 1.0059 |
| 2 KiB | 16 | mul_add_scatter | 1.0757 / 1.0115 |
| 2 KiB | 16 | mul_add_gather | 1.4521 / 1.1370 |
| 2 KiB | 16 | mul_add_matrix | 1.0253 / 1.0046 |
| 256 B | 0 | mul_add_scatter | 1.0987 / 1.0551 |
| 256 B | 0 | mul_add_gather | 3.5945 / 1.5208 |
| 256 B | 0 | mul_add_matrix | 1.0096 / 1.0234 |
| 256 B | 16 | mul_add_scatter | 0.9791 / 0.8837 |
| 256 B | 16 | mul_add_gather | 3.0679 / 1.5966 |
| 256 B | 16 | mul_add_matrix | 0.9355 / 1.0313 |
| 3 KiB | 0 | mul_add_scatter | 0.9386 / 1.0118 |
| 3 KiB | 0 | mul_add_gather | 1.2874 / 1.0610 |
| 3 KiB | 0 | mul_add_matrix | 1.0002 / 0.9954 |
| 3 KiB | 16 | mul_add_scatter | 0.9808 / 1.0091 |
| 3 KiB | 16 | mul_add_gather | 1.0187 / 1.0742 |
| 3 KiB | 16 | mul_add_matrix | 1.0123 / 0.9810 |
| 4 KiB | 0 | mul_add_scatter | 1.0181 / 1.0202 |
| 4 KiB | 0 | mul_add_gather | 1.1769 / 1.0454 |
| 4 KiB | 0 | mul_add_matrix | 1.0013 / 0.9963 |
| 4 KiB | 16 | mul_add_scatter | 1.0243 / 1.0107 |
| 4 KiB | 16 | mul_add_gather | 0.9702 / 1.0427 |
| 4 KiB | 16 | mul_add_matrix | 0.9975 / 0.9894 |
| 512 B | 0 | mul_add_scatter | 1.0512 / 1.0981 |
| 512 B | 0 | mul_add_gather | 2.7607 / 1.3363 |
| 512 B | 0 | mul_add_matrix | 1.0059 / 1.0142 |
| 512 B | 16 | mul_add_scatter | 1.0364 / 0.9850 |
| 512 B | 16 | mul_add_gather | 2.1931 / 1.2726 |
| 512 B | 16 | mul_add_matrix | 0.9715 / 1.0054 |
| 8 KiB | 0 | mul_add_scatter | 1.0183 / 1.0136 |
| 8 KiB | 0 | mul_add_gather | 1.0824 / 1.0241 |
| 8 KiB | 0 | mul_add_matrix | 1.0397 / 1.0113 |
| 8 KiB | 16 | mul_add_scatter | 1.0142 / 1.0066 |
| 8 KiB | 16 | mul_add_gather | 0.9469 / 1.0238 |
| 8 KiB | 16 | mul_add_matrix | 1.0355 / 1.0112 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | v3 speed factor |
| --- | --- |
| 0 | 1.0167 / 1.0011 |
| 16 | 1.0154 / 1.0030 |
| 32 | 1.0167 / 1.0024 |
| 48 | 1.0163 / 1.0019 |

### Scatter destination alignment

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | v3 speed factor |
| --- | --- |
| mul_add_scatter | 1.0050 / 1.0029 |
| mul_add_scatter_with | 1.0027 / 0.9999 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 1.0126 / 0.9520 |
| 16 KiB | 0.9957 / 1.0106 |
| 256 KiB | 1.0003 / 0.9987 |
| 4 KiB | 0.9785 / 0.9950 |
| 64 KiB | 1.0016 / 1.0073 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst = c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 1.0001 / 1.0003 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=262144; preparation=one-shot; row_bytes=16384; sources=16 | 0.9969 / 0.9970 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=65536; preparation=one-shot; row_bytes=4096; sources=16 | 0.9903 / 0.9849 |
| dst ^= c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 1.0001 / 0.9965 |
| systematic encode | alignment=page; destinations=2; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0061 / 1.0050 |
| systematic encode | alignment=page; destinations=2; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0325 / 1.0045 |
| systematic encode | alignment=page; destinations=2; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0050 / 0.9994 |
| systematic encode | alignment=page; destinations=4; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0017 / 1.0006 |
| systematic encode | alignment=page; destinations=4; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0019 / 1.0030 |
| systematic encode | alignment=page; destinations=4; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0004 / 1.0037 |
| systematic encode | alignment=page; destinations=6; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0025 / 1.0001 |
| systematic encode | alignment=page; destinations=6; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0039 / 1.0003 |
| systematic encode | alignment=page; destinations=6; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0026 / 1.0023 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
