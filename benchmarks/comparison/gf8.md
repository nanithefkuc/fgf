# Byte fields: `gf8b`, `gf8d`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf8.md), [v3](../v3/gf8.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| scalar mul | 1.0000 / 1.0000 |

### Packed operations

`gf8b`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 1.0756 / 1.0006 |
| add_assign | 256 KiB | 0.9987 / 1.0009 |
| add_assign | 8 MiB | 1.0036 / 0.9949 |
| add_assign_scalar | 4 KiB | 1.0239 / 1.0003 |
| add_assign_scalar | 256 KiB | 1.0006 / 1.0003 |
| add_assign_scalar | 8 MiB | 0.9877 / 1.0092 |
| mul_add | 4 KiB | 1.0277 / 0.9916 |
| mul_add | 256 KiB | 1.0004 / 1.0138 |
| mul_add | 8 MiB | 0.9687 / 0.9962 |
| mul_assign | 4 KiB | 1.0416 / 1.0029 |
| mul_assign | 256 KiB | 1.0001 / 1.0003 |
| mul_assign | 8 MiB | 0.9964 / 0.9999 |
| mul_elementwise | 4 KiB | 1.0462 / 1.0023 |
| mul_elementwise | 256 KiB | 0.9901 / 1.0132 |
| mul_elementwise | 8 MiB | 0.9935 / 0.9959 |
| mul_elementwise_assign | 4 KiB | 0.9758 / 0.8854 |
| mul_elementwise_assign | 256 KiB | 1.0008 / 1.0110 |
| mul_elementwise_assign | 8 MiB | 0.9818 / 1.0041 |
| mul_into | 4 KiB | 1.0080 / 1.0202 |
| mul_into | 256 KiB | 1.0012 / 1.0003 |
| mul_into | 8 MiB | 1.0003 / 1.0010 |
| sub_assign_scalar | 4 KiB | 1.0308 / 1.0001 |
| sub_assign_scalar | 256 KiB | 1.0007 / 1.0006 |
| sub_assign_scalar | 8 MiB | 0.9858 / 1.0018 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0003 / 1.0004 |
| mul_elementwise | 64 KiB | 1.0018 / 1.0003 |
| mul_elementwise_assign | 64 KiB | 1.0003 / 1.0012 |
| mul_into | 64 KiB | 1.0014 / 0.9999 |

### One-shot and prepared multiplication

Latency per call. The one-shot form derives the backend coefficient on every call; the prepared form reuses coefficients built outside the timed region.

| Row bytes | Case | v3 speed factor |
| --- | --- | --- |
| 1 KiB | mul_add | 1.0511 / 1.0031 |
| 1 KiB | mul_add_with | 1.0251 / 1.0096 |
| 128 B | mul_add | 1.3480 / 0.9585 |
| 128 B | mul_add_with | 1.3850 / 0.9997 |
| 16 B | mul_add | 1.3419 / 0.9589 |
| 16 B | mul_add_with | 1.3980 / 1.0022 |
| 16 KiB | mul_add | 0.9919 / 1.0091 |
| 16 KiB | mul_add_with | 0.9850 / 1.0095 |
| 2 KiB | mul_add | 0.9388 / 1.0038 |
| 2 KiB | mul_add_with | 0.9432 / 1.0057 |
| 256 B | mul_add | 1.1570 / 0.9656 |
| 256 B | mul_add_with | 1.2053 / 1.0031 |
| 32 B | mul_add | 1.3246 / 0.9499 |
| 32 B | mul_add_with | 1.4219 / 0.9970 |
| 4 KiB | mul_add | 0.9558 / 0.9993 |
| 4 KiB | mul_add_with | 0.9654 / 0.9927 |
| 512 B | mul_add | 1.1234 / 1.0489 |
| 512 B | mul_add_with | 1.1006 / 1.0377 |
| 64 B | mul_add | 1.2680 / 0.9563 |
| 64 B | mul_add_with | 1.3293 / 0.9930 |
| 64 KiB | mul_add | 1.0005 / 1.0005 |
| 64 KiB | mul_add_with | 0.9981 / 1.0003 |

### Scatter, gather, and matrix

64 KiB rows, ordinary `Vec<u8>` buffers. The column count is destinations for scatter and matrix, sources for gather.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather | 2 rows | 1.0126 / 1.0065 |
| mul_add_gather | 4 rows | 1.0324 / 1.0159 |
| mul_add_gather | 8 rows | 1.0296 / 1.0053 |
| mul_add_gather | 16 rows | 1.0166 / 1.0029 |
| mul_add_gather_with | 2 rows | 1.0128 / 1.0265 |
| mul_add_gather_with | 4 rows | 1.0343 / 1.0247 |
| mul_add_gather_with | 8 rows | 1.0280 / 1.0015 |
| mul_add_gather_with | 16 rows | 1.0174 / 1.0091 |
| mul_add_matrix | 2 rows | 0.9955 / 1.0019 |
| mul_add_matrix | 4 rows | 1.0049 / 1.0023 |
| mul_add_matrix | 8 rows | 0.9994 / 1.0011 |
| mul_add_matrix | 16 rows | 0.9937 / 0.9975 |
| mul_add_matrix_with | 2 rows | 1.0017 / 0.9988 |
| mul_add_matrix_with | 4 rows | 1.0016 / 1.0018 |
| mul_add_matrix_with | 8 rows | 1.0287 / 0.9967 |
| mul_add_matrix_with | 16 rows | 1.0198 / 0.9971 |
| mul_add_scatter | 2 rows | 0.9962 / 1.0000 |
| mul_add_scatter | 4 rows | 1.0020 / 1.0008 |
| mul_add_scatter | 8 rows | 1.0013 / 0.9987 |
| mul_add_scatter | 16 rows | 0.9909 / 1.0156 |
| mul_add_scatter_with | 2 rows | 0.9969 / 0.9998 |
| mul_add_scatter_with | 4 rows | 1.0023 / 0.9998 |
| mul_add_scatter_with | 8 rows | 1.0012 / 0.9986 |
| mul_add_scatter_with | 16 rows | 0.9918 / 1.0116 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_into_gather_with | 4 KiB | 1.0032 / 1.0401 |
| mul_into_gather_with | 16 KiB | 1.0085 / 1.0108 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9980 / 1.0171 |
| fill(0) + mul_add_matrix | 4 destinations | 1.0033 / 0.8418 |
| fill(0) + mul_add_matrix | 6 destinations | 1.0031 / 0.8996 |
| mul_add_matrix_at | 2 destinations | 1.0022 / 1.0157 |
| mul_add_matrix_at | 4 destinations | 0.9702 / 0.9985 |
| mul_add_matrix_at | 6 destinations | 0.9739 / 1.0145 |
| mul_add_matrix_with | 2 destinations | 1.0000 / 1.0031 |
| mul_add_matrix_with | 4 destinations | 0.9836 / 0.8171 |
| mul_add_matrix_with | 6 destinations | 1.0196 / 0.8870 |
| mul_into_matrix | 2 destinations | 0.9988 / 0.9884 |
| mul_into_matrix | 4 destinations | 1.0062 / 0.8641 |
| mul_into_matrix | 6 destinations | 0.9950 / 0.8983 |
| mul_into_matrix_with | 2 destinations | 0.9982 / 1.0309 |
| mul_into_matrix_with | 4 destinations | 0.9870 / 0.8624 |
| mul_into_matrix_with | 6 destinations | 1.0084 / 0.9274 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9807 / 1.0022 |
| 1 KiB | 16 | 1.0042 / 1.0178 |
| 1 KiB | 2 | 1.0057 / 1.0009 |
| 1 KiB | 32 | 0.9992 / 1.0000 |
| 1 KiB | 4 | 1.0125 / 1.0013 |
| 1 KiB | 8 | 1.0093 / 1.0185 |
| 64 B | 1 | 0.9509 / 1.0007 |
| 64 B | 16 | 0.9796 / 1.0033 |
| 64 B | 2 | 0.9909 / 0.9053 |
| 64 B | 32 | 1.0079 / 1.0009 |
| 64 B | 4 | 1.0850 / 0.9999 |
| 64 B | 8 | 0.9988 / 1.0104 |
| 64 KiB | 1 | 0.9995 / 0.9994 |
| 64 KiB | 16 | 0.9981 / 0.9875 |
| 64 KiB | 2 | 0.9999 / 0.9996 |
| 64 KiB | 32 | 1.0020 / 1.0000 |
| 64 KiB | 4 | 1.0001 / 1.0000 |
| 64 KiB | 8 | 0.9495 / 1.0104 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 1.0009 / 1.0063 |
| add_assign per row | 1 KiB | 16 | 0.9592 / 1.0012 |
| add_assign per row | 1 KiB | 2 | 0.9912 / 1.0025 |
| add_assign per row | 1 KiB | 32 | 0.9728 / 1.0035 |
| add_assign per row | 1 KiB | 4 | 0.8507 / 0.9998 |
| add_assign per row | 1 KiB | 8 | 0.9408 / 0.9973 |
| add_assign per row | 64 B | 1 | 0.9724 / 0.9542 |
| add_assign per row | 64 B | 16 | 0.8573 / 0.9519 |
| add_assign per row | 64 B | 2 | 0.9039 / 0.9531 |
| add_assign per row | 64 B | 32 | 0.8367 / 0.9110 |
| add_assign per row | 64 B | 4 | 0.8879 / 0.9621 |
| add_assign per row | 64 B | 8 | 0.8790 / 0.8988 |
| add_assign per row | 64 KiB | 1 | 1.0010 / 1.0003 |
| add_assign per row | 64 KiB | 16 | 0.9977 / 0.9875 |
| add_assign per row | 64 KiB | 2 | 0.9970 / 0.9994 |
| add_assign per row | 64 KiB | 32 | 1.0011 / 1.0004 |
| add_assign per row | 64 KiB | 4 | 0.9960 / 1.0007 |
| add_assign per row | 64 KiB | 8 | 0.9609 / 1.0172 |
| add_assign | 1 KiB | 1 | 0.9996 / 1.0006 |
| add_assign | 1 KiB | 16 | 0.9957 / 1.0003 |
| add_assign | 1 KiB | 2 | 1.1164 / 1.0055 |
| add_assign | 1 KiB | 32 | 0.9988 / 0.9994 |
| add_assign | 1 KiB | 4 | 1.0254 / 1.0010 |
| add_assign | 1 KiB | 8 | 0.9937 / 1.0179 |
| add_assign | 64 B | 1 | 0.9203 / 1.0463 |
| add_assign | 64 B | 16 | 1.0001 / 1.0021 |
| add_assign | 64 B | 2 | 0.9604 / 0.9057 |
| add_assign | 64 B | 32 | 1.1166 / 1.0032 |
| add_assign | 64 B | 4 | 1.2519 / 1.0003 |
| add_assign | 64 B | 8 | 0.9973 / 1.0003 |
| add_assign | 64 KiB | 1 | 0.9987 / 0.9996 |
| add_assign | 64 KiB | 16 | 0.9979 / 0.9874 |
| add_assign | 64 KiB | 2 | 0.9997 / 0.9997 |
| add_assign | 64 KiB | 32 | 1.0018 / 1.0000 |
| add_assign | 64 KiB | 4 | 0.9998 / 1.0001 |
| add_assign | 64 KiB | 8 | 0.9485 / 1.0076 |

### Network-size payloads

gf8b, ordinary `Vec<u8>` buffers; scatter rows use the payload length as the row pitch.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 64 B | 0.9601 / 1.0510 |
| add_assign | 256 B | 1.3901 / 0.9601 |
| add_assign | 512 B | 0.9992 / 1.0044 |
| add_assign | 1152 B | 1.0121 / 0.9991 |
| add_assign | 1168 B | 1.0071 / 1.0010 |
| add_assign | 1184 B | 1.0608 / 0.9999 |
| add_assign | 1200 B | 0.9780 / 0.9999 |
| add_assign | 1216 B | 1.0274 / 1.0031 |
| add_assign | 1232 B | 1.0410 / 1.0066 |
| add_assign | 1248 B | 1.0032 / 1.0011 |
| add_assign | 1400 B | 0.9820 / 1.0035 |
| mul_add | 64 B | 1.2951 / 0.9670 |
| mul_add | 256 B | 1.4307 / 0.9649 |
| mul_add | 512 B | 1.2974 / 1.0083 |
| mul_add | 1152 B | 1.0584 / 1.0026 |
| mul_add | 1168 B | 1.0901 / 0.9851 |
| mul_add | 1184 B | 1.0937 / 1.0026 |
| mul_add | 1200 B | 1.0963 / 0.9998 |
| mul_add | 1216 B | 1.0602 / 1.0007 |
| mul_add | 1232 B | 1.0788 / 1.0001 |
| mul_add | 1248 B | 1.0806 / 1.0022 |
| mul_add | 1400 B | 1.1832 / 1.0005 |
| mul_add_scatter (16 rows) | 64 B | 1.1601 / 1.3801 |
| mul_add_scatter (16 rows) | 256 B | 1.1091 / 1.1685 |
| mul_add_scatter (16 rows) | 512 B | 1.0668 / 0.9714 |
| mul_add_scatter (16 rows) | 1152 B | 1.0362 / 1.0376 |
| mul_add_scatter (16 rows) | 1168 B | 1.0613 / 1.0693 |
| mul_add_scatter (16 rows) | 1184 B | 1.0740 / 1.0313 |
| mul_add_scatter (16 rows) | 1200 B | 1.0962 / 1.0588 |
| mul_add_scatter (16 rows) | 1216 B | 1.0203 / 1.0704 |
| mul_add_scatter (16 rows) | 1232 B | 1.0706 / 1.0575 |
| mul_add_scatter (16 rows) | 1248 B | 1.0509 / 1.0450 |
| mul_add_scatter (16 rows) | 1400 B | 0.9495 / 1.0479 |
| mul_add_scatter (4 rows) | 64 B | 1.0631 / 1.2647 |
| mul_add_scatter (4 rows) | 256 B | 1.1404 / 1.1722 |
| mul_add_scatter (4 rows) | 512 B | 1.0093 / 0.9327 |
| mul_add_scatter (4 rows) | 1152 B | 1.0035 / 0.9914 |
| mul_add_scatter (4 rows) | 1168 B | 1.0741 / 1.0586 |
| mul_add_scatter (4 rows) | 1184 B | 1.0626 / 1.0019 |
| mul_add_scatter (4 rows) | 1200 B | 1.0493 / 1.0565 |
| mul_add_scatter (4 rows) | 1216 B | 1.0016 / 1.0187 |
| mul_add_scatter (4 rows) | 1232 B | 1.0427 / 1.0716 |
| mul_add_scatter (4 rows) | 1248 B | 1.0505 / 1.0289 |
| mul_add_scatter (4 rows) | 1400 B | 0.9757 / 1.0140 |
| mul_assign | 64 B | 1.0422 / 1.3859 |
| mul_assign | 256 B | 1.1002 / 1.2655 |
| mul_assign | 512 B | 1.0677 / 1.0007 |
| mul_assign | 1152 B | 1.0292 / 0.9925 |
| mul_assign | 1168 B | 1.0640 / 1.0147 |
| mul_assign | 1184 B | 1.0289 / 0.9931 |
| mul_assign | 1200 B | 1.0568 / 1.0196 |
| mul_assign | 1216 B | 1.0528 / 0.9888 |
| mul_assign | 1232 B | 1.0536 / 1.0185 |
| mul_assign | 1248 B | 1.0773 / 0.9851 |
| mul_assign | 1400 B | 1.0014 / 1.0462 |

### Scatter destination alignment

Eight destinations, ordinary `Vec<u8>` source; the destination starts at the stated offset from a 64-byte boundary.

| Row size | Destination offset mod 64 | v3 speed factor |
| --- | --- | --- |
| 256 KiB | 0 | 1.0057 / 0.9872 |
| 256 KiB | 16 | 1.0046 / 1.0097 |
| 64 KiB | 0 | 1.0006 / 0.9993 |
| 64 KiB | 16 | 1.0007 / 1.0000 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 1.0366 / 1.0599 |
| 16 KiB | 0.9609 / 0.9839 |
| 256 KiB | 1.0004 / 1.0001 |
| 4 KiB | 0.9551 / 0.9729 |
| 64 KiB | 1.0019 / 1.0004 |

### Byte-field density control

`gf8b` byte-per-element density control measured alongside the bit-packed buffers.

| Buffer bytes | v3 speed factor |
| --- | --- |
| 256 KiB | 0.9999 / 1.0001 |
| 4 KiB | 0.9908 / 1.0011 |
| 64 MiB | 1.0005 / 1.0045 |
| 8 MiB | 1.0012 / 0.9911 |

### Large destinations

Ordinary `Vec<u8>` buffers. The read-back row XOR-folds every written 64-bit word after `mul_into`.

| Buffer size | Case | v3 speed factor |
| --- | --- | --- |
| 1 MiB | mul_into | 1.0026 / 0.9919 |
| 1 MiB | mul_into + read | 0.9958 / 0.9991 |
| 1 MiB | mul_add | 1.0028 / 0.9934 |
| 32 MiB | mul_into | 0.9989 / 1.0273 |
| 32 MiB | mul_into + read | 1.0103 / 1.0083 |
| 32 MiB | mul_add | 1.0006 / 1.0003 |
| 8 MiB | mul_into | 0.8989 / 1.0003 |
| 8 MiB | mul_into + read | 0.9764 / 1.0062 |
| 8 MiB | mul_add | 1.0013 / 0.9893 |

## gf8d

### Scalar multiplication

Latency of the dependent multiply chain, per scalar operation.

| Operation | v3 speed factor |
| --- | --- |
| scalar mul | 0.9999 / 0.9998 |

### Aligned single-row operations

64-byte-aligned buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add | 64 KiB | 1.0004 / 1.0000 |
| mul_elementwise | 64 KiB | 1.0024 / 1.0011 |
| mul_elementwise_assign | 64 KiB | 1.0037 / 0.9997 |
| mul_into | 64 KiB | 1.0016 / 0.9998 |
| mul_into | 4 MiB | 0.9991 / 1.0001 |

### Aligned gather

Sixteen sources into one row, 64-byte-aligned buffers. `mul_into_gather` is the raw one-shot overwrite form.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| mul_add_gather_with | 4 KiB | 1.2264 / 1.0454 |
| mul_add_gather_with | 16 KiB | 1.0638 / 1.0090 |
| mul_into_gather | 4 KiB | 1.0032 / 0.9978 |
| mul_into_gather | 16 KiB | 1.0083 / 1.0122 |
| mul_into_gather_with | 4 KiB | 0.9706 / 1.0046 |
| mul_into_gather_with | 16 KiB | 1.0030 / 1.0155 |

### Overwrite matrix

Ten sources, 64 KiB rows, 64-byte-aligned buffers. The `fill(0)` cases include clearing the destination; `mul_add_matrix_at` uses contiguous row offsets.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| fill(0) + mul_add_matrix | 2 destinations | 0.9809 / 0.9998 |
| fill(0) + mul_add_matrix | 4 destinations | 0.9965 / 1.0031 |
| fill(0) + mul_add_matrix | 6 destinations | 0.9949 / 0.9989 |
| mul_into_matrix | 2 destinations | 1.0031 / 0.9903 |
| mul_into_matrix | 4 destinations | 0.9832 / 0.8479 |
| mul_into_matrix | 6 destinations | 0.9929 / 0.9161 |

### Fixed-offset multi-row operations

gf8d; every source and destination base carries the stated offset modulo 64. Scatter has four destinations, gather sixteen sources, matrix ten sources and four destinations.

| Row bytes | Base offset mod 64 | Case | v3 speed factor |
| --- | --- | --- | --- |
| 1 KiB | 0 | mul_add_scatter | 1.0194 / 1.0435 |
| 1 KiB | 0 | mul_add_gather | 2.0224 / 1.1626 |
| 1 KiB | 0 | mul_add_matrix | 1.0026 / 1.0108 |
| 1 KiB | 16 | mul_add_scatter | 1.0161 / 0.9984 |
| 1 KiB | 16 | mul_add_gather | 1.7963 / 1.1428 |
| 1 KiB | 16 | mul_add_matrix | 1.0048 / 1.0059 |
| 16 KiB | 0 | mul_add_scatter | 1.0025 / 1.0009 |
| 16 KiB | 0 | mul_add_gather | 1.0278 / 1.0120 |
| 16 KiB | 0 | mul_add_matrix | 0.9973 / 1.0010 |
| 16 KiB | 16 | mul_add_scatter | 1.0101 / 1.0052 |
| 16 KiB | 16 | mul_add_gather | 0.9970 / 1.0093 |
| 16 KiB | 16 | mul_add_matrix | 0.9954 / 1.0018 |
| 2 KiB | 0 | mul_add_scatter | 0.9832 / 1.0374 |
| 2 KiB | 0 | mul_add_gather | 1.5837 / 1.0968 |
| 2 KiB | 0 | mul_add_matrix | 1.0013 / 1.0018 |
| 2 KiB | 16 | mul_add_scatter | 1.0243 / 1.0083 |
| 2 KiB | 16 | mul_add_gather | 1.3214 / 1.1043 |
| 2 KiB | 16 | mul_add_matrix | 0.9936 / 1.0024 |
| 256 B | 0 | mul_add_scatter | 1.0618 / 1.0595 |
| 256 B | 0 | mul_add_gather | 3.6358 / 1.5357 |
| 256 B | 0 | mul_add_matrix | 1.0024 / 1.0247 |
| 256 B | 16 | mul_add_scatter | 1.1834 / 1.0391 |
| 256 B | 16 | mul_add_gather | 3.0935 / 1.4493 |
| 256 B | 16 | mul_add_matrix | 1.0012 / 1.0391 |
| 3 KiB | 0 | mul_add_scatter | 0.9506 / 1.0097 |
| 3 KiB | 0 | mul_add_gather | 1.3498 / 1.0786 |
| 3 KiB | 0 | mul_add_matrix | 1.0008 / 1.0004 |
| 3 KiB | 16 | mul_add_scatter | 0.9754 / 1.0053 |
| 3 KiB | 16 | mul_add_gather | 1.0080 / 1.0663 |
| 3 KiB | 16 | mul_add_matrix | 0.9967 / 1.0027 |
| 4 KiB | 0 | mul_add_scatter | 1.0139 / 1.0167 |
| 4 KiB | 0 | mul_add_gather | 1.1608 / 1.0424 |
| 4 KiB | 0 | mul_add_matrix | 1.0010 / 0.9945 |
| 4 KiB | 16 | mul_add_scatter | 1.0204 / 1.0245 |
| 4 KiB | 16 | mul_add_gather | 0.9835 / 1.0298 |
| 4 KiB | 16 | mul_add_matrix | 1.0076 / 0.9942 |
| 512 B | 0 | mul_add_scatter | 1.0521 / 1.0994 |
| 512 B | 0 | mul_add_gather | 2.8403 / 1.3322 |
| 512 B | 0 | mul_add_matrix | 0.9853 / 1.0186 |
| 512 B | 16 | mul_add_scatter | 1.0515 / 0.9950 |
| 512 B | 16 | mul_add_gather | 2.3386 / 1.2577 |
| 512 B | 16 | mul_add_matrix | 1.0090 / 1.0199 |
| 8 KiB | 0 | mul_add_scatter | 1.0132 / 1.0077 |
| 8 KiB | 0 | mul_add_gather | 1.1015 / 1.0231 |
| 8 KiB | 0 | mul_add_matrix | 0.9932 / 0.9999 |
| 8 KiB | 16 | mul_add_scatter | 0.9091 / 1.0079 |
| 8 KiB | 16 | mul_add_gather | 0.9814 / 1.0210 |
| 8 KiB | 16 | mul_add_matrix | 0.9963 / 0.9974 |

### Fixed-offset single-row multiplication

gf8d `mul_add`, 16 KiB; source and destination start at the stated offset from a 64-byte boundary.

| Base offset mod 64 | v3 speed factor |
| --- | --- |
| 0 | 1.0107 / 1.0017 |
| 16 | 1.0128 / 1.0042 |
| 32 | 1.0114 / 1.0020 |
| 48 | 1.0066 / 1.0036 |

### Scatter destination alignment

gf8d, one source and four 16 KiB destination rows, 64-byte-aligned buffers; raw coefficients against prepared coefficients through the plan.

| Operation | v3 speed factor |
| --- | --- |
| mul_add_scatter | 1.0008 / 1.0019 |
| mul_add_scatter_with | 1.0018 / 1.0005 |

### In-place scaling by buffer size

`mul_assign`, 64-byte-aligned buffers.

| Buffer size | v3 speed factor |
| --- | --- |
| 1 MiB | 0.9934 / 0.9695 |
| 16 KiB | 0.9681 / 0.9805 |
| 256 KiB | 1.0004 / 0.9991 |
| 4 KiB | 0.9676 / 0.9946 |
| 64 KiB | 1.0014 / 0.9919 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst = c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9990 / 0.9997 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=262144; preparation=one-shot; row_bytes=16384; sources=16 | 1.0182 / 1.0050 |
| dst = sum(c[i] * src[i]) | alignment=page; destinations=1; logical_bytes=65536; preparation=one-shot; row_bytes=4096; sources=16 | 0.9971 / 1.0002 |
| dst ^= c * src | alignment=page; logical_bytes=65536; preparation=one-shot; row_bytes=65536 | 0.9986 / 0.9996 |
| systematic encode | alignment=page; destinations=2; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0033 / 1.0154 |
| systematic encode | alignment=page; destinations=2; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0144 / 1.0015 |
| systematic encode | alignment=page; destinations=2; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0024 / 1.0032 |
| systematic encode | alignment=page; destinations=4; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0020 / 1.0004 |
| systematic encode | alignment=page; destinations=4; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0107 / 0.9979 |
| systematic encode | alignment=page; destinations=4; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0044 / 0.9992 |
| systematic encode | alignment=page; destinations=6; logical_bytes=163840; preparation=one-shot; row_bytes=16384; sources=10 | 1.0119 / 1.0007 |
| systematic encode | alignment=page; destinations=6; logical_bytes=40960; preparation=one-shot; row_bytes=4096; sources=10 | 1.0112 / 0.9997 |
| systematic encode | alignment=page; destinations=6; logical_bytes=655360; preparation=one-shot; row_bytes=65536; sources=10 | 1.0117 / 1.0104 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
