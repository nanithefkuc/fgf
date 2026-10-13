# Mersenne fields: `m31`, `qm31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/mersenne31.md), [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `v3_gfni_crypto`; v3: `v4x` / `v3_gfni_crypto` |
| Resolved field backends | v2 `m31`: `v4x` / `v3`; v2 `qm31`: `v4x` / `v3`; v3 `m31`: `v4x` / `v3`; v3 `qm31`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |
| Competitors | Willow Cove (i5-1135G7) v2: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Willow Cove (i5-1135G7) v3: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K) v2: Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8}; Golden Cove (i7-12700K) v3: Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## m31

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9965 / 1.0132 |
| add_assign | 256 KiB | 1.0068 / 0.9992 |
| add_assign | 8 MiB | 1.0112 / 0.9992 |
| add_assign_scalar | 4 KiB | 1.1197 / 1.0861 |
| add_assign_scalar | 256 KiB | 1.0856 / 1.0384 |
| add_assign_scalar | 8 MiB | 1.0181 / 1.0062 |
| mul_elementwise | 4 KiB | 0.6305 / 0.5744 |
| mul_elementwise | 256 KiB | 0.6657 / 0.5723 |
| mul_elementwise | 8 MiB | 0.9694 / 0.9361 |
| mul_elementwise_assign | 4 KiB | 0.6061 / 0.6102 |
| mul_elementwise_assign | 256 KiB | 0.6574 / 0.5924 |
| mul_elementwise_assign | 8 MiB | 0.9608 / 0.9044 |
| sub_assign_scalar | 4 KiB | 1.4000 / 1.3842 |
| sub_assign_scalar | 256 KiB | 1.3165 / 1.2462 |
| sub_assign_scalar | 8 MiB | 1.0962 / 0.8762 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9182 / 1.0020 |
| 1 KiB | 16 | 1.0059 / 0.9646 |
| 1 KiB | 2 | 0.9753 / 1.0012 |
| 1 KiB | 32 | 1.0049 / 0.9619 |
| 1 KiB | 4 | 1.0060 / 1.0081 |
| 1 KiB | 8 | 1.0005 / 0.9817 |
| 64 B | 1 | 1.0364 / 1.0408 |
| 64 B | 16 | 0.9175 / 1.0038 |
| 64 B | 2 | 1.0114 / 1.0020 |
| 64 B | 32 | 0.9755 / 0.9968 |
| 64 B | 4 | 0.9932 / 1.0147 |
| 64 B | 8 | 0.9934 / 1.0053 |
| 64 KiB | 1 | 1.0014 / 0.9609 |
| 64 KiB | 16 | 1.0079 / 0.9911 |
| 64 KiB | 2 | 0.9989 / 0.9990 |
| 64 KiB | 32 | 1.0010 / 1.0086 |
| 64 KiB | 4 | 1.0006 / 0.9934 |
| 64 KiB | 8 | 0.9875 / 1.0082 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 0.9626 / 1.0016 |
| add_assign per row | 1 KiB | 16 | 0.9938 / 1.0046 |
| add_assign per row | 1 KiB | 2 | 0.9578 / 0.9958 |
| add_assign per row | 1 KiB | 32 | 0.9969 / 1.0058 |
| add_assign per row | 1 KiB | 4 | 0.9621 / 1.0034 |
| add_assign per row | 1 KiB | 8 | 0.9668 / 1.0071 |
| add_assign per row | 64 B | 1 | 1.0000 / 1.0040 |
| add_assign per row | 64 B | 16 | 0.9687 / 1.0008 |
| add_assign per row | 64 B | 2 | 0.9999 / 0.9997 |
| add_assign per row | 64 B | 32 | 1.0331 / 0.9973 |
| add_assign per row | 64 B | 4 | 1.0116 / 1.0001 |
| add_assign per row | 64 B | 8 | 1.0005 / 1.0008 |
| add_assign per row | 64 KiB | 1 | 1.0035 / 0.9613 |
| add_assign per row | 64 KiB | 16 | 1.0083 / 0.9862 |
| add_assign per row | 64 KiB | 2 | 0.9997 / 0.9757 |
| add_assign per row | 64 KiB | 32 | 0.9962 / 1.0076 |
| add_assign per row | 64 KiB | 4 | 1.0002 / 0.9662 |
| add_assign per row | 64 KiB | 8 | 0.9887 / 1.0347 |
| add_assign | 1 KiB | 1 | 0.9360 / 1.0071 |
| add_assign | 1 KiB | 16 | 1.0064 / 0.9638 |
| add_assign | 1 KiB | 2 | 0.9816 / 0.9997 |
| add_assign | 1 KiB | 32 | 1.0075 / 0.9639 |
| add_assign | 1 KiB | 4 | 0.9909 / 1.0067 |
| add_assign | 1 KiB | 8 | 0.9995 / 0.9763 |
| add_assign | 64 B | 1 | 0.9994 / 1.0455 |
| add_assign | 64 B | 16 | 0.9385 / 1.0027 |
| add_assign | 64 B | 2 | 0.9309 / 1.0055 |
| add_assign | 64 B | 32 | 0.9780 / 0.9956 |
| add_assign | 64 B | 4 | 1.0279 / 1.0047 |
| add_assign | 64 B | 8 | 1.0757 / 1.0036 |
| add_assign | 64 KiB | 1 | 1.0058 / 0.9865 |
| add_assign | 64 KiB | 16 | 1.0080 / 0.9845 |
| add_assign | 64 KiB | 2 | 1.0017 / 0.9772 |
| add_assign | 64 KiB | 32 | 0.9964 / 0.9973 |
| add_assign | 64 KiB | 4 | 1.0008 / 0.9976 |
| add_assign | 64 KiB | 8 | 0.9883 / 1.0165 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=m31; lane_bytes=4; region_bytes=65536; scope=indexed acc += a * b | 1.0047 / 0.9959 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5913 / 0.6141 |
| dst += v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1246 / 1.1266 |
| dst -= v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.4007 / 1.4052 |
| dst = a * b | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5852 / 0.6032 |
| dst = c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.6898 / 0.6978 |

## qm31

### Packed operations

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9969 / 1.0076 |
| add_assign | 256 KiB | 1.0143 / 0.9854 |
| add_assign | 8 MiB | 1.0192 / 1.0014 |
| add_assign_scalar | 4 KiB | 1.1237 / 1.0749 |
| add_assign_scalar | 256 KiB | 1.0993 / 1.0448 |
| add_assign_scalar | 8 MiB | 1.0488 / 0.9480 |
| mul_elementwise | 4 KiB | 0.8981 / 0.9353 |
| mul_elementwise | 256 KiB | 0.8778 / 0.8724 |
| mul_elementwise | 8 MiB | 0.9782 / 0.9204 |
| mul_elementwise_assign | 4 KiB | 0.9300 / 0.9071 |
| mul_elementwise_assign | 256 KiB | 0.9291 / 0.9172 |
| mul_elementwise_assign | 8 MiB | 0.9545 / 0.9230 |
| sub_assign_scalar | 4 KiB | 1.3916 / 1.3021 |
| sub_assign_scalar | 256 KiB | 1.3377 / 1.2342 |
| sub_assign_scalar | 8 MiB | 1.1902 / 0.7710 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=qm31; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 0.5580 / 0.6722 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9379 / 0.8905 |
| dst += v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0887 / 1.1278 |
| dst -= v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.3571 / 1.4059 |
| dst = a * b | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.8851 / 0.9140 |
| dst = c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9110 / 0.9139 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
