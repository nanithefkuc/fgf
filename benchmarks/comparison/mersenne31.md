# Mersenne fields: `m31`, `qm31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/mersenne31.md), [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `-`; v3: `v4x` / `-` |
| Resolved field backends | v2 `m31`: `v4x` / -; v2 `qm31`: `v4x` / -; v3 `m31`: `v4x` / -; v3 `qm31`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |
| Competitors | Willow Cove (i5-1135G7) v2: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Willow Cove (i5-1135G7) v3: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16} |

## m31

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9965 / - |
| add_assign | 256 KiB | 1.0068 / - |
| add_assign | 8 MiB | 1.0112 / - |
| add_assign_scalar | 4 KiB | 1.1197 / - |
| add_assign_scalar | 256 KiB | 1.0856 / - |
| add_assign_scalar | 8 MiB | 1.0181 / - |
| mul_elementwise | 4 KiB | 0.6305 / - |
| mul_elementwise | 256 KiB | 0.6657 / - |
| mul_elementwise | 8 MiB | 0.9694 / - |
| mul_elementwise_assign | 4 KiB | 0.6061 / - |
| mul_elementwise_assign | 256 KiB | 0.6574 / - |
| mul_elementwise_assign | 8 MiB | 0.9608 / - |
| sub_assign_scalar | 4 KiB | 1.4000 / - |
| sub_assign_scalar | 256 KiB | 1.3165 / - |
| sub_assign_scalar | 8 MiB | 1.0962 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9182 / - |
| 1 KiB | 16 | 1.0059 / - |
| 1 KiB | 2 | 0.9753 / - |
| 1 KiB | 32 | 1.0049 / - |
| 1 KiB | 4 | 1.0060 / - |
| 1 KiB | 8 | 1.0005 / - |
| 64 B | 1 | 1.0364 / - |
| 64 B | 16 | 0.9175 / - |
| 64 B | 2 | 1.0114 / - |
| 64 B | 32 | 0.9755 / - |
| 64 B | 4 | 0.9932 / - |
| 64 B | 8 | 0.9934 / - |
| 64 KiB | 1 | 1.0014 / - |
| 64 KiB | 16 | 1.0079 / - |
| 64 KiB | 2 | 0.9989 / - |
| 64 KiB | 32 | 1.0010 / - |
| 64 KiB | 4 | 1.0006 / - |
| 64 KiB | 8 | 0.9875 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 0.9626 / - |
| add_assign per row | 1 KiB | 16 | 0.9938 / - |
| add_assign per row | 1 KiB | 2 | 0.9578 / - |
| add_assign per row | 1 KiB | 32 | 0.9969 / - |
| add_assign per row | 1 KiB | 4 | 0.9621 / - |
| add_assign per row | 1 KiB | 8 | 0.9668 / - |
| add_assign per row | 64 B | 1 | 1.0000 / - |
| add_assign per row | 64 B | 16 | 0.9687 / - |
| add_assign per row | 64 B | 2 | 0.9999 / - |
| add_assign per row | 64 B | 32 | 1.0331 / - |
| add_assign per row | 64 B | 4 | 1.0116 / - |
| add_assign per row | 64 B | 8 | 1.0005 / - |
| add_assign per row | 64 KiB | 1 | 1.0035 / - |
| add_assign per row | 64 KiB | 16 | 1.0083 / - |
| add_assign per row | 64 KiB | 2 | 0.9997 / - |
| add_assign per row | 64 KiB | 32 | 0.9962 / - |
| add_assign per row | 64 KiB | 4 | 1.0002 / - |
| add_assign per row | 64 KiB | 8 | 0.9887 / - |
| add_assign | 1 KiB | 1 | 0.9360 / - |
| add_assign | 1 KiB | 16 | 1.0064 / - |
| add_assign | 1 KiB | 2 | 0.9816 / - |
| add_assign | 1 KiB | 32 | 1.0075 / - |
| add_assign | 1 KiB | 4 | 0.9909 / - |
| add_assign | 1 KiB | 8 | 0.9995 / - |
| add_assign | 64 B | 1 | 0.9994 / - |
| add_assign | 64 B | 16 | 0.9385 / - |
| add_assign | 64 B | 2 | 0.9309 / - |
| add_assign | 64 B | 32 | 0.9780 / - |
| add_assign | 64 B | 4 | 1.0279 / - |
| add_assign | 64 B | 8 | 1.0757 / - |
| add_assign | 64 KiB | 1 | 1.0058 / - |
| add_assign | 64 KiB | 16 | 1.0080 / - |
| add_assign | 64 KiB | 2 | 1.0017 / - |
| add_assign | 64 KiB | 32 | 0.9964 / - |
| add_assign | 64 KiB | 4 | 1.0008 / - |
| add_assign | 64 KiB | 8 | 0.9883 / - |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=m31; lane_bytes=4; region_bytes=65536; scope=indexed acc += a * b | 1.0047 / - |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5913 / - |
| dst += v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1246 / - |
| dst -= v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.4007 / - |
| dst = a * b | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5852 / - |
| dst = c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.6898 / - |

## qm31

### Packed operations

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9969 / - |
| add_assign | 256 KiB | 1.0143 / - |
| add_assign | 8 MiB | 1.0192 / - |
| add_assign_scalar | 4 KiB | 1.1237 / - |
| add_assign_scalar | 256 KiB | 1.0993 / - |
| add_assign_scalar | 8 MiB | 1.0488 / - |
| mul_elementwise | 4 KiB | 0.8981 / - |
| mul_elementwise | 256 KiB | 0.8778 / - |
| mul_elementwise | 8 MiB | 0.9782 / - |
| mul_elementwise_assign | 4 KiB | 0.9300 / - |
| mul_elementwise_assign | 256 KiB | 0.9291 / - |
| mul_elementwise_assign | 8 MiB | 0.9545 / - |
| sub_assign_scalar | 4 KiB | 1.3916 / - |
| sub_assign_scalar | 256 KiB | 1.3377 / - |
| sub_assign_scalar | 8 MiB | 1.1902 / - |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=qm31; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 0.5580 / - |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9379 / - |
| dst += v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0887 / - |
| dst -= v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.3571 / - |
| dst = a * b | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.8851 / - |
| dst = c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9110 / - |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
