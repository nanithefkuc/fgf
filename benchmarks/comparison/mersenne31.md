# Mersenne fields: `m31`, `qm31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/mersenne31.md), [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `388cbb4a43d5d20e8a663e89c3b44380c7c4f8b7` |
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
| add_assign | 4 KiB | 1.0073 / 1.0105 |
| add_assign | 256 KiB | 1.0053 / 1.0036 |
| add_assign | 8 MiB | 0.9954 / 1.0013 |
| add_assign_scalar | 4 KiB | 1.1173 / 1.0857 |
| add_assign_scalar | 256 KiB | 1.0855 / 1.0474 |
| add_assign_scalar | 8 MiB | 1.0338 / 1.0007 |
| mul_elementwise | 4 KiB | 1.0503 / 0.8958 |
| mul_elementwise | 256 KiB | 1.0534 / 1.0097 |
| mul_elementwise | 8 MiB | 1.0072 / 0.9935 |
| mul_elementwise_assign | 4 KiB | 0.9977 / 0.8133 |
| mul_elementwise_assign | 256 KiB | 1.1678 / 0.8300 |
| mul_elementwise_assign | 8 MiB | 1.0529 / 1.0570 |
| sub_assign_scalar | 4 KiB | 1.4044 / 1.3849 |
| sub_assign_scalar | 256 KiB | 1.3241 / 1.2577 |
| sub_assign_scalar | 8 MiB | 1.1109 / 0.8754 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 0.9091 / 1.0014 |
| 1 KiB | 16 | 1.0046 / 1.0015 |
| 1 KiB | 2 | 0.9781 / 0.9976 |
| 1 KiB | 32 | 1.0140 / 1.0098 |
| 1 KiB | 4 | 0.9945 / 1.0036 |
| 1 KiB | 8 | 1.0001 / 1.0230 |
| 64 B | 1 | 0.9628 / 1.0005 |
| 64 B | 16 | 0.9078 / 1.0039 |
| 64 B | 2 | 0.9391 / 0.9657 |
| 64 B | 32 | 0.9795 / 0.9975 |
| 64 B | 4 | 0.9493 / 1.0040 |
| 64 B | 8 | 1.0172 / 1.0002 |
| 64 KiB | 1 | 1.0008 / 1.0065 |
| 64 KiB | 16 | 0.9947 / 1.0157 |
| 64 KiB | 2 | 1.0006 / 0.9992 |
| 64 KiB | 32 | 1.0319 / 0.9980 |
| 64 KiB | 4 | 1.0003 / 0.9946 |
| 64 KiB | 8 | 1.0353 / 1.0044 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 0.9440 / 0.9978 |
| add_assign per row | 1 KiB | 16 | 0.9843 / 1.0004 |
| add_assign per row | 1 KiB | 2 | 0.9497 / 0.9990 |
| add_assign per row | 1 KiB | 32 | 0.9929 / 1.0056 |
| add_assign per row | 1 KiB | 4 | 0.9735 / 1.0034 |
| add_assign per row | 1 KiB | 8 | 0.9682 / 0.9997 |
| add_assign per row | 64 B | 1 | 0.9990 / 1.0029 |
| add_assign per row | 64 B | 16 | 0.9474 / 1.0004 |
| add_assign per row | 64 B | 2 | 0.9996 / 1.0000 |
| add_assign per row | 64 B | 32 | 0.9959 / 1.0000 |
| add_assign per row | 64 B | 4 | 0.9777 / 1.0002 |
| add_assign per row | 64 B | 8 | 1.0371 / 1.0009 |
| add_assign per row | 64 KiB | 1 | 1.0030 / 1.0111 |
| add_assign per row | 64 KiB | 16 | 0.9956 / 1.0190 |
| add_assign per row | 64 KiB | 2 | 0.9992 / 0.9987 |
| add_assign per row | 64 KiB | 32 | 1.0306 / 0.9980 |
| add_assign per row | 64 KiB | 4 | 1.0004 / 0.9949 |
| add_assign per row | 64 KiB | 8 | 1.0362 / 0.9991 |
| add_assign | 1 KiB | 1 | 1.0271 / 1.0043 |
| add_assign | 1 KiB | 16 | 1.0076 / 1.0017 |
| add_assign | 1 KiB | 2 | 0.9756 / 1.0007 |
| add_assign | 1 KiB | 32 | 1.0162 / 1.0071 |
| add_assign | 1 KiB | 4 | 0.9941 / 1.0122 |
| add_assign | 1 KiB | 8 | 0.9979 / 1.0280 |
| add_assign | 64 B | 1 | 0.9998 / 1.0453 |
| add_assign | 64 B | 16 | 1.0233 / 1.0072 |
| add_assign | 64 B | 2 | 0.9998 / 1.0006 |
| add_assign | 64 B | 32 | 0.9761 / 0.9982 |
| add_assign | 64 B | 4 | 0.9591 / 1.0092 |
| add_assign | 64 B | 8 | 1.0158 / 1.0055 |
| add_assign | 64 KiB | 1 | 1.0068 / 1.0058 |
| add_assign | 64 KiB | 16 | 0.9946 / 1.0176 |
| add_assign | 64 KiB | 2 | 1.0004 / 1.0018 |
| add_assign | 64 KiB | 32 | 1.0331 / 0.9973 |
| add_assign | 64 KiB | 4 | 1.0007 / 0.9939 |
| add_assign | 64 KiB | 8 | 1.0357 / 0.9994 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=m31; lane_bytes=4; region_bytes=65536; scope=indexed acc += a * b | 1.0070 / 1.0021 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.8718 / 0.8264 |
| dst += v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1243 / 1.1266 |
| dst -= v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.4000 / 1.4055 |
| dst = a * b | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9354 / 0.8960 |
| dst = c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9635 / 0.8984 |

## qm31

### Packed operations

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9947 / 1.0077 |
| add_assign | 256 KiB | 1.0036 / 1.0222 |
| add_assign | 8 MiB | 1.0165 / 1.0128 |
| add_assign_scalar | 4 KiB | 1.1109 / 1.0749 |
| add_assign_scalar | 256 KiB | 1.0989 / 1.0722 |
| add_assign_scalar | 8 MiB | 1.0715 / 0.9473 |
| mul_elementwise | 4 KiB | 1.2975 / 1.2207 |
| mul_elementwise | 256 KiB | 1.2331 / 1.2120 |
| mul_elementwise | 8 MiB | 1.0177 / 1.1946 |
| mul_elementwise_assign | 4 KiB | 1.0995 / 1.2189 |
| mul_elementwise_assign | 256 KiB | 1.1458 / 1.2254 |
| mul_elementwise_assign | 8 MiB | 1.0946 / 1.3021 |
| sub_assign_scalar | 4 KiB | 1.4163 / 1.3054 |
| sub_assign_scalar | 256 KiB | 1.3371 / 1.2317 |
| sub_assign_scalar | 8 MiB | 1.2050 / 0.7702 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=qm31; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 1.9975 / 1.9079 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1707 / 1.0599 |
| dst += v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0903 / 1.1278 |
| dst -= v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.3581 / 1.4059 |
| dst = a * b | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.2894 / 1.2182 |
| dst = c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1714 / 1.1665 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
