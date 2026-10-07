# Mersenne fields: `m31`, `qm31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/mersenne31.md), [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| add_assign | 4 KiB | 1.0103 / 1.0050 |
| add_assign | 256 KiB | 1.0030 / 0.9954 |
| add_assign | 8 MiB | 1.0312 / 1.0032 |
| add_assign_scalar | 4 KiB | 1.1240 / 1.0968 |
| add_assign_scalar | 256 KiB | 1.0859 / 1.0411 |
| add_assign_scalar | 8 MiB | 1.0160 / 0.9957 |
| mul_elementwise | 4 KiB | 0.6277 / 0.5905 |
| mul_elementwise | 256 KiB | 0.6957 / 0.6581 |
| mul_elementwise | 8 MiB | 0.9818 / 1.0784 |
| mul_elementwise_assign | 4 KiB | 0.6063 / 0.6163 |
| mul_elementwise_assign | 256 KiB | 0.6620 / 0.6015 |
| mul_elementwise_assign | 8 MiB | 0.9605 / 0.8877 |
| sub_assign_scalar | 4 KiB | 1.4021 / 1.3824 |
| sub_assign_scalar | 256 KiB | 1.3264 / 1.2980 |
| sub_assign_scalar | 8 MiB | 1.0897 / 0.8799 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 1.0351 / 0.9991 |
| 1 KiB | 16 | 1.0065 / 0.9999 |
| 1 KiB | 2 | 1.0115 / 0.9954 |
| 1 KiB | 32 | 1.0071 / 1.0381 |
| 1 KiB | 4 | 1.0025 / 1.0058 |
| 1 KiB | 8 | 1.0040 / 1.0061 |
| 64 B | 1 | 0.9947 / 1.0343 |
| 64 B | 16 | 1.0424 / 0.9996 |
| 64 B | 2 | 1.0161 / 0.9557 |
| 64 B | 32 | 1.0118 / 0.9968 |
| 64 B | 4 | 0.9453 / 0.9914 |
| 64 B | 8 | 0.9707 / 0.9960 |
| 64 KiB | 1 | 1.0027 / 0.9651 |
| 64 KiB | 16 | 0.9921 / 1.0009 |
| 64 KiB | 2 | 0.9991 / 0.9995 |
| 64 KiB | 32 | 1.0151 / 1.0000 |
| 64 KiB | 4 | 1.0004 / 1.0021 |
| 64 KiB | 8 | 1.0022 / 1.0100 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 1.0413 / 0.9988 |
| add_assign per row | 1 KiB | 16 | 1.0302 / 1.0023 |
| add_assign per row | 1 KiB | 2 | 1.0419 / 0.9984 |
| add_assign per row | 1 KiB | 32 | 1.0121 / 1.0044 |
| add_assign per row | 1 KiB | 4 | 1.0317 / 0.9987 |
| add_assign per row | 1 KiB | 8 | 1.0285 / 0.9992 |
| add_assign per row | 64 B | 1 | 1.0001 / 1.0008 |
| add_assign per row | 64 B | 16 | 1.0874 / 1.0005 |
| add_assign per row | 64 B | 2 | 1.0000 / 1.0002 |
| add_assign per row | 64 B | 32 | 1.0223 / 0.9975 |
| add_assign per row | 64 B | 4 | 1.0116 / 1.0009 |
| add_assign per row | 64 B | 8 | 1.0060 / 1.0006 |
| add_assign per row | 64 KiB | 1 | 1.0027 / 0.9922 |
| add_assign per row | 64 KiB | 16 | 0.9918 / 1.0021 |
| add_assign per row | 64 KiB | 2 | 0.9998 / 0.9995 |
| add_assign per row | 64 KiB | 32 | 1.0178 / 1.0122 |
| add_assign per row | 64 KiB | 4 | 1.0005 / 0.9631 |
| add_assign per row | 64 KiB | 8 | 1.0032 / 1.0072 |
| add_assign | 1 KiB | 1 | 0.9564 / 1.0060 |
| add_assign | 1 KiB | 16 | 1.0092 / 0.9682 |
| add_assign | 1 KiB | 2 | 1.0009 / 0.9911 |
| add_assign | 1 KiB | 32 | 1.0085 / 1.0061 |
| add_assign | 1 KiB | 4 | 1.0095 / 1.0057 |
| add_assign | 1 KiB | 8 | 1.0098 / 0.9751 |
| add_assign | 64 B | 1 | 1.0015 / 1.0907 |
| add_assign | 64 B | 16 | 0.9549 / 1.0005 |
| add_assign | 64 B | 2 | 1.0088 / 1.0127 |
| add_assign | 64 B | 32 | 1.0040 / 0.9938 |
| add_assign | 64 B | 4 | 0.9710 / 0.9866 |
| add_assign | 64 B | 8 | 0.9753 / 0.9891 |
| add_assign | 64 KiB | 1 | 1.0039 / 1.0083 |
| add_assign | 64 KiB | 16 | 0.9935 / 1.0024 |
| add_assign | 64 KiB | 2 | 0.9998 / 0.9995 |
| add_assign | 64 KiB | 32 | 1.0148 / 1.0006 |
| add_assign | 64 KiB | 4 | 1.0005 / 0.9962 |
| add_assign | 64 KiB | 8 | 1.0026 / 1.0165 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=m31; lane_bytes=4; region_bytes=65536; scope=indexed acc += a * b | 1.0066 / 0.9992 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5918 / 0.6148 |
| dst += v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1243 / 1.1269 |
| dst -= v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.4003 / 1.4054 |
| dst = a * b | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.5857 / 0.6036 |
| dst = c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.6892 / 0.6975 |

## qm31

### Packed operations

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9979 / 1.0031 |
| add_assign | 256 KiB | 1.0048 / 0.9988 |
| add_assign | 8 MiB | 1.0336 / 0.9944 |
| add_assign_scalar | 4 KiB | 1.1206 / 1.0770 |
| add_assign_scalar | 256 KiB | 1.0991 / 1.0471 |
| add_assign_scalar | 8 MiB | 1.0584 / 0.9438 |
| mul_elementwise | 4 KiB | 0.8951 / 0.9302 |
| mul_elementwise | 256 KiB | 0.9029 / 0.9517 |
| mul_elementwise | 8 MiB | 0.9948 / 0.9769 |
| mul_elementwise_assign | 4 KiB | 0.9288 / 0.9057 |
| mul_elementwise_assign | 256 KiB | 0.9311 / 0.9173 |
| mul_elementwise_assign | 8 MiB | 0.9450 / 0.9193 |
| sub_assign_scalar | 4 KiB | 1.3967 / 1.3038 |
| sub_assign_scalar | 256 KiB | 1.3392 / 1.2295 |
| sub_assign_scalar | 8 MiB | 1.1950 / 0.7671 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=qm31; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 0.9290 / 0.8803 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9378 / 0.8920 |
| dst += v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0900 / 1.1277 |
| dst -= v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.3582 / 1.4051 |
| dst = a * b | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.8853 / 0.9162 |
| dst = c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9108 / 0.9092 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
