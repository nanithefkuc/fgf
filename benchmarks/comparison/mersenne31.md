# Mersenne fields: `m31`, `qm31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/mersenne31.md), [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| add_assign | 4 KiB | 1.0087 / 1.0058 |
| add_assign | 256 KiB | 1.0035 / 1.0015 |
| add_assign | 8 MiB | 1.0113 / 0.9987 |
| add_assign_scalar | 4 KiB | 1.1250 / 1.0858 |
| add_assign_scalar | 256 KiB | 1.0858 / 1.0387 |
| add_assign_scalar | 8 MiB | 1.0191 / 1.0115 |
| mul_elementwise | 4 KiB | 1.0415 / 0.8718 |
| mul_elementwise | 256 KiB | 1.1187 / 1.0019 |
| mul_elementwise | 8 MiB | 0.9897 / 1.0160 |
| mul_elementwise_assign | 4 KiB | 0.7399 / 0.8177 |
| mul_elementwise_assign | 256 KiB | 0.7862 / 0.7595 |
| mul_elementwise_assign | 8 MiB | 0.9923 / 1.0532 |
| sub_assign_scalar | 4 KiB | 1.3956 / 1.3829 |
| sub_assign_scalar | 256 KiB | 1.3262 / 1.2451 |
| sub_assign_scalar | 8 MiB | 1.1016 / 0.8809 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | v3 speed factor |
| --- | --- | --- |
| 1 KiB | 1 | 1.1220 / 0.9997 |
| 1 KiB | 16 | 1.0080 / 1.0001 |
| 1 KiB | 2 | 1.0214 / 0.9991 |
| 1 KiB | 32 | 1.0076 / 1.0065 |
| 1 KiB | 4 | 1.0080 / 1.0044 |
| 1 KiB | 8 | 1.0138 / 1.0036 |
| 64 B | 1 | 1.0372 / 1.0008 |
| 64 B | 16 | 1.1228 / 0.9998 |
| 64 B | 2 | 1.0006 / 1.0001 |
| 64 B | 32 | 1.0131 / 0.9989 |
| 64 B | 4 | 0.9956 / 0.9993 |
| 64 B | 8 | 0.9998 / 1.0148 |
| 64 KiB | 1 | 1.0016 / 0.9945 |
| 64 KiB | 16 | 0.9939 / 1.0149 |
| 64 KiB | 2 | 0.9997 / 0.9993 |
| 64 KiB | 32 | 1.0001 / 0.9977 |
| 64 KiB | 4 | 1.0004 / 0.9979 |
| 64 KiB | 8 | 1.0124 / 0.9773 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | v3 speed factor |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 1.0658 / 0.9976 |
| add_assign per row | 1 KiB | 16 | 1.0356 / 0.9981 |
| add_assign per row | 1 KiB | 2 | 1.0514 / 0.9941 |
| add_assign per row | 1 KiB | 32 | 1.0091 / 1.0082 |
| add_assign per row | 1 KiB | 4 | 1.0193 / 1.0025 |
| add_assign per row | 1 KiB | 8 | 1.0158 / 1.0024 |
| add_assign per row | 64 B | 1 | 1.0001 / 0.9938 |
| add_assign per row | 64 B | 16 | 1.0355 / 1.0007 |
| add_assign per row | 64 B | 2 | 0.9663 / 0.9768 |
| add_assign per row | 64 B | 32 | 0.9967 / 1.0012 |
| add_assign per row | 64 B | 4 | 1.0001 / 1.0012 |
| add_assign per row | 64 B | 8 | 1.0060 / 0.9938 |
| add_assign per row | 64 KiB | 1 | 1.0023 / 1.0361 |
| add_assign per row | 64 KiB | 16 | 0.9943 / 1.0117 |
| add_assign per row | 64 KiB | 2 | 0.9998 / 0.9727 |
| add_assign per row | 64 KiB | 32 | 1.0010 / 1.0080 |
| add_assign per row | 64 KiB | 4 | 1.0005 / 0.9619 |
| add_assign per row | 64 KiB | 8 | 1.0129 / 1.0112 |
| add_assign | 1 KiB | 1 | 1.1123 / 0.9986 |
| add_assign | 1 KiB | 16 | 1.0095 / 0.9989 |
| add_assign | 1 KiB | 2 | 1.0030 / 1.0001 |
| add_assign | 1 KiB | 32 | 1.0140 / 0.9991 |
| add_assign | 1 KiB | 4 | 1.0046 / 1.0040 |
| add_assign | 1 KiB | 8 | 1.0084 / 0.9756 |
| add_assign | 64 B | 1 | 1.0415 / 0.9597 |
| add_assign | 64 B | 16 | 1.1111 / 1.0108 |
| add_assign | 64 B | 2 | 1.0089 / 1.0011 |
| add_assign | 64 B | 32 | 1.0003 / 0.9949 |
| add_assign | 64 B | 4 | 1.0303 / 1.0309 |
| add_assign | 64 B | 8 | 1.0200 / 1.0041 |
| add_assign | 64 KiB | 1 | 1.0030 / 1.0080 |
| add_assign | 64 KiB | 16 | 0.9936 / 1.0066 |
| add_assign | 64 KiB | 2 | 0.9995 / 0.9991 |
| add_assign | 64 KiB | 32 | 1.0025 / 0.9967 |
| add_assign | 64 KiB | 4 | 1.0006 / 1.0018 |
| add_assign | 64 KiB | 8 | 1.0125 / 0.9813 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=m31; lane_bytes=4; region_bytes=65536; scope=indexed acc += a * b | 1.0043 / 1.0055 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.6849 / 0.8214 |
| dst += v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.1247 / 1.1267 |
| dst -= v | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.4005 / 1.4044 |
| dst = a * b | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.8632 / 0.8846 |
| dst = c * src | alignment=page; field=m31; lane_bytes=4; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.7580 / 0.9039 |

## qm31

### Packed operations

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 0.9988 / 1.0098 |
| add_assign | 256 KiB | 1.0036 / 1.0042 |
| add_assign | 8 MiB | 0.9991 / 0.9937 |
| add_assign_scalar | 4 KiB | 1.1236 / 1.0761 |
| add_assign_scalar | 256 KiB | 1.0995 / 1.0474 |
| add_assign_scalar | 8 MiB | 1.0766 / 0.9484 |
| mul_elementwise | 4 KiB | 1.0856 / 1.2305 |
| mul_elementwise | 256 KiB | 1.0492 / 1.2308 |
| mul_elementwise | 8 MiB | 0.9967 / 1.1422 |
| mul_elementwise_assign | 4 KiB | 1.1274 / 1.1968 |
| mul_elementwise_assign | 256 KiB | 1.1350 / 1.1729 |
| mul_elementwise_assign | 8 MiB | 1.0548 / 1.2840 |
| sub_assign_scalar | 4 KiB | 1.4179 / 1.3058 |
| sub_assign_scalar | 256 KiB | 1.3392 / 1.2310 |
| sub_assign_scalar | 8 MiB | 1.2257 / 0.7725 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=qm31; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 1.9901 / 1.8940 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0222 / 1.0585 |
| dst += v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0904 / 1.1306 |
| dst -= v | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.3583 / 1.4060 |
| dst = a * b | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0464 / 1.2185 |
| dst = c * src | alignment=page; field=qm31; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0357 / 1.1670 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
