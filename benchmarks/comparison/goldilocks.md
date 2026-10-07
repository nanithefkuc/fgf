# Goldilocks: `gld`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/goldilocks.md), [v3](../v3/goldilocks.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gdl`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `v3_gfni_crypto`; v3: `v4x` / `v3_gfni_crypto` |
| Resolved field backends | v2 `gld`: `v4x` / `v3`; v3 `gld`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |
| Competitors | Willow Cove (i5-1135G7) v2: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Willow Cove (i5-1135G7) v3: Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K) v2: Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8}; Golden Cove (i7-12700K) v3: Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## gld

### Packed operations

`gld`, ordinary `Vec<u8>` buffers.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| add_assign | 4 KiB | 1.0042 / 1.0243 |
| add_assign | 256 KiB | 0.9998 / 0.9985 |
| add_assign | 8 MiB | 1.0008 / 1.0028 |
| add_assign_scalar | 4 KiB | 1.0270 / 1.0100 |
| add_assign_scalar | 256 KiB | 1.0000 / 1.0022 |
| add_assign_scalar | 8 MiB | 0.9961 / 1.0002 |
| mul_elementwise | 4 KiB | 1.0004 / 0.9899 |
| mul_elementwise | 256 KiB | 0.9999 / 1.0013 |
| mul_elementwise | 8 MiB | 0.9953 / 1.0097 |
| mul_elementwise_assign | 4 KiB | 1.0069 / 1.0015 |
| mul_elementwise_assign | 256 KiB | 0.9999 / 0.9996 |
| mul_elementwise_assign | 8 MiB | 0.9910 / 0.9905 |
| sub_assign_scalar | 4 KiB | 1.0205 / 1.0001 |
| sub_assign_scalar | 256 KiB | 0.9998 / 1.0038 |
| sub_assign_scalar | 8 MiB | 0.9974 / 0.9991 |

### Indexed scalar multiplication and accumulation

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| acc += a * b | alignment=page; field=gld; lane_bytes=8; region_bytes=65536; scope=indexed acc += a * b | 0.8340 / 0.8011 |

### Native-layout public operations

The fgf arm's public calls from the native-layout harness; only version arms are compared. Layout conversion remains outside timing.

| Operation | Geometry | v3 speed factor |
| --- | --- | --- |
| dst += c * src | alignment=page; field=gld; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9999 / 1.0029 |
| dst += v | alignment=page; field=gld; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 1.0001 / 1.0002 |
| dst -= v | alignment=page; field=gld; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9999 / 1.0022 |
| dst = a * b | alignment=page; field=gld; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9998 / 1.0012 |
| dst = c * src | alignment=page; field=gld; lane_bytes=8; logical_bytes=131072; preparation=one-shot; region_bytes=65536 | 0.9997 / 1.0230 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
