# Wider binary fields: `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/wide-binary.md), [v3](../v3/wide-binary.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `gf`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `v3_gfni_crypto`; v3: `v4x` / `v3_gfni_crypto` |
| Resolved field backends | v2 `gf32b`: `v4x` / `v3_gfni_crypto`; v2 `gf64b`: `v4x` / `v3_gfni_crypto`; v2 `fp16`: `v4x` / `v3_gfni_crypto`; v2 `fp32`: `v4x` / `v3_gfni_crypto`; v2 `fp64`: `v4x` / `v3_gfni_crypto`; v3 `gf32b`: `v4x` / `v3_gfni_crypto`; v3 `gf64b`: `v4x` / `v3_gfni_crypto`; v3 `fp16`: `v4x` / `v3_gfni_crypto`; v3 `fp32`: `v4x` / `v3_gfni_crypto`; v3 `fp64`: `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |

## gf32b

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Operation | v3 speed factor |
| --- | --- |
| mul_add | 1.0716 / 1.0217 |

## gf64b

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Operation | v3 speed factor |
| --- | --- |
| mul_add | 1.0381 / 1.0138 |

## fp16

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Operation | v3 speed factor |
| --- | --- |
| mul_add | 1.0332 / 0.9962 |

## fp32

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Operation | v3 speed factor |
| --- | --- |
| mul_add | 1.0098 / 0.9974 |

## fp64

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers. One byte volume means one information volume: the wider fields carry the same payload per call.

| Operation | v3 speed factor |
| --- | --- |
| mul_add | 1.0109 / 1.0188 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
