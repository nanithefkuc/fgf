# Bit-packed GF(2): `gf1`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf1.md), [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `v3_gfni_crypto`; v3: `v4x` / `v3_gfni_crypto` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |

## gf1

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits [N/4, 7N/8); `bits::XorRange` is the prepared geometry that `bits::xor_range_with` applies.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| bits::XorRange 5/8 | 4 KiB | 0.9987 / 1.0003 |
| bits::XorRange 5/8 | 256 KiB | 1.0000 / 1.0030 |
| bits::XorRange 5/8 | 8 MiB | 1.0311 / 1.0002 |
| bits::XorRange 5/8 | 64 MiB | 1.0010 / 1.0024 |
| bits::and_into | 4 KiB | 0.9964 / 1.0227 |
| bits::and_into | 256 KiB | 1.0000 / 0.9559 |
| bits::and_into | 8 MiB | 1.0061 / 1.0154 |
| bits::and_into | 64 MiB | 0.9974 / 1.0014 |
| bits::dot_product | 4 KiB | 1.1218 / 1.0407 |
| bits::dot_product | 256 KiB | 1.0047 / 0.9994 |
| bits::dot_product | 8 MiB | 1.0120 / 1.1197 |
| bits::dot_product | 64 MiB | 0.9843 / 0.9859 |
| bits::weight | 4 KiB | 1.0363 / 1.0008 |
| bits::weight | 256 KiB | 0.9944 / 1.0000 |
| bits::weight | 8 MiB | 1.0004 / 0.9999 |
| bits::weight | 64 MiB | 0.9940 / 0.9994 |
| bits::xor_assign | 4 KiB | 1.0003 / 1.0019 |
| bits::xor_assign | 256 KiB | 0.9998 / 1.0000 |
| bits::xor_assign | 8 MiB | 1.0154 / 0.9906 |
| bits::xor_assign | 64 MiB | 1.0017 / 1.0061 |
| bits::xor_range 5/8 | 4 KiB | 1.0010 / 1.0077 |
| bits::xor_range 5/8 | 256 KiB | 1.0003 / 0.9994 |
| bits::xor_range 5/8 | 8 MiB | 1.0303 / 0.9992 |
| bits::xor_range 5/8 | 64 MiB | 1.0000 / 1.0011 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | v3 speed factor |
| --- | --- |
| bits::XorRange | 1.0778 / 1.0148 |
| bits::xor_range | 1.0212 / 1.0392 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
