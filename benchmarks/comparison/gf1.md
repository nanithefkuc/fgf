# Bit-packed GF(2): `gf1`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf1.md), [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Resolved process backend | v2: `v4x` / `-`; v3: `v4x` / `-` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Ratio | v3/v2 for throughput, v2/v3 for latency; above one means v3 is faster. Computed from unrounded host-local aggregates. |

## gf1

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits [N/4, 7N/8); `bits::XorRange` is the prepared geometry that `bits::xor_range_with` applies.

| Operation | Case | v3 speed factor |
| --- | --- | --- |
| bits::XorRange 5/8 | 4 KiB | 0.9987 / - |
| bits::XorRange 5/8 | 256 KiB | 1.0000 / - |
| bits::XorRange 5/8 | 8 MiB | 1.0311 / - |
| bits::XorRange 5/8 | 64 MiB | 1.0010 / - |
| bits::and_into | 4 KiB | 0.9964 / - |
| bits::and_into | 256 KiB | 1.0000 / - |
| bits::and_into | 8 MiB | 1.0061 / - |
| bits::and_into | 64 MiB | 0.9974 / - |
| bits::dot_product | 4 KiB | 1.1218 / - |
| bits::dot_product | 256 KiB | 1.0047 / - |
| bits::dot_product | 8 MiB | 1.0120 / - |
| bits::dot_product | 64 MiB | 0.9843 / - |
| bits::weight | 4 KiB | 1.0363 / - |
| bits::weight | 256 KiB | 0.9944 / - |
| bits::weight | 8 MiB | 1.0004 / - |
| bits::weight | 64 MiB | 0.9940 / - |
| bits::xor_assign | 4 KiB | 1.0003 / - |
| bits::xor_assign | 256 KiB | 0.9998 / - |
| bits::xor_assign | 8 MiB | 1.0154 / - |
| bits::xor_assign | 64 MiB | 1.0017 / - |
| bits::xor_range 5/8 | 4 KiB | 1.0010 / - |
| bits::xor_range 5/8 | 256 KiB | 1.0003 / - |
| bits::xor_range 5/8 | 8 MiB | 1.0303 / - |
| bits::xor_range 5/8 | 64 MiB | 1.0000 / - |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | v3 speed factor |
| --- | --- |
| bits::XorRange | 1.0778 / - |
| bits::xor_range | 1.0212 / - |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
