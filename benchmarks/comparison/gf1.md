# Bit-packed GF(2): `gf1`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf1.md), [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| bits::XorRange 5/8 | 4 KiB | 0.9921 / 0.9977 |
| bits::XorRange 5/8 | 256 KiB | 1.0000 / 1.0002 |
| bits::XorRange 5/8 | 8 MiB | 1.0084 / 1.0001 |
| bits::XorRange 5/8 | 64 MiB | 1.0045 / 0.9888 |
| bits::and_into | 4 KiB | 1.0021 / 1.0265 |
| bits::and_into | 256 KiB | 1.0004 / 0.9831 |
| bits::and_into | 8 MiB | 1.0188 / 0.9478 |
| bits::and_into | 64 MiB | 1.0003 / 0.9984 |
| bits::dot_product | 4 KiB | 1.1259 / 0.9270 |
| bits::dot_product | 256 KiB | 1.0047 / 0.9022 |
| bits::dot_product | 8 MiB | 0.9818 / 1.0041 |
| bits::dot_product | 64 MiB | 1.0093 / 1.0208 |
| bits::weight | 4 KiB | 0.9681 / 0.9841 |
| bits::weight | 256 KiB | 0.9940 / 1.0000 |
| bits::weight | 8 MiB | 0.9980 / 1.0000 |
| bits::weight | 64 MiB | 0.9989 / 1.0003 |
| bits::xor_assign | 4 KiB | 0.9972 / 0.9987 |
| bits::xor_assign | 256 KiB | 0.9999 / 1.0000 |
| bits::xor_assign | 8 MiB | 0.9959 / 1.0023 |
| bits::xor_assign | 64 MiB | 0.9980 / 0.9947 |
| bits::xor_range 5/8 | 4 KiB | 1.0037 / 0.9969 |
| bits::xor_range 5/8 | 256 KiB | 1.0001 / 1.0005 |
| bits::xor_range 5/8 | 8 MiB | 1.0094 / 1.0000 |
| bits::xor_range 5/8 | 64 MiB | 1.0019 / 0.9892 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | v3 speed factor |
| --- | --- |
| bits::XorRange | 1.0790 / 0.9996 |
| bits::xor_range | 1.0158 / 1.0363 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
