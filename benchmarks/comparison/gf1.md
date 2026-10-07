# Bit-packed GF(2): `gf1`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); hosts and shared method: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). Snapshots: [v2](../v2/gf1.md), [v3](../v3/gf1.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (canonical family `gf1`) |
| v2 | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| bits::XorRange 5/8 | 4 KiB | 1.0043 / 1.0218 |
| bits::XorRange 5/8 | 256 KiB | 0.9999 / 1.0009 |
| bits::XorRange 5/8 | 8 MiB | 1.0206 / 0.9990 |
| bits::XorRange 5/8 | 64 MiB | 1.0027 / 1.0179 |
| bits::and_into | 4 KiB | 1.0039 / 0.9938 |
| bits::and_into | 256 KiB | 1.0001 / 0.9629 |
| bits::and_into | 8 MiB | 0.9945 / 1.1302 |
| bits::and_into | 64 MiB | 1.0066 / 1.0052 |
| bits::dot_product | 4 KiB | 1.1036 / 1.0218 |
| bits::dot_product | 256 KiB | 1.0203 / 0.9997 |
| bits::dot_product | 8 MiB | 0.9784 / 1.1488 |
| bits::dot_product | 64 MiB | 1.0140 / 0.9833 |
| bits::weight | 4 KiB | 1.0072 / 1.0196 |
| bits::weight | 256 KiB | 0.9999 / 0.9999 |
| bits::weight | 8 MiB | 1.1077 / 0.9999 |
| bits::weight | 64 MiB | 1.0005 / 0.9998 |
| bits::xor_assign | 4 KiB | 0.9993 / 1.0011 |
| bits::xor_assign | 256 KiB | 0.9999 / 1.0000 |
| bits::xor_assign | 8 MiB | 0.9885 / 1.0129 |
| bits::xor_assign | 64 MiB | 0.9979 / 1.0027 |
| bits::xor_range 5/8 | 4 KiB | 0.9955 / 1.0072 |
| bits::xor_range 5/8 | 256 KiB | 1.0002 / 1.0032 |
| bits::xor_range 5/8 | 8 MiB | 1.0193 / 0.9992 |
| bits::xor_range 5/8 | 64 MiB | 1.0027 / 1.0173 |

### Short ranges

16-byte rows, bits [61, 69); latency per row pair, not per batch.

| Operation | v3 speed factor |
| --- | --- |
| bits::XorRange | 1.0619 / 0.9990 |
| bits::xor_range | 1.0095 / 1.0015 |

## Caveats

- Ratios are computed from unrounded host-local aggregates, never from the displayed cells.
- Differences comparable to control variability are unresolved, not proof of a faster implementation.
