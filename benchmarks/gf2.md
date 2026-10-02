# Bit-packed GF(2)

Public API timings for the `bits` module. Every result cell reports Tiger Lake
/ Golden Cove; shared hosts, toolchain, sampling, and number format:
[BENCHMARKS.md](../BENCHMARKS.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gf2-comp` |
| Resolved backend | `v4x` / `v3_gfni_crypto` |
| Throughput | Buffer bytes. Range XOR counts the entire packed buffer, not only the updated range. |

## Self-timings

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where `N` is
the number of packed bits.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.7 / 88.4 | 41.4 / 66.1 | 8.66 / 25.8 | 6.14 / 14.8 |
| `bits::and_into` | 22.3 / 66.5 | 19.3 / 47.7 | 5.87 / 19.6 | 4.66 / 10.9 |
| `bits::weight` | 115 / 14.5 | 67.9 / 15.0 | 38.6 / 15.0 | 18.0 / 14.4 |
| `bits::dot_product` | 18.1 / 38.3 | 17.8 / 43.5 | 10.7 / 38.4 | 8.91 / 18.9 |
| `bits::xor_range` | 53.4 / 89.5 | 65.9 / 74.5 | 20.8 / 39.9 | 9.73 / 23.9 |
| `bits::xor_range_with` | 57.8 / 100 | 66.0 / 74.3 | 20.9 / 39.9 | 9.82 / 24.4 |

### Short ranges

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.7 / 5.35 |
| `bits::xor_range_with` | 16-byte rows; bits `[61, 69)` | 4.45 / 2.08 |

## Competitors

No matched competitor measurements are available.
