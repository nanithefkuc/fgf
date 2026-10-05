# Bit-packed GF(2)

Public API timings for the `bits` module. Every result cell reports Tiger Lake
/ Golden Cove; shared hosts, toolchain, sampling, and number format:
[BENCHMARKS.md](../BENCHMARKS.md). The v2-versus-v3 comparison lives in
[v2 comparison](v2/gf2.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gf2-comp` |
| Resolved backend | `v4x` / `v3_gfni_crypto` |
| Throughput | Buffer bytes. Range XOR counts the entire packed buffer, not only the updated range. |

## Self-timings

### Buffer operations

Ordinary `Vec<u8>` buffers. Range XOR updates bits `[N/4, 7N/8)`, where `N` is
the number of packed bits; `bits::XorRange` is the prepared geometry that
`bits::xor_range_with` applies.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) | 64 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `bits::xor_assign` | 42.95 / 87.63 | 41.36 / 66.10 | 8.69 / 25.69 | 6.07 / 14.65 |
| `bits::and_into` | 22.39 / 68.50 | 19.32 / 46.92 | 5.93 / 19.18 | 4.64 / 10.94 |
| `bits::weight` | 119.44 / 14.69 | 67.86 / 15.01 | 38.59 / 15.01 | 18.01 / 14.44 |
| `bits::dot_product` | 20.33 / 39.83 | 18.28 / 43.44 | 11.28 / 38.18 | 8.76 / 18.52 |
| `bits::xor_range` | 53.07 / 89.63 | 65.93 / 74.43 | 21.04 / 39.85 | 9.75 / 23.94 |
| `bits::XorRange` | 57.28 / 99.65 | 66.04 / 74.51 | 21.06 / 39.86 | 9.83 / 24.24 |

### Short ranges

| Operation | Case | Latency per row pair (ns/op) |
| --- | --- | --- |
| `bits::xor_range` | 16-byte rows; bits `[61, 69)` | 11.70 / 5.35 |
| `bits::XorRange` | 16-byte rows; bits `[61, 69)` | 4.53 / 2.08 |

## Competitors

No matched competitor measurements are available.
