# Wider binary fields: `Gf32`, `Gf64`, and Fan–Paar towers

Public API timings for the binary tower fields above GF(2^16). Every result
cell reports Tiger Lake / Golden Cove; shared hosts, toolchain, sampling, and
number format: [BENCHMARKS.md](../BENCHMARKS.md). Comparisons against the
previous snapshot live in [v2 comparison](v2/wide-binary.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gf-comp` |
| Resolved backend | `v4x` / `v3_gfni_crypto` |

## Self-timings

### Single-row multiplication

256 KiB `mul_add`, ordinary `Vec<u8>` buffers.

| Field | Throughput (GiB/s) |
| --- | --- |
| `Gf32` | 15.77 / 30.70 |
| `Gf64` | 8.72 / 16.44 |
| `FanPaar16` | 9.57 / 17.03 |
| `FanPaar32` | 3.31 / 6.30 |
| `FanPaar64` | 1.24 / 2.21 |

## Competitors

No matched competitor measurements are available.
