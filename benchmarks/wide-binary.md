# Wider binary fields: `Gf32`, `Gf64`, and Fan–Paar towers

Public API timings for the binary tower fields above GF(2^16). Every result
cell reports Tiger Lake / Golden Cove; shared hosts, toolchain, sampling, and
number format: [BENCHMARKS.md](../BENCHMARKS.md).

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
| `Gf32` | 17.1 / 33.1 |
| `Gf64` | 9.09 / 17.0 |
| `FanPaar16` | 9.97 / 17.2 |
| `FanPaar32` | 3.71 / 6.34 |
| `FanPaar64` | 1.37 / 2.21 |

## Competitors

No matched competitor measurements are available.
