# Mersenne31 and `QuadMersenne31`

Public API timings for the Mersenne prime field, its quadratic extension, and
the matched Plonky3 panel. Every result cell reports Tiger Lake / Golden Cove;
shared hosts, toolchain, sampling, and number format:
[BENCHMARKS.md](../BENCHMARKS.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-m31-comp` |
| Resolved backend | `v4x` / `v3` |
| Competitor | Plonky3 0.7.0 |
| Competitor build | `external/prime-bench` with `fgf/simd512` and `-C target-cpu=native` |
| Plonky3 packing | `Mersenne31`: 16 / 8 elements; quadratic extension: 16 / 8 elements |
| Throughput | Self-timings count buffer bytes. Competitor cells count one operand pair, twice the region bytes. |

## Self-timings

Scalar latency is the `acc += a * b` row of the competitor table.

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Mersenne31` | `add_assign` | 29.5 / 34.9 | 23.2 / 32.6 | 8.20 / 21.4 |
| `Mersenne31` | `add_assign_scalar` | 40.2 / 52.3 | 41.2 / 53.0 | 24.2 / 21.9 |
| `Mersenne31` | `sub_assign_scalar` | 32.5 / 41.1 | 33.9 / 43.5 | 22.6 / 25.1 |
| `Mersenne31` | `mul_elementwise` | 22.8 / 27.8 | 21.3 / 32.4 | 5.79 / 20.2 |
| `Mersenne31` | `mul_elementwise_assign` | 25.0 / 33.6 | 21.2 / 34.1 | 8.38 / 22.7 |
| `QuadMersenne31` | `add_assign` | 20.3 / 38.7 | 18.3 / 30.5 | 7.74 / 23.7 |
| `QuadMersenne31` | `add_assign_scalar` | 29.6 / 52.9 | 27.2 / 51.4 | 18.8 / 26.9 |
| `QuadMersenne31` | `sub_assign_scalar` | 23.6 / 42.6 | 22.5 / 43.4 | 16.7 / 33.1 |
| `QuadMersenne31` | `mul_elementwise` | 11.8 / 14.4 | 11.7 / 14.7 | 5.17 / 14.2 |
| `QuadMersenne31` | `mul_elementwise_assign` | 7.24 / 15.0 | 7.37 / 15.1 | 6.62 / 14.1 |

### Row-wise addition

`Mersenne31` `add_assign_rows`, ordinary `Vec<u8>` buffers.

| Row bytes | Rows | `Mersenne31` (GiB/s) |
| --- | --- | --- |
| 64 | 1 | 5.17 / 9.44 |
| 64 | 2 | 9.78 / 14.7 |
| 64 | 4 | 16.5 / 21.0 |
| 64 | 8 | 20.2 / 30.3 |
| 64 | 16 | 27.3 / 33.4 |
| 64 | 32 | 26.6 / 36.3 |
| 1024 | 1 | 27.3 / 33.3 |
| 1024 | 2 | 26.6 / 36.4 |
| 1024 | 4 | 28.9 / 34.6 |
| 1024 | 8 | 29.7 / 36.2 |
| 1024 | 16 | 30.3 / 37.1 |
| 1024 | 32 | 29.6 / 35.5 |
| 65536 | 1 | 28.0 / 35.6 |
| 65536 | 2 | 29.7 / 37.5 |
| 65536 | 4 | 26.1 / 36.6 |
| 65536 | 8 | 25.5 / 32.7 |
| 65536 | 16 | 18.5 / 22.7 |
| 65536 | 32 | 18.3 / 21.7 |

## Competitors

Page-aligned 64 KiB regions on each library's native layout; conversions are
excluded. Arms alternate order within one process, and outputs are validated
before timing.

### Scalar

Indexed products with accumulation.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.22 / 1.54 | 2.10 / 0.80 |
| `QuadMersenne31` | `acc += a * b` | 8.98 / 4.38 | 6.98 / 2.90 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 73.3 / 76.8 | 65.5 / 72.7 |
| `Mersenne31` | `dst += c * src` | 55.6 / 61.1 | 49.8 / 57.0 |
| `Mersenne31` | `dst = a * b` | 59.5 / 68.3 | 59.8 / 69.4 |
| `Mersenne31` | `dst += v` | 87.6 / 108 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 70.3 / 86.4 | 103 / 153 |
| `QuadMersenne31` | `dst = c * src` | 31.1 / 37.3 | 30.6 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 22.9 / 28.4 | 27.0 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 27.0 / 31.1 | 24.1 / 26.4 |
| `QuadMersenne31` | `dst += v` | 62.5 / 108 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 49.9 / 86.4 | 103 / 153 |
