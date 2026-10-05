# Mersenne31 and `QuadMersenne31`

Public API timings for the Mersenne prime field, its quadratic extension, and
the matched Plonky3 panel. Every result cell reports Tiger Lake / Golden Cove;
shared hosts, toolchain, sampling, and number format:
[BENCHMARKS.md](../BENCHMARKS.md). The v2-versus-v3 comparison lives in
[v2 comparison](v2/mersenne31.md).

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
| `Mersenne31` | `add_assign` | 29.7 / 36.1 | 22.2 / 37.0 | 8.24 / 21.1 |
| `Mersenne31` | `add_assign_scalar` | 46.2 / 60.5 | 44.7 / 61.6 | 25.1 / 22.0 |
| `Mersenne31` | `sub_assign_scalar` | 46.3 / 60.6 | 44.9 / 61.5 | 25.0 / 22.0 |
| `Mersenne31` | `mul_elementwise` | 14.3 / 17.6 | 14.2 / 18.1 | 5.86 / 17.6 |
| `Mersenne31` | `mul_elementwise_assign` | 15.1 / 19.9 | 15.1 / 20.2 | 7.98 / 19.0 |
| `QuadMersenne31` | `add_assign` | 20.4 / 35.8 | 17.1 / 34.6 | 7.78 / 24.5 |
| `QuadMersenne31` | `add_assign_scalar` | 33.9 / 60.7 | 34.0 / 61.8 | 20.0 / 25.6 |
| `QuadMersenne31` | `sub_assign_scalar` | 33.3 / 60.7 | 33.8 / 62.0 | 19.9 / 25.6 |
| `QuadMersenne31` | `mul_elementwise` | 10.3 / 13.2 | 10.5 / 12.8 | 5.25 / 13.0 |
| `QuadMersenne31` | `mul_elementwise_assign` | 6.74 / 13.7 | 6.70 / 13.8 | 6.31 / 12.5 |

### Row-wise addition

`Mersenne31` `add_assign_rows`, ordinary `Vec<u8>` buffers.

| Row bytes | Rows | `Mersenne31` (GiB/s) |
| --- | --- | --- |
| 64 | 1 | 5.25 / 9.44 |
| 64 | 2 | 9.61 / 15.4 |
| 64 | 4 | 14.2 / 21.1 |
| 64 | 8 | 19.2 / 29.5 |
| 64 | 16 | 26.4 / 34.2 |
| 64 | 32 | 26.3 / 36.6 |
| 1024 | 1 | 26.2 / 34.2 |
| 1024 | 2 | 26.4 / 36.8 |
| 1024 | 4 | 28.7 / 35.9 |
| 1024 | 8 | 30.2 / 37.6 |
| 1024 | 16 | 31.0 / 38.4 |
| 1024 | 32 | 29.2 / 36.6 |
| 65536 | 1 | 27.5 / 37.0 |
| 65536 | 2 | 26.7 / 36.5 |
| 65536 | 4 | 26.1 / 36.7 |
| 65536 | 8 | 25.6 / 33.3 |
| 65536 | 16 | 18.7 / 22.3 |
| 65536 | 32 | 18.4 / 21.5 |

## Competitors

Page-aligned 64 KiB regions on each library's native layout; conversions are
excluded. Arms alternate order within one process, and outputs are validated
before timing.

### Scalar

Indexed products with accumulation.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Mersenne31` | `acc += a * b` | 4.21 / 1.54 | 2.10 / 0.80 |
| `QuadMersenne31` | `acc += a * b` | 9.70 / 4.95 | 6.96 / 2.91 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Mersenne31` | `dst = c * src` | 50.4 / 53.6 | 65.7 / 72.7 |
| `Mersenne31` | `dst += c * src` | 32.9 / 37.6 | 49.8 / 57.0 |
| `Mersenne31` | `dst = a * b` | 34.9 / 41.2 | 59.9 / 69.4 |
| `Mersenne31` | `dst += v` | 98.5 / 122 | 103 / 153 |
| `Mersenne31` | `dst -= v` | 98.5 / 122 | 103 / 153 |
| `QuadMersenne31` | `dst = c * src` | 29.2 / 34.2 | 30.6 / 36.1 |
| `QuadMersenne31` | `dst += c * src` | 21.4 / 25.3 | 27.0 / 31.8 |
| `QuadMersenne31` | `dst = a * b` | 23.9 / 28.5 | 24.1 / 26.2 |
| `QuadMersenne31` | `dst += v` | 68.2 / 122 | 103 / 153 |
| `QuadMersenne31` | `dst -= v` | 67.8 / 122 | 103 / 153 |
