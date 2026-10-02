# Goldilocks

Public API timings for the Goldilocks prime field and the matched Plonky3
panel. Every result cell reports Tiger Lake / Golden Cove; shared hosts,
toolchain, sampling, and number format: [BENCHMARKS.md](../BENCHMARKS.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-gdl-comp` |
| Resolved backend | `v4x` / `v3` |
| Competitor | Plonky3 0.7.0 |
| Competitor build | `external/prime-bench` with `fgf/simd512` and `-C target-cpu=native` |
| Plonky3 packing | 8 / 4 elements |
| Throughput | Self-timings count buffer bytes. Competitor cells count one operand pair, twice the region bytes. |

## Self-timings

Scalar latency is the `acc += a * b` row of the competitor table.

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Goldilocks` | `add_assign` | 30.0 / 21.9 | 27.3 / 22.5 | 8.23 / 21.8 |
| `Goldilocks` | `add_assign_scalar` | 44.2 / 39.3 | 44.4 / 40.8 | 25.9 / 33.7 |
| `Goldilocks` | `sub_assign_scalar` | 59.6 / 40.6 | 52.7 / 40.4 | 27.8 / 33.9 |
| `Goldilocks` | `mul_elementwise` | 10.8 / 10.6 | 11.3 / 11.3 | 5.65 / 10.6 |
| `Goldilocks` | `mul_elementwise_assign` | 11.0 / 11.1 | 11.3 / 11.3 | 7.27 / 10.0 |

## Competitors

Page-aligned 64 KiB regions on each library's native layout; conversions are
excluded. Arms alternate order within one process, and outputs are validated
before timing.

### Scalar

Indexed products with accumulation.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Goldilocks` | `acc += a * b` | 3.74 / 1.64 | 2.57 / 1.11 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Goldilocks` | `dst = c * src` | 25.1 / 23.8 | 27.5 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17.0 | 22.5 / 20.7 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.0 | 26.9 / 26.4 |
| `Goldilocks` | `dst += v` | 97.5 / 81.5 | 103 / 148 |
| `Goldilocks` | `dst -= v` | 103 / 81.0 | 103 / 148 |
