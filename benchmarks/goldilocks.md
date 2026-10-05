# Goldilocks

Public API timings for the Goldilocks prime field and the matched Plonky3
panel. Every result cell reports Tiger Lake / Golden Cove; shared hosts,
toolchain, sampling, and number format: [BENCHMARKS.md](../BENCHMARKS.md).
The v2-versus-v3 comparison lives at [v2 comparison](v2/goldilocks.md).

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

Scalar latency is the `scalar a * b` row of the competitor table.

### Packed operations

Ordinary `Vec<u8>` buffers.

| Field | Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- | --- |
| `Goldilocks` | `add_assign` | 33.9 / 21.6 | 27.3 / 22.5 | 8.02 / 21.9 |
| `Goldilocks` | `add_assign_scalar` | 44.2 / 38.9 | 44.4 / 40.8 | 26.2 / 33.7 |
| `Goldilocks` | `sub_assign_scalar` | 59.7 / 40.1 | 52.7 / 40.4 | 28.1 / 33.9 |
| `Goldilocks` | `mul_elementwise` | 10.4 / 10.7 | 11.3 / 11.2 | 5.80 / 10.6 |
| `Goldilocks` | `mul_elementwise_assign` | 11.0 / 10.9 | 11.3 / 11.1 | 7.42 / 9.92 |

## Competitors

Page-aligned 64 KiB regions on each library's native layout; conversions are
excluded. Arms alternate order within one process, and outputs are validated
before timing.

### Scalar

Indexed products with accumulation.

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| `Goldilocks` | `scalar a * b` | 4.48 / 2.05 | 2.57 / 1.11 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| `Goldilocks` | `dst = c * src` | 25.1 / 23.8 | 27.4 / 27.1 |
| `Goldilocks` | `dst += c * src` | 19.7 / 17.0 | 22.5 / 20.6 |
| `Goldilocks` | `dst = a * b` | 23.8 / 23.3 | 26.9 / 26.5 |
| `Goldilocks` | `dst += v` | 97.5 / 81.7 | 103 / 149 |
| `Goldilocks` | `dst -= v` | 103 / 80.9 | 103 / 147 |
