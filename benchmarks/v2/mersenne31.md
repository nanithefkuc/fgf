# Mersenne fields: `m31`, `qm31` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `m31`: `v4x` / `v3`; `qm31`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K): Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## Self-timings

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.5 / 34.9 | 28.4 / 35.9 | 7.95 / 21.3 |
| add_assign_scalar | 40.2 / 52.6 | 41.2 / 52.2 | 24.8 / 22.0 |
| mul_elementwise | 23.2 / 33.3 | 20.3 / 30.0 | 5.43 / 15.4 |
| mul_elementwise_assign | 25.1 / 33.0 | 22.7 / 34.0 | 7.92 / 22.1 |
| sub_assign_scalar | 32.5 / 41.3 | 33.9 / 43.5 | 23.0 / 25.0 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 20.0 / 33.5 | 18.4 / 33.0 | 7.77 / 24.2 |
| add_assign_scalar | 29.8 / 51.7 | 26.0 / 50.6 | 19.1 / 26.9 |
| mul_elementwise | 11.6 / 14.5 | 11.7 / 14.1 | 5.27 / 13.1 |
| mul_elementwise_assign | 7.26 / 15.1 | 7.38 / 14.9 | 6.61 / 13.6 |
| sub_assign_scalar | 23.6 / 42.7 | 21.2 / 43.1 | 16.8 / 33.1 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 26.0 / 33.7 |
| 1 KiB | 16 | 30.5 / 37.3 |
| 1 KiB | 2 | 26.1 / 36.6 |
| 1 KiB | 32 | 29.8 / 33.9 |
| 1 KiB | 4 | 29.0 / 34.6 |
| 1 KiB | 8 | 30.0 / 36.3 |
| 64 B | 1 | 5.55 / 11.7 |
| 64 B | 16 | 26.0 / 33.6 |
| 64 B | 2 | 6.27 / 19.9 |
| 64 B | 32 | 26.1 / 36.6 |
| 64 B | 4 | 17.6 / 25.4 |
| 64 B | 8 | 25.2 / 30.3 |
| 64 KiB | 1 | 28.2 / 35.7 |
| 64 KiB | 16 | 18.7 / 22.5 |
| 64 KiB | 2 | 29.7 / 38.5 |
| 64 KiB | 32 | 17.8 / 21.5 |
| 64 KiB | 4 | 26.1 / 36.6 |
| 64 KiB | 8 | 25.5 / 32.6 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 27.1 / 34.5 |
| add_assign per row | 1 KiB | 16 | 28.7 / 32.7 |
| add_assign per row | 1 KiB | 2 | 28.8 / 35.3 |
| add_assign per row | 1 KiB | 32 | 28.1 / 31.7 |
| add_assign per row | 1 KiB | 4 | 28.9 / 33.0 |
| add_assign per row | 1 KiB | 8 | 29.2 / 33.0 |
| add_assign per row | 64 B | 1 | 6.14 / 11.6 |
| add_assign per row | 64 B | 16 | 8.29 / 15.8 |
| add_assign per row | 64 B | 2 | 6.04 / 13.5 |
| add_assign per row | 64 B | 32 | 8.35 / 15.4 |
| add_assign per row | 64 B | 4 | 8.20 / 14.7 |
| add_assign per row | 64 B | 8 | 8.75 / 15.4 |
| add_assign per row | 64 KiB | 1 | 28.2 / 35.7 |
| add_assign per row | 64 KiB | 16 | 18.6 / 22.5 |
| add_assign per row | 64 KiB | 2 | 29.6 / 38.4 |
| add_assign per row | 64 KiB | 32 | 17.8 / 21.5 |
| add_assign per row | 64 KiB | 4 | 26.0 / 36.4 |
| add_assign per row | 64 KiB | 8 | 25.4 / 32.6 |
| add_assign | 1 KiB | 1 | 30.4 / 34.5 |
| add_assign | 1 KiB | 16 | 30.7 / 37.4 |
| add_assign | 1 KiB | 2 | 27.8 / 36.9 |
| add_assign | 1 KiB | 32 | 29.9 / 35.3 |
| add_assign | 1 KiB | 4 | 29.5 / 34.9 |
| add_assign | 1 KiB | 8 | 30.3 / 36.6 |
| add_assign | 64 B | 1 | 7.40 / 12.1 |
| add_assign | 64 B | 16 | 30.5 / 34.2 |
| add_assign | 64 B | 2 | 6.31 / 20.0 |
| add_assign | 64 B | 32 | 27.7 / 36.8 |
| add_assign | 64 B | 4 | 21.0 / 27.0 |
| add_assign | 64 B | 8 | 28.1 / 31.2 |
| add_assign | 64 KiB | 1 | 28.2 / 35.3 |
| add_assign | 64 KiB | 16 | 18.7 / 22.5 |
| add_assign | 64 KiB | 2 | 29.7 / 38.5 |
| add_assign | 64 KiB | 32 | 17.8 / 21.5 |
| add_assign | 64 KiB | 4 | 26.1 / 36.6 |
| add_assign | 64 KiB | 8 | 25.5 / 32.4 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.740 / 1.640 | 2.570 / 1.110 |
| m31 | scalar a * b | 4.219 / 1.546 | 2.107 / 0.801 |
| qm31 | scalar a * b | 9.003 / 4.365 | 6.963 / 2.908 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.9 | 22.5 / 20.6 |
| gld | dst += v | 97.5 / 81.6 | 103 / 148 |
| gld | dst -= v | 103 / 80.9 | 103 / 147 |
| gld | dst = a * b | 23.8 / 23.3 | 26.9 / 26.4 |
| gld | dst = c * src | 25.1 / 23.2 | 27.4 / 27.1 |
| m31 | dst += c * src | 55.6 / 61.1 | 49.8 / 57.1 |
| m31 | dst += v | 87.6 / 108 | 103 / 153 |
| m31 | dst -= v | 70.3 / 86.4 | 103 / 153 |
| m31 | dst = a * b | 59.5 / 68.2 | 59.8 / 69.3 |
| m31 | dst = c * src | 73.1 / 76.8 | 65.4 / 72.9 |
| qm31 | dst += c * src | 22.9 / 28.3 | 27.0 / 31.7 |
| qm31 | dst += v | 62.6 / 108 | 103 / 153 |
| qm31 | dst -= v | 49.9 / 86.4 | 103 / 153 |
| qm31 | dst = a * b | 27.0 / 31.1 | 24.1 / 26.2 |
| qm31 | dst = c * src | 32.1 / 37.6 | 30.5 / 36.1 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
