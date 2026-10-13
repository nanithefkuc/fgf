# Mersenne fields: `m31`, `qm31` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v2](../v2/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| add_assign | 29.3 / 33.9 | 28.2 / 34.4 | 8.23 / 20.5 |
| add_assign_scalar | 45.3 / 54.8 | 44.7 / 52.0 | 25.1 / 21.1 |
| mul_elementwise | 14.5 / 18.3 | 13.5 / 17.8 | 5.30 / 16.7 |
| mul_elementwise_assign | 15.2 / 19.4 | 15.0 / 19.3 | 7.61 / 19.3 |
| sub_assign_scalar | 45.7 / 54.9 | 44.9 / 52.0 | 25.2 / 21.1 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.0 / 32.2 | 18.6 / 31.2 | 7.92 / 24.2 |
| add_assign_scalar | 33.5 / 53.4 | 28.4 / 50.9 | 20.0 / 24.5 |
| mul_elementwise | 10.3 / 13.0 | 10.3 / 12.4 | 5.20 / 12.5 |
| mul_elementwise_assign | 6.73 / 13.1 | 6.85 / 13.1 | 6.26 / 12.2 |
| sub_assign_scalar | 32.9 / 53.4 | 28.2 / 50.9 | 20.0 / 24.5 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 26.0 / 32.3 |
| 1 KiB | 16 | 30.4 / 34.5 |
| 1 KiB | 2 | 26.2 / 35.1 |
| 1 KiB | 32 | 29.5 / 32.7 |
| 1 KiB | 4 | 29.3 / 33.8 |
| 1 KiB | 8 | 29.8 / 34.2 |
| 64 B | 1 | 5.77 / 11.6 |
| 64 B | 16 | 26.0 / 32.4 |
| 64 B | 2 | 10.4 / 19.2 |
| 64 B | 32 | 26.2 / 35.0 |
| 64 B | 4 | 17.9 / 24.7 |
| 64 B | 8 | 25.1 / 29.2 |
| 64 KiB | 1 | 27.9 / 32.9 |
| 64 KiB | 16 | 18.7 / 21.5 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 17.9 / 20.9 |
| 64 KiB | 4 | 26.1 / 35.0 |
| 64 KiB | 8 | 25.3 / 31.2 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 27.1 / 33.1 |
| add_assign per row | 1 KiB | 16 | 28.9 / 31.5 |
| add_assign per row | 1 KiB | 2 | 28.7 / 33.8 |
| add_assign per row | 1 KiB | 32 | 28.3 / 30.6 |
| add_assign per row | 1 KiB | 4 | 28.7 / 31.7 |
| add_assign per row | 1 KiB | 8 | 29.0 / 31.8 |
| add_assign per row | 64 B | 1 | 6.14 / 11.2 |
| add_assign per row | 64 B | 16 | 8.70 / 15.2 |
| add_assign per row | 64 B | 2 | 7.42 / 13.0 |
| add_assign per row | 64 B | 32 | 8.71 / 14.7 |
| add_assign per row | 64 B | 4 | 8.29 / 14.1 |
| add_assign per row | 64 B | 8 | 8.80 / 14.8 |
| add_assign per row | 64 KiB | 1 | 28.0 / 33.0 |
| add_assign per row | 64 KiB | 16 | 18.6 / 21.3 |
| add_assign per row | 64 KiB | 2 | 29.5 / 35.9 |
| add_assign per row | 64 KiB | 32 | 17.9 / 20.8 |
| add_assign per row | 64 KiB | 4 | 26.0 / 33.9 |
| add_assign per row | 64 KiB | 8 | 25.2 / 31.5 |
| add_assign | 1 KiB | 1 | 28.4 / 33.1 |
| add_assign | 1 KiB | 16 | 30.5 / 34.5 |
| add_assign | 1 KiB | 2 | 27.3 / 35.3 |
| add_assign | 1 KiB | 32 | 29.7 / 32.8 |
| add_assign | 1 KiB | 4 | 29.3 / 33.7 |
| add_assign | 1 KiB | 8 | 30.1 / 34.2 |
| add_assign | 64 B | 1 | 7.41 / 12.7 |
| add_assign | 64 B | 16 | 28.4 / 33.1 |
| add_assign | 64 B | 2 | 12.3 / 19.2 |
| add_assign | 64 B | 32 | 27.3 / 35.3 |
| add_assign | 64 B | 4 | 21.6 / 26.0 |
| add_assign | 64 B | 8 | 29.9 / 30.0 |
| add_assign | 64 KiB | 1 | 28.1 / 33.9 |
| add_assign | 64 KiB | 16 | 18.7 / 21.3 |
| add_assign | 64 KiB | 2 | 29.7 / 36.1 |
| add_assign | 64 KiB | 32 | 17.9 / 20.7 |
| add_assign | 64 KiB | 4 | 26.1 / 35.2 |
| add_assign | 64 KiB | 8 | 25.3 / 31.4 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.478 / 2.149 | 2.564 / 1.162 |
| m31 | scalar a * b | 4.181 / 1.612 | 2.092 / 0.881 |
| qm31 | scalar a * b | 16.13 / 6.768 | 7.046 / 3.037 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.8 |
| gld | dst += v | 97.5 / 78.4 | 103 / 141 |
| gld | dst -= v | 103 / 77.8 | 103 / 141 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.5 / 26.0 |
| m31 | dst += c * src | 32.9 / 36.0 | 49.8 / 54.7 |
| m31 | dst += v | 98.5 / 117 | 103 / 147 |
| m31 | dst -= v | 98.5 / 116 | 103 / 147 |
| m31 | dst = a * b | 34.8 / 39.5 | 59.9 / 66.5 |
| m31 | dst = c * src | 50.4 / 51.4 | 65.4 / 69.7 |
| qm31 | dst += c * src | 21.4 / 24.2 | 27.0 / 30.5 |
| qm31 | dst += v | 68.1 / 117 | 103 / 147 |
| qm31 | dst -= v | 67.7 / 116 | 103 / 147 |
| qm31 | dst = a * b | 23.9 / 27.4 | 24.1 / 25.3 |
| qm31 | dst = c * src | 29.2 / 32.8 | 30.6 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
